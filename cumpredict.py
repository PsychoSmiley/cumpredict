#!/usr/bin/env python3
"""cumpredict -- live arousal prediction + an exploring edging policy for a Bluetooth toy.

PREDICTOR and DIRECTOR are separate and never merged, and no predictor output enters the
Director's reward -- a policy scored by the perception it is steering learns to hack it.
Predictor: grey-box ODE, ~10 params, fit offline by CEM, pure inference during a session; a
recurrence, not a buffer, so there is no window to slide. Director: policy over (stroke_lo,
stroke_hi, half_stroke_s), exploring online. Keypresses are the only ground truth: they SCORE
every cycle (its peak press, less a cum), correct the belief about predictor bias, and drive
cycle accounting (a 9 closes the cycle, applies REFRACTORY, and under --finish-on-cum ends the
session); they never issue a device command. A cycle he never rated is missing, not zero -- unless
it hit a deadline, which is an outcome the clock reports and he does not. Presses are EDGE-
triggered: a key held across several commands is one press, not one per command.

Modality is detected from the device. Linear: rate = 1/dt is a real stroke rate. Vibrator:
"hard" is a level HELD, not swung, so rate is a constant and amplitude carries the drive --
those runs are UNCALIBRATED (`a` and `g` were fit on stroke depth) and record as VIB_*.csv,
which `train` refuses to load alongside linear recordings (see training_sessions).

No safety shield: the policy is supposed to risk orgasm, and a shield exists to prevent one. The
only brakes are CUM_COST, TIMEOUT_COST and the 60s cold-start ramp limiter in the live loop.
Ctrl-C stops the toy and keeps every row already written.

Data: [session,time_elapse,intensity,tired_level]; time_elapse = MEASURED seconds the command
occupied /60 (perf_counter, send latency included -- not the duration that was requested, which is
logged beside it as cmd_dur), intensity = the travel actually commanded, |target - previous
target|, tired_level = sparse 0..1 label, 1.0 = cum, blank where he said nothing. The model is
personal and not shipped -- refit from sessions.csv. Recordings written before this revision hold
the REQUESTED duration and the nominal hi-lo in those two columns; they differ by the BLE send
latency and by whatever the actuator was doing at a cycle boundary.
Deps: live  numpy + buttplug-py==0.3.0 keyboard
      train numpy pandas torch [scipy]     -- `live` imports none of the train-only ones
"""
import argparse, glob, hashlib, json, math, os, sys, time
from contextlib import suppress
import numpy as np
# pandas is imported INSIDE the two loaders that need it. It costs 292 ms and is
# reachable only from `train`; `live` must not pay for the trainer's dependencies,
# which is the same rule that keeps torch out of the session path.

BETA, PRESS_BOOST = 4.0, 8.0          # label weighting: near-cum emphasis, real-press boost
DATA = "sessions.csv"                 # ONE corpus: `live` appends to it, `train` reads it. A
                                      # vibrator run goes to a VIB_-prefixed sibling instead --
                                      # its intensity is a held amplitude, not a stroke travel,
                                      # and one file holding both fits one exponent to two toys.
CKPT = "checkpoint"                   # model, policy, eval and logs
CONFIG, POLICY = CKPT + "/model.json", CKPT + "/policy.json"
VIB_RATE_REF = 2.0                    # drive rate for a vibrator; placeholder, unfitted
PARK_S = 0.5                          # slow move to a known position before anything is recorded

def _interp_on_rise(lab, t_s):
    """Sparse labels -> dense target: ramp UP between rising presses, step DOWN."""
    T = len(lab)
    press_idx = np.where(~np.isnan(lab))[0]
    y, is_press = np.zeros(T), np.zeros(T, dtype=bool)
    if not len(press_idx):
        return y.astype(np.float32), is_press, press_idx, np.array([])
    is_press[press_idx] = True
    vals = lab[press_idx]
    pi = np.r_[0, press_idx] if press_idx[0] else press_idx   # a virtual 0-press at t=0 IS the
    pv = np.r_[0.0, vals] if press_idx[0] else vals            # loop's rising branch, not a case
    for j, (a, va) in enumerate(zip(pi, pv)):
        b, vb = (pi[j + 1], pv[j + 1]) if j + 1 < len(pi) else (T - 1, va)
        y[a] = va
        if vb > va:
            y[a:b + 1] = va + (vb - va) * (t_s[a:b + 1] - t_s[a]) / max(t_s[b] - t_s[a], 1e-9)
        else:
            y[a:b], y[b] = va, vb
    return np.clip(y, 0, 1).astype(np.float32), is_press, press_idx, vals

def session_from_raw(name, dur_s, intensity, tired):
    dur_s = np.clip(np.asarray(dur_s, float), 1e-3, None)
    intensity = np.clip(np.asarray(intensity, float), 0, 1)
    tired = np.asarray(tired, float)
    t_s = np.cumsum(dur_s)
    y, is_press, press_idx, press_vals = _interp_on_rise(tired, t_s)
    w = dur_s * (1.0 + BETA * y)
    w[is_press] *= PRESS_BOOST
    return {"name": name, "intensity": intensity.astype(np.float32), "y": y,
            "w": w.astype(np.float32), "is_press": is_press, "dur_s": dur_s.astype(np.float32),
            "t_s": t_s, "press_idx": press_idx,
            "press_vals": press_vals.astype(np.float32), "raw_dur_s": dur_s.copy(),
            "raw_intensity": intensity.copy(), "raw_tired": tired.copy()}

def _from_frame(name, df):
    import pandas as pd
    df.columns = [c.strip() for c in df.columns]
    for c in ("time_elapse", "intensity", "tired_level"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["time_elapse", "intensity"]).reset_index(drop=True)
    if len(df) < n0:
        print(f"  {name}: dropped {n0 - len(df)} unparseable row(s) -- a stray header or a torn "
              f"line means this recording was concatenated out of more than one file. Check it "
              f"for a replayed block; one is how a duplicate got into this corpus unnoticed.")
    # press_vals is returned UNCLIPPED, so this is the only place a mis-scaled column is caught.
    # A file on the raw 0-9 scale makes cum_onsets read a 7 as an orgasm and press_metrics report
    # an MAE of 8 against a predictor bounded in [0,1], while the band table silently drops every
    # row -- three wrong answers and no error. The column already holds hand-typed values off the
    # k/9 grid, so it does get edited by hand.
    lab = df["tired_level"].dropna()
    lab = lab[(lab < 0.0) | (lab > 1.0)]
    if len(lab):
        sys.exit(f"{name}: {len(lab)} tired_level values outside [0,1] (e.g. "
                 f"{[round(float(v), 3) for v in lab[:5]]}) -- the column is his keypress divided "
                 f"by 9, not the raw digit.")
    return session_from_raw(name, df["time_elapse"].to_numpy(float) * 60.0,
                            df["intensity"].to_numpy(float), df["tired_level"].to_numpy(float))

def load_sessions(path):
    import pandas as pd
    if os.path.isdir(path):
        return [_from_frame(os.path.basename(p)[:-4], pd.read_csv(p))
                for p in sorted(glob.glob(os.path.join(path, "*.csv")))]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "session" not in df.columns:
        return [_from_frame(os.path.splitext(os.path.basename(path))[0], df)]
    return [_from_frame(str(n), g.drop(columns=["session"]))
            for n, g in sorted(df.groupby("session", sort=False), key=lambda kv: kv[0])]

def replayed_rows(sess, w=128):
    """First window of `w` consecutive rows that appears twice in one recording, as (i, j).

    12.3% of this corpus was once a 1567-row block pasted in twice, and the only fingerprint was a
    stray CSV header that the loader silently dropped. It survived nine months and six review
    rounds; it inflated the headline LOSO number by 18% and hid how fragile the fit underneath it
    was. A rolling hash over the whole corpus costs 16 ms.

    Non-overlapping windows only (`i - j >= w`). A held command repeats its own row at shift 1 for
    as long as he leaves it there -- one real session holds 181 identical rows, the Keon pinned at
    its 0.13 s floor -- and that is data, not a paste."""
    n = len(sess["raw_dur_s"])
    if n < 2 * w:
        return None
    rows = list(zip(sess["raw_dur_s"].tolist(), sess["raw_intensity"].tolist(),
                    np.nan_to_num(sess["raw_tired"], nan=-1.0).tolist()))
    h = [hash(r) & 0xFFFFFFFFFFFFFFFF for r in rows]
    B, M = 1000003, (1 << 64) - 59
    pw, cur, seen = pow(B, w - 1, M), 0, {}
    for k in range(w):
        cur = (cur * B + h[k]) % M
    seen[cur] = 0
    for i in range(1, n - w + 1):
        cur = ((cur - h[i - 1] * pw) * B + h[i + w - 1]) % M
        j = seen.get(cur)
        if j is not None and i - j >= w and rows[j:j + w] == rows[i:i + w]:
            return j, i
        seen.setdefault(cur, i)
    return None


def training_sessions(path):
    """load_sessions plus the two gates that only matter when the rows become training evidence.
    Both are about training evidence only; `live` records whatever happened.

    A session with no keypress at all interpolates to y=0 everywhere and STILL carries full
    duration weight, so it enters the fit as dense evidence that arousal was zero all session --
    the one thing the Director's "missing, not zero" rule exists to forbid. Drop it, loudly.
    A vibrator run is uncalibrated against `a`/`g` (fit on stroke depth) and its `intensity` is a
    held amplitude, not a travel; mixing VIB_* with LIVE_* fits one exponent to two actuators."""
    sessions = load_sessions(path)
    # Derived from the session NAMES, so it covers a single file as well as a directory. It used
    # to be gated on os.path.isdir, which meant it never ran on the default single-file corpus --
    # the one `live` appends to and the only one anyone uses. A corpus this file demonstrably gets
    # built by concatenation is exactly where a VIB_ run can end up filed beside linear ones.
    kinds = {("vib" if s["name"].upper().startswith("VIB") else "linear") for s in sessions}
    if len(kinds) > 1:
        sys.exit(f"{path} mixes VIB_* and linear recordings -- a vibrator's intensity is a "
                 f"held amplitude, not a stroke travel. Train them from separate directories.")
    keep = []
    for s in sessions:
        dup = replayed_rows(s)
        if dup:
            sys.exit(f"{s['name']}: rows {dup[0]} and {dup[1]} begin identical blocks -- this "
                     f"recording contains a REPLAY, not two similar stretches. Training on it "
                     f"counts the same evidence twice and inflates every held-out number. Repair "
                     f"the file before training.")
        if len(s["press_idx"]):
            keep.append(s)
        else:
            print(f"  skipping {s['name']}: no keypresses -- unrated is missing, not zero")
    return keep

def augment_raw(sess, rng):
    """Labels untouched; keeps the true session start (the causal integrator must never see a
    cold start mid-session) and time-warps to decorrelate stroke rate from outcome."""
    dur, inten, tired = sess["raw_dur_s"].copy(), sess["raw_intensity"].copy(), sess["raw_tired"].copy()
    if rng.random() < 0.5:
        # Never truncate past the FIRST keypress. training_sessions refuses a press-less session
        # because it interpolates to y=0 everywhere while carrying full duration weight -- dense
        # evidence that arousal was zero all session. Augmented copies never pass through that
        # gate, so a session whose first press lands past 60% of its rows can produce one here:
        # measured 181 of 500 seeds on a synthetic case. Not reachable on today's corpus (the
        # latest first press is at 22.4%) and one late-starting evening away from it.
        lo = max(int(0.6 * len(dur)), int(sess["press_idx"][0]) + 1)
        k = int(rng.integers(lo, len(dur) + 1))
        dur, inten, tired = dur[:k], inten[:k], tired[:k]
    inten = np.clip(inten + rng.uniform(-0.05, 0.05, len(inten)), 0, 1)
    dur = dur * np.exp(rng.uniform(-0.10, 0.10, len(dur)))
    dur = dur * float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    return session_from_raw(sess["name"] + "_aug", dur, inten, tired)

# Closed-form SEQUENTIAL update per stroke. Each line is the exact solution of its own linear ODE
# with the other states held over dt, applied in order -- an operator split, not the exact flow of
# the coupled system: de reads the already-updated E, H_ss the already-updated A. It is therefore
# not a semigroup (one 2.5s step differs from two 1.25s ones by up to ~8e-3 in A at the fitted
# params), which is fine while command cadence stays in the range it was fit on, and is a real
# caveat if that cadence ever changes. This recurrence, not the coupled ODE, IS the model.
#   P  = intensity^a * rate^b                     stimulation power
#   E <- E + (1-exp(-dt/tauE))*(P-E)              fast excitation
#   de = softplus(g*(E - thr - kH*H))             drive excess above threshold
#   A <- A_ss + (A-A_ss)*exp(-(rho*de+lam)*dt)    slow arousal,      A_ss = rho*de/(rho*de+lam)
#   H <- H_ss + (H-H_ss)*exp(-dH*dt)              slow habituation,  H_ss = hH*A/dH
# Below threshold de~0 and arousal only decays, so "slow == safe" comes from the structure
# rather than from having enough negative sessions.
CORE_NAMES = ["a", "b", "tauE", "g", "thr", "rho", "lam", "kH", "hH", "dH"]
CORE_BOUNDS = np.array([
    [1.00, 3.0],    # a     >=1 so stroke depth really modulates drive
    [0.20, 1.5],    # b     capped; b~3 let one stroke max out arousal
    [10.0, 60.0],   # tauE  floored so response isn't instant
    [0.20, 10.0],   # g
    [0.00, 2.5],    # thr
    [1e-4, 0.02],   # rho   capped so buildup stays gradual
    [0.015, 0.06],  # lam   floored so backing off actually lowers arousal
    [0.00, 5.0],    # kH
    [0.00, 0.05],   # hH    fits to 0 on this data -> no state slower than ~6min
    [1e-4, 0.02],   # dH
])

def softplus(x):
    return np.logaddexp(0.0, x)

def ode(E, A, H, P, dt, tauE, g, thr, rho, lam, kH, hH, dH):
    """One closed-form sequential step: the REFERENCE form of the recurrence, and no longer the
    only one. core_scan calls it, and invariants' vibrator arm checks against it directly.

    Two other copies of this math exist and a change here must be mirrored in BOTH: fit_core's
    eval_pop inlines it over preallocated buffers, and CoreStreamer.step writes it out again in
    libm scalars. Only the streamer's copy is gated -- invariants() measures streaming-vs-batch
    parity on every export and refuses to write a model unless it is exactly 0.0. eval_pop's copy
    is guarded by nothing but the fit staying byte-reproducible."""
    E = E + (1.0 - np.exp(-dt / tauE)) * (P - E)
    de = softplus(g * (E - thr - kH * H)); k = rho * de + lam; A_ss = rho * de / k
    A = A_ss + (A - A_ss) * np.exp(-k * dt); H_ss = hH * A / dH
    return E, A, H_ss + (H - H_ss) * np.exp(-dH * dt)

def core_scan(intensity, rate, dt, a, b, *core):
    """intensity/rate/dt: [B,T]; params: [B]. -> A: [B,T]."""
    P = np.clip(intensity, 1e-4, 1.0) ** a.reshape(-1, 1) * np.clip(rate, 1e-4, None) ** b.reshape(-1, 1)
    B, T = intensity.shape
    E = np.zeros(B); A = np.zeros(B); H = np.zeros(B)
    Ao = np.empty((B, T))
    for t in range(T):
        E, A, H = ode(E, A, H, P[:, t], dt[:, t], *core)
        Ao[:, t] = A
    return Ao

class CoreStreamer:
    """Online core, bit-identical to core_scan; ~10 lines to port to JS. rate_ref: None for a
    stroker (1/dt is a real stroke rate); a constant for a vibrator, where dt is only how long a
    level is held so 1/dt would measure nothing but the re-send cadence."""
    def __init__(self, params, rate_ref=None):
        self.__dict__.update(zip(CORE_NAMES, [params[k] for k in CORE_NAMES]
                                 if isinstance(params, dict) else list(params)))
        self.rate_ref = rate_ref
        self.E = self.A = self.H = 0.0
        self._core = tuple(getattr(self, k) for k in CORE_NAMES[2:])   # fixed for this
                                                                       # streamer's lifetime

    def refract(self):
        """The post-orgasm dip, applied to both states at once -- the live loop and the probe
        must dip his body by the same rule, so there is one place that says what the rule is."""
        self.A *= REFRACTORY
        self.E *= REFRACTORY

    def step(self, intensity, dt):
        intensity = min(max(float(intensity), 1e-4), 1.0)
        dt = max(float(dt), 1e-3)
        rate = self.rate_ref if self.rate_ref is not None else 1.0 / dt
        P = intensity ** self.a * max(rate, 1e-4) ** self.b
        # ode() written out in libm scalars: 3.30 us -> 1.25 us, and this runs 5.6M times per
        # train. Every np.exp here was a ufunc dispatch on a single float. Bit-identical, checked
        # rather than assumed -- np.exp vs math.exp and np.logaddexp(0,x) vs the branched log1p
        # agree to 0 ulp over 8M draws spanning the ranges CORE_BOUNDS allows. Every exponent is
        # provably <= 0 there (tauE>=10, k>=lam>=0.015, dH>=1e-4, and the softplus branch always
        # hands exp a non-positive argument), so math.exp cannot raise where np.exp returned inf.
        #
        # This does make the streamer a second written copy of ode(). Unlike eval_pop's copy, it
        # is checked on EVERY export: invariants() measures streaming-vs-batch parity and refuses
        # to write a model unless it is exactly 0.0.
        tauE, g, thr, rho, lam, kH, hH, dH = self._core
        E = self.E + (1.0 - math.exp(-dt / tauE)) * (P - self.E)
        x = g * (E - thr - kH * self.H)
        de = x + math.log1p(math.exp(-x)) if x > 0.0 else math.log1p(math.exp(x))
        k = rho * de + lam
        A_ss = rho * de / k
        A = A_ss + (self.A - A_ss) * math.exp(-k * dt)
        H_ss = hH * A / dH
        self.E, self.A, self.H = E, A, H_ss + (self.H - H_ss) * math.exp(-dH * dt)
        return min(max(float(A), 0.0), 1.0)   # scalar: np.clip here was a ufunc dispatch

CEM_ITERS, CEM_POP, CEM_ELITE, CEM_RESTARTS = 40, 80, 0.15, 2
CEM_CHUNKS = (80, 40, 20, 16, 10, 8, 5, 4, 2, 1)   # divisors of CEM_POP; see chunk_for
FIT_MEM_TARGET = 1.0e9   # bytes of scan buffer one fit_core aims to stay under; see chunk_for
LOSO_MEM_BUDGET = 12e9   # bytes of scan buffer the whole LOSO pool may hold at once


def fit_core_bytes(n_sessions, n_rows, T, pop=CEM_POP):
    """Peak scan-buffer footprint of one fit_core, measured against tracemalloc to within 2.5% at
    9/18/27/36 sessions.

    Ao is still allocated at the PADDED width, so the single longest recording in the corpus sets
    that term for every short session after it; the packed terms are charged per live row instead.
    Either way it grows ~18 MB per SESSION at full population, which is why cpu_count() folds of it
    is what runs the box out of memory rather than out of time.

    Recheck this against tracemalloc if eval_pop's buffers ever change -- it is only useful while it
    describes buffers that exist, and the previous version outlived its by 1.8x."""
    n = 3 * n_sessions
    return 8 * (pop * n * T          # Ao, the one buffer still padded
                + 12 * n_rows * pop  # Pf/Al/Eh/Df: the state-free block, packed to live rows only
                + 2 * pop * T        # pb/ev, the loss accumulator -- was (S,n,T), two 96 MB slabs
                + 10 * n * T)        # pad()'s five grids, their reordered copies, ys/ws


def chunk_for(n_sessions, n_rows, T):
    """Largest population slice whose buffers fit FIT_MEM_TARGET -- a divisor of CEM_POP, because
    eval_pop's buffers are allocated once at the first width they see and a ragged final slice
    would not fit them.

    Adaptive rather than fixed because the two costs pull opposite ways and swap over. Slicing
    costs fixed Python per extra pass over T -- 13.6 s to 22.4 s on today's 9-session fit, where
    memory was never scarce -- while NOT slicing at 365 sessions means a fold wants 8.5 GB and the
    pool drops to one worker. So: full population until the corpus is big enough to care.

    The fit is unaffected either way, which is the only reason this is allowed to depend on the
    machine at all: the two reductions that pick the elite set run over T and over n, and slicing
    touches neither axis' length nor its layout, so numpy's pairwise summation sees identical
    shapes. Verified bit-identical -- not close, equal -- at every divisor from 80 down to 2."""
    # Every candidate must divide CEM_POP: eval_pop's buffers are sized once at the first width
    # they see, so a ragged final slice is a broadcast error 20 minutes into a train rather than
    # at import. Asserted here so changing CEM_POP cannot quietly invalidate this list.
    assert all(CEM_POP % c == 0 for c in CEM_CHUNKS) and 1 in CEM_CHUNKS, (
        f"CEM_CHUNKS {CEM_CHUNKS} must all divide CEM_POP={CEM_POP} and must contain 1 -- "
        f"without it this loop falls off the end and returns None, which becomes a slice width")
    for c in CEM_CHUNKS:
        if c == 1 or fit_core_bytes(n_sessions, n_rows, T, c) <= FIT_MEM_TARGET:
            return c


N_AUG, REST_PENALTY = 2, 3.0

def cem(bounds, eval_pop, rng, iters=CEM_ITERS, pop=CEM_POP, elite=CEM_ELITE, restarts=CEM_RESTARTS):
    """Gradient-free: autograd through a 5000-step scan is pathologically slow on this box."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    n = len(lo)
    best, best_loss, k = None, np.inf, max(3, int(pop * elite))
    for _ in range(restarts):
        mean = np.clip((lo + hi) / 2 + 0.25 * (hi - lo) * rng.standard_normal(n), lo, hi)
        std = (hi - lo) / 4.0
        for _ in range(iters):
            P = np.clip(mean + std * rng.standard_normal((pop, n)), lo, hi)
            if best is not None:
                P[0] = best
            L = eval_pop(P)
            idx = np.argsort(L)[:k]
            mean, std = P[idx].mean(0), P[idx].std(0) + 1e-3 * (hi - lo)
            if L[idx[0]] < best_loss:
                best_loss, best = L[idx[0]], P[idx[0]].copy()
    return best, best_loss

def pad(sessions):
    T, B = max(len(s["dur_s"]) for s in sessions), len(sessions)
    inten = np.zeros((B, T)); rate = np.ones((B, T)); dt = np.zeros((B, T))
    y = np.zeros((B, T)); w = np.zeros((B, T))    # w stays zero outside the fill, so padded
    for i, s in enumerate(sessions):              # steps carry no loss weight and need no mask
        n, d = len(s["dur_s"]), s["dur_s"].astype(float)
        inten[i, :n] = s["intensity"]; rate[i, :n] = 1.0 / d; dt[i, :n] = d
        y[i, :n] = s["y"]; w[i, :n] = s["w"]
    return inten, rate, dt, y, w

def fit_core(sessions, rng):
    """Sessions differ 13.5x in length (295 rows to 3986), so 69% of the padded grid is inert
    no-ops (dt=0 leaves every state untouched). Sorting rows long-first makes the live ones a contiguous prefix, so each
    timestep computes only what is still running; the clips and the weight totals are invariant
    across the 80-member population and are hoisted out.

    The loss tail obeys the same rule as the scan: it used to walk all n*T cells of a (S,n,T) pair
    of 96 MB buffers -- six passes over three quarters padding, 42% of the call -- and now computes
    each session over its live prefix into an (S,T) accumulator. The REDUCTION still sums T terms
    in numpy's own order, which is the part that cannot move: the elite set is decided in the last
    ulp. Same reason the ascontiguousarray below is load-bearing -- a fancy-indexed array is
    F-ordered, and reducing over it drifts by 1 ulp, which is enough to change the elite set.

    Verified the way that claim has to be verified: every population and every loss vector of a
    full fit compared with tobytes(), 6400 float64 per fit, on the whole corpus and on folds
    0/4/8, chunked and unchunked. No mismatch."""
    pool = list(sessions) + [augment_raw(s, rng) for s in sessions for _ in range(N_AUG)]
    inten, rate, dt, y, wm = pad(pool)
    n, T = inten.shape
    lens = np.array([len(s["dur_s"]) for s in pool])
    order = np.argsort(-lens, kind="stable")
    inv, ls = np.argsort(order, kind="stable"), lens[order]
    Ic, Rc = np.clip(inten, 1e-4, 1.0)[order].T.copy(), np.clip(rate, 1e-4, None)[order].T.copy()
    Dt, ys, ws = dt[order].T.copy(), y[order], wm[order]
    wsum = wm.sum(1)[order]      # act[t] = sessions still running at t
    act = np.bincount(np.minimum(ls, T), minlength=T + 1)[::-1].cumsum()[::-1][1:]

    acti = [int(v) for v in act]
    # every (t, session-still-running) pair, packed once. The per-step blocks below are VIEWS into
    # it, so the ten quantities that do not read a state -- the two powers, the two exponentials
    # and their arithmetic -- are computed in one vectorised pass instead of 5553 short ones.
    Icol = np.concatenate([Ic[t, :r] for t, r in enumerate(acti)])[:, None]
    Rcol = np.concatenate([Rc[t, :r] for t, r in enumerate(acti)])[:, None]
    Dcol = np.concatenate([Dt[t, :r] for t, r in enumerate(acti)])[:, None]
    K = len(Dcol)
    offs = np.r_[0, np.cumsum(acti)]
    lsl, Z = ls.tolist(), np.array(0.0)
    buf = {}
    chunk = chunk_for(len(sessions), int(lens.sum()) // (1 + N_AUG), T)

    def eval_pop(P):
        if len(P) > chunk:                     # see chunk_for: bit-identical, purely a memory lever
            return np.concatenate([eval_pop(P[i:i + chunk]) for i in range(0, len(P), chunk)])
        S = P.shape[0]
        if not buf:
            buf["Pf"] = np.empty((K, S)); buf["Al"] = np.empty((K, S)); buf["Eh"] = np.empty((K, S))
            buf["Df"] = np.repeat(Dcol[:, 0], S)
            sl = [(int(offs[t]) * S, (int(offs[t]) + r) * S) for t, r in enumerate(acti)]
            buf["Dv"] = [buf["Df"][a:b] for a, b in sl]
            buf["Pv"] = [buf["Pf"].reshape(-1)[a:b] for a, b in sl]
            buf["Alv"] = [buf["Al"].reshape(-1)[a:b] for a, b in sl]
            buf["Ehv"] = [buf["Eh"].reshape(-1)[a:b] for a, b in sl]
            buf["Ao"] = np.zeros((T, n * S))
            buf["Aov"] = [buf["Ao"][t, :r * S] for t, r in enumerate(acti)]
            buf["E"] = np.empty(n * S); buf["A"] = np.empty(n * S); buf["H"] = np.empty(n * S)
            buf["u1"] = np.empty(n * S); buf["u2"] = np.empty(n * S); buf["u3"] = np.empty(n * S)
            # The padded tail of a row is a constant: y=0, w=0, and an A the scan never writes.
            # Computing it once with the same ops lets the reduction below still sum T terms in
            # numpy's own order -- the elite set is decided in the last ulp -- while the
            # elementwise work that feeds it only touches steps that are still running.
            z = np.zeros(1); np.clip(z, 0, 1, z); np.subtract(0.0, z, z)
            q = np.empty(1); np.multiply(z, 0.75, q); np.multiply(z, -0.25, z)
            np.maximum(q, z, q); np.multiply(q, 0.0, q)
            buf["pad"] = float(q[0])
            buf["pb"] = np.empty((S, T)); buf["ev"] = np.empty((S, T)); buf["row"] = np.empty((S, n))
        Ao, E, A, H = buf["Ao"], buf["E"], buf["A"], buf["H"]
        Pf, Al, Eh = buf["Pf"], buf["Al"], buf["Eh"]
        Dv, Pv, Alv, Ehv, Aov = buf["Dv"], buf["Pv"], buf["Alv"], buf["Ehv"], buf["Aov"]
        pa, pb_, pt, pd = P[:, 0], P[:, 1], P[:, 2], P[:, 9]
        np.power(Icol, pa, Eh); np.power(Rcol, pb_, Pf); np.multiply(Eh, Pf, Pf)   # P
        np.divide(Dcol, pt, Al); np.negative(Al, Al); np.exp(Al, Al)
        np.subtract(1.0, Al, Al)                                                   # 1-exp(-dt/tauE)
        np.multiply(pd, Dcol, Eh); np.negative(Eh, Eh); np.exp(Eh, Eh)             # exp(-dH*dt)
        E.fill(0.0); A.fill(0.0); H.fill(0.0)
        tiles = [np.tile(P[:, i], n) for i in range(3, 10)]
        cur = -1
        for t in range(T):
            r = acti[t]
            if r != cur:
                cur, m = r, r * S
                g, thr, rho, lam, kH, hH, dH = [x[:m] for x in tiles]
                u1, u2, u3 = buf["u1"][:m], buf["u2"][:m], buf["u3"][:m]
                Ev, Av, Hv = E[:m], A[:m], H[:m]
            d = Dv[t]
            np.subtract(Pv[t], Ev, u1); np.multiply(Alv[t], u1, u1)
            np.add(Ev, u1, Ev)                                # E
            np.multiply(kH, Hv, u1)
            np.subtract(Ev, thr, u2); np.subtract(u2, u1, u2)
            np.multiply(g, u2, u2); np.logaddexp(Z, u2, u2)
            np.multiply(rho, u2, u2)                          # rho*de, once
            np.add(u2, lam, u1)                               # k
            np.divide(u2, u1, u2)                             # A_ss
            np.negative(u1, u1); np.multiply(u1, d, u1); np.exp(u1, u1)
            np.subtract(Av, u2, u3); np.multiply(u3, u1, u3)
            np.add(u2, u3, Av)                                # A
            np.multiply(hH, Av, u1); np.divide(u1, dH, u1)
            np.subtract(Hv, u1, u3); np.multiply(u3, Ehv[t], u3)
            np.add(u1, u3, Hv)                                # H
            np.copyto(Aov[t], Av)
        pbb, evb, row = buf["pb"], buf["ev"], buf["row"]
        pbb.fill(buf["pad"])
        for j in range(n - 1, -1, -1):     # shortest session first, so each write covers the last
            L = lsl[j]                     # one and the padded tail is never overwritten
            ev = evb[:, :L]; pv = pbb[:, :L]
            np.copyto(ev, Ao[:L, j * S:(j + 1) * S].T)
            np.clip(ev, 0, 1, ev); np.subtract(ys[j, :L], ev, ev)
            np.multiply(ev, 0.75, pv); np.multiply(ev, -0.25, ev)
            np.maximum(pv, ev, pv); np.multiply(pv, ws[j, :L], pv)
            np.sum(pbb, 1, out=row[:, j])                     # pinball tau=.75
        # Per SESSION, not per row: dividing by that session's own weight total and then
        # averaging gives every session exactly 1/n of the objective regardless of length, its 2
        # augmented copies included. Deliberate, and worth stating because it is a 108x spread in
        # per-row influence -- a row of the 295-row session counts 38x what a row of the 3986-row
        # one does -- and because the LOSO eval below does the OPPOSITE, weighting every press
        # equally. Objective and metric disagreeing about what a session is worth is fine as long
        # as it is on purpose.
        #
        # Measured, 4 seeds x 9 folds both ways: pooled-row weighting looks 11% better overall and
        # 66% better in the TOP band, sign-consistent at every seed -- and the whole effect is ONE
        # session (LVL-Gradiant). Clustered at the session level, where this corpus is actually
        # independent, shipped is worse in 3 of 9 and the only band that clears session noise is
        # low 0.0-0.4, where shipped WINS. Not a reason to change it; a reason not to trust a
        # seed-paired comparison on nine sessions.
        row = row / (wsum + 1e-8)
        g, thr, rho, lam = P[:, 3], P[:, 4], P[:, 5], P[:, 6]
        de = softplus(g * (-thr))                       # resting sanity: zero-stim must decay,
        rest = rho * de / (rho * de + lam)              # a leaky gate stalls the policy forever
        return np.ascontiguousarray(row[:, inv]).mean(1) + REST_PENALTY * rest ** 2
    return cem(CORE_BOUNDS, eval_pop, rng)

def predict_core(params, sess):
    inten, rate, dt, _, _ = pad([sess])
    return np.clip(core_scan(inten, rate, dt, *params[:, None])[0], 0, 1)

def stream_session(params, sess):
    st = CoreStreamer(params)
    return np.array([st.step(sess["intensity"][i], sess["dur_s"][i])
                     for i in range(len(sess["dur_s"]))])

def fit_first_cum_hazard(sessions, params):
    """FIRST-cum rate = h0*exp(sharp*(A-1)), max-likelihood, no-cum sessions censored. A thin head
    on the rollout the ODE already gives, not a second model.

    Deliberately first-event, and the name says so: each session is truncated at its first merged
    onset, so a session that recorded two contributes one. On the current corpus that is 4 events
    from 4 of 9 sessions, and the corpus holds exactly those 4.

    This docstring used to say the data held 5 and explain the discarded one away as "it follows an
    orgasm". It did not. It WAS one of the four, replayed 847.9 s later inside a 1567-row block
    that had been pasted into LVL-fast twice -- 12.3% of the corpus, removed 2026-08-15. The
    explanation was a story told about an artefact, which is the failure mode to watch for here:
    with 4 events, any accident large enough to add a fifth is also large enough to look like data.
    Four events is coarse enough that it is only ever used to SIMULATE -- nothing live reads it.

    It is not inert, though. cmd_train writes it into model.json, so it sits inside the artefact
    whose sha1 is cfg_hash: move it and `live` disowns both the encoder and the policy prior. And
    encoder_dataset draws every simulated orgasm from it, so these four events shape all 3000
    training samples of the exported encoder. Four events is a thin basis for that much."""
    tracks = []
    for s in sessions:
        A = stream_session(params, s)
        c = cum_onsets(s["t_s"], s["press_idx"], s["press_vals"])
        end = len(A) if not c else int(np.searchsorted(s["t_s"], c[0])) + 1
        tracks.append((np.clip(A[:end], 0, 1), s["dur_s"][:end].astype(float), bool(c)))
    sharps, h0s = np.arange(4.0, 26.0), np.exp(np.linspace(np.log(1e-5), np.log(0.5), 90))
    nll = np.zeros((len(sharps), len(h0s)))
    for i, sharp in enumerate(sharps):
        for A, d, ev in tracks:
            ex = np.exp(sharp * (A - 1.0))            # independent of h0; hoisting is the whole win
            nll[i] += h0s * float((ex * d).sum())
            if ev:
                nll[i] -= np.log(np.maximum(h0s * ex[-1], 1e-12))
    i, j = np.unravel_index(nll.argmin(), nll.shape)
    # The core fit prints `bound-pinned:`; this printed nothing. With four events the likelihood is
    # barely identified -- on session-bootstraps of this corpus the unconstrained MLE leaves the
    # grid in 35% of resamples, and dropping LVL-fast alone (66% of total exposure) pins h0 at the
    # 0.5 ceiling with sharp=22, measured through this very grid. The answer still goes into
    # model.json and seeds every simulated orgasm the encoder trains on, so when it is the grid's
    # answer rather than the data's, say so.
    if i in (0, len(sharps) - 1) or j in (0, len(h0s) - 1):
        print(f"  WARNING: hazard pinned at a scan edge (sharp={sharps[i]:.0f}, "
              f"h0={h0s[j]:.5f}) -- the likelihood wants to leave the grid.")  # argmin takes the first min in the same
    return float(h0s[j]), float(sharps[i])            # C order the loops visited -> same tie-break

def load_config(path):
    if not os.path.exists(path):
        sys.exit(f"no model at {path} -- train one first:\n  python cumpredict.py train --data sessions.csv")
    raw = open(path).read()
    return json.loads(raw), hashlib.sha1(raw.encode()).hexdigest()[:10]

def write_json(path, obj):
    """Staged and swapped, like drop_session -- for the same reason. policy.json is ~160 KB,
    mostly RLS covariance, written in cmd_live's `finally`, which is precisely where a second Ctrl-C lands
    when he thinks a disconnect has hung. Truncating it there makes the NEXT evening refuse to
    start, because the prior is loaded without a guard."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

CUM_THR, MERGE_GAP_S = 0.99, 120.0

def cum_onsets(t_s, press_idx, press_vals):
    out = []
    for tm in sorted(float(t_s[i]) for i, v in zip(press_idx, press_vals) if v >= CUM_THR):
        if not out or tm - out[-1] > MERGE_GAP_S:
            out.append(tm)
    return out

def press_metrics(press_idx, press_vals, pred):
    tau = None
    if len(press_idx) > 2:
        with suppress(Exception):
            from scipy.stats import kendalltau
            tau = round(float(kendalltau(pred[press_idx], press_vals).correlation), 3)
    return round(float(np.mean(np.abs(pred[press_idx] - press_vals))), 3), tau

POLICY_NAMES = ["aim", "rel", "lo", "span", "dur_hi", "dur_lo", "lfo_s", "lfo_d"]
POLICY_BOUNDS = np.array([
    # Both setpoints are FRACTIONS of the interval this policy can actually reach, never absolute
    # arousal. They used to be absolute, and that made most of the search space unreachable rather
    # than merely bad: `target` ran to 0.97 against a global ceiling of 0.939, while a shallow-slow
    # window settles near 0.58 -- below `target`'s FLOOR of 0.70. About half of all draws were
    # therefore 300-second stalls decided before the first stroke, booking -3.0 no matter what he
    # did, and a measured 25.3% of real trials timed out. A fraction of [ease floor, build ceiling]
    # is reachable from either side by construction, so a deadline now means the body did something
    # the model did not predict -- which is news -- instead of arithmetic anyone could have done.
    [0.55, 0.97],   # aim      how far up that interval to build, 0.97 = all but its last 3%
    [0.25, 0.85],   # rel      how far back down toward the floor to release, as a fraction of aim
    [0.00, 0.45],   # lo       stroke window bottom; lo=0.45 -> shallowest reachable, 0 -> full
    [0.35, 1.00],   # span     window height, hi = min(1, lo+span)
    [0.13, 0.45],   # dur_hi   half-stroke seconds building (0.13 = Keon hardware max)
    [0.45, 2.50],   # dur_lo   half-stroke seconds easing
    [1.50, 8.00],   # lfo_s    LFO period
    [0.00, 0.35],   # lfo_d    LFO depth
])
POLICY_SPAN = POLICY_BOUNDS[:, 1] - POLICY_BOUNDS[:, 0]   # the box's width, once: three copies of
                                                     # this normalisation drift otherwise
BAND_MIN = 0.05   # a policy whose ease stroke drives almost as hard as its build stroke still gets
                  # somewhere to turn around -- half a keypress level -- rather than a zero-width
                  # band that opens and closes on the same command.
REFRACTORY = 0.35   # post-cum dip; unfitted (5 recorded onsets), not a CORE param
# A cycle is worth exactly what he PRESSED during it: peak, the highest real non-orgasm press, on
# the 0..8 grid the keyboard emits. So the best possible cycle scores PEAK_MAX = 8/9 = 0.889 and
# both costs below are quoted in units of that -- the whole reward now lives on a 0..0.889 scale.
PEAK_MAX = 8 / 9.0
CUM_COST = 2.5      # 2.8x PEAK_MAX: the old "~3 good cycles", carried onto the press scale.
                    # 6x that (the old 120-vs-20) made it too timid to approach at all.
BIAS_LR = 0.25
# Deadlines, now a BACKSTOP rather than the main teacher. Both phases wait on corrected(), so a
# setpoint the body never reaches keeps the cycle open forever: nothing is scored, CUM_COST never
# applies, and the session degenerates into one endless trial. _band makes every setpoint reachable
# under the MODEL, which is the whole point of the fraction parameterization -- so a deadline that
# still fires means the real body moved slower than the model says it can, or stopped moving.
# That is news about him, and it is still booked as a failure so CEM learns away from it.
MAX_BUILD_S, MAX_EASE_S = 300.0, 180.0
OUTAGE_CENSOR_S = 20.0   # a gap longer than this is a radio fault, not a technique
# A timed-out cycle forfeits its peak AND pays this -- strictly below the worst real cycle there
# is, a cum with no non-orgasm press at 0 - CUM_COST = -2.5. TIMEOUT_COST > CUM_COST is the
# load-bearing part: it means even a policy that made him cum EVERY cycle outranks one that stalls
# every cycle, so stalling can never become a refuge from the cum penalty. (The old integral reward
# paid stalling outright -- 480*0.8^8 = 80 against a median real cycle of 7. Press-only removes the
# accrual; this ordering removes the refuge.)
TIMEOUT_COST = 3.0

def outcome(peak, cum, timed_out):
    """The single definition of what a closed cycle is worth, so the truth table below and
    close_cycle cannot drift apart. A deadline forfeits the peak; a cum with no non-orgasm press
    is still real evidence and is scored on the cost alone."""
    return (0.0 if timed_out or peak is None else float(peak)) \
        - (CUM_COST if cum else 0.0) - (TIMEOUT_COST if timed_out else 0.0)

# There are FOUR outcome classes, not three, and this is the whole truth table:
#   timeout+cum -5.500 < timeout -3.000 < any cum [-2.500, -1.611] < clean cycle [0.000, +0.889]
# Left: stalling is never a refuge, and stalling into an orgasm is not a way out of the deadline.
# Right: CUM_COST > PEAK_MAX, so no cum can ever outrank even the worst cycle that kept him this
# side of the edge. The elite set only ever RANKS, so -5.5 buys nothing over -3.0 except the
# guarantee that a timeout is never improved by also coming; it is one failed trial either way.
def outcome_ordering():
    """(ok, timeouts, cums, cleans), every class evaluated through outcome() itself so the table
    can never come to describe a different function from the one close_cycle calls. Checked at
    import AND again by invariants() before any export: the import assert cannot see a constant
    that a caller rebound afterwards, and the export gate cannot see one that was wrong all along."""
    peaks = (None, 0.0, PEAK_MAX)
    cleans = [outcome(p, False, False) for p in peaks if p is not None]
    cums = [outcome(p, True, False) for p in peaks]
    tos = [outcome(p, c, True) for p in peaks for c in (False, True)]
    ok = bool(0.0 < PEAK_MAX < CUM_COST < TIMEOUT_COST
              and max(tos) < min(cums) and max(cums) < min(cleans))
    return ok, tos, cums, cleans

assert outcome_ordering()[0], "outcome ordering broken"
# Behaviour cells survive the archive that introduced them: nothing keeps a champion per cell any
# more, but the grid is still how the Director asks "have I just done this". Any scalar optimiser
# converges -- CEM and a GP surrogate both ended a 3-hour run doing long full strokes 38% of the
# time and working the tip 1.8% of the time -- so the anti-repeat is what stops a confident head
# from settling, and it needs a definition of "the same thing" that is about what he FELT rather
# than about which knobs moved.
DESC_NAMES = ["travel_build", "dur_build", "travel_ease", "dur_ease", "target", "cycle_s"]
# Two of these six are one axis. `travel_ease` is 0.25*span EXACTLY (the clip in settle can never
# bind: lo <= 0.45 and 0.25*span <= 0.25), and `travel_build` is min(1-lo, span), so on the 65.2%
# of the box where span <= 1-lo they are related by a constant factor of 4 -- measured to 2.8e-16.
# So the anti-repeat has five axes, not six, and `lo` (base versus tip, the distinction _draw's
# comment actually cares about) reaches the grid only through travel_build's clip. `lfo_s` reaches
# it not at all: swept across its whole range every descriptor moves by exactly 0.0, so two
# policies differing only in a 5x change of LFO period share a cell. Fixing either means a retrain
# and a new cell grid -- prior() already refuses a policy.json whose DESC_NAMES differ, loudly.
NBIN = 4              # bins per descriptor axis -> 4^6 = 4096 cells, of which at least 1486
                      # (36%) have a preimage: measured over 2M uniform draws and still creeping,
                      # so a floor rather than a count. The EXTENT_N sample that DEFINES the grid
                      # touches only 997 of them, which is what earlier comments here quoted as
                      # though it were the reachable set. Travel, duration and turning point are
                      # coupled through setpoints(), so most of the box is genuinely not a place --
                      # but see DESC_NAMES: two of the six axes are one axis over most of the box,
                      # so the grid is doing less work than its axis count suggests.
EXTENT_N = 20000      # fixed-seed sample that fixes what a cell MEANS, per set of core params
EPS_UNIFORM = 0.25    # fraction drawn uniformly regardless -- a floor under exploration that no
                      # amount of head confidence can close

def setpoints(P, core, rate_ref=None):
    """The arousal band a policy is graded against: where its ease and build strokes SETTLE, and
    the target/release placed strictly inside that interval. THE fixed point -- _band reads it for
    the running policy and descriptors() for a whole candidate batch, so the number a cycle is
    graded against and the number the cell grid and the encoder read cannot disagree.

    Broadcasts on shape: [n,8] gives arrays, a bare 8-vector gives scalars. Every operation below
    is a ufunc, so a 0-d input stays 0-d all the way through.

    E converges to the drive P, which freezes de, k and A_ss; kH feeds H back into the drive
    (0.1302 in the deployed fit -- but hH is exactly 0, so H is identically zero and the iteration
    is done on its first pass) and is solved by damped iteration rather than assumed away. Same clamps and the
    same rate_ref convention as CoreStreamer.step, because a ceiling quoted for a different
    stimulus than the one the streamer integrates is not this machine's ceiling. The deadline used
    to answer "can this policy get there?" by burning 300 seconds of his evening first."""
    a, b, tauE, g, thr, rho, lam, kH, hH, dH = (float(core[k]) for k in CORE_NAMES)
    aim, rel, lo, span, dur_hi, dur_lo, _, lfo_d = np.asarray(P, float).T
    tb = np.minimum(1.0, lo + span) - lo
    te = np.minimum(1.0, lo + 0.25 * span) - lo
    # The LFO-AVERAGE half-stroke, not its shortest: a ceiling is what a stimulus held forever
    # reaches, and dur_hi*(1-lfo_d) is touched for an instant at the top of each sweep.
    db, de_ = dur_hi * (1 - 0.5 * lfo_d), dur_lo * (1 + 0.5 * lfo_d)

    def settle(hi, travel, dur):
        """One half-cycle's ceiling, plus the relaxation rate -- the speed half of what a cycle
        feels like.

        `hi` is the amplitude this half of the cycle actually HOLDS, and a vibrator is driven by
        that rather than by a travel. Passing lo+span for both halves -- the build amplitude, for
        the ease phase too -- made settle(build) and settle(ease) identical under a vibrator, so
        hi_A collapsed onto lo_A, every band was pinned to BAND_MIN, and the `target` axis
        disagreed with what the cycle is actually graded against by up to 0.30. That fed the cell
        grid, the anti-repeat key and the heads' features, so everything the Director knew about a
        vibrator was wrong. It was wrong HERE and right in _band for exactly as long as those were
        two functions."""
        stim = hi if rate_ref is not None else travel
        rate = rate_ref if rate_ref is not None else 1.0 / np.maximum(dur, 1e-3)
        P_ = np.clip(stim, 1e-4, 1.0) ** a * np.maximum(rate, 1e-4) ** b
        # Damped fixed point -- kH*hH/dH is large enough at the bound corners that the undamped
        # map oscillates. A FIXED pass count, never an early exit: `np.max(|A2-A|) < eps` is a
        # BATCH-WIDE vote, so a singleton stops when ITS element stalls while a batch runs until
        # the slowest one does, and on a damped oscillation a stall is not convergence -- A
        # saturates in H, so a sweeping H holds A still for one step and then moves on. Measured
        # at kH=2/hH=.02/dH=1e-3: setpoints(P)[i] and setpoints(P[i]) diverged by up to 0.097 on
        # 25361 of 43310 policies -- in the one number _band and descriptors() exist to SHARE.
        #
        # NOT a converged fixed point at the bound corners: kH*hH/dH reaches 25000 there and the
        # damped map is chaotic, so 64 passes and 4096 passes disagree by up to 0.92. What the
        # FIXED count buys is determinism -- a singleton and a batch always agree, run to run --
        # which is the property _band and descriptors() actually need from each other.
        #
        # ONE pass is exact whenever H cannot reach the drive, and there are TWO ways that happens:
        # kH == 0 (H is computed and then multiplied out) or hH == 0 (H_ss is 0, so H never leaves
        # the zero it starts at -- H <- H + 0.25*(0 - H) is 0.75*H from 0). Keying this on kH alone
        # was right for the fit it was written against and wrong for the next one: the refit gave
        # kH=0.130 with hH=EXACTLY 0, so this ran 64 bit-identical passes and cost 22x on every
        # setpoints() call. A fit with both non-zero still pays the full 64 rather than quietly
        # disagreeing with itself.
        onepass = kH == 0.0 or hH == 0.0
        A = H = np.zeros_like(P_)
        for _ in range(1 if onepass else 64):
            de = np.logaddexp(0.0, g * (P_ - thr - kH * H))
            A2 = rho * de / (rho * de + lam)
            H = H + 0.25 * (hH * A2 / dH - H)
            A = A2
        # `rho*de + lam` IS the k that goes with the A being returned -- A = rho*de/(rho*de+lam)
        # by the line above, in BOTH branches. Recomputing it from the post-loop H made the
        # non-shortcut arm quote a rate one iteration ahead of its own A: measured 143x off at the
        # bound corners, into the axis descriptors() divides by. No test executes that arm (every
        # core in both suites has hH == 0.0), which is how it stayed wrong.
        return A, rho * de + lam

    hiA, kb = settle(np.minimum(1.0, lo + span), tb, db)
    loA, ke = settle(np.minimum(1.0, lo + 0.25 * span), te, de_)
    hiA = np.maximum(hiA, loA + BAND_MIN)
    tgt = loA + aim * (hiA - loA)
    return loA, hiA, tgt, loA + rel * (tgt - loA), kb, ke, tb, db, te, de_

def descriptors(P, core, rate_ref=None):
    """Behaviour of a batch of policies, [n,8] -> [n,6]: how far the machine travels and how fast
    on each half of the cycle, where it turns around, and how long that takes. Vectorised because
    every candidate costs arithmetic and none of them cost a stroke.

    The first four axes are ACTUATOR facts and survive any refit. `target` and `cycle_s` come from
    the fitted core, which is why the online heads are pinned to the predictor that produced it."""
    loA, hiA, tgt, rls, kb, ke, tb, db, te, de_ = setpoints(P, core, rate_ref)
    tB = -np.log(np.maximum((hiA - tgt) / np.maximum(hiA - rls, 1e-9), 1e-9)) / kb
    tE = -np.log(np.maximum((rls - loA) / np.maximum(tgt - loA, 1e-9), 1e-9)) / ke
    # The 480 cap has never bound: the worst tB+tE anywhere in POLICY_BOUNDS measures 163 s.
    # Kept as a guard on the two logs above, which go to +inf as a setpoint approaches its ceiling.
    return np.stack([tb, db, te, de_, tgt, np.minimum(tB + tE, MAX_BUILD_S + MAX_EASE_S)], 1)

ENCODER = CKPT + "/encoder.json"
# ---------------------------------------------------------------- the online half
# READ THIS BEFORE PROPOSING A SIMPLER DIRECTOR.
#
# The goal is a Director that is SURPRISING and adapts to him tonight -- not one that is reliable.
# A fixed rule that scores the same is not a substitute. Reviewers reliably answer a question that
# was not asked here ("the problem is small, use math"), and the reason they can is that the reward
# metric is blind to the goal: reward is his peak keypress per cycle, so it measures how high he
# got and CANNOT see whether the machine repeated itself or adapted. A reward tie between a rule
# and a learner is evidence about the metric, not about the learner. Measured, for the record: the
# original hand-written wave controller ties this Director on reward at 0.1 sigma, and runs one
# fixed pattern every evening forever.
#
# Two halves, and the split is the only reason this is trainable on 218 keypresses:
#   FROZEN  a CfC (closed-form continuous-time) encoder over the last few cycles -> 32 features.
#           Pretrained by `train` on plant DYNAMICS only -- what a policy reaches and how long it
#           takes -- so it holds no opinion about what he likes. Exported as plain JSON and run
#           here in numpy: `live` never imports torch, and the same ten lines port to JavaScript.
#   ONLINE  a Bayesian linear head per outcome (press, cum, timeout) over those features, updated
#           by recursive least squares the moment a cycle closes. THIS is what learns him, and one
#           keypress moves it materially -- which is why it works at ~15 rated cycles an evening
#           where a network trained by gradient descent cannot.
#
# Why continuous time specifically: his presses arrive whenever he feels like it, and the same 0.8
# means different things at 00:40 and at 24:00. dt sits INSIDE the cell's gate, so elapsed time
# rescales the state update structurally instead of being one input column the net must learn to
# treat as special. Command durations in his corpus span 36x (p1 0.130s, p99 4.671s) with 9x jumps
# between consecutive commands, so the clock genuinely moves here.
#
# What was tried and lost, so it is not re-proposed: LSTM/GRU/CfC as the PREDICTOR (7-25% worse
# than the ten-parameter ODE); GP-BO over the policy (ties, because reward is a 7-rung staircase
# with 15 distinct values and a surrogate has no smooth surface to interpolate); per-COMMAND neural
# control (-15 to -225 sigma -- one command moves arousal 0.00056 against a press quantum of 0.111,
# so a single stroke is 199x below the resolution of the only ground truth in the system). The
# learnable unit is the cycle, because it is the smallest action whose effect he can report.
#
# Hard rules any change must keep: his keypress is the only ground truth, and predictor output may
# be a FEATURE but never a training target; no body state (E/A/H) crosses a session.

RLS_SIGMA, RLS_P0, RLS_Q, NN_CAND = 0.25, 1.0, 1e-4, 48
NF_ENC, NH_ENC, HIST_ENC, RECENT_CELLS = 32, 24, 6, 8   # feature width, cycles the encoder reads, anti-repeat

def lecun(x):
    return 1.7159 * np.tanh(0.666 * x)

def load_encoder(predictor_hash, say=print):
    """The exported CfC, or None with a REASON. Never returns None quietly: if the online head is
    not running tonight, he is told which of the three reasons applies before the session starts."""
    if not os.path.exists(ENCODER):
        say(f"online learning OFF -- no {ENCODER}. It is built by `train`, alongside the model.")
        return None
    try:
        e = json.load(open(ENCODER))
    except Exception as ex:
        say(f"online learning OFF -- {ENCODER} unreadable ({type(ex).__name__}).")
        return None
    if e.get("predictor") != predictor_hash:
        say(f"online learning OFF -- encoder was trained against predictor "
            f"{e.get('predictor')}, this model is {predictor_hash}. Re-export it after a refit.")
        return None
    try:
        e["w"] = {k: np.array(v, float) for k, v in e["w"].items()}
        e["dmin"], e["drange"] = np.array(e["dmin"], float), np.array(e["drange"], float)
    except Exception as ex:      # the fourth way to fail, and it used to fail SILENTLY: cmd_live
        say(f"online learning OFF -- {ENCODER} is malformed "   # then printed "reason above" with
            f"({type(ex).__name__}: {ex}). Rebuild it with `train`.")   # no reason above it
        return None
    # Check the weights the forward pass will actually index, HERE. The recurrent loop only runs
    # once a cycle has closed, so a missing cell.* key used to pass every check, print "online
    # learning ON", drive the toy, and raise 45 seconds into the evening.
    need = ["cell.bb.weight", "cell.bb.bias", "mlp.0.weight", "mlp.0.bias",
            "mlp.2.weight", "mlp.2.bias"] + [f"cell.{k}.{w}" for k in ("ff1", "ff2", "ta", "tb")
                                             for w in ("weight", "bias")]
    # cfc_features caches its stacked weights on this dict under "_cell4". Nothing underscore-
    # prefixed is part of the format, so drop any the file carries rather than let encoder.json
    # hand the forward pass a "cache" of its own choosing.
    for k in [k for k in e if k.startswith("_")]:
        del e[k]
    lack = [k for k in need if k not in e["w"]]
    if lack:
        say(f"online learning OFF -- {ENCODER} is missing {lack[:3]}. Rebuild it with `train`.")
        return None
    if (e["w"]["cell.bb.weight"].shape[1] != 11 + NH_ENC
            or e["w"]["mlp.2.weight"].shape[0] != NF_ENC
            # Each head's ROW COUNT too, not just the ends of the net. cfc_features stacks these
            # four into one matvec and slices the result by NH_ENC, so a pair of compensating
            # corruptions that still sums to 4*NH_ENC used to return a wrong answer where the old
            # four-matvec form raised. Cheap to check; the failure it prevents is silent.
            # ...and their BIASES, which are the more dangerous half: cfc_features stacks those
            # with a 1-D np.concatenate, and that never raises on mismatched lengths. A pair of
            # compensating bias corruptions summing to 4*NH_ENC returned wrong features with no
            # exception at all -- exactly what checking only the weights was supposed to prevent.
            or any(e["w"][f"cell.{k}.weight"].shape[0] != NH_ENC
                   or e["w"][f"cell.{k}.bias"].shape != (NH_ENC,) for k in KEYS4)):
        say(f"online learning OFF -- {ENCODER} was built for a different feature size "
            f"(expects 11+{NH_ENC} in, {NF_ENC} out). Rebuild it with `train`.")
        return None
    return e

KEYS4 = ("ff1", "ff2", "ta", "tb")     # stacked into one matvec by cfc_features


def cfc_features(enc, hist, pol_n, desc_n):
    """The frozen encoder, forward only. hist: [T,11] cycles already run (may be empty);
    pol_n/desc_n: [n,8] and [n,6], normalized. -> [n,32].

    This is the whole runtime cost of the network: a few matmuls per cycle, about a millisecond,
    once every ~45 seconds. It is also the entire JavaScript port."""
    w = enc["w"]
    # ff1/ff2/ta/tb all read the SAME z, so they are one stacked matvec -- bit-identical, since
    # every output element is still the same row dotted with the same z. Cached on the encoder
    # dict itself and never on id(): a freed dict hands its id to the next one, which during
    # development quietly served another encoder's weights.
    if "_cell4" not in enc:
        enc["_cell4"] = (np.concatenate([w[f"cell.{k}.weight"] for k in KEYS4], 0),
                         np.concatenate([w[f"cell.{k}.bias"] for k in KEYS4]))
    W4, b4 = enc["_cell4"]
    Wbb, bbb, n = w["cell.bb.weight"], w["cell.bb.bias"], NH_ENC
    h = np.zeros(n)
    if len(hist):
        v = np.empty(11 + n)               # np.r_ rebuilt this every step, at 3x concatenate
        for row in hist:                   # one step per cycle already run this session
            v[:11] = row; v[11:] = h
            u = W4 @ lecun(Wbb @ v + bbb) + b4
            gate = 1.0 / (1.0 + np.exp(-(u[2 * n:3 * n] * (row[9] * 300.0) + u[3 * n:])))
            h = np.tanh(u[:n]) * (1.0 - gate) + np.tanh(u[n:2 * n]) * gate   # cycle's REAL seconds
    X = np.concatenate([np.repeat(h[None], len(pol_n), 0), pol_n, desc_n], 1)
    a = X @ w["mlp.0.weight"].T + w["mlp.0.bias"]
    a = a / (1.0 + np.exp(-a))                                        # SiLU
    return np.tanh(a @ w["mlp.2.weight"].T + w["mlp.2.bias"])

class RLS:
    """Bayesian linear head. Six lines, no gradient, constant memory, and one keypress moves it
    materially -- which is the only reason online learning is possible at this sample size."""

    def __init__(self, n, w=None, P=None):
        self.w = np.array(w, float) if w is not None else np.zeros(n)
        self.P = np.array(P, float) if P is not None else np.eye(n) * RLS_P0

    def update(self, phi, y):
        Pp = self.P @ phi
        K = Pp / (RLS_SIGMA ** 2 + float(phi @ Pp))
        self.w = self.w + K * (y - float(phi @ self.w))
        self.P -= np.outer(K, Pp)              # forgetting term straight onto the diagonal:
        self.P.flat[::len(K) + 1] += RLS_Q     # same arithmetic, no 32x32 identity built for it
        self._L = None                         # the factor below describes the P that just moved

    def sample(self, PHI, rng):
        """Thompson: score with one draw from the posterior, so exploration decays on its own as
        the belief tightens and there is no epsilon to hand-tune.

        The factor is kept until update() replaces P. P moves by a rank-1 downdate PLUS a full-rank
        RLS_Q*I, and a diagonal add has no exact O(n^2) factor update -- a pure-numpy rank-1
        downdate measured 21x SLOWER than just asking LAPACK again. So the choice is recompute or
        cache, and cheap caching is what an unrated cycle (samples, never updates) gets for free."""
        if getattr(self, "_L", None) is None:
            A = self.P.copy()
            A.flat[::len(self.w) + 1] += 1e-9
            self._L = np.linalg.cholesky(A)
        return PHI @ (self.w + self._L @ rng.standard_normal(len(self.w)))

    def dump(self):
        return {"w": self.w.tolist(), "P": self.P.tolist()}


class Director:
    """One build->ease cycle = one trial, scored ONLY by what he pressed: `peak`, the highest
    real non-orgasm press during the cycle, less CUM_COST if he came and TIMEOUT_COST if a deadline
    fired. No predictor output enters the score -- corrected() still drives the phase transitions,
    but the policy is never graded by the model it is steering. An unrated cycle that ran to its
    release is MISSING, not zero: it teaches the heads nothing and stores no trial. An unrated
    cycle that hit a DEADLINE is not missing -- the deadline itself is the outcome, measured by a
    clock rather than by him, so it scores.

    A cycle is a transaction, and the two halves are separate calls on purpose:
    commit_observation() advances the clocks and may close/resample; next_action() reads self.p at
    call time. Fusing them is how the first command of every new policy used to be built from the
    previous policy's knobs, at every ease->build and every cum boundary."""

    _EXTENT = {}     # see _extent: the descriptor grid, cached per (core, modality)

    def __init__(self, rng, core, prior=None, encoder=None):
        self.rng = rng
        self.core = core          # the plant, read-only: needed to know what each policy can REACH
        prior = prior or {}
        # A log that STEERS: _draw reads the last RECENT_CELLS cells out of this to strike out
        # anything he has just had. Kept as dicts because "what was the reward" is only one of the
        # questions a later analysis asks of a cycle. Tolerant of the old (policy, reward) pairs.
        # Gated with everything else: _draw reads cells out of this log, so a trial written under
        # different knobs or different descriptor axes names a coordinate that no longer exists.
        # It used to be restored unconditionally, which seeded the anti-repeat with nonsense.
        _ok = prior.get("names") == POLICY_NAMES and prior.get("desc") == DESC_NAMES
        self.trials = [t if isinstance(t, dict) else {"p": list(t[0]), "reward": float(t[1])}
                       for t in (prior.get("trials", []) if _ok else [])]
        # Cells only mean anything under the knob list AND the core that produced them: the last
        # two descriptor axes are computed from the fit. A different action space or a different
        # predictor is a different map, so neither the heads nor the cycle history survive it.
        self.mod = str(prior.get("modality") or "linear")   # provisional until set_modality
        # Derived here as well as in set_modality: a prior that already says "vib" makes set_modality
        # a no-op, and the band sampled below would otherwise be quoted for a stroker's 1/dt.
        self.rate_ref = VIB_RATE_REF if self.mod == "vib" else None
        self.bias = 0.0
        self.t = self.phase_t = self.build_s = 0.0
        self.cycles, self.timeouts, self.phase = 0, 0, "build"
        self.enc = encoder
        # The heads persist across evenings: what he likes is policy knowledge, not body state, so
        # it accumulates. No E/A/H ever crosses a session -- that rule is about the BODY and it is
        # untouched. Only restored under the same knob names and the same descriptor axes, because
        # weights learned over one feature map mean nothing over another.
        # All three or none, and only at the width this build uses. A file carrying two of the
        # three left the third fresh at P0=1.0 beside two tight ones, and the proposal then
        # subtracted TIMEOUT_COST times a sample drawn from that noise; a file from a
        # different NF_ENC aborted the session with a raw matmul error and no reason.
        _hd = prior.get("heads") or {}
        hd = _hd if (_ok and set(_hd) == {"press", "cum", "to"}
                     and all(len(v.get("w", ())) == NF_ENC for v in _hd.values())) else {}
        self.heads = {k: RLS(NF_ENC, **hd.get(k, {})) for k in ("press", "cum", "to")} \
            if encoder else None
        self.hist = [np.array(r, float) for r in (prior.get("cycle_hist") or [])][-HIST_ENC:] \
            if _ok else []
        self._extent()
        self._sample()

    def _extent(self):
        """What a cell MEANS: the observed range of each descriptor axis over the reachable space.
        Fixed seed, so two runs of the same core agree on the grid to the bit, and a cell recorded
        in policy.json still names the same behaviour tomorrow. Recomputed per core because the
        last two axes move when the model is refit -- which is why the encoder is pinned to it."""
        key = (tuple(float(self.core[k]) for k in CORE_NAMES), self.rate_ref)
        if key not in Director._EXTENT:      # pure in (core, rate_ref) and fixed-seed, so the
            B = descriptors(np.random.default_rng(0).uniform(   # answer cannot differ between
                POLICY_BOUNDS[:, 0], POLICY_BOUNDS[:, 1], (EXTENT_N, len(POLICY_NAMES))),
                self.core, self.rate_ref)                       # calls. At most two keys live.
            Director._EXTENT[key] = (B.min(0), np.maximum(B.max(0) - B.min(0), 1e-9))
        dmin, drange = Director._EXTENT[key]
        self.dmin, self.drange = dmin.copy(), drange.copy()

    def features(self, P, desc=None):
        """Frozen encoder over this session's cycles, crossed with each candidate policy."""
        # Scaled by the extent the ENCODER was trained on, never this session's. self.dmin is the
        # CELL grid and it is allowed to move with the actuator -- set_modality re-measures it for
        # a vibrator. The net is frozen, so feeding it units it never saw is feeding it noise:
        # measured 0.023 of feature range on vib, where features live in [-1,1]. One field was
        # doing two jobs that only coincide on linear.
        H = np.array(self.hist[-HIST_ENC:], float) if self.hist else np.zeros((0, 11))
        b = descriptors(P, self.core, self.rate_ref) if desc is None else desc
        return cfc_features(self.enc, H, (P - POLICY_BOUNDS[:, 0]) / POLICY_SPAN,
                            (b - self.enc["dmin"]) / self.enc["drange"])

    def cells(self, P, desc=None):
        """Cells for a BATCH. cell() computes descriptors for one policy; calling it in a loop over
        48 candidates cost 35ms of the proposal against 0.8ms batched, because descriptors() is
        vectorised and the loop threw that away."""
        b = descriptors(P, self.core, self.rate_ref) if desc is None else desc
        ix = np.clip((b - self.dmin) / self.drange * NBIN, 0, NBIN - 1).astype(int)
        return [",".join(map(str, r)) for r in ix.tolist()]   # tolist: str() on numpy scalars 2x

    def cell(self, p):
        """One policy's cell. Delegates, because _draw compares strings from cells() against
        strings this wrote into the trial log -- two copies of the binning that silently stop
        matching would disable the anti-repeat with no crash and no log line."""
        return self.cells(np.asarray(p, float)[None])[0]


    def _draw(self):
        """Propose the next stroke pattern: a Thompson draw from the online heads, or a uniform
        one EPS_UNIFORM of the time.

        This used to be MAP-Elites -- mutate a champion drawn from a per-behaviour-cell archive.
        The archive is gone. It never proposed anything once the heads were live, and measured on
        120 latent bodies its own proposal was worse than uniform random: +0.435 against +0.456,
        because clipping a Gaussian at the bounds piled 15.4% of knob values onto the walls. The
        heads carry what it was for -- they generalise reward ACROSS the space instead of storing
        one champion per cell and never interpolating, which is what 1639 cells against ~15 rated
        cycles an evening could never do."""
        lo, hi = POLICY_BOUNDS[:, 0], POLICY_BOUNDS[:, 1]
        if self.heads is None:
            return self.rng.uniform(lo, hi)   # bootstrap only; see __init__
        if self.rng.random() < EPS_UNIFORM:
            # Its OWN features. Returning here without setting _phi left the previous policy's
            # features in place, so close_cycle taught all three heads this cycle's outcome
            # against the last cycle's inputs -- measured on 12 of 59 consecutive cycles, the
            # rate EPS_UNIFORM predicts. One update in four was wrong.
            q = self.rng.uniform(lo, hi)      # a floor under exploration no confidence can close
            self._phi = self.features(q[None])[0]
            return q
        C = self.rng.uniform(lo, hi, (NN_CAND, len(lo)))
        B = descriptors(C, self.core, self.rate_ref)   # ONE pass: the encoder and the cell grid
        PHI = self.features(C, B)                      # scale it differently, they do not recompute
        # one posterior draw per head, so exploration decays on its own as the belief tightens
        val = (self.heads["press"].sample(PHI, self.rng)
               - CUM_COST * np.clip(self.heads["cum"].sample(PHI, self.rng), 0, 1)
               - TIMEOUT_COST * np.clip(self.heads["to"].sample(PHI, self.rng), 0, 1))
        # Anti-repeat, the one job the archive genuinely did: strike out anything whose behaviour
        # cell was used in the last RECENT_CELLS cycles, so a confident head cannot settle onto a
        # single technique. .get, not [] -- an older policy.json has trials with no cell key.
        recent = [c for c in (t.get("cell") for t in self.trials[-RECENT_CELLS:]) if c]
        fresh = np.array([c not in recent for c in self.cells(C, B)])
        if fresh.any():
            val = np.where(fresh, val, -1e9)
        k = int(np.argmax(val))
        self._phi = PHI[k]
        return C[k]

    def _band(self):
        """Turn `aim` and `rel` into the two arousal setpoints THIS policy will be judged against.

        setpoints() puts both strictly inside (lo_A, hi_A) -- where arousal settles under this
        policy's ease and build strokes -- so build can always climb to its target and ease can
        always fall to its release, and release < target holds by construction, which is what stops
        a draw from asking for a band that is already closed. It is the same call descriptors()
        makes for the candidate batch, so the band a cycle is GRADED against and the band the cell
        grid and the encoder describe are one computation rather than two that have to agree.

        Frozen for the whole cycle at the bias of the instant it was sampled. commit_observation
        compares corrected(A), so the band is quoted in those same coordinates; recomputing it as
        the bias moved would grade the trial against a contract it never started on."""
        # next_action reads THIS, not self.p: unpacking eight numpy scalars and running
        # min/sin/round on them costs 2.13 us against 1.04 us for Python floats, 267 times a
        # cycle. _band is the only safe place to cache it -- _sample and encoder_dataset's one()
        # are the sole writers of self.p and both call _band immediately after, so a stale _pf is
        # only reachable down a path that would already be grading against the wrong band.
        self._pf = tuple(map(float, self.p))
        _lo_A, _hi_A, target, release = setpoints(self.p, self.core, self.rate_ref)[:4]
        self.target = float(np.clip(target + self.bias, 0.0, 1.0))
        self.release = min(float(np.clip(release + self.bias, 0.0, 1.0)), self.target)

    def _sample(self):
        self.p = self._draw()
        self._band()
        # build_s too: it is only assigned at the build->ease transition, so a cycle ending in
        # an orgasm before it ever eased kept the LAST cycle's number -- measured 45.0 s
        # recorded for a 5.0 s cycle, on the orgasm cycle specifically, straight into the
        # encoder's time gate.
        self.phase_t = self.build_s = 0.0
        self.peakA = 0.0      # highest PREDICTED arousal this cycle: the encoder's
                              # history column, and the quantity it was trained on
        self.peak = None      # None, not 0.0 -- "he never rated it" and "he rated it 0" are
        self.timed_out = False   # different facts, and only the second one is evidence

    def censor_cycle(self):
        """The context the cycle was running in no longer exists (different actuator, different
        hardware). Its evidence describes a machine that is not attached any more, so store NO trial
        and start a fresh cycle in `build`. Censored, never transferred."""
        self.phase = "build"
        self._sample()

    def set_modality(self, tag):
        """The toy is only identified after the prior has been loaded, and a stroke window is not an
        amplitude, so the heads learned on one actuator do not describe the other. Idempotent, so a mid-session reconnect to
        the same toy does not throw away the trial that is already running.

        A real change is an EPISODE boundary, not a relabel: the cycle in flight is censored under
        the key it actually ran on, the phase restarts, and reward history is dropped -- rewards
        earned by swinging a stroker do not rank amplitudes held by a vibrator."""
        # Unconditional, and ahead of the branch: reset_prior resamples, and a band derived from a
        # stroker's 1/dt would then be handed to a vibrator. Idempotent on a reconnect either way.
        self.rate_ref = VIB_RATE_REF if tag == "vib" else None
        self._extent()   # a vibrator's descriptors are a different map: re-measure what a cell means
        if tag != self.mod:
            self.mod, self.phase = tag, "build"
            self.reset_prior()

    def note_press(self, press):
        """The ONLY evidence a trial is ever scored on. An orgasm press is excluded here -- it
        arrives separately as `cum`, and "how high did he report" and "did he go over" are two
        different facts about the cycle."""
        press = float(press)
        if press < CUM_THR:
            self.peak = press if self.peak is None else max(self.peak, press)

    def close_cycle(self, cum=False):
        peak = self.peak
        self.timeouts += int(self.timed_out)
        self.cycles += 1
        rated = self.peak is not None or cum or self.timed_out
        r = outcome(self.peak, cum, self.timed_out) if rated else None
        # Scored unless the cycle produced no outcome at all. A press is an outcome; so is a cum;
        # so is a DEADLINE -- and that last one is the whole reason TIMEOUT_COST exists, so
        # requiring a press first meant the exact pathology the deadline was built to teach against
        # (drive at full depth toward a target the body never reaches, while he is too far gone or
        # too bored to rate anything) was the one case the heads never saw. Timing is measured by the
        # clock, not by the predictor, so scoring it reopens no reward circularity.
        if rated:                            # else MISSING: no learning signal
            # Everything a later learner could want about this cycle, not just its score. `target`,
            # `release` and `bias` are predictor-derived and are recorded as CONTEXT -- they are
            # inputs to a future model, never terms in the reward, which stays press-only. That
            # distinction is the whole of what round 2 removed and it is preserved here.
            if self.heads is not None:
                # THE ONLY LEARNING THAT HAPPENS LIVE, and every target is something HE did:
                # the press he gave, whether he came, whether a deadline fired. The predictor's
                # opinion is a feature, never a label -- the line round 2 drew, still held.
                self.heads["press"].update(self._phi, 0.0 if self.timed_out else float(peak or 0.0))
                self.heads["cum"].update(self._phi, float(cum))
                self.heads["to"].update(self._phi, float(self.timed_out))
            self.trials.append({"p": self.p.tolist(), "reward": r, "cell": self.cell(self.p),
                                "peak": self.peak, "cum": bool(cum), "timeout": self.timed_out,
                                "bias": round(self.bias, 4), "target": round(self.target, 4),
                                "release": round(self.release, 4), "mod": self.mod,
                                "build_s": round(self.build_s, 2),
                                "ease_s": round(self.phase_t, 2)})
        if self.heads is not None:     # what the encoder reads next cycle: what this one DID
            # peakA, NOT his press. encoder_dataset trains this column on the arousal a cycle
            # reached; feeding it a keypress at runtime is a different quantity on a different
            # scale in the same slot, and the encoder is dynamics-only by design -- what he felt
            # about it is the RLS heads' job, and they get it as their target.
            self.hist.append(np.r_[(self.p - POLICY_BOUNDS[:, 0]) / POLICY_SPAN, self.peakA,
                                   min(self.phase_t + self.build_s, 300.) / 300., float(cum)])
            self.hist = self.hist[-HIST_ENC:]
        self.phase = "build"
        self._sample()

    def observe(self, press, A):
        self.bias += BIAS_LR * ((press - A) - self.bias)

    def corrected(self, A):
        return min(max(float(A) + self.bias, 0.0), 1.0)

    def commit_observation(self, A, dt, cum=False):
        """Book what the command just executed actually did: advance the clocks by the time it
        really took, then close the cycle if it ended. `dt` is the measured period, and it is the
        SAME number the predictor integrated -- one system, one clock, so an outage cannot age the
        body without ageing the phase deadlines with it.

        This closes the cycle but never builds a command. close_cycle resamples, so the caller must
        ask next_action() afterwards to get the command the NEW policy wants; that ordering is the
        single owner of the cycle boundary, and no outside caller can resample first and hand this
        cycle's presses to a policy that did not earn them."""
        Ah = self.corrected(A)
        self.peakA = max(self.peakA, float(A))
        self.t += dt; self.phase_t += dt
        if cum:
            self.close_cycle(True)
        elif self.phase == "build" and (Ah >= self.target or self.phase_t > MAX_BUILD_S):
            self.timed_out |= self.phase_t > MAX_BUILD_S
            self.phase, self.phase_t, self.build_s = "ease", 0.0, self.phase_t
        elif self.phase == "ease" and (Ah <= self.release or self.phase_t > MAX_EASE_S):
            self.timed_out |= self.phase_t > MAX_EASE_S
            self.close_cycle(False)

    def next_action(self):
        """The command the CURRENT policy asks for. Every knob is read from self.p HERE, at call
        time: read them once at the top of a call that can also resample and the first physical
        command of every new trial is the old trial's."""
        _aim, _rel, lo, span, dur_hi, dur_lo, lfo_s, lfo_d = self._pf  # setpoints are self.target /
        # self.release, frozen by _band at sample time and deliberately not re-read here
        w = 0.5 + 0.5 * math.sin(2 * math.pi * self.t / lfo_s)
        if self.phase == "build":
            hi, dur = min(1.0, lo + span), dur_hi * (1.0 - lfo_d * w)
        else:
            hi, dur = min(1.0, lo + 0.25 * span), dur_lo * (1.0 + lfo_d * w)
        return round(float(lo), 3), round(float(hi), 3), round(max(float(dur), 0.10), 3)

    def reset_prior(self):
        """Scores earned under different terms are not comparable, so the trial log is dropped --
        and so are the heads and the cycle history. The archive used to be keyed per modality for
        exactly this reason: a weight learned swinging a stroker does not rank amplitudes held by a
        vibrator, and the encoder's features are read off a stroke window that a vibrator does not
        have. Forgetting is the correct behaviour here; the alternative is ranking this toy by
        evidence collected on a different one.

        A predictor or reward change is handled elsewhere and more bluntly: cmd_live refuses to
        start at all when the encoder does not match the model."""
        self.trials = []
        self.hist = []
        self.heads = {k: RLS(NF_ENC) for k in ("press", "cum", "to")} if self.enc else None
        self._sample()

    def prior(self, predictor, modality):
        return {"predictor": predictor, "modality": modality,
                "names": POLICY_NAMES, "reward": [CUM_COST, TIMEOUT_COST], "desc": DESC_NAMES,
                # Windowed at 120, and that window is load-bearing: _draw's anti-repeat reads
                # cells out of the tail, so this is what lets "don't repeat yourself" survive a
                # restart. 120 is far more than RECENT_CELLS, so the window is still chosen for
                # file size -- but it is no longer only a file-size decision.
                "trials": self.trials[-120:],
                # the online head accumulates ACROSS evenings: it is preference, not body state
                "heads": {k: v.dump() for k, v in self.heads.items()} if self.heads else {},
                "cycle_hist": [r.tolist() for r in self.hist[-HIST_ENC:]]}

def encoder_dataset(params, haz, rng, n=3000, hs=HIST_ENC):
    """What the encoder learns from: run cycles on the NOMINAL plant and record, for each policy,
    the history that preceded it and what it went on to reach. Dynamics only -- no keypress, no
    reward, no preference. The frozen half must know how this body MOVES and nothing about taste,
    because taste is the online head's job and it learns that from him."""
    p = dict(zip(CORE_NAMES, params))
    hist = np.zeros((n, hs, 11), np.float32)
    pol = rng.uniform(POLICY_BOUNDS[:, 0], POLICY_BOUNDS[:, 1], (n, len(POLICY_NAMES)))
    y = np.zeros((n, 2), np.float32)

    def one(d, q, st):
        d.p = np.asarray(q, float); d._band()
        d.phase, d.phase_t, d.peak, d.timed_out = "build", 0.0, None, False
        lo, hi, dur = d.next_action()
        t, c0, pk = 0.0, d.cycles, 0.0
        cum = False
        # Derived, not a magic 600: the two deadlines are what actually bound a cycle, so this
        # cannot drift away from them. It is a backstop against a future change that stops those
        # deadlines closing a cycle at all -- unreachable today (a cycle is bounded at ~485 s) and
        # loud rather than silent if that ever stops being true, because a sample truncated
        # quietly is a training row that says a policy timed out when it did not.
        while d.cycles == c0 and t < MAX_BUILD_S + MAX_EASE_S + 30.0:
            A = st.step(hi - lo, dur); t += dur; pk = max(pk, A)
            # Orgasms drawn from the fitted hazard, not assumed away. This column was a hardcoded
            # 0.0 for every one of the 18000 history rows, so its weights never left their
            # initialisation -- and live sets it on precisely the cycle the heads most need read
            # correctly.
            cum = rng.random() < cum_hazard(A, dur, *haz)
            if cum:
                st.refract()
            d.commit_observation(A, dur, cum)
            if d.cycles != c0:
                break
            lo, hi, dur = d.next_action()
        if d.cycles == c0:
            raise RuntimeError(f"encoder_dataset: a cycle ran {t:.0f}s without closing, past both "
                               f"deadlines ({MAX_BUILD_S:.0f}+{MAX_EASE_S:.0f}s). The rollout and "
                               f"the deadlines disagree; the training set would be wrong.")
        return pk, t, float(cum)

    # ONE Director for the whole corpus. Constructing one costs an _extent(), and building 3000
    # of them re-measured an identical descriptor grid 3000 times. Director._EXTENT has since
    # memoised that per (core, modality) -- 40 ms on the first call, 0.2 ms after -- so the reuse
    # here is worth 0.6 s rather than the four minutes it was worth when it was written. Kept
    # because its state is reset per sample by `one()` anyway, so one Director is also simpler.
    d = Director(rng, p)
    for b in range(n):
        st = CoreStreamer(p)
        for k in range(hs):
            q = rng.uniform(POLICY_BOUNDS[:, 0], POLICY_BOUNDS[:, 1])
            pk, t, cq = one(d, q, st)
            hist[b, k] = np.r_[(q - POLICY_BOUNDS[:, 0]) / POLICY_SPAN,
                               pk, min(t, 300.) / 300., cq]
        pk, t, _ = one(d, pol[b], st)
        y[b] = [pk, min(t, 300.) / 300.]
    return hist, pol, y


def train_encoder(params, haz, rng, iters=1500):
    """Fit the frozen CfC encoder. torch is imported HERE and only here -- `live` never needs it,
    because the weights are exported as plain JSON and the forward pass runs in numpy.

    The cell is the closed-form continuous-time formulation of Hasani & Lechner, backbone included:
    all four heads -- the two feature branches AND the two that form the time gate -- read a
    non-linear projection of [input, hidden], not the raw concatenation. Omitting that backbone
    linearises exactly the mechanism that lets the cell rescale its own time constants."""
    import torch
    import torch.nn as nn
    # The core fit is byte-reproducible and this was not: same seed, same data, 24 vs 8 threads
    # moved a weight by 3.8e-01 -- a completely different net -- because the intra-op split
    # changes the reduction order. Pinning makes encoder.json a function of the corpus alone,
    # verified identical across separate processes. Free, and then some: 24 threads was the
    # SLOWEST setting measured on this box (17.1 s against 12.4 s at 8).
    torch.set_num_threads(8)

    class Cell(nn.Module):
        def __init__(s, din, h, bb=128):
            super().__init__()
            s.h = h
            s.bb = nn.Linear(din + h, bb)
            s.ff1, s.ff2 = nn.Linear(bb, h), nn.Linear(bb, h)
            s.ta, s.tb = nn.Linear(bb, h), nn.Linear(bb, h)

        def forward(s, x, hid, ts):
            z = 1.7159 * torch.tanh(0.666 * s.bb(torch.cat([x, hid], -1)))   # LeCun tanh
            g = torch.sigmoid(s.ta(z) * ts + s.tb(z))
            return torch.tanh(s.ff1(z)) * (1.0 - g) + torch.tanh(s.ff2(z)) * g

    class Enc(nn.Module):
        def __init__(s):
            super().__init__()
            s.cell = Cell(11, NH_ENC)
            s.mlp = nn.Sequential(nn.Linear(NH_ENC + 8 + 6, 64), nn.SiLU(),
                                  nn.Linear(64, NF_ENC), nn.Tanh())
            s.head = nn.Linear(NF_ENC, 2)

        def phi(s, hist, pol, desc, keep=None):
            h = torch.zeros(pol.shape[0], NH_ENC)
            for t in range(hist.shape[1]):
                nh = s.cell(hist[:, t], h, hist[:, t, 9:10] * 300.0)  # the cycle's REAL seconds
                h = nh if keep is None else torch.where(keep[:, t:t + 1], nh, h)
            return s.mlp(torch.cat([h, pol, desc], -1))

        def forward(s, hist, pol, desc, keep=None):
            return s.head(s.phi(hist, pol, desc, keep))

    torch.manual_seed(0)
    hist, pol, y = encoder_dataset(params, haz, rng)
    d0 = Director(np.random.default_rng(0), dict(zip(CORE_NAMES, params)))
    H = torch.tensor(hist)
    Pt = torch.tensor((pol - POLICY_BOUNDS[:, 0]) / POLICY_SPAN, dtype=torch.float32)
    Dt = torch.tensor((descriptors(pol, d0.core) - d0.dmin) / d0.drange, dtype=torch.float32)
    Y = torch.tensor(y)
    net = Enc()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    T_HIST, NV = H.shape[1], 300     # the last NV samples are never trained on; see the print
    for _ in range(iters):
        i = torch.tensor(rng.integers(0, len(Y) - NV, 256))
        # Live feeds 0..HIST_ENC cycles, not always T_HIST. A session's first cycles have NO
        # history, and every refit moves the predictor hash, which drops the stored cycle_hist and
        # puts the next session back at zero -- so this is not a rare corner, it is the first two
        # cycles of the first evening after every train. Measured on the shipped encoder: at T=0
        # it scored 0.0332 against 0.0110 for predicting the mean, i.e. 3x WORSE than useless, on
        # exactly the cycles the RLS heads are being updated from. Masking a random-length prefix
        # off each sample trains the identical function on a wider input set -- skipping the first
        # k steps IS starting the recurrence at step k, because h starts at zero. T=0 improved
        # 12.5x, T=1 13x, and T>=2 did not move outside seed noise.
        L = torch.tensor(rng.integers(0, T_HIST + 1, 256))
        keep = torch.arange(T_HIST)[None] >= (T_HIST - L[:, None])
        loss = ((net(H[i], Pt[i], Dt[i], keep) - Y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Genuinely held out, at last. This printed the last minibatch's TRAINING loss; I replaced
    # that with a resample and labelled it HELD OUT, which it was not -- j indexed the same 3000
    # rows the loop had trained on ~128 times each. A number that names itself wrong is worse than
    # the number it replaced. NV rows are excluded from the loop above and scored at the same
    # random-length prefixes live actually feeds, so this measures the thing the encoder does.
    with torch.no_grad():
        jl = torch.tensor(rng.integers(0, T_HIST + 1, NV))
        jk = torch.arange(T_HIST)[None] >= (T_HIST - jl[:, None])
        held = ((net(H[-NV:], Pt[-NV:], Dt[-NV:], jk) - Y[-NV:]) ** 2).mean()
    print(f"encoder  held-out MSE={held.item():.5f}  "
          f"{sum(q.numel() for q in net.parameters())} params")
    return ({k: v.detach().numpy().round(6).tolist() for k, v in net.state_dict().items()},
            d0.dmin.tolist(), d0.drange.tolist())

def cum_hazard(A, dt, h0, sharp):
    """Terminates rollouts, so a policy cannot ride a deterministic ceiling forever."""
    return 1.0 - math.exp(-dt * h0 * math.exp(sharp * (min(A, 1.0) - 1.0)))

def invariants(sessions, params):
    """Deterministic properties of the model about to be exported, each with an answer that is
    right or wrong independently of any policy, any simulated body and any taste: the streamer and
    the batch scan are one recurrence and must agree to the bit, zero stimulation must decay, and
    the four outcome classes must rank in the one order the whole reward is built on. This is the
    ONLY check entitled to block an export, and everything it looks at it can actually decide."""
    p = dict(zip(CORE_NAMES, params))
    # Both a random draw AND the params actually being exported. A random vector exercises corners
    # the fit never reaches; the deployed one is the only vector that ships. It used to be the
    # random draw alone, so the gate tested a model nobody runs.
    rp = np.random.default_rng(7).uniform(CORE_BOUNDS[:, 0], CORE_BOUNDS[:, 1])
    worst = max(max(float(np.max(np.abs(predict_core(pv, s) - stream_session(pv, s))))
                    for s in sessions)
                for pv in (rp, np.asarray(params, float)))
    # stream_session always builds rate_ref=None, so the vibrator arm of CoreStreamer.step had no
    # batch counterpart and NOTHING compared it to anything -- deleting that branch outright still
    # exported clean. It has no batch twin by construction, so check it against ode() directly.
    sv = CoreStreamer(p, rate_ref=VIB_RATE_REF)
    E = A = H = 0.0
    for it, dt in zip(np.linspace(0.0, 1.0, 200), np.linspace(0.05, 5.0, 200)):
        a1 = sv.step(it, dt)
        P_ = min(max(float(it), 1e-4), 1.0) ** p["a"] * max(VIB_RATE_REF, 1e-4) ** p["b"]
        E, A, H = ode(E, A, H, P_, max(float(dt), 1e-3), *[p[k] for k in CORE_NAMES[2:]])
        worst = max(worst, abs(a1 - min(max(float(A), 0.0), 1.0)))

    def decay(A0, seconds):
        st = CoreStreamer(p); st.E, st.A = (0.9 if A0 > 0.4 else 0.0), float(A0)
        for _ in range(int(seconds)):
            A = st.step(0.10, 1.0)
        return A
    drift, stop = decay(0.0, 600.0), decay(0.75, 120.0)
    order, tos, cums, cleans = outcome_ordering()
    ok = bool(worst == 0.0 and drift <= 0.40 and stop <= 0.40 and order)
    print(f"\ninvariants  parity={worst:.1e}  rest_drift={drift:.3f}  decay_from_0.75={stop:.3f}")
    print(f"  outcome order  timeout<={max(tos):+.3f} < cum<={max(cums):+.3f} < "
          f"clean>={min(cleans):+.3f}  {'held' if order else 'BROKEN'}")
    print("  " + ("PASS -- the exported model obeys them exactly."
                  if ok else "FAIL -- an invariant is violated; the model is not exported."))
    return ok

SIM_PRESS_P = 0.5   # policy_probe only: chance he rates a level change. Never a live knob.
PROBE_SEEDS, PROBE_S, PROBE_STEPS = 6, 25 * 60.0, 20000

def policy_probe(params, haz, encoder=None):
    """A SIMULATION, printed and then dropped on the floor. It returns nothing, it makes no claim,
    and cmd_train runs it only after the export has already happened -- so there is no shape in
    which a number from here can withhold a model.

    Structurally circular, which is exactly why it decides nothing: the plant IS the CoreStreamer
    the Director observes, the hazard was fitted to that same model, and the presses the reward
    needs are faked by quantizing the plant's own arousal onto the 0..8 keyboard grid. Model, body
    and grader are one object. It says what the mechanics DID over a fixed rollout -- whether
    cycles close, stall or tip -- and nothing whatever about a policy, about hardware, about BLE
    timing, or about a person. The faked press must never reach cmd_live, where the only admissible
    press is one he actually made.

    SPARSE on purpose: he rates when the level he would report CHANGES, and only SIM_PRESS_P of
    those times. Pressing every stroke deletes the sparse-feedback condition, and with it every
    bug that only shows up when a cycle carries no press at all."""
    p = dict(zip(CORE_NAMES, params))
    if encoder is None:      # the probe drives a real Director, and there is only the neural one
        return print("policy_probe not run -- no encoder given")
    cyc, cums, to, npress, nstep, capped = [], [], [], 0, 0, 0
    for seed in range(PROBE_SEEDS):
        r = np.random.default_rng(seed)
        st, d = CoreStreamer(p), Director(r, p, encoder=encoder)
        t, n, lvl, steps = 0.0, 0, -1, 0
        lo, hi, dur = d.next_action()        # same order as the live loop: act, observe, act again
        # Bounded by STEPS as well as by the horizon: a policy that asks for a zero-length command
        # advances t by nothing, and an unbounded report is a report that can hang a train run.
        while t < PROBE_S and steps < PROBE_STEPS:
            A = st.step(hi - lo, dur); t += dur; nstep += 1; steps += 1
            q = min(int(A * 9.0), 8)
            if q != lvl and r.random() < SIM_PRESS_P:
                d.note_press(q / 9.0); npress += 1
            lvl = q
            cum = r.random() < cum_hazard(d.corrected(A), dur, *haz)
            if cum:
                n += 1; st.refract()
            d.commit_observation(A, dur, cum)
            lo, hi, dur = d.next_action()
        capped += int(steps >= PROBE_STEPS)
        cyc.append(d.cycles); cums.append(n); to.append(d.timeouts)
    rate, pf = sum(to) / max(sum(cyc), 1), npress / max(nstep, 1)
    print("policy_probe (report only -- a simulated body and simulated presses; decides nothing)")
    print(f"  {PROBE_SEEDS}x{PROBE_S / 60:.0f}min: cycles={cyc} cums={cums} timeouts={to} "
          f"timeout_rate={rate:.2f} press_rate={pf:.1%}"
          + (f"  rollouts stopped at the {PROBE_STEPS}-step cap: {capped}" if capped else ""))

EPOCH, EPOCH_WARN_AT = CKPT + "/epoch.json", 3

def dataset_digest(path):
    """Content-addressed, and deliberately blind to filenames and to file order: renaming a FILE or
    re-sorting a directory does not make a new corpus, so neither can reset the count of how many
    times this one has been looked at. One changed byte does.

    Renaming a SESSION ID does reset it, because the id is content. That happened on 2026-08-15
    (LIVE_<timestamp> -> LIVE_n) and priced nine much-examined sessions as never looked at, even
    though every numeric array the loader returns was byte-identical across the change. The counter
    measures bytes, not looks, and there is no cheap fix -- so read it as a floor."""
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    parts = sorted(hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files)
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]

def next_session_id(path, prefix, claim_dir=None):
    """LIVE_1, LIVE_2, ... -- an index, never a clock.

    The id used to be strftime("%Y%m%d_%H%M%S"), which names the minute and second he masturbated.
    That is a behavioural fingerprint, and this corpus is meant to be publishable: four ids from
    one night give the duration and the spacing, and across enough datasets that kind of column is
    exactly what re-identifies people. An index carries the one thing the corpus actually needs --
    which recording is which -- and nothing about when.

    Rename them to anything you like afterwards; only the prefix and the digits are read back, and
    a name that parses as neither simply never collides.
    """
    n = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            for ln in f:
                a, _, _ = ln.partition(",")
                # isdecimal, not isdigit: 95 BMP characters are isdigit and NOT accepted by int()
                # (superscripts, Ethiopic numerals), and a long enough run trips Python's own
                # int/str conversion limit. Either raised HERE -- before cmd_live's try/finally
                # opens -- which leaves the toy connected, parked and never sent a stop.
                d = a[len(prefix):]
                if a.startswith(prefix) and d.isdecimal() and len(d) < 19:
                    n = max(n, int(d))
    # Claim it before returning, so two processes started in the same window cannot both get it.
    # Nothing is written to the corpus until the first command lands, so the id was unowned for as
    # long as the 90 s device scan takes -- and drop_session matches on `sid + ","`, so the second
    # process ending unrated deleted BOTH evenings. Measured: 3 rated rows, all removed.
    # A claim left behind by a crash simply advances the index by one, which costs nothing.
    if claim_dir:
        os.makedirs(claim_dir, exist_ok=True)
        while True:
            sid = f"{prefix}{n + 1}"
            try:
                os.close(os.open(os.path.join(claim_dir, sid + ".claim"),
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return sid
            except FileExistsError:
                n += 1
    return f"{prefix}{n + 1}"


def drop_session(path, sid):
    """Un-append one recording from the corpus. Rows are flushed as they are written, because a
    crash must never cost the evening -- so the only way to reject a session after the fact is to
    rewrite the file without it. Staged and swapped, so a failure here cannot eat the corpus.

    newline="" on BOTH handles. Text mode reads this file's CRLF as LF and the write passes it
    through untranslated, so rejecting one session rewrote all 12697 line endings -- which is a
    different dataset_digest for a corpus nobody changed, the precise laundering bump_epoch's
    docstring says cannot happen. It also left the body LF while csv.writer kept appending CRLF."""
    with open(path, encoding="utf-8", newline="") as f:
        keep = [ln for ln in f if not ln.startswith(sid + ",")]
    if keep and not keep[-1].endswith("\n"):
        keep[-1] += "\r\n"          # a torn last row must not glue itself to tonight's first
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(keep)
    os.replace(tmp, path)

def bump_epoch(path, sessions):
    """One evaluation of one corpus, counted. Every LOSO table, every fitloss, every held-out MAE
    and every side-by-side of two candidates is another look at the SAME nine sessions, and looks
    do not come back: past a handful, whatever is chosen is chosen partly by this corpus's noise
    rather than by the body it came from. Nothing here blocks anything -- it prices the evidence.

    Counts are kept per dataset digest and persist, so looking at another corpus and back cannot
    launder them, and this runs BEFORE the fit so a run abandoned after reading the numbers has
    still spent its look."""
    digest = dataset_digest(path)
    rec = {}
    if os.path.exists(EPOCH):
        with suppress(Exception):
            rec = json.load(open(EPOCH))
    seen = dict(rec.get("datasets") or {})
    now, prev = float(time.time()), seen.get(digest) or {}
    n = int(prev.get("evaluations", 0)) + 1
    seen[digest] = {"evaluations": n, "first_seen": float(prev.get("first_seen", now)),
                    "last_seen": now}
    out = {"dataset": digest, "evaluations": n, "datasets": seen,
           "n_sessions": len(sessions),
           "n_rows": int(sum(len(s["dur_s"]) for s in sessions)),
           "n_presses": int(sum(len(s["press_idx"]) for s in sessions)),
           "n_cum_events": int(sum(len(cum_onsets(s["t_s"], s["press_idx"], s["press_vals"]))
                                   for s in sessions))}
    write_json(EPOCH, out)
    return out

LOSO_SEEDS = (0, 1, 2, 3)   # see cmd_train: one seed reports noise as if it were a measurement


def loso_fold(arg):
    """One held-out fold at one CEM seed. Top-level and taking one picklable argument so it can
    cross a process boundary on Windows spawn, where a closure cannot. Its own generator, so the
    fold depends on which session is held out and on which seed -- not on how many folds ran
    first, and therefore not on whether any of them ran at the same time."""
    sessions, i, seed = arg
    held = sessions[i]
    fp, loss = fit_core([s for j, s in enumerate(sessions) if j != i],
                        np.random.default_rng(seed))
    pred = predict_core(fp, held)
    mae, tau = press_metrics(held["press_idx"], held["press_vals"], pred)
    return ({"session": held["name"], "peak": round(float(pred.max()), 3),
             "n_press": int(len(held["press_idx"])), "press_mae": mae,
             "kendall_tau": tau, "fitloss": round(float(loss), 4)},
            np.abs(pred[held["press_idx"]] - held["press_vals"]), float(pred.max()))

def cmd_train(args):
    sessions = training_sessions(args.data)
    if not sessions:
        sys.exit(f"no usable sessions found in {args.data}")
    ep = bump_epoch(args.data, sessions)
    print(f"epoch {ep['dataset']}: evaluation #{ep['evaluations']} of {ep['n_sessions']} sessions "
          f"/ {ep['n_rows']} rows / {ep['n_presses']} presses / {ep['n_cum_events']} cum events")
    if ep["evaluations"] > EPOCH_WARN_AT:
        print(f"  WARNING: this corpus has now been evaluated {ep['evaluations']} times. Anything "
              f"chosen between models on the numbers below is no longer independent evidence -- "
              f"the choice is partly a fit to this corpus's noise. Record new sessions.")
    if len(sessions) > 1:
        # Held out one session at a time and scored ONLY on real keypresses: the interpolated
        # target is an assumption of the training loss, so scoring against it would grade the
        # model on its own prior. Pooled press-MAE weights every PRESS equally -- deliberately the
        # opposite of eval_pop's objective, which weights every SESSION equally; see the note there.
        # Every fold starts from its OWN generator at the same seed. One generator threaded through
        # the loop made fold N's fit depend on how many augmentation draws the earlier folds
        # happened to consume -- i.e. on session ORDER and LENGTH -- so held-session identity and
        # optimizer noise were confounded, and CEM seed alone is known to move this fit materially.
        #
        # That independence is also what makes the folds SEPARABLE: a fold is defined by which
        # session is held out and by nothing else, so running them concurrently cannot change any
        # number here. map() preserves order, so even the printed table is unchanged, byte for
        # byte. Nine sequential CEM fits are ~85% of `train`'s wall clock; this is the one place in
        # the program with real parallelism available for free.
        from concurrent.futures import ProcessPoolExecutor
        # Folds are independent by construction and map() preserves order, so HOW MANY run at once
        # cannot move a number below -- verified bit-identical at 32, 3 and 1 workers. But each
        # one carries the full scan buffers, and ProcessPoolExecutor() defaults to cpu_count():
        # 32 x ~18 MB per session is 7.8 GB at 9 sessions, 20 GB at 30, 36 GB at 109 -- against
        # 31.8 GB of RAM. Unbudgeted, `train` stops working somewhere past 100 sessions; the cap
        # first binds at 17, where it drops the pool to 30 workers.
        rows, T = sum(len(s["dur_s"]) for s in sessions), max(len(s["dur_s"]) for s in sessions)
        # chunk_for is a STEP function of n_rows, so it must not be asked a question no job in the
        # pool will ask it. Two ways that bit: fit_core charges chunk_for against the AUGMENTED
        # pool it builds (augment_raw truncates, so ~0.9x these rows), and a FOLD is one session
        # SMALLER than the deployed fit, which can put it a step ABOVE -- the pool's biggest job is
        # not always its biggest corpus. So charge the bytes at the largest job and the chunk at
        # the most generous one, and per_fold BOUNDS the pool instead of estimating one member of
        # it. Measured worst case for the naive form over ~30k shapes: 1.80x over budget, first
        # reachable about 17 sessions from now.
        lo_rows = int(0.73 * (rows - T))     # (1 + N_AUG*0.6)/(1 + N_AUG): augment_raw's floor
        per_fold = fit_core_bytes(len(sessions), rows, T,
                                  max(chunk_for(len(sessions), rows, T),
                                      chunk_for(max(1, len(sessions) - 1), lo_rows, T))) + 77e6
        nw = max(1, min(os.cpu_count() or 1, int(LOSO_MEM_BUDGET // per_fold)))
        if nw < min(os.cpu_count() or 1, len(sessions)):
            print(f"  LOSO pool capped at {nw} workers: one fold needs {per_fold / 1e9:.2f} GB")
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [[ex.submit(loso_fold, (sessions, i, sd)) for i in range(len(sessions))]
                    for sd in LOSO_SEEDS]
            # The deployed fit is a tenth job by this code's own argument: own generator, same
            # seed, never depends on whether LOSO ran. It used to run serially AFTER the pool shut
            # down, alone on one core for 21 s. Folds are submitted first on purpose -- fit-first
            # schedules marginally better but reorders what test_p4 asserts.
            deployed = ex.submit(fit_core, sessions, np.random.default_rng(0))
            runs = [[f.result() for f in row] for row in futs]   # same order as map()
        out = runs[0]                     # the printed table stays LOSO_SEEDS[0], as it always was
        folds, errs = [f for f, _, _ in out], [e for _, e, _ in out]
        # peak is carried back UNROUNDED for this line and rounded only into the JSON, because
        # round(x,3) then format(.2f) is not always format(.2f) -- the fold record must not be able
        # to change a printed digit.
        for f, _, peak in out:
            print(f"  held={f['session']:<28} peak={peak:.2f} fitloss={f['fitloss']:.4f} "
                  f"n={f['n_press']:<3d} MAE={f['press_mae']} tau={f['kendall_tau']}")
        pooled = np.concatenate(errs) if any(len(e) for e in errs) else np.array([])
        pooled_mae = round(float(pooled.mean()), 3) if len(pooled) else None
        # Across CEM seeds, because ONE seed reports optimizer noise as if it were a measurement.
        # Measured on this corpus: the same code on the same data spans 0.189 to 0.200 with nothing
        # changed but the seed, which is wider than most differences anyone would act on. A model
        # comparison inside that band is not a comparison. The printed fold table stays at
        # LOSO_SEEDS[0] so it remains one coherent run rather than an average of nine.
        spread = [float(np.concatenate([e for _, e, _ in r]).mean()) for r in runs]
        sd = float(np.std(spread, ddof=1)) if len(spread) > 1 else 0.0
        print(f"\nLOSO held-out press-MAE={np.mean(spread):.3f} +/- {sd:.3f} "
              f"over {len(pooled)} presses / {len(folds)} folds x {len(LOSO_SEEDS)} CEM seeds"
              f"\n  per-seed: {', '.join(f'{v:.3f}' for v in spread)}"
              f"   <- anything inside this band is seed noise, not a result")
        # Band-wise, because one pooled number hides WHERE the model is wrong, and the bands are
        # not interchangeable: the low band is the only one validated live, while the top band is
        # the one the Director actually steers by. errs comes back in submission order and
        # training_sessions drops sessions with no press, so every fold contributes both arrays.
        labs = np.concatenate([s["press_vals"] for s in sessions]) if len(pooled) else pooled
        bands = {}
        for a, b, nm in ((0, .4, "low 0.0-0.4"), (.4, .7, "mid 0.4-0.7"), (.7, 1.01, "TOP 0.7-1.0")):
            m = (labs >= a) & (labs < b)
            bands[nm] = round(float(pooled[m].mean()), 3) if m.any() else None
            print(f"  {nm}: n={int(m.sum()):3d}" + (f"  MAE={bands[nm]}" if m.any() else ""))
        # pooled_press_mae is the MEAN over seeds, matching the headline. It used to be runs[0] --
        # the single-seed number the multi-seed change exists to stop reporting -- sitting under
        # the most obviously-named key in the file, which is where anyone reads it from.
        write_json(CKPT + "/loso.json",
                   {"pooled_press_mae": round(float(np.mean(spread)), 4),
                    "pooled_sd": round(sd, 4), "pooled_per_seed": [round(v, 4) for v in spread],
                    "seeds": list(LOSO_SEEDS), "pooled_press_mae_seed0": pooled_mae,
                    "bands": bands, "n_press": int(len(pooled)), "folds": folds})
    # its own generator too, so the deployed fit never depends on whether LOSO ran at all
    params, loss = (deployed.result() if len(sessions) > 1 else
                    fit_core(sessions, np.random.default_rng(0)))
    print(f"\nfitloss={loss:.4f} { {n: round(float(v), 4) for n, v in zip(CORE_NAMES, params)} }")
    pinned = [n for n, v, (lo, hi) in zip(CORE_NAMES, params, CORE_BOUNDS)
              if abs(v - lo) < 1e-9 or abs(v - hi) < 1e-9]
    if pinned: print(f"bound-pinned: {pinned}  (bounds are doing the work, not the data)")
    h0, sharp = fit_first_cum_hazard(sessions, params)
    print(f"hazard h0={h0:.5f}/s sharp={sharp:.0f}  "
          f"(median time-to-cum {math.log(2) / (h0 * math.exp(sharp * -0.1)) / 60:.1f}min at A=0.90)")
    # Stage, check, then swap: a failed run must not leave a rejected model loadable by `live`,
    # and must not destroy the previous known-good one.
    cand = CONFIG + ".candidate"
    write_json(cand, {"core_names": CORE_NAMES, "hazard": [h0, sharp],
                      "core_params": {n: float(v) for n, v in zip(CORE_NAMES, params)}})
    if not invariants(sessions, params):
        print(f"NOT exported -- an invariant failed. Candidate kept at {cand}; {CONFIG} unchanged.")
        return 1
    os.replace(cand, CONFIG)
    # The encoder is pinned to the model file that was just written: refit the core and this hash
    # moves, `live` sees the mismatch and refuses to start rather than steering on stale features.
    ew, dmin, drange = train_encoder(params, (h0, sharp), np.random.default_rng(0))
    write_json(ENCODER, {"predictor": load_config(CONFIG)[1], "w": ew,
                         "dmin": dmin, "drange": drange})
    print(f"exported {ENCODER}")
    print(f"exported {CONFIG}")
    # Below the swap on purpose: a report that runs after the file has already moved cannot
    # withhold it -- not by returning False, not by returning garbage, not by crashing. A failure
    # here is a broken analysis, and a broken analysis is never a reason to withhold a model.
    try:
        policy_probe(params, (h0, sharp), encoder=load_encoder(load_config(CONFIG)[1]))
    except Exception as e:
        print(f"policy_probe crashed ({type(e).__name__}: {e}) -- {CONFIG} is already exported")

MODE_BANNER = {
    "label": "0-9 = YOUR arousal (9 = cumming). The keys never touch the toy -- they are the only\n"
             "  thing the Director is scored on. Ctrl-C stops.",
    "auto":  "AUTO -- no key is read. The prediction is fed back as its own label, so the Director\n"
             "  is graded on the output of the model it is steering: it will settle wherever that\n"
             "  model already believes the edge is and stay there. No orgasm can be detected, so\n"
             "  nothing ever makes it back off. NOTHING is recorded and policy memory is not saved.",
    "manual": "MANUAL -- 0-9 DRIVE THE TOY (0 = stop, 9 = deepest and fastest). The Director is out\n"
              "  of the loop entirely. No arousal label exists, so nothing is recorded and policy\n"
              "  memory is not saved; this is a readout of how the predictor answers your own\n"
              "  stroking. Ctrl-C stops.",
}

def cmd_live(args):
    """Ctrl-C stops the toy; every row is flushed as written.

    All timing here is perf_counter, never monotonic: on Windows `monotonic` is GetTickCount64
    with 15.6 ms granularity, which is a tenth of the shortest half-stroke the Keon can take.
    Measured on this box, a 0.13 s movement window timed by monotonic overshoots by 10.4 ms --
    8% of the stroke spent with the toy already stopped -- and the `wall` written into the corpus
    collapses 60 real measurements onto 4 distinct values, injecting +-4% of pure clock artefact
    into the column the ODE is fitted on.

    Three modes, and which one is running decides what is allowed to persist:
      label  (default)  he presses, the Director drives, the session is APPENDED to --data
      --auto            the prediction is its own label; records nothing, saves no policy
      --manual          the keypad drives the toy; no Director at all, records nothing

    Only `label` writes. A trial graded on the predictor's own output is not evidence about him,
    and a run where the keypad was a throttle produced no arousal report at all -- letting either
    reach sessions.csv or policy.json would put the model's opinion into the corpus that trains it.
    """
    import asyncio, csv
    from collections import deque
    try:
        import keyboard
    except Exception:
        keyboard = None
        print("WARNING: no 'keyboard' package -- labels disabled, recording kept regardless.")
    from buttplug.client import ButtplugClient, ButtplugClientWebsocketConnector

    cfg, cfg_hash = load_config(CONFIG)
    streamer = CoreStreamer(cfg["core_params"])
    # policy memory only -- the BODY starts cold every run (no E/A/H carried between days)
    prior = None
    if os.path.exists(POLICY):
        try:                       # the only load here that used to raise. A hand edit or a disk
            prior = json.load(open(POLICY))   # error would end every future evening at startup
        except Exception as e:                # with a traceback instead of a sentence.
            print(f"policy memory at {POLICY} is unreadable ({type(e).__name__}) -- starting "
                  f"fresh. Move it aside if you want to keep it.")
    # NO SILENT FALLBACK. If the online head cannot run, the session does not start. Running a
    # different Director without saying so means an evening that quietly stopped adapting and he
    # would have no way to tell -- which is the one failure mode he asked to never have.
    enc = load_encoder(cfg_hash)
    if enc is None:
        sys.exit("online learning cannot start (reason above).\n"
                 f"  fix: python cumpredict.py train --data {args.data}")
    director = Director(np.random.default_rng(), cfg["core_params"], prior, encoder=enc)
    print(f"online learning ON -- CfC encoder {NF_ENC}d, 3 RLS heads, "
          f"{len(director.hist)} cycles of carried history")
    os.makedirs(CKPT, exist_ok=True)
    logf = open(CKPT + "/live.log", "a")
    t_log0 = time.perf_counter()

    def log(m):
        """Timestamped by ELAPSED session time, never by the clock.

        How long he has been going is the quantity this log is actually read for -- "the timeout
        fired at +38m" answers a question, "the timeout fired at 01:34" does not. And the wall
        clock answers a question nobody should be able to ask of this file: what time of night he
        does this, how long he lasts, how often. The session ids stopped being timestamps for the
        same reason; this was the last clock left anywhere near the recording."""
        el = time.perf_counter() - t_log0
        logf.write(f"[+{int(el // 60):3d}m{el % 60:04.1f}s] {m}\n"); logf.flush()

    mode = "manual" if args.manual else "auto" if args.auto else "label"
    log(f"START mode={mode} config={cfg_hash} finish_on_cum={args.finish_on_cum} "
        f"policy_prior={bool(prior)}")
    is_vib, dev_key, last_pos = [False], [None], [None]

    def caps_of(d):
        m = getattr(d, "allowed_messages", {})
        return [c for c, k in (("vibrate", "VibrateCmd"), ("linear", "LinearCmd"),
                               ("rotate", "RotateCmd")) if k in m]

    def key_of(d):
        """Hardware identity, not just modality: two linear toys are not one experiment."""
        return f"{getattr(d, 'name', '?')}[{','.join(caps_of(d))}]"

    def attach_device(dev):
        """Runs on connect AND on every reconnect. Everything that is a property of the machine
        rather than of the session is decided here, and each of them is a context boundary."""
        nonlocal prior
        m = getattr(dev, "allowed_messages", {})
        is_vib[0] = "LinearCmd" not in m and ("VibrateCmd" in m or "RotateCmd" in m)
        streamer.rate_ref = VIB_RATE_REF if is_vib[0] else None
        log(f"MODALITY {'VIBRATION' if is_vib[0] else 'LINEAR'} caps={caps_of(dev)}")
        # Modality is only known now, after the policy prior was loaded. Rewards earned under a
        # different predictor, actuator, action space or reward scale are not comparable to the
        # ones this session will earn -- discard, don't rank. reset_prior drops the heads and the
        # cycle history with them, because a weight learned on a stroker cannot rank amplitudes.
        tag = "vib" if is_vib[0] else "linear"
        if prior and (prior.get("predictor") != cfg_hash or prior.get("modality") != tag
                      or prior.get("names") != POLICY_NAMES
                      or prior.get("reward") != [CUM_COST, TIMEOUT_COST]):
            log(f"POLICY RESET: prior={prior.get('predictor')}/{prior.get('modality')}"
                f"/{prior.get('reward')} now={cfg_hash}/{tag}/{[CUM_COST, TIMEOUT_COST]}")
            print("policy prior discarded -- different model, modality, action space or reward")
            director.reset_prior()
        # CONSUME the prior. It is a startup input, not a standing condition: a reconnect re-runs
        # this function, and re-testing the same stale file would fire the reset a second time,
        # deleting everything the session had learned since the first one.
        prior = None
        # only now can the right cell grid be measured; a no-op on a reconnect to the same
        # toy, so the trial already running survives it
        director.set_modality(tag)
        k = key_of(dev)
        if dev_key[0] is not None and k != dev_key[0]:
            # Same policy, same heads, same trial, DIFFERENT hardware is not one experiment.
            log(f"DEVICE CHANGED {dev_key[0]} -> {k}: censoring the cycle in flight")
            print(f"\n different toy ({k}) -- the running trial is discarded, not transferred")
            director.censor_cycle()
        dev_key[0] = k
        last_pos[0] = None            # wherever the actuator physically is, we no longer know
        print("modality: VIBRATION (auto) -- arousal readout UNCALIBRATED, trust your presses"
              if is_vib[0] else "modality: LINEAR (auto)")

    def harden(client):
        """buttplug-py 0.3.0 answers a request by setting a result on the future the caller is
        waiting on -- and drive()'s wait_for CANCELS that future when a send runs long. The late
        Ok then lands on a cancelled future, InvalidStateError escapes _consumer_handler's try
        (which wraps only recv), and the read loop DIES. Measured: after the first timeout, 0 of
        the next 3 commands got through, and every one after that timed out too, forever -- with
        the Ctrl-C that follows hanging on a stop that awaits a reply nobody will send. One slow
        command must not cost the evening. Popping also stops _msg_tasks growing by one entry per
        command for the whole session."""
        async def _handle(msg):
            f = client._msg_tasks.pop(getattr(msg, "id", None), None)
            if f is not None:
                if not f.done():          # cancelled by wait_for: the caller has given up, and
                    f.set_result(msg)     # setting a result on it is what used to raise
                return
            with suppress(Exception):     # a DeviceRemoved for an index we no longer hold used
                await client._parse_message(msg)      # to kill the read loop the same way

        client._handle_message = _handle

    async def setup():
        client = ButtplugClient("CumPredict")
        harden(client)                    # before connect, so no message can arrive unprotected
        await client.connect(ButtplugClientWebsocketConnector(args.intiface))
        print("connected; scanning -- power the toy on whenever (90s)")
        await client.start_scanning()
        waited = 0.0
        while waited < 90.0 and not client.devices:
            await asyncio.sleep(2.0); waited += 2.0
        await client.stop_scanning()
        devs = dict(client.devices)

        async def bail(msg):
            with suppress(Exception):
                await client.disconnect()
            raise RuntimeError(msg)
        if not devs:
            await bail(f"no device found -- is Intiface running at {args.intiface}?")
        for i, d in devs.items():
            print(f"  device {i}: {d.name} [{', '.join(caps_of(d)) or 'none'}]")
        # The toy we were already running is the first choice: a reconnect that silently lands on
        # a different actuator continues the session on hardware the trial in flight never touched.
        # Then linear, the modality the model was fit on. Then whatever is there.
        same = [d for d in devs.values() if key_of(d) == dev_key[0]]
        m = [d for d in devs.values() if "LinearCmd" in getattr(d, "allowed_messages", {})]
        dev = (same or m or list(devs.values()))[0]
        if not caps_of(dev):
            await bail(f"'{dev.name}' exposes no actuator this build can drive "
                       f"({sorted(getattr(dev, 'allowed_messages', {}))}). Driving it would record "
                       f"a full session of strokes no actuator ever performed.")
        if len(devs) > 1:
            print(f"  -> using '{dev.name}'")
        return client, dev

    up = [False]

    async def drive(dev, lo, hi, dur):
        """Alternate lo<->hi so the window is a real stroke: lo=0.45,hi=1.0 is the shallowest
        window the bounds allow, lo=0,hi=0.4 is base-only. Vibrators take hi as amplitude and
        ignore lo. Returns the target actually commanded -- LinearCmd is a (duration, POSITION)
        primitive, so the movement is |target - wherever it was|, which is hi-lo only while the
        window is unchanged and the parity is intact."""
        msgs = getattr(dev, "allowed_messages", {})
        for attempt in (1, 2):
            try:
                if "LinearCmd" in msgs:
                    tgt = hi if up[0] else lo
                    await asyncio.wait_for(dev.send_linear_cmd((int(max(dur, .05) * 1000), tgt)),
                                           max(1.0, dur))
                    up[0] = not up[0]     # flip only on success, else retry sends the other end
                    return tgt
                if "VibrateCmd" in msgs:
                    await asyncio.wait_for(dev.send_vibrate_cmd(float(hi)), 5.0)
                elif "RotateCmd" in msgs:
                    await asyncio.wait_for(dev.send_rotate_cmd((float(hi), True)), 5.0)
                # None, not a success, when the device has no actuator this build knows: control
                # used to fall off the end of the try and return float(hi), so the loop recorded
                # 370 rows of strokes that no actuator performed. setup() now refuses such a
                # device outright; this is the second door on the same room.
                return float(hi) if ("VibrateCmd" in msgs or "RotateCmd" in msgs) else None
            except Exception as e:
                log(f"CMD ERROR ({attempt}/2) {type(e).__name__}: {e}")
                if attempt == 1:
                    await asyncio.sleep(0.3)
        return None

    async def park(dev):
        """One deliberate move to a known position, outside any trial and before any row exists.
        LinearCmd is (duration, POSITION), so the movement is |target - wherever it already was|;
        at connect that is unknown, and a stroke measured from an invented origin is invented data.
        Parking buys an exact travel for every row after it. Vibrators need none -- an amplitude is
        absolute. If it fails, the first stroke establishes the position instead and forfeits its
        own row."""
        if "LinearCmd" not in getattr(dev, "allowed_messages", {}):
            return
        if await drive(dev, 0.0, 0.0, PARK_S) is not None:
            await asyncio.sleep(PARK_S)
            last_pos[0], up[0] = 0.0, False   # next command goes to lo, as it does from cold
            log("PARKED at 0.00")

    # A scan code is a key POSITION, and two different keys share each of ten of them. Both
    # collisions land inside 0-9, and both are live on this machine:
    #   POSITION -- the navigation cluster IS the numpad. Page Up is scan 73, exactly like numpad
    #               9, and 9 means he came. Measured end to end: one Page Up logged *** CUM ***,
    #               wrote tired_level=1.00 into the corpus, applied REFRACTORY, and charged
    #               CUM_COST to a head that is kept across evenings. Home/End/arrows give 7/1/8264.
    #   LAYOUT   -- GetKeyboardLayoutNameW reports 0000040C (French) here. The unshifted number
    #               row types & e " ' ( - e _ c a, and keyed by POSITION the c scored 1.0 -- so
    #               typing "garcon" in ANY window recorded an orgasm. The hook is global, so which
    #               window has focus is irrelevant.
    # So: the KEYPAD is read by position, which is what a keypad is for and is correct with
    # NumLock either way; the number row is read by what the key actually TYPED.
    #
    # That is NOT a guard against ordinary typing, and an earlier version of this comment claimed
    # it was. Measured on this layout: Caps Lock is a SECOND SHIFT for the number row --
    #     scan 10    none 'c'    shift '9'    caps '9'    shift+caps 'c'
    # so with Caps Lock on, a bare 9 typed in any window rates him, and Shift+9 -- the habit this
    # design encourages -- silently records nothing. There is no state on screen that says which
    # way round it currently is. Hence the warning at startup below, and hence: THE KEYPAD IS THE
    # RELIABLE INPUT. It is immune to layout, to Caps Lock and to NumLock.
    # keyboard fills both fields in before the callback -- measured 46 ns, the same as before.
    NAMED = {str(i): i / 9.0 for i in range(10)}    # what the key TYPED -- the number row
    LEVEL_OF = {}                                   # where the key SITS -- the keypad
    man = [0.0]      # manual: the level his last keypress asked for, 0..1
    pressq = deque()
    down = set()

    def on_key(e):
        """His keypress, captured in keyboard's own listener thread the instant it happens.

        Polling could not see it. read_press() only ran inside the movement window, so the whole
        BLE round trip -- 23% of a session at the 150ms round trip this file elsewhere reports
        observing -- sampled nothing, and a tap that landed there vanished with no log line. That
        is silent loss of the only ground truth in the system, worst in the fast build phase where
        his rating matters most.

        Auto-repeat is not a new press: `down` edge-triggers, so holding 9 for a second is still
        exactly one orgasm. deque.append is atomic under the GIL, so no lock is needed between this
        thread and the loop."""
        # The release is UNCONDITIONAL, before any name lookup. `keyboard` resolves a key's name
        # under the modifier state at that instant, so Shift+9 arrives as '9' on the way down and
        # -- if he lets go of Shift first -- as 'c' on the way up. Gating the discard on a
        # recognised value stranded that scan code in `down` and killed the key for the rest of
        # the session: measured 6 deliberate presses recorded as 1.
        if e.event_type == "up":
            down.discard(e.scan_code)
            return
        v = LEVEL_OF.get(e.scan_code) if e.is_keypad else NAMED.get(e.name)
        if v is None:
            return
        if e.scan_code not in down:
            down.add(e.scan_code)
            pressq.append(v)

    if keyboard is not None:
        # NO SILENT FALLBACK, for the same reason the encoder has none. Both of these used to be
        # `with suppress(Exception)`. A failed name table left LEVEL_OF keyed by STRINGS against
        # int scan codes; a failed hook left no hook at all. Either one ran a whole evening in
        # which not one press was seen -- and then the `en == 0` branch DELETED the recording,
        # saying nothing. Both reproduced in the mock.
        try:
            # Filtered to the ten KEYPAD positions. key_to_scan_codes("9") returns (73, 10) here --
            # the numpad key AND the number-row key -- so this table held the whole number row too,
            # keyed by position. Inert only because is_keypad is never true for scans 2-11: it was
            # the AZERTY-by-position bug fully loaded, one condition away from firing, inside a
            # table whose own name says it is where the key SITS.
            KEYPAD_SC = {71, 72, 73, 75, 76, 77, 79, 80, 81, 82}
            LEVEL_OF = {c: i / 9.0 for i in range(10)
                        for c in keyboard.key_to_scan_codes(str(i)) if c in KEYPAD_SC}
            keyboard.hook(on_key)
        except Exception as e:
            sys.exit(f"the keyboard hook cannot start ({type(e).__name__}: {e}).\n"
                     f"  Every press would be lost and the recording deleted as unrated at the\n"
                     f"  end of the session. Fix `keyboard`, or run --auto / --manual.")

    if keyboard is not None and sys.platform == "win32":
        with suppress(Exception):
            import ctypes
            if ctypes.windll.user32.GetKeyState(0x14) & 1:
                print("\n  CAPS LOCK IS ON. On this keyboard layout that makes the bare number row\n"
                      "  type digits, so anything you type in ANY window rates you -- and Shift+9\n"
                      "  types a letter, so it records nothing. Turn it off, or use the KEYPAD,\n"
                      "  which is immune to layout, Caps Lock and NumLock.\n")
                log("CAPSLOCK ON at start -- number row inverted")

    def manual_action():
        """0 stops; 1-9 ramp depth and speed together, which is the one knob a keypad can carry.
        dur runs from the slowest half-stroke the policy space allows down to the Keon's hardware
        maximum, so his hand and the Director are bounded by the same actuator rather than by
        taste. lo is 0 -- he asked for depth, and depth is measured from the bottom."""
        d_lo, d_hi = POLICY_BOUNDS[POLICY_NAMES.index("dur_hi")]
        return 0.0, float(man[0]), float(d_hi - (d_hi - d_lo) * man[0])

    def drain():
        """Everything he pressed since the last call, oldest first."""
        out = []
        while pressq:
            out.append(pressq.popleft())
        return out

    async def main_loop():
        try:
            client, dev = await setup()
        except Exception as e:
            log(f"SETUP ERROR: {e}")
            sys.exit(f"SETUP ERROR: {e}")
        # The hook is installed before the 90 s scan, so anything typed while waiting for the toy
        # to be switched on would otherwise land on command 1 as a label. Drop the QUEUE only --
        # `down` tracks which key is physically held, and clearing that while he is holding one
        # would make the next auto-repeat read as a fresh press.
        pressq.clear()
        attach_device(dev)
        await park(dev)
        # The destination is decided AFTER the toy identifies itself, because the modality picks
        # the file. Appended, not created: one corpus, one schema, `session` separating the runs.
        # time_elapse is the MEASURED period the command occupied; the requested movement time and
        # every internal state go to live.log instead, so the corpus keeps exactly the four columns
        # the loader reads and a recording made tonight needs no conversion to be trained on.
        dest = os.path.join(*os.path.split(args.data)[:-1],
                            ("VIB_" if is_vib[0] else "") + os.path.basename(args.data))
        sid = next_session_id(dest, "VIB_" if is_vib[0] else "LIVE_", claim_dir=CKPT)
        f = w = None
        if mode == "label":
            fresh = not os.path.exists(dest) or os.path.getsize(dest) == 0
            f = open(dest, "a", newline="")
            w = csv.writer(f)
            if fresh:
                w.writerow(["session", "time_elapse", "intensity", "tired_level"])
        print(f"\n--- {MODE_BANNER[mode]} ---")
        print(f"  -> appending to {os.path.abspath(dest)} as session {sid}\n" if w
              else "  -> nothing will be written\n")
        A, t0, fails, maxlab, esum, en = 0.0, time.perf_counter(), 0, -1.0, 0.0, 0
        outage, censored = 0.0, False   # dead seconds since the last command that actually went out
        vib_at_start = is_vib[0]        # dest and sid were chosen from this; see the guard below
        lo, hi, dur = manual_action() if mode == "manual" else director.next_action()
        try:
            while True:
                ramp = time.perf_counter() - t0
                if ramp < 60.0 and mode != "manual":   # never slam cold -- except when the hand on
                    hi = min(hi, lo + (0.30 + 0.60 * ramp / 60.0) * (hi - lo))   # the throttle is
                                                                                # his, not a policy's
                t_send = time.perf_counter()
                pos = await drive(dev, lo, hi, dur)
                if pos is None:
                    fails += 1
                    if fails >= 2:
                        log("CONNECTION LOST -- reconnecting")
                        print("\n toy disconnected -- reconnecting...")
                        try:
                            with suppress(Exception):
                                await client.disconnect()
                            client, nd = await setup()
                            # attach_device sets modality at its top and device identity at its
                            # bottom; a raise between them used to leave the loop driving the NEW
                            # toy with the OLD toy's last position, so the next row's travel was
                            # measured between two different actuators and entered the corpus as
                            # the stimulus that caused whatever he pressed.
                            # dev BEFORE park: park is awaited, and a Ctrl-C inside it is a
                            # BaseException the handler below does not catch -- the finally
                            # then stopped the toy we had just stopped driving, leaving the
                            # one actually attached to him holding its amplitude. park(nd)
                            # drives nd explicitly, so identity is still settled first.
                            attach_device(nd); dev = nd; await park(nd); fails = 0
                            t0 = time.perf_counter()   # re-arm the cold-start ramp: "never slam cold"
                        except Exception as e:      # is about re-engagement, not about process start
                            log(f"RECONNECT FAILED: {e}")
                            await asyncio.sleep(3.0)
                    out_dt = max(time.perf_counter() - t_send, dur)
                    outage += out_dt
                    # Whatever he pressed at a toy that was NOT MOVING rated nothing. The failure
                    # path `continue`s before drain(), so those presses used to sit in the queue
                    # and land on the first command after recovery -- against the fresh cycle
                    # censor_cycle() had just created to replace the one it threw away. Measured:
                    # one 9 during a 30 s dead radio arrived as *** CUM at PRED=0.03 *** on cycle
                    # 0, charged CUM_COST, fired refract() and moved bias by +0.244 in one step.
                    # Discarded rather than delivered: a press describes a stimulus, and there was
                    # none. Same drain also clears anything typed during a reconnect's 90 s scan.
                    dead = drain()
                    if dead:
                        log(f"DISCARDED {len(dead)} press(es) made while the radio was down: {dead}")
                    A = streamer.step(0.0, out_dt)             # decay over the real outage, and
                    if mode == "manual":                       # age the deadlines by the same
                        lo, hi, dur = manual_action()          # clock, not by the nominal one
                    else:
                        # A dead radio is not the policy's fault, and charging it correctly needs
                        # both of these. Censor BEFORE committing: commit_observation can itself
                        # CLOSE and score the cycle -- from `ease`, where A decaying over a dead
                        # radio is exactly what crosses `release` -- so censoring afterwards
                        # censors the cycle that REPLACED the one just booked, and the policy is
                        # credited (measured +0.440, head moved 0.172) for a release the outage
                        # produced. And censor on ACCUMULATED dead time: an outage arrives as many
                        # short failures, not one long one. Intiface killed costs ~4 s a command,
                        # never over OUTAGE_CENSOR_S, and 480 s of that still booked a -3.000
                        # TIMEOUT against a policy that never actuated anything -- into heads that
                        # persist across evenings, permanently teaching him to avoid a technique
                        # that was fine. Once censored nothing is committed until the radio is
                        # back, so no clock advances and no deadline can fire on dead time.
                        if outage > OUTAGE_CENSOR_S:
                            if not censored:
                                log(f"OUTAGE {outage:.0f}s -- censoring the cycle, not scoring it")
                                director.censor_cycle()
                                censored = True
                        else:
                            director.commit_observation(A, out_dt)
                        lo, hi, dur = director.next_action()
                    continue
                fails, outage, censored = 0, 0.0, False
                # The movement window starts when the command is out, not when we began sending it.
                # Measured from before the send, a 150ms BLE round trip on a 130ms stroke left zero
                # time to move and the next command preempted a stroke that never happened.
                t_move = time.perf_counter()
                while time.perf_counter() - t_move < dur:
                    await asyncio.sleep(0.005)
                presses = drain()   # everything that landed while the command ran is his and
                                    # belongs to it. Draining inside the loop only split one list
                                    # into pieces that were concatenated straight back together --
                                    # 500 calls per command at dur=2.5s -- because the hook thread
                                    # only ever appends, so nothing can be lost by waiting.
                wall = time.perf_counter() - t_send
                # linear: the drive is the travel actually commanded, |target - previous target|.
                # That is hi-lo only while the window and the parity hold; at a cycle boundary, a
                # phase change or a ramp step the machine moves a different distance, and the
                # nominal number would enter the corpus as the stimulus that caused what he felt.
                # vibrator: the drive is the held AMPLITUDE (hi); lo is not actuated at all.
                if is_vib[0]:
                    stim = hi
                elif last_pos[0] is None:
                    # Position unknown (session start, or a reconnect): this command IS the move
                    # that establishes it, and travel from an invented origin is invented data.
                    # Drive it, learn where we are -- and keep anything he pressed while it ran,
                    # because a press is his, not the actuator's. Only the ROW is forfeited.
                    stim = None; last_pos[0] = pos
                else:
                    stim = abs(pos - last_pos[0]); last_pos[0] = pos
                A = streamer.step(0.0 if stim is None else stim, wall)
                cum = False
                if len(presses) > 1:
                    log(f"MULTI-PRESS {presses} inside one {dur:.2f}s command; the row keeps the "
                        f"last, the Director gets all of them")
                if mode == "manual":
                    for label in presses:          # the keypad IS the throttle; nothing is a label
                        man[0] = label
                        log(f"MANUAL level={label:.2f} PRED={A:.2f}")
                elif mode == "auto":
                    # The label IS the prediction, quantised onto the same 0..8 grid a keyboard
                    # emits and capped one level below cumming -- a model reading 1.0 is a model
                    # at its ceiling, not an orgasm, and treating it as one would close the cycle
                    # and charge CUM_COST for something nobody felt. observe() is skipped on
                    # purpose: there is no bias between a prediction and itself to learn.
                    director.note_press(min(int(A * 9.0), 8) / 9.0)
                else:
                    for label in presses:
                        maxlab = max(maxlab, label)
                        esum += abs(A - label); en += 1
                        director.observe(label, A)  # corrects the belief about predictor bias, and
                        director.note_press(label)  # separately IS this cycle's entire score
                        log(f"YOU={label:.2f} PRED={A:.2f} bias={director.bias:+.2f} "
                            f"phase={director.phase}")
                        if label >= 1.0 and not cum:   # one command can only close one cycle, so
                            log(f"*** CUM at PRED={A:.2f} cycle={director.cycles} ***")  # it dips
                            cum = True                                                   # once
                            streamer.refract()
                if stim is None:
                    log(f"POSITIONING move to {pos:.2f}: presses kept, row not recorded because "
                        f"the travel that caused them is unknown")
                else:
                    # Everything the corpus does not read goes to the log instead of into a wider
                    # CSV: one schema, and a session recorded tonight is trainable as it stands.
                    log(f"ROW pred={A:.4f} stim={stim:.3f} wall={wall:.4f} pos={pos:.3f} "
                        f"cmd_dur={dur:.4f} win_lo={lo:.3f} win_hi={hi:.3f} phase={director.phase} "
                        f"E={streamer.E:.4f} H={streamer.H:.4f} bias={director.bias:+.4f} "
                        f"cycle={director.cycles}")
                    # dest and sid were chosen from the modality at startup, and a reconnect can
                    # land on the other kind: measured, a stroker dropping out and coming back as a
                    # vibrator put 67 held amplitudes into the linear corpus under a LIVE_ id.
                    # training_sessions refuses exactly that mixture ACROSS files and cannot see it
                    # INSIDE one, so it would be silent and permanent once trained on. Stop
                    # recording rather than mislabel: the rows already written are honest.
                    if w is not None and is_vib[0] != vib_at_start:
                        log(f"MODALITY CHANGED mid-session -- closing "
                            f"{os.path.basename(dest)}; the rest of this evening is not recorded")
                        print(f"\n different modality after reconnect -- recording stopped, "
                              f"{os.path.basename(dest)} keeps what was already written")
                        f.close(); f = w = None
                    if w is not None:
                        w.writerow([sid, wall / 60.0, round(stim, 3),
                                    "" if not presses else f"{presses[-1]:.2f}"])
                        f.flush()
                bar = "#" * int(A * 22)
                if mode == "manual":
                    print(f"\r {bar:<22} {A:0.2f} | MANUAL level {man[0]:.2f} "
                          f"| depth {hi:.2f} @ {dur:.2f}s   ", end="", flush=True)
                else:
                    mae = f"MAE {esum / en:.2f}" if en else "MAE --"
                    print(f"\r {bar:<22} {A:0.2f} | you "
                          f"{maxlab if maxlab >= 0 else ('auto' if mode == 'auto' else '--')} "
                          f"| {mae} | {director.phase.upper():5s} c{director.cycles} "
                          f"win {lo:.2f}-{hi:.2f} bias{director.bias:+.2f}   ", end="", flush=True)
                # Row and console are written first: both describe the command that just ran, under
                # the policy and the phase that ran it. Only then is the cycle allowed to end -- and
                # if it does, next_action() is what asks the newly sampled policy what IT wants.
                if mode == "manual":
                    lo, hi, dur = manual_action()
                    continue
                director.commit_observation(A, wall, cum)
                if cum and args.finish_on_cum:      # the trial is credited and closed above; the
                    log("FINISH-ON-CUM -- stopping after the cum row was written")
                    break                           # cum row is the best label in the dataset
                lo, hi, dur = director.next_action()
        except KeyboardInterrupt:
            log("QUIT (Ctrl-C)")
        except Exception as e:
            import traceback
            log(f"LOOP CRASH: {traceback.format_exc()}")
            print(f"\nLOOP CRASH: {e}")
            raise
        finally:
            # EVERY step independent, and the policy write FIRST -- ahead of every await in the
            # block. asyncio.run delivers Ctrl-C by CANCELLING this coroutine, so `except
            # KeyboardInterrupt` above never fires and this block resumes only when the loop next
            # schedules it. A second Ctrl-C while it is suspended at an await aborts the loop and
            # the frame is NEVER RESUMED -- no suppress() can save a coroutine that is not resumed.
            # Measured with the stop first: two Ctrl-C 0.5 s apart executed not one statement here.
            # The toy was not stopped, the corpus file was not closed, and the evening's heads were
            # gone. This step has no await, so nothing can preempt it, and write_json stages to a
            # temp file and swaps, so an interrupt inside it cannot truncate policy.json.
            #
            # Rows are flushed as they are written and survive anything; the heads exist ONLY here.
            if mode == "label":
                saved = False
                with suppress(Exception, KeyboardInterrupt):
                    write_json(POLICY, director.prior(cfg_hash, director.mod))
                    saved = True
                if not saved:      # the last silent fallback in this file, and this is the one
                    print(f"\nPOLICY NOT SAVED -- {POLICY} still holds the previous evening. "
                          f"Tonight's learning is lost; the recording is not.")
            else:
                # Trials graded on the predictor's own output, or a run with no Director at all,
                # are not evidence about him. Ranking tomorrow's real policies against them would
                # put the model's opinion of itself into the memory that chooses what he feels.
                print(f"\n{mode}: nothing recorded, policy memory left as it was")
            with suppress(Exception, KeyboardInterrupt):
                # Bounded, and only now: this is the statement that turns the toy off, and also the
                # one most likely to stall -- a dead link is usually WHY he hit Ctrl-C, and
                # buttplug-py waits on a server reply with no timeout of its own.
                await asyncio.wait_for(dev.send_stop_device_cmd(), 3.0)
            with suppress(Exception):
                if keyboard is not None:
                    keyboard.unhook_all()
            with suppress(Exception):
                if f is not None:
                    f.close()
            with suppress(Exception):
                log(f"STOP mode={mode} cycles={director.cycles} "
                    f"bias={director.bias:+.3f} labeled={en}")
            # `en`, not `maxlab`: a session he spent pressing 0 is a session of real, hard-won
            # negative evidence and has maxlab == 0.0. Deleting it conflated "he gave no labels"
            # with "he repeatedly reported zero" -- the one distinction the rest of this file is
            # built on. Only a recording with no keypress at all is worth nothing.
            if w is not None and en == 0 and keyboard is not None:
                with suppress(Exception):
                    drop_session(dest, sid)
                    print(f"\nNO KEYPRESS all session -- {sid} removed from "
                          f"{os.path.basename(dest)}. Unrated is missing, not zero, so there is "
                          f"nothing here to train on.")
            elif w is not None:
                print(f"\nsaved {en} labels -> {os.path.abspath(dest)} as session {sid}")
            with suppress(Exception):
                logf.close()
            with suppress(Exception):
                await asyncio.wait_for(client.disconnect(), 5.0)
    asyncio.run(main_loop())

def main():
    ap = argparse.ArgumentParser(description="arousal prediction + exploring edging policy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="fit, LOSO eval, hazard, invariants -> " + CONFIG)
    tr.add_argument("--data", default=DATA)
    lv = sub.add_parser("live", help="drive the toy (Intiface + buttplug-py + keyboard)")
    lv.add_argument("--data", default=DATA, help="the session is APPENDED here, where train reads it")
    lv.add_argument("--finish-on-cum", action="store_true", help="stop at the first orgasm")
    lv.add_argument("--intiface", default="ws://127.0.0.1:12345")
    m = lv.add_mutually_exclusive_group()
    m.add_argument("--auto", action="store_true",
                   help="no keys read; the prediction is its own label. Records nothing")
    m.add_argument("--manual", action="store_true",
                   help="0-9 drive the TOY, no Director in the loop. Records nothing")
    args = ap.parse_args()
    sys.exit({"train": cmd_train, "live": cmd_live}[args.cmd](args) or 0)

if __name__ == "__main__":
    main()
