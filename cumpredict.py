#!/usr/bin/env python3
"""
cumpredict -- live arousal prediction + inference-driven edging for a Bluetooth stroker.

A tiny MECHANISTIC STATE-SPACE MODEL reads the stroke stream of a linear stroker
(e.g. Kiiroo Keon via Intiface Central, buttplug protocol) and outputs a live arousal
estimate 0..1 (1.0 = orgasm). A separate WAVE-CONTROLLER POLICY drives the toy from
that prediction: build -> ease off at the (escalating) edge -> optionally finish.
The human never signals anything; his sparse 0-9 key presses are ground-truth labels
only and never touch the model or the device.

Design notes (the why -- the what is below in code):
  * Predictor = grey-box ODE, ~10 physical params (excitation -> threshold-gated slow
    arousal accumulator + habituation; exact closed-form per-stroke updates). With only
    a handful of sessions, structure beats capacity: a GRU trained on the same data
    collapsed to a constant on held-out sessions; the physics extrapolates. It is also,
    deliberately, a liquid network (CfC) with frozen time-constants -- the upgrade path
    is to unfreeze them, not to rewrite.
  * The fit is GRADIENT-FREE (CEM over a batched NumPy scan): autograd through a
    5000-step sequential scan is pathologically slow on Windows/WDDM; CEM fits in
    seconds and a resting-sanity penalty keeps the gate from leaking (a model whose
    arousal cannot decay is a controller that stalls in brake forever).
  * Predictor and Director are SEPARATE and never merged -- otherwise the policy
    reward-hacks the perception it is judged by.
  * Labels: interpolate-on-rise targets (a press lags the felt level), per-second loss
    weighting, near-cum emphasis, press-instant boosting, and physically-plausible
    augmentation (jitter + time-warp) to decorrelate stroke rate from outcome.
  * Evaluation is EVENT-LEVEL and honest: cum-detection lead time, false alarms/hour,
    cum-vs-no-cum separation, leave-one-session-out with the held-out session untouched.
    Per-row MSE rewards flat-lining; don't use it. When re-verifying a refit on an old
    recording, --recompute (logged predictions reflect the model AT RECORD TIME).

Data: one combined CSV with columns [session, time_elapse, intensity, tired_level]
(time_elapse = stroke duration in MINUTES; intensity = commanded reach 0..1;
tired_level = sparse manual label 0..1, 1.0 = cum), or a directory of per-session CSVs.

The trained model is NOT shipped -- it is personal (a different body = a different fit).
Reproduce it from the included sessions.csv, then run live:
  python cumpredict.py train --data sessions.csv          # fit -> writes artifacts/config_core.json
  python cumpredict.py live                                # drive the toy with that model
  python cumpredict.py report --file SESSION.csv [--recompute]   # score a recording
  python cumpredict.py sim                                 # offline safety check (run BEFORE live)
  python cumpredict.py parity                              # streaming == batch, exactly
To use your OWN data: record sessions (live writes one CSV per session), merge them into a
sessions.csv with a `session` column, and retrain -- the model does not transfer between people.

Live keys: 0-9 = your arousal (9 = cumming), q = quit safely. The device is always
stopped on exit; if the toy drops mid-session it auto-reconnects and resumes.
Deps: numpy pandas [matplotlib scipy]  + for live: buttplug-py==0.3.0 keyboard
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- data prep
BETA = 4.0          # near-cum loss emphasis: weight *= 1 + BETA*y
PRESS_BOOST = 8.0   # extra weight on real key-press rows (the only true labels)


def _interp_on_rise(lab, t_s):
    """Dense target from a sparse label array: ramp UP between rising presses, step DOWN."""
    T = len(lab)
    press_idx = np.where(~np.isnan(lab))[0]
    y = np.zeros(T, dtype=np.float64)
    is_press = np.zeros(T, dtype=bool)
    if len(press_idx) == 0:
        return y.astype(np.float32), is_press, press_idx, np.array([])
    is_press[press_idx] = True
    press_vals = lab[press_idx]

    i0, v0 = press_idx[0], press_vals[0]
    if i0 > 0 and v0 > 0:                      # leading ramp 0 -> first press
        seg = slice(0, i0 + 1)
        span = max(t_s[i0] - t_s[0], 1e-9)
        y[seg] = v0 * (t_s[seg] - t_s[0]) / span
    else:
        y[: i0 + 1] = v0

    for j in range(len(press_idx)):
        a, va = press_idx[j], press_vals[j]
        if j + 1 < len(press_idx):
            b, vb = press_idx[j + 1], press_vals[j + 1]
        else:
            b, vb = T - 1, va                  # hold last value to end of session
        y[a] = va
        if b == a:
            continue
        if vb > va:                            # rising -> linear ramp in time
            seg = slice(a, b + 1)
            span = max(t_s[b] - t_s[a], 1e-9)
            y[seg] = va + (vb - va) * (t_s[seg] - t_s[a]) / span
        else:                                  # falling/flat -> hold, step down at the press
            y[a:b] = va
            y[b] = vb
    return np.clip(y, 0.0, 1.0).astype(np.float32), is_press, press_idx, press_vals


def session_from_raw(name, dur_s, intensity, tired_sparse):
    """Build a session dict from raw arrays (used for real files AND augmented copies)."""
    dur_s = np.clip(np.asarray(dur_s, dtype=np.float64), 1e-3, None)
    intensity = np.clip(np.asarray(intensity, dtype=np.float64), 0.0, 1.0)
    tired_sparse = np.asarray(tired_sparse, dtype=np.float64)
    t_s = np.cumsum(dur_s)
    y, is_press, press_idx, press_vals = _interp_on_rise(tired_sparse, t_s)
    w = dur_s * (1.0 + BETA * y)
    w[is_press] *= PRESS_BOOST
    return {
        "name": name, "intensity": intensity.astype(np.float32),
        "y": y, "w": w.astype(np.float32), "is_press": is_press,
        "dur_s": dur_s.astype(np.float32), "t_s": t_s, "t_min": t_s / 60.0,
        "press_idx": press_idx, "press_vals": press_vals.astype(np.float32),
        "raw_dur_s": dur_s.copy(), "raw_intensity": intensity.copy(), "raw_tired": tired_sparse.copy(),
    }


def _clean_frame(df):
    df.columns = [c.strip() for c in df.columns]
    for c in ("time_elapse", "intensity", "tired_level"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["time_elapse", "intensity"]).reset_index(drop=True)


def _session_from_frame(name, df):
    return session_from_raw(name, df["time_elapse"].to_numpy(dtype=np.float64) * 60.0,
                            df["intensity"].to_numpy(dtype=np.float64),
                            df["tired_level"].to_numpy(dtype=np.float64))


def load_sessions(data_path):
    """Sessions from a combined CSV (with a `session` column) or a directory of CSVs."""
    if os.path.isdir(data_path):
        return [_session_from_frame(os.path.basename(p)[:-4], _clean_frame(pd.read_csv(p)))
                for p in sorted(glob.glob(os.path.join(data_path, "*.csv")))]
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    if "session" not in df.columns:
        return [_session_from_frame(os.path.splitext(os.path.basename(data_path))[0], _clean_frame(df))]
    return [_session_from_frame(str(name), _clean_frame(g.drop(columns=["session"])))
            for name, g in sorted(df.groupby("session", sort=False), key=lambda kv: kv[0])]


def augment_raw(sess, rng, warp=True):
    """One physically-plausible augmented copy. Real labels stay untouched; the copy always
    keeps the true session start so the causal integrator never sees a mid-session cold start."""
    dur = sess["raw_dur_s"].copy()
    inten = sess["raw_intensity"].copy()
    tired = sess["raw_tired"].copy()
    n = len(dur)
    if rng.random() < 0.5:                     # end-truncation
        k = int(rng.integers(int(0.6 * n), n + 1))
        dur, inten, tired = dur[:k], inten[:k], tired[:k]
    inten = np.clip(inten + rng.uniform(-0.05, 0.05, len(inten)), 0.0, 1.0)
    dur = dur * np.exp(rng.uniform(-0.10, 0.10, len(dur)))
    if warp:                                   # global time-warp 0.7x..1.4x
        dur = dur * float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    return session_from_raw(sess["name"] + "_aug", dur, inten, tired)


# ------------------------------------------------------------------------ mechanistic core
# State per stroke of duration dt, exact closed-form update (each ODE is linear given the
# drive held constant over the stroke):
#   P   = intensity^a * rate^b                      stimulation power
#   E  <- E + (1-exp(-dt/tauE))*(P - E)             fast excitation (leaky-tracked P)
#   de  = softplus(g*(E - thr - kH*H))              drive EXCESS above threshold
#   A  <- A_ss + (A-A_ss)*exp(-(rho*de+lam)*dt)     slow arousal accumulator, A_ss=rho*de/(rho*de+lam)
#   H  <- H_ss + (H-H_ss)*exp(-dH*dt)               slow habituation, H_ss=hH*A/dH
# When E < thr (slow/weak strokes), de~0 and arousal only DECAYS -- "slow == safe" comes
# from physics, not from having enough negative data.
CORE_NAMES = ["a", "b", "tauE", "g", "thr", "rho", "lam", "kH", "hH", "dH"]
CORE_BOUNDS = np.array([
    [1.00, 3.0],     # a    intensity exponent; >=1 so stroke DEPTH really modulates drive
    [0.20, 1.5],     # b    rate exponent; capped -- b~3 made one stroke max out arousal
    [10.0, 60.0],    # tauE excitation time-constant (s); floor so response isn't instant
    [0.20, 10.0],    # g    excitation-excess gain
    [0.00, 2.5],     # thr  excitation threshold
    [1e-4, 0.02],    # rho  arousal accumulation /s; capped so buildup is gradual
    [0.015, 0.06],   # lam  arousal decay /s; floor so backing off actually lowers arousal
    [0.00, 5.0],     # kH   habituation threshold-shift
    [0.00, 0.05],    # hH   habituation build rate
    [1e-4, 0.02],    # dH   habituation decay /s
])


def softplus(x):
    return np.logaddexp(0.0, x)


def core_scan(intensity, rate, dt, a, b, tauE, g, thr, rho, lam, kH, hH, dH):
    """intensity/rate/dt: [B,T]; each param: [B]. Returns A,E,H: [B,T]."""
    a2 = a.reshape(-1, 1); b2 = b.reshape(-1, 1)
    P = np.clip(intensity, 1e-4, 1.0) ** a2 * np.clip(rate, 1e-4, None) ** b2
    alphaE = 1.0 - np.exp(-dt / tauE.reshape(-1, 1))
    decayH = np.exp(-dH.reshape(-1, 1) * dt)
    B, T = intensity.shape
    E = np.zeros(B); A = np.zeros(B); H = np.zeros(B)
    Aout = np.empty((B, T)); Eout = np.empty((B, T)); Hout = np.empty((B, T))
    for t in range(T):
        E = E + alphaE[:, t] * (P[:, t] - E)
        de = softplus(g * (E - thr - kH * H))
        k = rho * de + lam
        A_ss = rho * de / k
        A = A_ss + (A - A_ss) * np.exp(-k * dt[:, t])
        H_ss = hH * A / dH
        H = H_ss + (H - H_ss) * decayH[:, t]
        Aout[:, t] = A; Eout[:, t] = E; Hout[:, t] = H
    return Aout, Eout, Hout


def scan_params(intensity, rate, dt, params_row):
    return core_scan(intensity, rate, dt, *[params_row[:, i] for i in range(len(CORE_NAMES))])


class CoreStreamer:
    """Online (per-stroke) core -- bit-identical to core_scan; ~10 lines to port to JS."""
    def __init__(self, params, theta=0.8, alarm_debounce_s=15.0):
        p = [params[k] for k in CORE_NAMES] if isinstance(params, dict) else list(params)
        (self.a, self.b, self.tauE, self.g, self.thr,
         self.rho, self.lam, self.kH, self.hH, self.dH) = p
        self.theta = theta
        self.debounce = alarm_debounce_s
        self.E = 0.0; self.A = 0.0; self.H = 0.0
        self._above_since = None
        self._t = 0.0

    def step(self, intensity, dt):
        """One stroke. Returns (arousal 0..1, debounced_alarm_bool)."""
        intensity = min(max(float(intensity), 1e-4), 1.0)
        dt = max(float(dt), 1e-3)
        P = intensity ** self.a * max(1.0 / dt, 1e-4) ** self.b
        self.E += (1.0 - np.exp(-dt / self.tauE)) * (P - self.E)
        de = softplus(self.g * (self.E - self.thr - self.kH * self.H))
        k = self.rho * de + self.lam
        A_ss = self.rho * de / k
        self.A = A_ss + (self.A - A_ss) * np.exp(-k * dt)
        H_ss = self.hH * self.A / self.dH
        self.H = H_ss + (self.H - H_ss) * np.exp(-self.dH * dt)
        self._t += dt
        A = float(np.clip(self.A, 0.0, 1.0))
        if A >= self.theta:
            if self._above_since is None:
                self._above_since = self._t
            alarm = (self._t - self._above_since) >= self.debounce
        else:
            self._above_since = None
            alarm = False
        return A, alarm


# ------------------------------------------------------------------------------- director
class Director:
    """Wave-controller policy (build/brake/finish) acting only on the predicted arousal.
    Continuous proportional depth/speed + LFO (no bang-bang); a LOADING state L makes it
    brake earlier & harder each successive edge (the momentum of edging); an online brake
    GAIN ramps on near-misses. finish_after=N: after N edges, never brake again -- drive
    to climax at the device's max stroke rate."""
    def __init__(self, setpoint=0.70, band=0.20, finish_after=None):
        self.setpoint = setpoint
        self.band = max(0.08, band)
        self.finish_after = finish_after
        self.L = 0.0
        self.gain = 1.0
        self.edges = 0
        self.phase = "build"
        self.t = 0.0
        self.prevA = 0.0
        self.hi = setpoint

    def _wave(self, period):
        return 0.5 + 0.5 * math.sin(2.0 * math.pi * self.t / period)

    def step(self, A, dt):
        """arousal A, dt seconds -> (reach 0..1, duration_s)."""
        self.t += dt
        slope = (A - self.prevA) / max(dt, 1e-3)
        self.prevA = A
        self.L += dt * (0.03 * max(0.0, A - (self.setpoint - 0.20))) - dt * 0.001 * self.L
        # escalating brake, CAPPED, with a floored exit so it can always climb back out
        self.hi = self.setpoint - min(0.15 * self.L * self.gain, 0.15)
        lo = max(self.hi - self.band, 0.22)
        if self.phase == "build" and (A >= self.hi or (A >= self.hi - 0.05 and slope > 0.06)):
            self.phase = "brake"; self.edges += 1
            if slope > 0.08 or A > self.hi + 0.03:
                self.gain = min(3.0, self.gain * 1.3)
        elif self.phase == "brake" and A <= lo:
            self.phase = "build"
        if self.finish_after is not None and self.edges >= self.finish_after:
            self.phase = "finish"
        if self.phase == "finish":       # device max (~0.13s/half-stroke), full reach
            reach, dur = 1.0, 0.13
        elif self.phase == "build":      # moderate depth, fast undulating pace
            gap = max(0.0, self.hi - A)
            reach = (0.40 + 0.30 * min(1.0, gap * 3.0)) * (0.85 + 0.15 * self._wave(2.5))
            dur = 0.14 + 0.10 * self._wave(3.0)
        else:                            # ease off: shallow + slow
            ease = min(0.90, 0.5 + 0.3 * self.L * self.gain)
            reach = max(0.03, (1.0 - ease) * 0.4) * (0.6 + 0.4 * self._wave(4.0))
            dur = 0.5 + 0.4 * self._wave(4.0)
        return round(min(1.0, max(0.02, reach)), 3), round(dur, 3)


# ---------------------------------------------------------------------------- event metrics
CUM_THR = 0.99
EDGE_THR = 0.78
MERGE_GAP_MIN = 2.0
DEBOUNCE_S = 15.0
HIT_MIN_LEAD_S = 15.0
HIT_MAX_LEAD_S = 300.0
NEAR_EDGE_S = 120.0


def merge_cum_onsets(t_s, press_idx, press_vals):
    times = sorted(float(t_s[i]) for i, v in zip(press_idx, press_vals) if v >= CUM_THR)
    merged = []
    for tm in times:
        if not merged or tm - merged[-1] > MERGE_GAP_MIN * 60.0:
            merged.append(tm)
    return merged


def edge_times(t_s, press_idx, press_vals):
    return [float(t_s[i]) for i, v in zip(press_idx, press_vals) if EDGE_THR <= v < CUM_THR]


def detect_alarms(pred, t_s, theta, debounce_s=DEBOUNCE_S, require_rising=True):
    above = pred >= theta
    onsets = []
    i, n = 0, len(pred)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if t_s[j - 1] - t_s[i] >= debounce_s or (j - i) >= 3:
                if not require_rising or i == 0 or pred[i] >= pred[max(i - 3, 0)]:
                    onsets.append(float(t_s[i]))
            i = j
        else:
            i += 1
    return onsets


def score_session(sess, pred, theta):
    t_s = sess["t_s"]
    cums = merge_cum_onsets(t_s, sess["press_idx"], sess["press_vals"])
    edges = edge_times(t_s, sess["press_idx"], sess["press_vals"])
    alarms = detect_alarms(pred, t_s, theta)
    detections, leads = [], []
    for c in cums:
        prior = [a for a in alarms if HIT_MIN_LEAD_S <= (c - a) <= HIT_MAX_LEAD_S]
        if prior:
            detections.append(True); leads.append(c - max(prior))
        else:
            detections.append(False)
    false_alarms = edge_hits = 0
    for a in alarms:
        # a valid early-warning lead (up to HIT_MAX_LEAD_S before a cum) or a just-after alarm
        # is a detection, not a false alarm
        if any(-30.0 <= (c - a) <= HIT_MAX_LEAD_S for c in cums):
            continue
        if any(abs(a - e) <= NEAR_EDGE_S for e in edges):
            edge_hits += 1        # detecting an approach-to-edge is the product, not a false alarm
        else:
            false_alarms += 1
    dur_h = float(t_s[-1]) / 3600.0
    return {"session": sess["name"], "peak": round(float(pred.max()), 3),
            "n_cum": len(cums), "detected": int(sum(detections)),
            "leads_s": [round(l, 1) for l in leads],
            "median_lead_s": round(float(np.median(leads)), 1) if leads else None,
            "edge_detections": edge_hits, "false_alarms": false_alarms,
            "fa_per_hour": round(false_alarms / dur_h, 2) if dur_h > 0 else None}


def press_metrics(press_idx, press_vals, pred):
    """(MAE, Kendall tau) vs the real key presses -- the only true labels."""
    if len(press_idx) == 0:
        return None, None
    mae = float(np.mean(np.abs(pred[press_idx] - press_vals)))
    tau = None
    try:
        from scipy.stats import kendalltau
        if len(press_idx) > 2:
            tau = float(kendalltau(pred[press_idx], press_vals).correlation)
    except Exception:
        pass
    return round(mae, 3), (round(tau, 3) if tau is not None else None)


def separation_margin(sessions, preds):
    cum_peaks, neg_peaks = [], []
    for s, p in zip(sessions, preds):
        peak = float(np.max(p))
        if merge_cum_onsets(s["t_s"], s["press_idx"], s["press_vals"]):
            cum_peaks.append(peak)
        else:
            neg_peaks.append(peak)
    if not cum_peaks or not neg_peaks:
        return None
    return round(min(cum_peaks) - max(neg_peaks), 3)


# ------------------------------------------------------------------------------ CEM fitting
CEM_ITERS = 40
CEM_POP = 80
CEM_ELITE = 0.15
CEM_RESTARTS = 2
N_AUG = 2           # augmented copies per training session in the fit pool
REST_PENALTY = 3.0  # resting-sanity: zero-stim arousal must decay to ~0 (or edging stalls)
THETA_GRID = np.round(np.linspace(0.50, 0.95, 19), 3)


def pad(sessions):
    T = max(len(s["dur_s"]) for s in sessions)
    B = len(sessions)
    inten = np.zeros((B, T)); rate = np.ones((B, T)); dt = np.zeros((B, T))
    y = np.zeros((B, T)); w = np.zeros((B, T)); mask = np.zeros((B, T))
    for i, s in enumerate(sessions):
        n = len(s["dur_s"]); d = s["dur_s"].astype(np.float64)
        inten[i, :n] = s["intensity"]; rate[i, :n] = 1.0 / d; dt[i, :n] = d
        y[i, :n] = s["y"]; w[i, :n] = s["w"]; mask[i, :n] = 1.0
    return inten, rate, dt, y, w * mask


def cem_fit(train_sessions, rng):
    pool = list(train_sessions) + [augment_raw(s, rng) for s in train_sessions for _ in range(N_AUG)]
    inten, rate, dt, y, wm = pad(pool)
    n = inten.shape[0]
    ndim = CORE_BOUNDS.shape[0]
    lo, hi = CORE_BOUNDS[:, 0], CORE_BOUNDS[:, 1]

    def eval_pop(P):
        S = P.shape[0]
        bi = np.tile(inten, (S, 1)); br = np.tile(rate, (S, 1)); bd = np.tile(dt, (S, 1))
        by = np.tile(y, (S, 1)); bwm = np.tile(wm, (S, 1))
        A, _, _ = scan_params(bi, br, bd, np.repeat(P, n, axis=0))
        e = by - np.clip(A, 0.0, 1.0)
        pin = np.maximum(0.75 * e, -0.25 * e) * bwm         # pinball tau=0.75: an alarm wants
        row = pin.sum(1) / (bwm.sum(1) + 1e-8)              # the upper arousal estimate
        data = row.reshape(S, n).mean(1)
        g, thr, rho, lam = P[:, 3], P[:, 4], P[:, 5], P[:, 6]
        de_rest = softplus(g * (-thr))
        Ass_rest = rho * de_rest / (rho * de_rest + lam)
        return data + REST_PENALTY * Ass_rest ** 2

    best, best_loss = None, np.inf
    k = max(3, int(CEM_POP * CEM_ELITE))
    for _ in range(CEM_RESTARTS):
        mean = np.clip((lo + hi) / 2 + 0.25 * (hi - lo) * rng.standard_normal(ndim), lo, hi)
        std = (hi - lo) / 4.0
        for _ in range(CEM_ITERS):
            P = np.clip(mean + std * rng.standard_normal((CEM_POP, ndim)), lo, hi)
            if best is not None:
                P[0] = best
            L = eval_pop(P)
            idx = np.argsort(L)[:k]
            elite = P[idx]
            mean, std = elite.mean(0), elite.std(0) + 1e-3 * (hi - lo)
            if L[idx[0]] < best_loss:
                best_loss, best = L[idx[0]], P[idx[0]].copy()
    return best, best_loss


def predict_core(params, sess):
    inten, rate, dt, _, _ = pad([sess])
    A, _, _ = scan_params(inten, rate, dt, params[None, :])
    return np.clip(A[0], 0.0, 1.0)


def export_config(path, params, alarm_theta, extra=None):
    cfg = {"model": "core", "core_names": CORE_NAMES,
           "core_params": {n: float(v) for n, v in zip(CORE_NAMES, params)},
           "alarm_theta": alarm_theta, **(extra or {})}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"exported {path}")


def load_config(path):
    if not os.path.exists(path):
        sys.exit(f"no model at {path} -- train one first:\n"
                 f"  python cumpredict.py train --data sessions.csv")
    with open(path) as f:
        raw = f.read()
    import hashlib
    return json.loads(raw), hashlib.sha1(raw.encode()).hexdigest()[:10]


# ------------------------------------------------------------------------------ subcommands
def cmd_train(args):
    sessions = load_sessions(args.data)
    rng = np.random.default_rng(0)
    print(f"CORE (CEM) pop={CEM_POP} iters={CEM_ITERS} restarts={CEM_RESTARTS} "
          f"sessions={[s['name'] for s in sessions]}\n")
    if not sessions:
        sys.exit(f"no sessions found in {args.data}")
    if len(sessions) < 2:
        print("only 1 session -- skipping leave-one-out; fitting the deployed config directly.")
        params, loss = cem_fit(sessions, rng)
        print(f"fitloss={loss:.4f} params:", {n: round(float(v), 4) for n, v in zip(CORE_NAMES, params)})
        export_config(args.config, params, args.shield)
        return

    fold_preds, fold_params = [], []
    for held in sessions:
        params, loss = cem_fit([s for s in sessions if s is not held], rng)
        fold_preds.append(predict_core(params, held))
        fold_params.append(params)
        pp = {n: round(float(v), 3) for n, v in zip(CORE_NAMES, params)}
        print(f"  held={held['name']:<28} peak={fold_preds[-1].max():.2f} fitloss={loss:.4f} {pp}")

    margin = separation_margin(sessions, fold_preds)
    sweep = []
    for th in THETA_GRID:
        tot = dict(cum=0, det=0, fa=0, edge=0)
        for s, p in zip(sessions, fold_preds):
            r = score_session(s, p, th)
            tot["cum"] += r["n_cum"]; tot["det"] += r["detected"]
            tot["fa"] += r["false_alarms"]; tot["edge"] += r["edge_detections"]
        sweep.append({"theta": float(th), **tot})
    theta = sorted(sweep, key=lambda r: (-r["det"], r["fa"], -r["theta"]))[0]["theta"]

    print("\nthreshold sweep (LOSO aggregate):")
    for r in sweep:
        print(f"  theta={r['theta']:.2f} det={r['det']}/{r['cum']} FA={r['fa']} edge={r['edge']}"
              + ("  <==" if r["theta"] == theta else ""))

    print(f"\n=== LOSO @ theta={theta:.2f}  (separation margin={margin}) ===")
    results = []
    for s, p in zip(sessions, fold_preds):
        r = score_session(s, p, theta)
        r["press_mae"], r["kendall_tau"] = press_metrics(s["press_idx"], s["press_vals"], p)
        results.append(r)
        print(f"  {r['session']:<28} peak={r['peak']:.2f} cum={r['detected']}/{r['n_cum']} "
              f"lead={r['median_lead_s']}s edges={r['edge_detections']} FA/h={r['fa_per_hour']} "
              f"pressMAE={r['press_mae']} tau={r['kendall_tau']}")

    try:                                       # plot is best-effort; headless-safe
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncol, nrow = 2, math.ceil(len(sessions) / 2)
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 3.75 * nrow))
        axes = np.array(axes).ravel()
        for k, (s, p) in enumerate(zip(sessions, fold_preds)):
            ax = axes[k]; tm = s["t_min"]
            ax.plot(tm, p, lw=1.4, label="pred")
            ax.plot(tm, s["y"], lw=1.0, alpha=0.5, label="target")
            if len(s["press_idx"]):
                ax.scatter(tm[s["press_idx"]], s["press_vals"], s=10, c="k", zorder=5)
            ax.axhline(theta, ls=":", c="r", lw=0.8)
            for c in merge_cum_onsets(s["t_s"], s["press_idx"], s["press_vals"]):
                ax.axvline(c / 60.0, c="g", ls="--", lw=1)
            ax.set_title(s["name"], fontsize=8); ax.set_ylim(-0.05, 1.08); ax.legend(fontsize=6)
        fig.suptitle(f"LOSO  theta={theta:.2f}  separation={margin}")
        fig.tight_layout()
        png = os.path.join(args.out_dir, "loso_core.png")
        os.makedirs(args.out_dir, exist_ok=True)
        fig.savefig(png, dpi=110)
        print(f"\nsaved {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    # fresh seed for the deployed fit so `train` and `fit` export the SAME model (the LOSO folds
    # above advance `rng`); with a fixed seed the fit is deterministic and byte-reproducible.
    final_params, _ = cem_fit(sessions, np.random.default_rng(0))
    export_config(args.config, final_params, theta, {"separation_margin": margin})
    with open(os.path.join(args.out_dir, "loso_core_results.json"), "w") as f:
        json.dump({"theta": theta, "separation_margin": margin, "sweep": sweep, "folds": results,
                   "fold_params": [{n: float(v) for n, v in zip(CORE_NAMES, p)} for p in fold_params]},
                  f, indent=2)


def cmd_fit(args):
    sessions = load_sessions(args.data)
    if not sessions:
        sys.exit(f"no sessions found in {args.data}")
    print(f"fitting on {len(sessions)} sessions: {[s['name'] for s in sessions]}")
    params, loss = cem_fit(sessions, np.random.default_rng(0))
    print(f"fitloss={loss:.4f}")
    print("params:", {n: round(float(v), 4) for n, v in zip(CORE_NAMES, params)})
    export_config(args.config, params, args.shield, {"fit": "final-only"})


def cmd_parity(args):
    """Streaming inference must match the batch scan exactly (the JS-port contract)."""
    rng = np.random.default_rng(7)
    params = rng.uniform(CORE_BOUNDS[:, 0], CORE_BOUNDS[:, 1])
    worst = 0.0
    for s in load_sessions(args.data):
        inten = s["intensity"].astype(np.float64)[None, :]
        dur = s["dur_s"].astype(np.float64)[None, :]
        A_batch, _, _ = core_scan(inten, 1.0 / dur, dur, *[np.array([v]) for v in params])
        st = CoreStreamer(dict(zip(CORE_NAMES, params)))
        A_stream = np.array([st.step(inten[0, t], dur[0, t])[0] for t in range(dur.shape[1])])
        d = float(np.max(np.abs(np.clip(A_batch[0], 0, 1) - A_stream)))
        worst = max(worst, d)
        print(f"  {s['name']:<28} max|batch-stream|={d:.2e}")
    print(f"\nworst parity error = {worst:.2e}  ->  {'PASS' if worst < 1e-9 else 'FAIL'}")
    return 0 if worst < 1e-9 else 1


def cmd_sim(args):
    """Offline closed-loop check: does backing off lower arousal, and does the Director
    cycle instead of stalling? Run this before ever going live on a new config."""
    cfg, _ = load_config(args.config)
    params = cfg["core_params"]

    def sp(x):
        return math.log1p(math.exp(-abs(x))) + max(x, 0.0)
    de_rest = sp(params["g"] * (0.0 - params["thr"]))
    k = params["rho"] * de_rest + params["lam"]
    print(f"RESTING ATTRACTOR (toy off): A_ss = {params['rho'] * de_rest / k:.3f}  (tau = {1.0/k:.0f}s)")

    def brake_run(A0, seconds, reach=0.10, dur=1.0):
        s = CoreStreamer(params)
        s.E, s.A = (0.9 if A0 > 0.4 else 0.0), float(A0)
        t = 0.0
        while t < seconds:
            A, _ = s.step(reach, dur)
            t += dur
        return A
    drift = brake_run(0.0, 600.0)
    a_stop = brake_run(0.75, 120.0)
    ok = drift <= 0.40 and a_stop <= 0.40      # both must end below a typical edge_low
    print(f"  brake test: 10min gentle-strokes from 0 = {drift:.3f}  ({'PASS' if drift <= .4 else 'FAIL'})")
    print(f"  brake test: 120s backing-off from 0.75  = {a_stop:.3f}  ({'PASS' if a_stop <= .4 else 'FAIL'})")

    def closed_loop(setpoint, minutes=8.0):
        s = CoreStreamer(params)
        d = Director(setpoint=setpoint, band=0.20)
        t, A, last_dt, trace = 0.0, 0.0, 0.20, []
        while t < minutes * 60.0:
            reach, dur = d.step(A, last_dt)
            A, _ = s.step(reach, dur)
            last_dt = dur; t += dur
            trace.append((t, A, d.phase, d.edges))
        return trace
    print("\nsetpoint sweep (want edges>3, cycling, no stall):")
    stalled_any = False
    for spnt in (0.45, 0.55, 0.65, 0.75):
        tr = closed_loop(spnt)
        longest = cur = 0.0
        prev = 0.0
        for (t, A, ph, e) in tr:
            cur = cur + (t - prev) if ph == "brake" else 0.0
            longest = max(longest, cur); prev = t
        edges = tr[-1][3]
        stalled = longest > 180.0
        stalled_any |= stalled
        print(f"  setpoint={spnt:.2f}: edges={edges:2d} longest_brake={longest:3.0f}s"
              + ("  STALLED" if stalled else ""))
    print(f"\nVERDICT: {'PASS -- safe to run live' if ok and not stalled_any else 'FAIL -- refit before going live'}")
    return 0 if ok and not stalled_any else 1


def cmd_report(args):
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        sys.exit(f"cannot read {args.file}: {e}")
    df.columns = [c.strip() for c in df.columns]
    if "session" in df.columns:
        if not args.session:
            sys.exit(f"combined CSV: pass --session NAME (one of: "
                     f"{sorted(df['session'].astype(str).unique())})")
        df = df[df["session"].astype(str) == args.session].drop(columns=["session"])
    for c in ("time_elapse", "intensity", "tired_level", "pred_arousal"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_elapse", "intensity"]).reset_index(drop=True)
    if len(df) == 0:
        sys.exit("no usable rows (wrong --session name, or empty file)")

    if not args.recompute and "pred_arousal" in df.columns and df["pred_arousal"].notna().any():
        print("(scoring LOGGED predictions -- the model AT RECORD TIME; --recompute for current)")
        pred = df["pred_arousal"].to_numpy(dtype=float)
    else:
        print("(recomputing predictions with the current config)")
        cfg, _ = load_config(args.config)
        s = CoreStreamer(cfg["core_params"])
        dur = np.clip(df["time_elapse"].to_numpy(float) * 60.0, 1e-3, None)
        inten = df["intensity"].to_numpy(float)
        pred = np.array([s.step(inten[i], dur[i])[0] for i in range(len(df))])

    dur = np.clip(df["time_elapse"].to_numpy(float) * 60.0, 1e-3, None)
    t_s = np.cumsum(dur)
    lab = df["tired_level"].to_numpy(float) if "tired_level" in df.columns else np.full(len(df), np.nan)
    pidx = np.where(~np.isnan(lab))[0]
    print(f"rows={len(df)} length={t_s[-1]/60:.1f}min presses={len(pidx)}")
    if len(pidx) == 0:
        print("no labels to score against."); return
    mae, tau = press_metrics(pidx, lab[pidx], pred)
    print(f"overall press-MAE = {mae}  Kendall tau = {tau}  "
          f"peak_pred={pred.max():.2f}  peak_label={lab[pidx].max():.2f}")
    err = np.abs(pred[pidx] - lab[pidx])
    for lo_b, hi_b, name in [(0.0, 0.4, "low  0.0-0.4"), (0.4, 0.7, "mid  0.4-0.7"), (0.7, 1.01, "TOP  0.7-1.0")]:
        m = (lab[pidx] >= lo_b) & (lab[pidx] < hi_b)
        print(f"  {name}: n={m.sum():3d}" + (f"  MAE={err[m].mean():.3f}" if m.any() else ""))
    if "phase" in df.columns:
        ph = df["phase"].astype(str).to_numpy()
        i = 0
        while i < len(ph):
            if ph[i] == "brake":
                j = i
                while j < len(ph) and ph[j] == "brake":
                    j += 1
                print(f"  brake {t_s[i]:.0f}-{t_s[j-1]:.0f}s: pred {pred[i]:.2f}->{pred[j-1]:.2f}")
                i = j
            else:
                i += 1


def cmd_live(args):
    """The live loop: Director (or random) drives the toy, the streamer runs pure
    inference, your 0-9 presses are logged as ground truth only, everything lands in a
    per-session timestamped CSV + state persists across sessions."""
    import asyncio
    import csv
    import random
    try:
        import keyboard
    except Exception:
        keyboard = None
        print("WARNING: 'keyboard' package unavailable -- labels and q-to-quit are DISABLED; "
              "the recording is kept regardless, stop with Ctrl-C.")
    from buttplug.client import ButtplugClient, ButtplugClientWebsocketConnector

    cfg, cfg_hash = load_config(args.config)
    theta = cfg.get("alarm_theta", 0.8)
    streamer = CoreStreamer(cfg["core_params"], theta=theta)
    setpoint = args.setpoint if args.setpoint is not None else round(theta - 0.15, 3)
    low = args.low if args.low is not None else max(0.30, round(setpoint - 0.15, 3))
    band = setpoint - low
    finish_after = args.finish_after if args.mode == "finish" else None
    director = Director(setpoint=setpoint, band=band, finish_after=finish_after)
    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, "live.log"), "a")

    def log(msg):
        logf.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); logf.flush()

    state_path = os.path.join(args.out_dir, "state.json")
    try:                                        # restore learned/slow state, bridged over the gap
        with open(state_path) as f:
            st = json.load(f)
        gap = max(0.0, time.time() - st.get("ts", 0.0))
        director.gain = float(st.get("gain", 1.0))
        director.L = float(st.get("L", 0.0)) * math.exp(-0.001 * gap)
        streamer.H = float(st.get("H", 0.0)) * math.exp(-streamer.dH * gap)
        log(f"STATE restored (gap={gap/3600:.1f}h): gain={director.gain:.2f} L={director.L:.3f}")
    except FileNotFoundError:
        log("STATE none (first session)")
    except Exception as e:
        log(f"STATE load failed: {e}")

    log(f"START mode={args.mode} setpoint={setpoint} band={band:.2f} shield={args.shield} "
        f"finish_after={finish_after} config={cfg_hash}")
    print(f"mode={args.mode}  setpoint={setpoint}  shield={args.shield}  band={band:.2f}")

    def read_key_level():
        if keyboard is None:
            return None, False, False
        try:
            if keyboard.is_pressed("q"):
                return None, False, True
            for i in range(10):
                if keyboard.is_pressed(str(i)):
                    return i / 9.0, (i == 9), False
        except Exception:
            pass
        return None, False, False

    def bar(x, n=22):
        fill = int(round(min(max(x, 0), 1) * n))
        return "[" + "#" * fill + "-" * (n - fill) + "]"

    async def setup_device():
        log(f"connecting to Intiface at {args.intiface} ...")
        print(f"connecting to {args.intiface} ...")
        connector = ButtplugClientWebsocketConnector(args.intiface)
        client = ButtplugClient("CumPredict Live")
        await client.connect(connector)
        print("connected; scanning... power the toy on whenever (up to 90s)")
        await client.start_scanning()
        waited = 0.0
        while waited < 90.0 and not client.devices:
            await asyncio.sleep(2.0)
            waited += 2.0
        await client.stop_scanning()
        log(f"scan done after {waited:.0f}s: {[d.name for d in client.devices.values()]}")
        if not client.devices:
            try:
                await client.disconnect()          # don't leak this client on a failed (re)connect
            except Exception:
                pass
            raise RuntimeError("No device found. Is Intiface running and the toy powered on?")
        return client, list(client.devices.values())[0]

    stroke_up = [False]

    async def command_stroke(device, level, duration):
        """Alternate base(0.0) <-> reach(=level) so `level` = stroke DEPTH and the device
        always returns to base -> real full strokes. One failed BLE command is retried,
        never fatal. Returns the commanded position, or None if the link is dead."""
        msgs = getattr(device, "allowed_messages", {})
        for attempt in (1, 2):
            try:
                # 5s timeout so a wedged-but-connected Intiface becomes a failure (-> reconnect),
                # never an indefinite hang of the whole loop.
                if "LinearCmd" in msgs:
                    target = min(1.0, max(0.0, float(level) if stroke_up[0] else 0.0))
                    stroke_up[0] = not stroke_up[0]
                    await asyncio.wait_for(
                        device.send_linear_cmd((int(max(duration, 0.05) * 1000), target)), 5.0)
                    return target
                if "VibrateCmd" in msgs:
                    await asyncio.wait_for(device.send_vibrate_cmd(float(level)), 5.0)
                elif "RotateCmd" in msgs:
                    await asyncio.wait_for(device.send_rotate_cmd((float(level), True)), 5.0)
                return float(level)
            except Exception as e:                     # includes asyncio.TimeoutError
                log(f"COMMAND ERROR ({attempt}/2): {type(e).__name__}: {e}")
                if attempt == 1:
                    await asyncio.sleep(0.3)
        return None

    async def safe_stop(device):
        try:
            await device.send_stop_device_cmd()
        except Exception:
            pass

    async def main_loop():
        try:
            client, device = await setup_device()
        except Exception as e:
            import traceback
            log(f"SETUP ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            sys.exit(f"SETUP ERROR: {e}  (is Intiface Central running at {args.intiface}?)")
        print(f"device: {device.name}   capabilities: {list(getattr(device, 'allowed_messages', {}))}")

        out_csv = os.path.join(args.record_dir, "LIVE_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
        os.makedirs(args.record_dir, exist_ok=True)
        f = open(out_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["time_elapse", "intensity", "tired_level", "pred_arousal", "phase",
                         "E", "H", "L", "gain", "eff_hi", "alarm", "shield", "cmd_pos", "wall_dt"])

        print("\n--- your 0-9 = ground-truth only (never drives anything). q = quit ---\n")
        last_label, err_sum, err_n, t0, last_hb = "--", 0.0, 0, time.time(), 0.0
        max_label = -1.0
        arousal_prev, last_dt, prev_edges = 0.0, 0.20, 0
        shield_active = False
        consec_fail = 0
        driven = args.mode in ("edge", "finish")
        try:
            while True:
                if driven:
                    intensity, duration = director.step(arousal_prev, last_dt)
                    finishing = director.phase == "finish"     # in finish mode climax is the goal
                else:
                    intensity, duration = round(random.uniform(0, 1), 2), random.uniform(0.15, 0.80)
                    finishing = False
                # SAFETY SHIELD (all modes except finish): hard-brake above threshold, outside the
                # policy -- applies to random data-collection too, not just the Director modes.
                if finishing:
                    shield_active = False
                else:
                    if arousal_prev >= args.shield:
                        shield_active = True
                    if shield_active:
                        intensity, duration = min(intensity, 0.08), max(duration, 1.2)
                        if arousal_prev < setpoint - band:
                            shield_active = False
                ramp = time.time() - t0                        # 60s ramp-in (all modes): never slam cold
                if ramp < 60.0:
                    intensity = min(intensity, 0.30 + 0.60 * ramp / 60.0)

                t_stroke = time.time()
                cmd_pos = await command_stroke(device, intensity, duration)
                if cmd_pos is None:                      # link died -> reconnect, resume in place
                    consec_fail += 1
                    if consec_fail >= 2:
                        log("CONNECTION LOST -- auto-reconnecting (state preserved)...")
                        print("\n toy disconnected -- reconnecting...")
                        try:
                            try:
                                await client.disconnect()
                            except Exception:
                                pass
                            client, device = await setup_device()
                            consec_fail = 0
                            log("RECONNECTED -- resuming where we left off")
                        except Exception as e:
                            log(f"RECONNECT FAILED: {e}; retrying in 3s")
                            await asyncio.sleep(3.0)
                    # advance the model by the ACTUAL outage time (reconnect/scan can take up to
                    # ~90s), not one stroke duration, so predicted arousal decays with the real body
                    outage = max(time.time() - t_stroke, duration)
                    arousal_prev, _ = streamer.step(0.0, outage)
                    last_dt = duration
                    continue
                consec_fail = 0

                label, is_cum, quit_flag = None, False, False
                while time.time() - t_stroke < duration:
                    lv, cf, qf = read_key_level()
                    if lv is not None:
                        label, is_cum = lv, cf
                    if qf:
                        quit_flag = True; break
                    await asyncio.sleep(0.005)
                wall_dt = time.time() - t_stroke
                if quit_flag:
                    log("QUIT (q pressed by user)")
                    break

                arousal, alarm = streamer.step(intensity, duration)   # pure inference
                arousal_prev, last_dt = arousal, duration
                if driven and director.edges > prev_edges:
                    prev_edges = director.edges
                    log(f"EDGE #{director.edges} brake@pred={arousal:.2f} L={director.L:.2f} gain={director.gain:.2f}")
                    if finish_after is not None and director.edges >= finish_after:
                        log("FINISH PHASE ENGAGED -- no more braking")

                if label is not None:
                    last_label = f"{label:.2f}"
                    max_label = max(max_label, label)
                    err_sum += abs(arousal - label); err_n += 1
                    log(f"YOU={label:.2f} PRED={arousal:.2f} |err|={abs(arousal-label):.2f} phase={director.phase}")
                    if is_cum:
                        log(f"*** CUM (pressed 9) at PRED={arousal:.2f} edges={director.edges} "
                            f"L={director.L:.2f} -- model NOT reset (pure inference) ***")

                writer.writerow([duration / 60.0, intensity, "" if label is None else f"{label:.2f}",
                                 round(arousal, 4), director.phase if driven else "random",
                                 round(streamer.E, 4), round(streamer.H, 4), round(director.L, 4),
                                 round(director.gain, 3), round(director.hi, 4), int(alarm),
                                 int(shield_active), round(cmd_pos, 3), round(wall_dt, 4)])
                f.flush()

                mae = (err_sum / err_n) if err_n else None
                maes = f"MAE {mae:.2f}" if mae is not None else "MAE --"
                extra = (f"{director.phase.upper():6s} e{director.edges} L{director.L:.1f}"
                         if driven else "RND")
                print(f"\r INFER {bar(arousal)} {arousal:0.2f} | YOU {last_label} | {maes} | {extra}"
                      + ("  !ALARM" if alarm else "") + "   ", end="", flush=True)

                if time.time() - last_hb > 8:
                    log(f"HB t={int(time.time()-t0)}s pred={arousal:.2f} phase={director.phase} "
                        f"edges={director.edges} L={director.L:.2f} gain={director.gain:.2f}")
                    last_hb = time.time()
        except KeyboardInterrupt:
            log("QUIT (Ctrl-C)")
        except Exception as e:
            import traceback
            log(f"LOOP CRASH: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            print(f"\nLOOP CRASH: {e}")
            raise
        finally:
            await safe_stop(device)
            log(f"STOP edges={director.edges} L={director.L:.2f} gain={director.gain:.2f} "
                f"labeled={err_n} maxlabel={max_label:.2f}")
            try:
                with open(state_path, "w") as sf:
                    json.dump({"ts": time.time(), "gain": director.gain, "L": director.L,
                               "H": streamer.H, "edges": director.edges, "setpoint": setpoint}, sf, indent=2)
            except Exception as e:
                log(f"STATE save failed: {e}")
            f.close()
            # auto-delete only when labeling was POSSIBLE but unused (no label above 0); never
            # delete when the keyboard hook was unavailable -- the labels couldn't be captured.
            if max_label <= 0.0 and keyboard is not None:
                try:
                    os.remove(out_csv)
                except Exception:
                    pass
                print(f"\ndeleted garbage recording (no labels above 0): {os.path.basename(out_csv)}")
            else:
                print(f"\nsaved recording -> {os.path.abspath(out_csv)}")
            logf.close()
            try:
                await client.disconnect()
            except Exception:
                pass

    asyncio.run(main_loop())


# ------------------------------------------------------------------------------------ main
def main():
    common = argparse.ArgumentParser(add_help=False)   # shared flags, valid AFTER the subcommand
    common.add_argument("--data", default="sessions.csv",
                        help="combined CSV (session,time_elapse,intensity,tired_level) or a directory of CSVs")
    common.add_argument("--config", default="artifacts/config_core.json", help="model file to write/read")
    common.add_argument("--out-dir", default="artifacts")
    common.add_argument("--shield", type=float, default=0.78,
                        help="hard-brake arousal threshold (outside the policy; also the exported alarm theta)")

    ap = argparse.ArgumentParser(description="live arousal prediction + inference-driven edging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", parents=[common], help="fit + LOSO eval, write the model")
    sub.add_parser("fit", parents=[common], help="fit-only rerun of the model")
    sub.add_parser("parity", parents=[common], help="streaming-vs-batch exactness test")
    sub.add_parser("sim", parents=[common], help="offline brake test + Director cycling check")
    rp = sub.add_parser("report", parents=[common], help="score one session recording")
    rp.add_argument("--file", required=True)
    rp.add_argument("--session", help="session name when --file is a combined CSV")
    rp.add_argument("--recompute", action="store_true",
                    help="ignore logged predictions; re-run the current model")
    lv = sub.add_parser("live", parents=[common], help="drive the toy (needs Intiface + buttplug-py + keyboard)")
    lv.add_argument("--mode", choices=["edge", "finish", "random"], default="finish")
    lv.add_argument("--setpoint", type=float, default=0.65, help="edge target 0..1")
    lv.add_argument("--low", type=float, default=0.45, help="rebuild threshold 0..1")
    lv.add_argument("--finish-after", type=int, default=5, help="finish mode: edges before climax drive")
    lv.add_argument("--record-dir", default="recordings")
    lv.add_argument("--intiface", default="ws://127.0.0.1:12345")

    args = ap.parse_args()
    rc = {"train": cmd_train, "fit": cmd_fit, "parity": cmd_parity,
          "sim": cmd_sim, "report": cmd_report, "live": cmd_live}[args.cmd](args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
