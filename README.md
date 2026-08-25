# cumpredict - [Demo](https://psychosmiley.github.io/cumpredict/)

### How does it work?
The *JOI* stack contains two modules:
- Director: a CfC **N**eural**N**etwork (close to liquid NN) that **controls the toy** toward the edge and adapts to your input (optional: keypad, mic or slider) in real time, using different patterns and learning from you on the fly, "`your input vs what the Predictor says`". Each new session is logged in your browser to build a personal dataset.
- Predictor: much like weather forecasting, except it predicts your arousal level up to climax. Once you have gathered a [dataset](./sessions.csv), you can retrain it to suit you best, ladies.


To use the demo you'll need to connect a sex toy over Web Bluetooth or an [Intiface](https://intiface.com/#intiface-central) local server.

- `v0.1`-`v0.4`: Python, torch, local saved model, mechanistic; `v0.2`-`v0.4` switched to liquid NN, CfC+ODE Predictor.  `v0.5`: switched to HTML.
