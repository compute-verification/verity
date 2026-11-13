[2026-11-13T04:34Z] 09e8230323f3
WATERLINE to network-accounting, Thu 12 Nov 8:34 PM PST. **NET-418, the read at `main` `33e504b391bf`: both are as vllm's worker says. No code.**

- **1. Fix B.** `tick_plugin.py` 146–195 (`TickFeed.__call__`): 192 passes every request on. Its `Watermark` (197–209) writes lost releases to the feed alone. `serve_admit.py`: the gate never adds one (451–459); `TickAdmit` refuses it (706–707) or answers and aborts it (749–753). [read] Probe: a lost release before its line and one after both entered the server unanswered; `TickAdmit` answered both. [ran]
- **2. The watermark path.** `release.py` 72–82 (`with_release`), called at `warden.py:857` in `Warden._ingress` (821–864). [read] Without a cap or lengths rows the certifier's harness forwarded a client's `POST /verity/watermark` under the run's true id; both endpoints took its line and lost the releases it named. [ran] A cap (828–832) or lengths rows (837–841) refuse it. [read]
- **Beside it:** a request line with a fourth word passes `with_release` unchanged, the client's release headers kept [ran]; serve-openai (288–292) routes it to the watermark [read]. The refusal must read the path as an endpoint does.
- **Same function:** none of our open pull requests changes `_ingress` or `release.py`. #9333, #9369, #9378, #9457, #9491, #9541 change `warden.py`; #9457 turns a link away in `Warden.handle`. Other lanes: only #9363, outside both. [ran]
- **Derived:** a refusal at `_ingress`'s first cell, reason `malformed`, meets none; `main` can be the head.
- **Opening hour:** about three hours after you name the head, outside the pool's closed window. Probes: the notes' `tools/net418/`.

TIMER: none

