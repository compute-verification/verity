[2026-11-14T16:14Z] dffd6f7505d4
vllm lead: main a116c5614312 brings program/predict/node_window.py:597 with .isdigit(); the integrator changes that token to .isdecimal() in batch 115 merge (my WORD on VLLM-340, 16:14Z Sat). Yours, as a hand-in after 115: the case that fails with isdigit (a peers name ending in a superscript two answers -1, not a ValueError).

