[2026-11-08T12:09Z] 8e59d65b4989
BINDTERM 12:10Z: #8015's records commit d07a440d93 is pushed on top of a28977b63e. If you haven't merged a28977b63e yet, merge d07a440d93 instead.
- It changes only Security/lean-audit.json and Security/Properties/lean-audit.json.
- They were regenerated over the restacked tree with `lean_changed.py --records Security --update --moved` (r20261108-113548-c76d). Security, Security/Proofs and Security/Properties pass, and no SampledProofs name is left in either lock.
- INLINE: Security is 914 (cap 917), Security/Properties 132 (cap 132).
- It merges cleanly with next.
The quick suites wait for a slot, with 11 runs ahead on both cpu-11 and cpu-7. The mark follows them, ETA about 13:15Z; unless a suite fails, the marked head is d07a440d93. I'll send it when it's marked.

