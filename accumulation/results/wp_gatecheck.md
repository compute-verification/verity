# Weight-presence certificate: gate-level check of the op-level rule

Generated 2026-09-16 13:30:36 by `accumulation.redteam.wp_gatecheck`.  For every chained weight tensor of the tiny registry training circuits, `Up_lb` (op-level between work, `bounds/lower_wp.py`) must not exceed the smallest true upstream work `T_e` of any element's update gate inside the gate-level hull (`exact.solve.flatten`).

| cell | gates | chained tensors | skipped (partial reads) | violations | min_e T_e / Up_lb |
|---|---|---|---|---|---|
| tiny local-sgd K=2 Q_step=2 seq=2 | 5981 | 11 | 0 | 0 | - |
| tiny local-sgd K=2 Q_step=4 seq=2 | 10517 | 11 | 0 | 0 | - |
| tiny local-sgd K=2 Q_step=4 seq=4 | 11525 | 11 | 0 | 0 | - |
| tiny local-sgd K=3 Q_step=2 seq=2 | 8873 | 22 | 0 | 0 | - |
| tiny2 local-sgd K=2 Q_step=2 seq=2 | 12933 | 20 | 0 | 0 | [5.809, 10.494] |
| tiny2 local-sgd K=2 Q_step=4 seq=2 | 22765 | 20 | 0 | 0 | [5.802, 10.488] |
| tiny2 local-sgd K=2 Q_step=4 seq=4 | 25645 | 20 | 0 | 0 | [6.889, 12.660] |
| tiny2 local-sgd K=3 Q_step=2 seq=2 | 19209 | 40 | 0 | 0 | [5.809, 10.494] |
| tiny2-L3 local-sgd K=2 Q_step=2 seq=2 | 18781 | 29 | 0 | 0 | [1.785, 10.494] |
| tiny2-L3 local-sgd K=2 Q_step=4 seq=2 | 33093 | 29 | 0 | 0 | [1.784, 10.488] |
| tiny2-L3 local-sgd K=2 Q_step=4 seq=4 | 37413 | 29 | 0 | 0 | [1.817, 12.660] |
| tiny2-L3 local-sgd K=3 Q_step=2 seq=2 | 27905 | 58 | 0 | 0 | [1.785, 10.494] |
| tiny2-L4 local-sgd K=2 Q_step=2 seq=2 | 24629 | 38 | 0 | 0 | [1.428, 10.494] |
| tiny2-L4 local-sgd K=2 Q_step=4 seq=2 | 43421 | 38 | 0 | 0 | [1.427, 10.488] |
| tiny2-L4 local-sgd K=2 Q_step=4 seq=4 | 49181 | 38 | 0 | 0 | [1.439, 12.660] |
| tiny2-L4 local-sgd K=3 Q_step=2 seq=2 | 36601 | 76 | 0 | 0 | [1.428, 10.494] |
| tiny2-L4-kv2 local-sgd K=2 Q_step=2 seq=2 | 26613 | 38 | 0 | 0 | [1.426, 11.012] |
| tiny2-L4-kv2 local-sgd K=2 Q_step=4 seq=2 | 46813 | 38 | 0 | 0 | [1.425, 11.006] |
| tiny2-L4-kv2 local-sgd K=2 Q_step=4 seq=4 | 52573 | 38 | 0 | 0 | [1.437, 13.179] |
| tiny2-L4-kv2 local-sgd K=3 Q_step=2 seq=2 | 39545 | 76 | 0 | 0 | [1.426, 11.012] |

**Total violations: 0.**
