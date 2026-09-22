# Unsupported selected VUs by kind and operator — `V5-TINY-bex0-inc` (653 units; program `Serve_v1{C={6f196795e564},LP=4,STEPS=8}`, built in 0.6 s)

| kind | count | operator(s) | bucket | owner |
|---|---|---|---|---|
| `F32Mul_v1|batch` | 111 | post 62, pre 49 | rest | open (integrator to assign) |
| `Bf16ToF32_v1|batch` | 103 | post 58, pre 44, t=10/final 1 | rest | open (integrator to assign) |
| `SiluMulBf16_v1|batch` | 69 | post 69 | CK-ELEM four | CK-ELEM |
| `F2fpBf16_v1|batch` | 69 | post 28, pre 21, attn 19, t=3/final 1 | mixed | A5/6 (attn part) / open |
| `F32Add_v1|primitive` | 48 | post 26, pre 22 | rest | open (integrator to assign) |
| `F32Mul_v1|primitive` | 48 | pre 25, post 23 | rest | open (integrator to assign) |
| `ArgmaxStep_v1|scan` | 45 | t=7/final 9, t=11/final 8, t=6/final 7, t=3/final 6, t=10/final 6, t=5/final 3, t=9/final 3, t=4/final 2, t=8/final 1 | rest | open (integrator to assign) |
| `Bf16AddF2fp_v1|batch` | 40 | post 24, pre 16 | CK-ELEM four | CK-ELEM |
| `F32MulFtz_v1|batch` | 31 | attn 31 | attention body | A5/6 |
| `Const32[0x00000000]_v1|primitive` | 29 | attn 29 | attention body | A5/6 |
| `AmpereBF16TcDot16_v1|batch` | 21 | attn 21 | attention body | A5/6 |
| `RopeOut_v1|batch` | 15 | pre 15 | CK-ELEM four | CK-ELEM |
| `RopeOutAdd_v1|batch` | 12 | pre 12 | CK-ELEM four | CK-ELEM |
| `MufuEx2Ftz_v1|primitive` | 2 | attn 2 | attention body | A5/6 |
| `F2fpBf16_v1|primitive` | 2 | attn 2 | attention body | A5/6 |
| `F32Max_v1|primitive` | 2 | attn 2 | attention body | A5/6 |
| `DotBf16_v1{K=64}|call` | 2 | attn 2 | attention body | A5/6 |
| `Const32[0x3e38aa3b]_v1|primitive` | 1 | attn 1 | attention body | A5/6 |
| `Fa2InvSum_v1|primitive` | 1 | attn 1 | attention body | A5/6 |
| `F32MulFtz_v1|primitive` | 1 | attn 1 | attention body | A5/6 |
| `F32AddFtz_v1|primitive` | 1 | attn 1 | attention body | A5/6 |

| bucket | units | share |
|---|---|---|
| rest | 355 | 54.4 % |
| CK-ELEM four (buildable) | 136 | 20.8 % |
| attention body (interior, uncommitted -- InteriorRefuser / head VU PROPOSED) | 93 | 14.2 % |
| mixed (attention + other operators) | 69 | 10.6 % |

By operator: `post` 290, `pre` 204, `attn` 112, `t=7/final` 9, `t=11/final` 8, `t=3/final` 7, `t=6/final` 7, `t=10/final` 7, `t=5/final` 3, `t=9/final` 3, `t=4/final` 2, `t=8/final` 1

counts of SELECTED unsupported VUs in retained RUs (the record's own list), located by the verifier's ProgramLocator; not a population census
