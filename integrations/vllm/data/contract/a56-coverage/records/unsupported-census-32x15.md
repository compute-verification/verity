# Unsupported selected VUs by kind and operator — `V5-32x15-bex0-inc` (1479 units; program `Serve_v1{C={6f196795e564},LP=32,STEPS=15}`, built in 2.6 s)

| kind | count | operator(s) | bucket | owner |
|---|---|---|---|---|
| `Bf16ToF32_v1|batch` | 238 | post 134, pre 104 | rest | open (integrator to assign) |
| `F32Mul_v1|batch` | 203 | post 101, pre 100, t=31/final 1, t=35/final 1 | rest | open (integrator to assign) |
| `SiluMulBf16_v1|batch` | 152 | post 152 | CK-ELEM four | CK-ELEM |
| `F2fpBf16_v1|batch` | 145 | attn 51, pre 49, post 45 | mixed | A5/6 (attn part) / open |
| `Bf16AddF2fp_v1|batch` | 113 | post 61, pre 51, t=42/final 1 | CK-ELEM four | CK-ELEM |
| `F32Add_v1|primitive` | 99 | post 60, pre 39 | rest | open (integrator to assign) |
| `AmpereBF16TcDot16_v1|batch` | 97 | attn 97 | attention body | A5/6 |
| `F32Mul_v1|primitive` | 91 | post 54, pre 37 | rest | open (integrator to assign) |
| `F32MulFtz_v1|batch` | 56 | attn 56 | attention body | A5/6 |
| `ArgmaxStep_v1|scan` | 51 | t=44/final 7, t=38/final 6, t=42/final 6, t=32/final 5, t=35/final 4, t=41/final 4, t=37/final 3, t=39/final 3, t=40/final 3, t=43/final 3, t=34/final 2, t=36/final 2, t=45/final 2, t=33/final 1 | rest | open (integrator to assign) |
| `Const32[0x00000000]_v1|primitive` | 46 | attn 46 | attention body | A5/6 |
| `RopeOut_v1|batch` | 44 | pre 44 | CK-ELEM four | CK-ELEM |
| `RopeOutAdd_v1|batch` | 37 | pre 37 | CK-ELEM four | CK-ELEM |
| `F32AddFtz_v1|primitive` | 22 | attn 22 | attention body | A5/6 |
| `DotBf16_v1{K=64}|call` | 20 | attn 20 | attention body | A5/6 |
| `F32Max_v1|primitive` | 19 | attn 19 | attention body | A5/6 |
| `F2fpBf16_v1|primitive` | 15 | attn 15 | attention body | A5/6 |
| `MufuEx2Ftz_v1|primitive` | 14 | attn 14 | attention body | A5/6 |
| `F32FmaSubFtz_v1|primitive` | 11 | attn 11 | attention body | A5/6 |
| `GatherBf16x49152_v1|batch` | 2 | t=6/embed 1, t=44/embed 1 | rest | open (integrator to assign) |
| `Const32[0x3e38aa3b]_v1|primitive` | 2 | attn 2 | attention body | A5/6 |
| `GuardNegInfZero_v1|primitive` | 1 | attn 1 | attention body | A5/6 |
| `Const16[0x0000]_v1|primitive` | 1 | attn 1 | attention body | A5/6 |

| bucket | units | share |
|---|---|---|
| rest | 684 | 46.2 % |
| CK-ELEM four (buildable) | 346 | 23.4 % |
| attention body (interior, uncommitted -- InteriorRefuser / head VU PROPOSED) | 304 | 20.6 % |
| mixed (attention + other operators) | 145 | 9.8 % |

By operator: `post` 607, `pre` 461, `attn` 355, `t=42/final` 7, `t=44/final` 7, `t=38/final` 6, `t=32/final` 5, `t=35/final` 5, `t=41/final` 4, `t=37/final` 3, `t=39/final` 3, `t=40/final` 3, `t=43/final` 3, `t=34/final` 2, `t=36/final` 2, `t=45/final` 2, `t=6/embed` 1, `t=31/final` 1, `t=33/final` 1, `t=44/embed` 1

counts of SELECTED unsupported VUs in retained RUs (the record's own list), located by the verifier's ProgramLocator; not a population census
