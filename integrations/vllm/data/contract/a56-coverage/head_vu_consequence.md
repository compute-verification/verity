# Whole-head attention VU (S55, PROPOSED) -- count and exposure consequence

Probability of selecting an unsupported check -- completion risk, `1 - (1 - s)^n`, exact arithmetic.  `after` presumes BOTH the
partition change (head unit; new digests, counts, certificates -- the user's decision) AND approval of the fa2 guest `33cba058`

## 6500x1599_P8099

source: lane/record-6k:out/record-6k/expected_work.json (D4 RECORD-6K; MODELED by census_v2.small_scaling.terms(6500, 1599))

* attention-body VUs per RU: **98,386,178,040** of 103,029,958,196 (95.4928%) -> **2,186,730** head units (P x NL x NH); attention VU count / 44,992
* population per RU: 103,029,958,196 -> 4,645,966,886

| scenario | n_R | N (4 RU) | P(unsupported selected) per RU @2^-14 / 2^-19 / 2^-30 | E[#unsupported selected] per RU @2^-14 / 2^-19 / 2^-30 |
|---|---|---|---|---|
| approved set as-built (9fd00e60), per-primitive attention VUs | 101,556,712,720 | 406,226,850,880 | 1.00000000 / 1.00000000 / 1.00000000 | 6198529.82 / 193704.057 / 94.5820591 |
| head VU adopted (PROPOSED partition change) AND fa2 guest 33cba058 approved (PROPOSED) | 3,170,534,680 | 12,682,138,720 | 1.00000000 / 1.00000000 / 0.94780614 | 193514.079 / 6047.31498 / 2.95279052 |
| after + SiluMulBf16/Bf16AddF2fp fallbacks (elem_builder, transferred to CK-ELEM) accounted | 2,521,174,744 | 10,084,698,976 | 1.00000000 / 1.00000000 / 0.90444248 | 153880.294 / 4808.75920 / 2.34802695 |
| head VU adopted, fa2 NOT approved: every head unit is unsupported (no other checker) | 3,172,721,410 | 12,690,885,640 | 1.00000000 / 1.00000000 / 0.94791233 | 193647.546 / 6051.48584 / 2.95482707 |

Per attention-body kind (before: each is its own unsupported set; after: absorbed into the head unit):

| kind | count per RU | share | unsupported before | P(selected) per RU before @2^-14 / 2^-19 / 2^-30 |
|---|---|---|---|---|
| `AmpereBF16TcDot16_v1|batch` | 35,490,648,960 | 34.4469% | True | 1.0000000000 / 1.0000000000 / 0.9999999999 |
| `DotBf16_v1{K=64}|call` | 8,856,256,500 | 8.5958% | True | 1.0000000000 / 1.0000000000 / 0.9997382266 |
| `F2fpBf16_v1|primitive` | 8,856,256,500 | 8.5958% | True | 1.0000000000 / 1.0000000000 / 0.9997382266 |
| `F32MulFtz_v1|primitive` | 9,062,052,930 | 8.7956% | True | 1.0000000000 / 1.0000000000 / 0.9997838838 |
| `MufuEx2Ftz_v1|primitive` | 8,993,527,200 | 8.7290% | True | 1.0000000000 / 1.0000000000 / 0.9997696417 |
| `F32FmaSubFtz_v1|primitive` | 8,856,256,500 | 8.5958% | True | 1.0000000000 / 1.0000000000 / 0.9997382266 |
| `F32Max_v1|primitive` | 8,854,069,770 | 8.5937% | True | 1.0000000000 / 1.0000000000 / 0.9997376930 |
| `F32AddFtz_v1|primitive` | 8,305,398,450 | 8.0611% | True | 1.0000000000 / 1.0000000000 / 0.9995627498 |
| `F32FmaFtz_v1|primitive` | 549,082,800 | 0.5329% | True | 1.0000000000 / 1.0000000000 / 0.4003284601 |
| `F32MulFtz_v1|batch` | 139,950,720 | 0.1358% | True | 1.0000000000 / 1.0000000000 / 0.1222024255 |
| `GuardNegInfZero_v1|primitive` | 139,457,430 | 0.1354% | True | 1.0000000000 / 1.0000000000 / 0.1217990620 |
| `F32SubFtz_v1|primitive` | 137,270,700 | 0.1332% | True | 1.0000000000 / 1.0000000000 / 0.1200087386 |
| `Fa2InvSum_v1|primitive` | 2,186,730 | 0.0021% | True | 1.0000000000 / 0.9845610301 / 0.0020344786 |
| `Const32[0x3e38aa3b]_v1|primitive` | 139,457,430 | 0.1354% | True | 1.0000000000 / 1.0000000000 / 0.1217990620 |
| `Const16[0x0000]_v1|primitive` | 4,305,420 | 0.0042% | True | 1.0000000000 / 0.9997286075 / 0.0040017065 |

## 32x15

source: lane/partition:out/census/census_small.json (D2 census_v2, SMALL B0 Serve_v1{LP=32,STEPS=15}, partition 56ebcdf5...)

* attention-body VUs per RU: **4,300,560** of 166,196,840 (2.5876%) -> **12,690** head units (P x NL x NH); attention VU count / 339
* population per RU: 166,196,840 -> 161,908,970

| scenario | n_R | N (4 RU) | P(unsupported selected) per RU @2^-14 / 2^-19 / 2^-30 | E[#unsupported selected] per RU @2^-14 / 2^-19 / 2^-30 |
|---|---|---|---|---|
| approved set as-built (9fd00e60), per-primitive attention VUs | 23,061,640 | 92,246,560 | 1.00000000 / 0.99999999 / 0.02124881 | 1407.57080 / 43.9865875 / 0.02147782 |
| head VU adopted (PROPOSED partition change) AND fa2 guest 33cba058 approved (PROPOSED) | 18,761,080 | 75,044,320 | 1.00000000 / 0.99999999 / 0.01732085 | 1145.08544 / 35.7839202 / 0.01747261 |
| after + SiluMulBf16/Bf16AddF2fp fallbacks (elem_builder, transferred to CK-ELEM) accounted | 14,988,856 | 59,955,424 | 1.00000000 / 0.99999999 / 0.01386247 | 914.847167 / 28.5889739 / 0.01395945 |
| head VU adopted, fa2 NOT approved: every head unit is unsupported (no other checker) | 18,773,770 | 75,095,080 | 1.00000000 / 0.99999999 / 0.01733247 | 1145.85998 / 35.8081245 / 0.01748443 |

Per attention-body kind (before: each is its own unsupported set; after: absorbed into the head unit):

| kind | count per RU | share | unsupported before | P(selected) per RU before @2^-14 / 2^-19 / 2^-30 |
|---|---|---|---|---|
| `AmpereBF16TcDot16_v1|batch` | 1,607,040 | 0.9669% | True | 1.0000000000 / 0.9533549501 / 0.0014955531 |
| `DotBf16_v1{K=64}|call` | 304,560 | 0.1833% | True | 0.9999999915 / 0.4406067996 / 0.0002836033 |
| `F2fpBf16_v1|primitive` | 304,560 | 0.1833% | True | 0.9999999915 / 0.4406067996 / 0.0002836033 |
| `F32MulFtz_v1|primitive` | 12,690 | 0.0076% | True | 0.5390933590 / 0.0239137028 / 0.0000118184 |
| `MufuEx2Ftz_v1|primitive` | 304,560 | 0.1833% | True | 0.9999999915 / 0.4406067996 / 0.0002836033 |
| `F32FmaSubFtz_v1|primitive` | 304,560 | 0.1833% | True | 0.9999999915 / 0.4406067996 / 0.0002836033 |
| `F32Max_v1|primitive` | 291,870 | 0.1756% | True | 0.9999999816 / 0.4269019020 / 0.0002717881 |
| `F32AddFtz_v1|primitive` | 295,110 | 0.1776% | True | 0.9999999849 / 0.4304326217 / 0.0002748048 |
| `F32MulFtz_v1|batch` | 812,160 | 0.4887% | True | 0.9999999999 / 0.7875553389 / 0.0007560969 |
| `GuardNegInfZero_v1|primitive` | 12,690 | 0.0076% | True | 0.5390933590 / 0.0239137028 / 0.0000118184 |
| `Fa2InvSum_v1|primitive` | 12,690 | 0.0076% | True | 0.5390933590 / 0.0239137028 / 0.0000118184 |
| `Const32[0x3e38aa3b]_v1|primitive` | 12,690 | 0.0076% | True | 0.5390933590 / 0.0239137028 / 0.0000118184 |
| `Const16[0x0000]_v1|primitive` | 25,380 | 0.0153% | True | 0.7875650683 / 0.0472555404 / 0.0000236366 |

What `after` does NOT remove: `Const32[0x00000000]` (shared with the rest of the body; kept), the RMSNorm / rope / sampler / gather kinds
(CK-ELEM's), `ArgmaxStep`.  What `after` requires: (1) the user adopts the head unit as the attention VU (partition change); (2) the fa2
guest is approved (its `tc_dot16` NaN extension is total, `fa2.rs:254-298`, = the registered primitive; the approved guest's is not); (3) the
record commits V rows (`qkv_proj.gemm/port=0`) -- present in the REC profile, absent in `profile-hash`.
