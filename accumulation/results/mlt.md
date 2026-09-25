### 12a. Exact optimum vs closed form (small instances)

| d | L | T | b_w / b_a (B) | a^0 | Gamma (MACs) | gates | closed-form LB | exact I* | proven optimal | best rectangle (m x t) | I* / LB | formula regime |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 4 | 1 / 1 | literal | 2 | 16 | 16.00 | 24 | yes | 24 (1 x 2) | 1.50 | no |
| 1 | 4 | 4 | 1 / 1 | literal | 4 | 16 | 16.00 | 16 | yes | 16 (2 x 2) | 1.00 | yes |
| 1 | 4 | 4 | 1 / 1 | literal | 8 | 16 | 11.31 | 12 | yes | 12 (2 x 4) | 1.06 | yes |
| 1 | 4 | 6 | 1 / 1 | literal | 4 | 24 | 24.00 | 24 | yes | 24 (2 x 2) | 1.00 | yes |
| 1 | 4 | 6 | 1 / 1 | literal | 6 | 24 | 19.60 | 20 | yes | 20 (2 x 3) | 1.02 | yes |
| 1 | 4 | 6 | 1 / 1 | literal | 12 | 24 | 13.86 | 14 | yes | 14 (4 x 3) | 1.01 | yes |
| 1 | 6 | 4 | 1 / 1 | literal | 3 | 24 | 24.00 | 32 | no (bound 29) | 32 (3 x 1) | 1.33 | no |
| 1 | 6 | 4 | 1 / 1 | literal | 6 | 24 | 19.60 | 20 | yes | 20 (3 x 2) | 1.02 | yes |
| 1 | 6 | 4 | 1 / 1 | literal | 9 | 24 | 16.00 | 18 | yes | 18 (2 x 4) | 1.12 | yes |
| 1 | 6 | 4 | 1 / 1 | literal | 12 | 24 | 13.86 | 14 | yes | 14 (3 x 4) | 1.01 | yes |
| 1 | 3 | 8 | 1 / 1 | literal | 3 | 24 | 24.00 | 32 | no (bound 29) | 32 (3 x 1) | 1.33 | no |
| 1 | 3 | 8 | 1 / 1 | literal | 6 | 24 | 19.60 | 20 | yes | 20 (3 x 2) | 1.02 | yes |
| 1 | 3 | 8 | 1 / 1 | literal | 12 | 24 | 14.00 | 14 | yes | 14 (3 x 4) | 1.00 | yes |
| 1 | 4 | 6 | 2 / 1 | literal | 4 | 24 | 24.00 | 36 | yes | 36 (2 x 2) | 1.50 | no |
| 1 | 4 | 6 | 2 / 1 | literal | 8 | 24 | 24.00 | 26 | yes | 28 (2 x 3) | 1.08 | yes |
| 1 | 4 | 6 | 2 / 1 | literal | 12 | 24 | 19.60 | 20 | yes | 20 (2 x 6) | 1.02 | yes |
| 1 | 4 | 6 | 1 / 2 | literal | 4 | 24 | 33.94 | 36 | yes | 36 (2 x 2) | 1.06 | yes |
| 1 | 4 | 6 | 1 / 2 | literal | 8 | 24 | 24.00 | 24 | yes | 24 (4 x 2) | 1.00 | yes |
| 1 | 4 | 6 | 1 / 2 | literal | 12 | 24 | 20.00 | 20 | yes | 20 (4 x 3) | 1.00 | yes |
| 2 | 2 | 4 | 1 / 4 | literal | 4 | 32 | 64.00 | 96 | yes | 96 (1 x 1) | 1.50 | no |
| 2 | 2 | 4 | 1 / 4 | literal | 8 | 32 | 64.00 | 64 | yes | 64 (2 x 1) | 1.00 | yes |
| 2 | 2 | 4 | 1 / 4 | literal | 16 | 32 | 48.00 | 48 | yes | 48 (2 x 2) | 1.00 | yes |
| 2 | 3 | 2 | 1 / 4 | literal | 6 | 24 | 48.00 | 64 | yes | 72 (1 x 1) | 1.33 | no |
| 2 | 3 | 2 | 1 / 4 | literal | 12 | 24 | 39.19 | 40 | yes | 40 (3 x 1) | 1.02 | yes |
| 2 | 2 | 4 | 1 / 2 | literal | 4 | 32 | 32.00 | 64 | no (bound 60) | 64 (1 x 1) | 2.00 | no |
| 2 | 2 | 4 | 1 / 2 | literal | 8 | 32 | 32.00 | 48 | yes | 48 (1 x 2) | 1.50 | no |
| 2 | 2 | 4 | 1 / 2 | literal | 16 | 32 | 32.00 | 32 | yes | 32 (2 x 2) | 1.00 | yes |
| 2 | 2 | 3 | 1 / 1 | literal | 4 | 24 | 14.00 | 36 | no (bound 34) | 36 (1 x 1) | 2.57 | no |
| 2 | 2 | 3 | 1 / 1 | literal | 8 | 24 | 14.00 | 26 | yes | 28 (1 x 2) | 1.86 | no |
| 1 | 4 | 4 | 1 / 1 | token | 4 | 20 | 13.00 | 19 | yes | 24 (1 x 2) | 1.46 | yes |
| 1 | 4 | 4 | 1 / 1 | token | 8 | 20 | 9.49 | 14 | yes | 16 (2 x 2) | 1.48 | yes |
| 1 | 3 | 6 | 1 / 1 | token | 4 | 24 | 13.00 | 21 | yes | 24 (3 x 1) | 1.62 | yes |
| 1 | 3 | 6 | 1 / 1 | token | 8 | 24 | 10.00 | 15 | yes | 15 (3 x 2) | 1.50 | yes |
| 2 | 2 | 3 | 1 / 4 | token | 10 | 30 | 11.00 | 27 | yes | 27 (2 x 1) | 2.45 | no |
| 2 | 2 | 3 | 1 / 4 | token | 20 | 30 | 11.00 | 19 | yes | 19 (2 x 2) | 1.73 | no |

### 12b. Read-off at the draft's Kimi-K3 Gamma

| Circuit | b_w / b_a (bits) | optimal IU: layers x positions | novel bits per position: LB / U | slowdown NCI/alpha: LB / U | alpha_C = size(M) / size(step) | training rho >= alpha_C x LB |
|---|---|---|---|---|---|---|
| draft's M_{L,T} as written: 93 square 7168x7168 layers, a^0 = d-wide novel input | 16 / 16 | 93 x 9.9e+07 | 1.15e+05 / 1.15e+05 | 79,283x / 79,283x | 0.015 | 1,203x |
| draft's M_{L,T} as written: 93 square 7168x7168 layers, a^0 = d-wide novel input | 4.25 / 16 | 93 x 9.9e+07 | 1.15e+05 / 1.15e+05 | 78,895x / 78,895x | 0.015 | 1,197x |
| same, a^0 derived from the 32-bit token | 16 / 16 | 93 x 9.9e+07 | 785 / 801 | 539x / 550x | 0.015 | 8x |
| same, a^0 derived from the 32-bit token | 4.25 / 16 | 93 x 9.9e+07 | 232 / 236 | 159x / 162x | 0.015 | 2x |
| Kimi-K3 matmul weights, routing fully clusterable (rho = 1: IU holds only active weights) | 16 / 16 | 47 x 8.9e+06 | 2.94e+05 / 3.01e+05 | 9,180x / 9,390x | 0.334 | 3,063x |
| Kimi-K3 matmul weights, routing fully clusterable (rho = 1: IU holds only active weights) | 4.25 / 16 | 93 x 4.5e+06 | 9.67e+04 / 9.88e+04 | 3,020x / 3,086x | 0.334 | 1,008x |
| Kimi-K3 matmul weights, balanced routing (rho = 26.7: IU holds all experts of its layers) | 16 / 16 | 12 x 3.5e+07 | 2e+06 / 2.05e+06 | 62,385x / 64,153x | 0.334 | 20,813x |
| Kimi-K3 matmul weights, balanced routing (rho = 26.7: IU holds all experts of its layers) | 4.25 / 16 | 19 x 2.2e+07 | 9.74e+05 / 9.88e+05 | 30,417x / 30,847x | 0.334 | 10,148x |
