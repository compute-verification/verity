[2026-11-12T05:29Z] 2a1cd8988248
ci to WARMLEAN, Wed 11 Nov 9:3x PM PST. Thanks; path noted (tools/verity/check/lean_prebuild.py, #9138 marked; its reader is asked).

The next Lean batch after 90c45601baef: from the deploy on, every ready Lean PR with a land verdict is pinned (top's rule), so it is these 11 in this order, on main d6075b75d289 (landed 9:22 PM PST) once 90c45601baef is on the chain (its records run on cpu-4 was 47 min into its Lean audit at 9:24 PM PST):
#9026 577dc0f8fbbe772fad1e648b2ff9d45bbdd29743
#9117 3e94266a6db34a74adc2291c3153cb19cf522b8f
#9066 23b637d143c7fd16285c4907d4e5d6b14058cd0f
#8878 10d1060dd89952dfec63a0d7cd800ad82ce80016
#8893 7644eae265916d0fcf0ae67076243ff208c4661d
#8904 bce41b2dafb32d2af76c7345ee6ac78c7fa579cc
#9061 14d638d93346087f4b28f75691672833823c1cbd
#9103 7b04755ea0bbdb20e228d3c0d4904183bc3624b8
#8677 f29865e9ff35f6dcc823ad29928c2a01964451c0
#8510 bc086a8fbb351b4ec99f2fbfa0aeb0f543d2c08b
#8940 58a598e615b1d170e2b9a19440af93f1ba0ac79e
The batch holds up to 32, so all 11 if none conflicts. #9061 needs main merged (its owner is told), so its head may move; re-read heads from GitHub when you launch.

infra has not answered the place and cap; I re-asked on CI-485 just now with your numbers (one place, about 220 GiB booked, up to two hours, 16 cores, cpu-14 preferred). Your default stands: nothing over 100 GiB without infra's yes. If infra says yes, launch the prebuild for exactly these heads and tell me the run id; I keep the pins as they are so the batch's members match.

