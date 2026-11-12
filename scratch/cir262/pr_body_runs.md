
## circuit-check

Run `r20261112-173637-e88d` on `1b61bd883e`, a check slot, rc 0 for each of its three passes. The files circuit-check reads
here (`verity/definitions/registry/flashmla_dsv32.py`, its sidecar, `numerics/flashmla_dsv32_twin.py`, `registry/flashmla.py`,
`numerics/flashmla_twin.py`) are unchanged from that commit to this head: the two commits after it touch only the capture
script and its test. The update pass (`--update-pins`, run `r20261112-152809-d14e`) wrote nothing: the module defines no
new piece.

The sidecar (`verity/definitions/registry/flashmla_dsv32.circuit-check.py`): 12 targets, 0 failures, 0 unchecked, all 12
compared with the reference on random vectors.

~~~text
ok  FlashMlaV32KvRow_v1{DN=512,DR=64,QT=128}                                     failures=0 unchecked=0 q_call=2118u/42050b
ok  FlashMlaV32Block_v1{DK=576,DV=512,BN=64,FIRST=True,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}   failures=0 unchecked=0 q_call=2495u/78816b
ok  FlashMlaV32Block_v1{DK=576,DV=512,BN=64,FIRST=False,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}  failures=0 unchecked=0 q_call=3014u/95424b
ok  FlashMlaV32SplitEnd_v1{DV=512,PARTIAL=False}                                 failures=0 unchecked=0 q_call=1034u/24865b
ok  FlashMlaV32SplitEnd_v1{DV=512,PARTIAL=True}                                  failures=0 unchecked=0 q_call=522u/16673b
ok  FlashMlaV32CombineWeights_v1{NS=2}                                           failures=0 unchecked=0 q_call=18u/545b
ok  FlashMlaV32CombineWeights_v1{NS=32}                                          failures=0 unchecked=0 q_call=198u/6305b
ok  FlashMlaV32CombineValues_v1{NS=2,G=256}                                      failures=0 unchecked=0 q_call=768u/20480b
ok  FlashMlaV32CombineValues_v1{NS=32,G=256}                                     failures=0 unchecked=0 q_call=8448u/266240b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,2),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}    failures=0 unchecked=0 q_call=6543u/199105b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,1,3),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}  failures=0 unchecked=0 q_call=10602u/327907b
ok  FlashMlaV32Decode_v1{NH=2,SPLITS=(0,1,3),DN=512,DR=64,QT=128,BN=64,DOT=…,SCALE=0.0625}  failures=0 unchecked=0 q_call=427860u/8729414b
circuit-check: 12 target(s), 0 failure(s), 0 unchecked
~~~

Heads at served sizes (32, 16 and 1 splits) and two combines: 5 targets, 0 failures, 0 unchecked, all on random vectors.

~~~text
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,1,…,32),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}   failures=0 unchecked=0 q_call=113638u/3594433b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,2,…,32),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}   failures=0 unchecked=0 q_call=105302u/3328177b
ok  FlashMlaV32DecodeHead_v1{SPLITS=(0,32),DK=576,DV=512,BN=64,DOT=…,SCALE=0.0625}       failures=0 unchecked=0 q_call=96963u/3061825b
ok  FlashMlaV32CombineValues_v1{NS=16,G=256}                                          failures=0 unchecked=0 q_call=4352u/135168b
ok  FlashMlaCombine_v1{NS=16,D=512}                                                   failures=0 unchecked=0 q_call=8817u/273859b
circuit-check: 5 target(s), 0 failure(s), 0 unchecked
~~~

One whole token at 16 splits: 0 failures, 1 unchecked. The warning is `vectors/edge-only`: at this size the token was
compared with the reference on edge vectors only, and the partition check did not run on the whole. Its units are the
targets above.

~~~text
unchecked  FlashMlaV32Decode_v1{NH=8,SPLITS=(0,2,…,32),DN=512,DR=64,QT=128,BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625}  failures=0 unchecked=1
unchecked vectors/edge-only: 1 over 1 target(s)
circuit-check: 1 target(s), 0 failure(s), 1 unchecked (warnings, which never fail)
~~~
