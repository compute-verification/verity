[2026-11-19T04:31Z] 8e4a3dcbec36
The three circuit-check rows asked by the network lane, for Thursday's class (4 slots, prompt 27, 2 outputs, Llama 3.2 1B, stop id 128001); as their worker wrote them, not applied anywhere:

--- adapters/vllm/verity_vllm/definitions/registry/io_shell.circuit-check.py
  ("SelectEntry_v1:64", lambda: bind(X.SelectEntry, V=128256, C=16, KO=64, FIRST=False, IDB=17, NP=1024,
-                                     STOP=(128001,)))]
+                                     STOP=(128001,))),
+ ("SelectEntry_v1:2:27", lambda: bind(X.SelectEntry, V=128256, C=16, KO=2, FIRST=False, IDB=17, NP=27,
+                                      STOP=(128001,)))]
# only if the served-shape branch lands, after SlotEntry_v1:64:
+ ("SlotEntry_v1:2:np1:stop", lambda: bind(X.SlotEntry, KO=2, IDB=17, NP=1, STOP=(128001,))),
--- adapters/vllm/verity_vllm/definitions/registry/boolean_io_shell.circuit-check.py
  ("ScanSlots_v3:4:64:1024", lambda: _scan_slots(9353, (376, 168, 200, 1528), 37387, 4, 1024)),
+ ("ScanSlots_v3:4:2:27", lambda: _scan_slots(9090, (376, 168, 200, 474), 36333, 4, 27)),

Closure digests by their run: SelectEntry_v1 at that binding 6a5429d9...89384; SlotEntry_v1{NP=1, STOP=[128001]} 5bf30fb7...; ScanSlots_v3{NS=9090, NP=27} 1a9128a0....
Their circuit-check run with the rows applied: 0 failures, 4 warnings: redundant gates 3 in each entry; dead gates 228 in the scan-slots; for SelectEntry_v1 at this binding no realization was compared on a random vector.
Each sidecar's docstring should name its new row.

