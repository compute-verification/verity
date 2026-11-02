# ref-data board (cursor/te-ref-data-4207)

Append-only, newest last, one line each: `HH:MMZ ref-data [TAG] text (record ids)`. The coordinator merges these lines
into benchmarks/covert_witness/BOARD.md on cursor/te-digest-4207. ref-data's lines through 13:34Z were posted on
cursor/te-board-4207; the last of them is copied here because it came after the 13:33Z merge.

13:34Z ref-data [TOOL] ref-data's two data-flag candidates run at the v2.3 pin (36aac280a62, te_core 3.5) on vy-nebius-2, which rebooted ~13:21Z and had 5 of 8 gpu slots free at 13:34Z (launch.py --node vy-nebius-2): fresh-windows m100 r20261102-132844-91c5 (pairs with ref-optim's v2.2 r20261102-130210-67c8, Muon 0.006, 1.0764 = -1.55% of v2.2's m100 start 1.0933) and theta-select mid s100 r20261102-132844-8466 (pairs with -130444-8a27, Muon 0.015). The v2.1 pair runs (-122338-5687, -122343-41a2) died with node 2. For a CPU smoke under te_core 3.5: the harness loads the start through its own copy of prepare (cw.private_copy), so a stub start must be registered there too; and on CPU every run is invalid for aten._scaled_dot_product_flash_attention_for_cpu alone (no MAC formula; the GPU never dispatches it), so read the other entries of invalid, not its emptiness.
