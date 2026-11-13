# CIR-274 DENSE: design notes (choices taken, alternatives set aside)

- **COLS width.** Every COLS family of the half (the all-reduce, the fused all-reduce's residual part, ResidualAdd,
  MoeSum) is cut at W = `_cols(N, grain.norm)`, the widest halving of N within 256: 192 at N = 6144, and 192 at the
  prefill all-reduce's N = 32·6144. That is the norms' apply width (`scale_norm`'s G). So an apply block reads a block
  row as it is, and a sum of squares reads the 32 blocks through one `AllGather_v1{WORLD=32,N=192}`.
  - Set aside: W = 96, where the own partial's gather of 96 GEMM coordinates is one level, not two. That gives 64
    blocks, and the apply at G = 192 then reads 2 rows cut 2 ways. It has fewer gather units for o_proj's and
    down_proj's partials (64 against 96 a node) but twice the residual units. Not measured.
  - A width wider than the reader's G is refused by `Piece.rows`, because the apply block would cut a row.
- **Own partial wired, peers free.** `parts` is one port, outer WORLD = 8, so it is cut 8 ways. Rank r's row is wired
  where its writer is lowered, and the peers' rows are free. `_Builder.feed` takes only a port whose rows are all fed,
  so the frame gained `NodeWindow.feed_rows`, which writes `IF.Wire`s with `cut=(i, 1)` and sets the cut itself, and
  `Free.rows` (a frame commit). `check_wires` and `check_inputs` take it [test].
  - Under the frame's checks (frame commit `7ebb59cc81`): `NodeWindow.row_fed` keeps each row-fed port's cut and its
    rows' unit ranges. Refused by name: a second cut at once (`feed_rows`); a port also fed whole (`_Builder.feed`, or
    left free whole by `feed`) and a row of a unit fed twice (free twice, wired and free) before `settle`
    (`_check_rows`); a cut changed after `settle`. One test each, plus two showing what `check_wires` refuses on its
    own: a wire on a row past the cut (``cut``) and one whose writer row is another width (``width``) [test, local].
    `settle`'s rows-a-unit check and `check_proof_block` read the cut from `b.cuts`, and the after-settle check holds
    that it is still the `feed_rows` cut. Taken over a smaller form (folding `feed_rows` into `_Builder.feed` as a
    per-row feed) because that is a hunk in `window_builder.py`, which is on `main`.
  - **What no structure check catches:** a wire on a row that exists, has its width and has no other wire, but is the
    wrong one: rank r's partial on row i ≠ r, or column block a's writer unit on block a′'s reader unit. `check_wires`
    sees rows, widths, bounds and overlap, never which writer a row should be. `test_wires_refused_on_their_own` moves
    rank 0's wire to row 3 and the Circuit is built. Only a check on values sees it (item 5's notes).
- **Gathers past two levels.** The LM head's 19,360 one-word slices are 160 chunks of 121 (161 > 128 rows), so
  `_gather` now recurses on its chunk rows (a frame commit): AllGather{121,1} ×160, then {80,121} ×2, then
  {2,9680}. Where two levels fit, the units are unchanged (the frame's tests pass unchanged).
- **The FINAL fused norm** (outside every layer, by its weight's label): the residual blocks, then `RmsNormMnnvl_v1`
  whole on r. The LM head reads its one row.
- **The LM head's x** is fed one entry a unit (19,360 entries), which `_lead` folds into one feed. Its weight is read
  as rows of K (radix (N, M), step (K, 0)).
- **RoPE's cs** is a registered read of the weights record at the position's 64 words, one step a member where the
  members' rows are evenly spaced.
