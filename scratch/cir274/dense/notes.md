# CIR-274 DENSE: design notes (choices taken, alternatives set aside)

- **COLS width.** Every COLS family of the half (the all-reduce, the fused all-reduce's residual part, ResidualAdd,
  MoeSum) is cut at W = `_cols(N, grain.norm)`, the widest halving of N within 256: 192 at N = 6144, and 192 at the
  prefill all-reduce's N = 32·6144. That is the norms' apply width (`scale_norm`'s G). So an apply block reads a block
  row as it is, and a sum of squares reads the 32 blocks through one `AllGather_v1{WORLD=32,N=192}`.
  - Set aside: W = 96, where the own partial's gather of 96 GEMM coordinates is one level, not two. That gives 64
    blocks, and the apply at G = 192 then reads 2 rows cut 2 ways. It has fewer gather units for o_proj's and
    down_proj's partials (64 against 96 a node) but twice the residual units.
  - **Measured (both halves, 43 ids, 2 outputs, 4 layers, rank 0):** W = 192 is `r20261113-022216-daa9`, W = 96 is
    `r20261113-022216-58ba` (`_cols` patched to 96, so the fused norm's `RmsNormMnnvlApply` is at G = 96 too).

    |  | W = 192 | W = 96 |
    |---|---|---|
    | units | 5,137,384 | 5,150,344 (+12,960) |
    | templates | 105 | 106 |
    | wire patterns | 13,238 | 13,232 |

    The units that change, 192 → 96:

    | template | W = 192 | W = 96 |
    |---|---|---|
    | AllReduceMnnvl | 32 | 64 |
    | AllReduceResidualMnnvl | 256 | 512 |
    | RmsNormMnnvlApply | 224 | 448 |
    | MoeSum | 1,408 | 2,816 |
    | ResidualAdd | 1,408 | 2,816 |
    | `AllGather{2,96}` | 3,040 | 1,408 |
    | `AllGather{6,32}` | 11,264 | — |
    | `AllGather{3,32}` | — | 22,528 |
    | `AllGather{32,192}` | 620 | 604 |
    | `AllGather{64,96}` | — | 16 |

    Gather units total 56,574 at W = 192 and 66,206 at W = 96.

    Rows a unit are the same at both widths: AllReduceMnnvl 9, AllReduceResidualMnnvl 10, MoeSum 9, ResidualAdd 3,
    RmsNormMnnvlApply 4. The frame's `RMSNormApplyCudaVector8{G=192}` goes from 4 to 5, because it reads 96-wide rows.
    96 is not fewer units. Every COLS family doubles, and MoeSum's 96-word `y` row is gathered from 3 of the expert W's
    32-word rows instead of 6, but by twice as many gathers (`{3,32}` 22,528 against `{6,32}` 11,264). So **192
    stays**.
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
- **A check on values (item 5: sized, not built).**
  - *Nothing on main checks a node window's wires on values.*
    - `wiring.check` / `check_drawn` compare a reader row's committed string with the writer row its wire names:
      consistency along the wires, never whether a wire names the right writer.
    - `window_circuit`'s tests check its own `Lowered` class circuit.
    - `IF.Window.record` + `mismatches()` evaluate each unit's Definition (`evaluate_batch`, `"reference"`) on recorded
      rows, wired inputs taken from the writers' recorded outputs, and take an `IF.Circuit` as it is. But they need
      every unit's rows, and nothing on main produces a node window's rows: `window_rows.Rows` gives rows to
      `window_circuit`'s `Lowered`. So no existing check runs on it as it is, and none was run.
  - *The smallest such check:*
    1. Evaluate the Program once on random weights and tokens, node by node, as `window_rows.Slot` does
       (`numerics.rows.ROWS` kernels, else `evaluate_batch`).
    2. Give each writer row its words from its node's value through its `Piece`.
    3. Evaluate only the units of my families, and the gathers feeding them, with `evaluate_batch(kernel="reference")`
       on the rows their wires name.
    4. Compare each unit's outputs with the node words its `Piece` names.

    It catches the wrong-but-valid row above, except a permutation the Definition is invariant to (which is then
    harmless).
  - *Size* [local, the test's `glm_request(32, 1)` model]:
    - the weights record is 343,569,992 words dense (0.69 GB as u16) and 937,471,240 with a MoE layer (1.87 GB);
    - 41 / 59 nodes and 163,494 / 158,470 window units;
    - my families' units, dense: AllReduceMnnvl 1,056, AllReduceRmsNormMnnvl 196, RoPEHeadGptJ 16, SiluMul 2,
      Gemm 38,720 (the LM head), AllGather 16, gathers 932.
  - *Time* [local probe, 64 and 256 units, one core, reference kernel]:

    | template | per unit |
    |---|---|
    | `Gemm_v2{K=6144,N=1}` | 35 ms |
    | `AllReduceResidualMnnvl{N=192}` | 33 ms |
    | `SiluMul_v2{I=1536}` | 25 ms |
    | RoPE | 1.6 ms |
    | a gather | 0.14 ms |

    Everything but the LM head is about 1 min; the LM head's 38,720 units are about 23 min on one core, or 36 s for
    1,024 sampled. The Program's own evaluation is not measured.
  - *Memory* [inferred]: about 4 GB dense, about 8 GB with MoE. A remote run under 16 GB.
  - *Covered first:* the dense 1-layer Program covers 6 of my 8 families: AllReduceMnnvl, AllReduceRmsNormMnnvl, RoPE,
    SiluMul, the LM head and AllGather. MoeSum and ResidualAdd need a MoE layer.
