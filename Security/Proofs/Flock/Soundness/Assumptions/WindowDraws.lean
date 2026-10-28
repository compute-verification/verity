import Definitions.Flock.Audit.Window

/-!
# The deployed sampling window's draw files

Assumption (`flock-window-draws-pinned`): the draw files of one deployed sampling window, named by their SHA-512 digests,
read as the run recorded them. It is a parse premise: Lean doesn't read the files' bytes, so what they read as is
assumed, and hashing and parsing them checks it. `PinnedWindowDraws` states it, and the window decomposition's honest
lemma (`Flock.Guarantees.WindowBound_honest`, `Proofs/Flock/EndToEnd/WindowDraw`) takes it.
-/

namespace FlockSoundness.Assumptions

open FlockSoundness.Audit.Window

/-- **The draw files of the deployed sampling window read as recorded** (`flock-window-draws-pinned`): `files d` is the
draw file whose bytes have SHA-512 digest `d` (lowercase hex), as `json.loads` reads it.

* The window's draw file (`rank0/draw.json`) is the window's one draw: `stratified`, population 3744, one unit from each
  of three strata, the `SparseMlaSplitPart_v1` units at `T=64` (`[0, 2496)`), at `T=16` (`[2496, 3120)`) and the
  `SparseMlaCombine_v1` units (`[3120, 3744)`), drawing the units 421, 2824 and 3270.
* The three sessions' draw files (`session-m1/draw.json`, `session-m10/draw.json`, `session-m12/draw.json`) are `subset`
  draws of one unit: local unit 165 of 256, 328 of 512 and 150 of 624.

Rationale: the digests name the bytes the run recorded, and reading them (`sha512sum`, then `json.loads`) checks what
they read as. Nothing else is assumed: that each session's file is its member's share of the window's draw is
`Flock.Guarantees.WindowBound_honest`'s to prove, from the members' layout. -/
def PinnedWindowDraws (files : String → Option DrawFile) : Prop :=
  files "ca5031c6ca161ee6e3d542e280ebf3a5ec0eacb4af56db5061d4bd04d41102949c53d5cab3d9a2f200d77a851ecabcbc14205adcfdb7e1119abae3c87eedf220" =
      some ⟨"stratified", 3744, 1,
        [⟨"SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"},INV={\"fn\":\"Fa3InvSum_v1\"},SCALE={\"f64\":\"0x1.0000000000000p-4\"}}",
            [(0, 2496)], 1⟩,
          ⟨"SparseMlaSplitPart_v1{T=16,DR=64,DV=512,BN=64,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"},INV={\"fn\":\"Fa3InvSum_v1\"},SCALE={\"f64\":\"0x1.0000000000000p-4\"}}",
            [(2496, 3120)], 1⟩,
          ⟨"SparseMlaCombine_v1{NSPLIT=5,DV=512,INV={\"fn\":\"Fa3InvSum_v1\"}}", [(3120, 3744)], 1⟩],
        [421, 2824, 3270]⟩ ∧
    files "20495de9fbad861572ade8c086ffbc8d5cd52c5659a805a3e46104c267f9233e6b516da7633dbda18c1565b7efd7ecdced5209d0ee7b0c63fd9a2156bcc7de95" =
      some ⟨"subset", 256, 1, [], [165]⟩ ∧
    files "d047a561298f60ea07a72e7f7915650db0661441aa5bcd5d8967f0989f3e4ce7e4f018eb066cb1f663319ee6e3440180e33b1d80ac7f7e18290626c72eb18759" =
      some ⟨"subset", 512, 1, [], [328]⟩ ∧
    files "bb1690fe0ceed558c068e1d2ced851113d4a46a8d9067760fde29bdbc2abcbd9b9aa110142b079fc8afa59f2070f7caba3eaf3ffcf3831992a054a2c06a37763" =
      some ⟨"subset", 624, 1, [], [150]⟩

end FlockSoundness.Assumptions
