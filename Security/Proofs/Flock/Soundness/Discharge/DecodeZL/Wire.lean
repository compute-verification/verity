import Proofs.Flock.Soundness.Discharge.Exec.FramesRep
import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Proofs.Flock.Soundness.Discharge.Exec.Decode

/-!
# The per-Kind decoder (R11b)

`DecodesAll` asks for a decoder `dec : ℕ → ByteArray → List Msg` from a round's retained bytes to the messages the
round carries. A round retains the encodings of its chunk's frames (`Exec.RepRounds`), and frames are recoverable from
their bytes when each one's length fits its header (`encs_inj`). So the decoder recovers the frames, then reads each
message the round carries off them by its `Wire`: a number of field values (an `.one` frame is one value, a `.vec` frame
its values), or one `.bytes` frame.

* `Wire`, `readAll`, `framesOf`, `decRound`, `decOf`: the decoder. `decRound_encs`: on a chunk's bytes it reads the
  chunk's frames.
* `coinPat`, `groupPat`, `rdChunks`, `Carries`: a phase's frames carry its messages, in the groups its spine cuts them
  into at its coins, then a tail the proof file carries. `Carries.append` glues two phases.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

/-- **How a message is read off its round's frames**: from the round's next `k` field values, or from its next `.bytes`
frame. -/
inductive Wire : Type 1
  | vals (M : Type) (k : ℕ) (rd : List F128 → M)
  | bytes (M : Type) (rd : ByteArray → M)

namespace Wire

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- A pair of field elements, from two values. -/
def ff : Wire := .vals (𝔽 × 𝔽) 2 fun vs => (fOf vs[0]!, fOf vs[1]!)

/-- A field element, from one value. -/
def f : Wire := .vals 𝔽 1 fun vs => fOf vs[0]!

/-- A vector of `n` field elements, from `n` values. -/
def v (n : ℕ) : Wire := .vals (Fin n → 𝔽) n fun vs i => fOf vs[i.val]!

/-- A pair of vectors of `n` field elements, from `2n` values. -/
def vv (n : ℕ) : Wire := .vals ((Fin n → 𝔽) × (Fin n → 𝔽)) (2 * n) fun vs =>
  (fun i => fOf vs[i.val]!, fun i => fOf vs[n + i.val]!)

/-- A pair of extension-field elements, from the four values of `kFr`. -/
def kk : Wire := .vals (𝕂 × 𝕂) 4 fun vs => (gOf ⟨vs[0]!, vs[1]!⟩, gOf ⟨vs[2]!, vs[3]!⟩)

/-- A cap, from one byte string: the 64-byte entries it concatenates (`capOfRoot`). -/
noncomputable def cap : Wire := .bytes (Cap Dig) capOfRoot

end Wire

/-- **The zerocheck's and lincheck's wires**, one list per round: the statement binding's round and the second
zerocheck coin's carry nothing, the third round `(round1_ab, round1_c)`, each residual round its pair, the round of the
lincheck's first coin `(final_a, final_b)`, its second coin's nothing, each lincheck round its pair, the last
`z_partial`. -/
def wZL (m kLog : ℕ) : List (List Wire) :=
  [[], [], [.vv 64]] ++ List.replicate (m - 6) [.ff] ++ [[.ff], []] ++ List.replicate (kLog - 6) [.ff] ++ [[.v 64]]

/-- A frame's field values: one for `.one`, its values for `.vec`. -/
def Frame.vals : Frame → List F128
  | .dom _ => [] | .label _ => [] | .one v => [v] | .vec vs => vs | .bytes _ => [] | .sq _ _ => []

/-- A frame's byte string, for `.bytes`. -/
def Frame.byteStr : Frame → List ByteArray
  | .dom _ => [] | .label _ => [] | .one _ => [] | .vec _ => [] | .bytes b => [b] | .sq _ _ => []

/-- The field values of a list of frames, in order. -/
def valsOf (fs : List Frame) : List F128 := fs.flatMap Frame.vals

/-- The `.bytes` frames' strings of a list of frames, in order. -/
def bytesOf (fs : List Frame) : List ByteArray := fs.flatMap Frame.byteStr

/-- **Read messages by their wires**: each `.vals` wire takes the next `k` values, each `.bytes` wire the next string. -/
def readAll : List Wire → List F128 → List ByteArray → List Msg
  | [], _, _ => []
  | .vals M k rd :: ws, vs, bs => ⟨M, rd (vs.take k)⟩ :: readAll ws (vs.drop k) bs
  | .bytes M rd :: ws, vs, bs => ⟨M, rd (bs.headD ByteArray.empty)⟩ :: readAll ws vs bs.tail

open Classical in
/-- **The frames a round's bytes encode**: the well-formed frames whose bytes they are (unique, `encs_inj`), or none. -/
noncomputable def framesOf (B : ByteArray) : List Frame :=
  if h : ∃ fs : List Frame, (∀ f ∈ fs, f.WF) ∧ encs fs = B.data.toList then Classical.choose h else []

/-- **A round's messages**: its frames, read by its wires. -/
noncomputable def decRound (w : List Wire) (B : ByteArray) : List Msg :=
  readAll w (valsOf (framesOf B)) (bytesOf (framesOf B))

/-- **The decoder of a wire table**: round `j` read by the table's `j`-th wires (none past its end). -/
noncomputable def decOf (W : List (List Wire)) (j : ℕ) (B : ByteArray) : List Msg := decRound (W.getD j []) B

theorem framesOf_encs {c : List Frame} (hc : ∀ f ∈ c, f.WF) : framesOf ⟨(encs c).toArray⟩ = c := by
  have h : ∃ fs : List Frame, (∀ f ∈ fs, f.WF) ∧ encs fs = (ByteArray.mk (encs c).toArray).data.toList :=
    ⟨c, hc, by simp⟩
  unfold framesOf
  rw [dite_eq_left_of_eq_true (eq_true h)]
  exact encs_inj (Classical.choose_spec h).1 hc ((Classical.choose_spec h).2.trans (by simp))

/-- **On a chunk's bytes the decoder reads the chunk's frames**, when they are well formed. -/
theorem decRound_encs {c : List Frame} (hc : ∀ f ∈ c, f.WF) (w : List Wire) :
    decRound w ⟨(encs c).toArray⟩ = readAll w (valsOf c) (bytesOf c) := by
  unfold decRound; rw [framesOf_encs hc]

@[simp] theorem decRound_nil (B : ByteArray) : decRound [] B = [] := by
  simp [decRound, readAll]

/-- Whether a spine step is a coin. -/
def isCoin : Exec.Kind → Bool
  | .msg _ => false
  | .coin _ => true

/-- **A spine's coin pattern**: `true` at each coin, `false` at each message. -/
def coinPat (σ : List Exec.Kind) : List Bool := σ.map isCoin

/-- **The pattern of message groups**: groups of `ns` messages, each closed by a coin, then `t` trailing messages. -/
def groupPat (ns : List ℕ) (t : ℕ) : List Bool :=
  ns.flatMap (fun k => List.replicate k false ++ [true]) ++ List.replicate t false

/-- **The messages a phase's rounds carry**: chunk `j` of the phase's frames, as its bytes, read by wires `j`. -/
noncomputable def rdChunks (W : List (List Wire)) (F : List Frame) : List (List Msg) :=
  List.zipWith (fun w c => decRound w ⟨(encs c).toArray⟩) W (chunks F).1

/-- **A phase's frames carry its messages** (the chunk lemma each phase proves): its frames end at a closing squeeze,
it has one wire list per chunk, and its messages `ms` are its rounds' decoded messages, then `tail`, in the groups its
spine `σ` cuts them into at its coins. -/
def Carries (W : List (List Wire)) (σ : List Exec.Kind) (F : List Frame) (ms tail : List Msg) : Prop :=
  (chunks F).2 = [] ∧ W.length = (chunks F).1.length ∧ ms = (rdChunks W F).flatten ++ tail ∧
    coinPat σ = groupPat ((rdChunks W F).map List.length) tail.length

theorem chunks_append_of_closed {F₁ : List Frame} (h : (chunks F₁).2 = []) (F₂ : List Frame) :
    chunks (F₁ ++ F₂) = ((chunks F₁).1 ++ (chunks F₂).1, (chunks F₂).2) := by
  rw [chunks_append]
  unfold glue
  rcases e : (chunks F₂).1 with _ | ⟨c, cs⟩
  · simp [h]
  · simp [h]

theorem rdChunks_append {W₁ W₂ : List (List Wire)} {F₁ : List Frame} (h : (chunks F₁).2 = [])
    (hl : W₁.length = (chunks F₁).1.length) (F₂ : List Frame) :
    rdChunks (W₁ ++ W₂) (F₁ ++ F₂) = rdChunks W₁ F₁ ++ rdChunks W₂ F₂ := by
  unfold rdChunks
  rw [chunks_append_of_closed h, List.zipWith_append hl]

/-- **Two phases glue**: when the first's messages have no tail. -/
theorem Carries.append {W₁ W₂ : List (List Wire)} {σ₁ σ₂ : List Exec.Kind} {F₁ F₂ : List Frame} {ms₁ ms₂ t : List Msg}
    (h₁ : Carries W₁ σ₁ F₁ ms₁ []) (h₂ : Carries W₂ σ₂ F₂ ms₂ t) :
    Carries (W₁ ++ W₂) (σ₁ ++ σ₂) (F₁ ++ F₂) (ms₁ ++ ms₂) t := by
  obtain ⟨p₁, l₁, e₁, c₁⟩ := h₁
  obtain ⟨p₂, l₂, e₂, c₂⟩ := h₂
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [chunks_append_of_closed p₁]; exact p₂
  · rw [chunks_append_of_closed p₁, List.length_append, List.length_append, l₁, l₂]
  · rw [rdChunks_append p₁ l₁, e₁, e₂]; simp
  · rw [rdChunks_append p₁ l₁, coinPat, List.map_append, ← coinPat, ← coinPat, c₁, c₂]
    simp [groupPat]

end FlockSoundness.Discharge.DecodeZL
