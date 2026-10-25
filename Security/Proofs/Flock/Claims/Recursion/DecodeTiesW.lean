import Proofs.Flock.Claims.Recursion.DecodeTiesVals
import Proofs.Flock.C3.Reps.SentKeys
import Proofs.Flock.VBridge.Family.Runs
import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements

/-!
# A w that holds the rounds' values decodes to the proof

`RunsFrom` asks that the proof a rep's frames absorb be the decoded one: its zerocheck (`ZcProofOf`, but `final_c`,
which no frame absorbs), lincheck (`LcProofOf`), ring switches, Ligerito's messages, out-of-domain values and final
`yr` are `decoded sh w comb cons`'s. They are when w holds the rounds' values (`WHolds`): below the rounds' elements,
w's entry `i` is the proof's value at the `i`-th (`keyVal`), which is what the frames absorb (`vals_repFrames_rounds`).

* **One element** (`valOf_keyVal`): `valOf` reads an element of w's names from its first slot (`valOf_w`); that slot
  is below the rounds' elements unless the element is a running sum, so its value is the element's `keyVal`.
* **The fields** (`zcProofOf_of`, `lcProofOf_of`, `rs_of`, `msgs_of`, `ood_of`, `yr_of`): each field of `symbolic` is
  its elements' variables, at the proof's sizes (`ProofSizes`); each element is one of w's names (`ab_mem`, …,
  `ligRounds_has`, the latter at two levels or more).
* **w from the frames** (`chunks_vals`, `wHolds_of_frames`): a rep's chunks absorb its frames' values, none pending,
  so a w that starts with its rounds' values, in order, holds them.
-/

namespace FlockDecodeTies

open Flock
open Flock.RecAlgebra (Key rounds wNames wi accNames decoded valOf symbolic ZcProofOf LcProofOf)
open FlockSoundness.Discharge.DecodeZL (ProofSizes)
open FlockSoundness.Refine (Frame)
open FlockSoundness.Discharge.Exec (chunks chunks_flatten repFrames repFrames_chunks_snd)
open FlockDeploy (vals)

/-! ## w at the rounds' values -/

/-- **w holds the rounds' values**: below the rounds' elements, w's entry `i` is the decoded proof's value at the
rounds' `i`-th element (`keyVal`). -/
def WHolds (sh : Shape) (w : Array F128) (p : Proof) (T : F128) : Prop :=
  ∀ i (hi : i < (rounds sh).flatten.size), w[i]! = keyVal p T (rounds sh).flatten[i]

/-- **A rounds' element reads its value**: at a w that holds the rounds' values, `valOf` at one of w's names that is
not a running sum (nor a transpose, a comb element or a consistency sum, which w doesn't hold) is its `keyVal`. -/
theorem valOf_keyVal {sh : Shape} {w comb : Array F128} {cons : ℕ → ℕ → F128} {p : Proof} {T : F128}
    (hw : WHolds sh w p T) {x : Key} (hx : x ∈ wNames sh) (hacc : x ∉ accNames)
    (h₁ : ∀ k u, x ≠ .rsT k u) (h₂ : ∀ l, x ≠ .comb l) (h₃ : ∀ ℓ l, x ≠ .cons ℓ l) :
    valOf sh w comb cons x = keyVal p T x := by
  obtain ⟨j, hj⟩ : ∃ j, (wNames sh).finIdxOf? x = some j :=
    Option.isSome_iff_exists.1 (by simp [Array.isSome_finIdxOf?, hx])
  have hi : wi sh x = some j.val := by
    simp only [wi, Array.idxOf?, hj, Option.map_some]
  rw [FlockC3.Reps.valOf_w hi h₁ h₂ h₃]
  obtain ⟨hj1, -⟩ := Array.finIdxOf?_eq_some_iff.1 hj
  have hj2 : ((rounds sh).flatten ++ accNames)[j.val]'(by simpa [wNames] using j.isLt) = x := by
    rw [← hj1]; rfl
  by_cases hl : j.val < (rounds sh).flatten.size
  · rw [hw j hl, ← hj2, Array.getElem_append_left hl]
  · refine absurd ?_ hacc
    rw [← hj2, Array.getElem_append_right (by omega)]
    exact Array.getElem_mem _

/-- An array of `n` entries is its entries, read by `f`, in order. -/
theorem eq_map_range {α : Type} [Inhabited α] {a : Array α} {n : ℕ} {f : ℕ → α} (hs : a.size = n)
    (h : ∀ i < n, f i = a[i]!) : a = (Array.range n).map f := by
  apply Array.ext (by simp [hs])
  intro i h1 h2
  simp only [Array.getElem_map, Array.getElem_range]
  rw [h i (by omega), getElem!_pos a i h1]

theorem rsEach_get {sh : Shape} {p : Proof} (hp : ProofSizes sh p) {k : ℕ} (hk : k < sh.claims) :
    (p.ringSwitches[k]!).size = 128 := by
  rw [getElem!_pos p.ringSwitches k (by rw [hp.rs]; exact hk)]
  exact hp.rsEach _ (Array.getElem_mem _)

theorem yr_size {sh : Shape} {p : Proof} (hp : ProofSizes sh p) :
    p.ligerito.finalProof.yr.size = 2 <<< sh.sched.ext := by
  rw [hp.yr, Nat.shiftLeft_eq, Nat.pow_succ, Nat.mul_comm]

/-! ## The decoded proof's fields -/

section fields

variable {sh : Shape} {w comb : Array F128} {cons : ℕ → ℕ → F128} {p : Proof} {T : F128}

/-- **The zerocheck the frames absorb is the decoded one's** (`ZcProofOf`), with `final_c`, which no frame absorbs,
as given. -/
theorem zcProofOf_of (hw : WHolds sh w p T) (hp : ProofSizes sh p) {vc : F128} (hfc : p.zerocheck.finalC = vc) :
    ZcProofOf (decoded sh w comb cons) vc p.zerocheck where
  ab := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var]
    exact eq_map_range hp.r1ab fun i hi =>
      valOf_keyVal hw (FlockC3.Reps.ab_mem sh hi) (by simp [accNames]) (by simp) (by simp) (by simp)
  c := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var]
    exact eq_map_range hp.r1c fun i hi =>
      valOf_keyVal hw (FlockC3.Reps.c_mem sh hi) (by simp [accNames]) (by simp) (by simp) (by simp)
  rounds := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]
    refine eq_map_range hp.zcRounds fun i hi => ?_
    rw [valOf_keyVal hw (FlockC3.Reps.g1_mem hi) (by simp [accNames]) (by simp) (by simp) (by simp),
      valOf_keyVal hw (FlockC3.Reps.ginf_mem hi) (by simp [accNames]) (by simp) (by simp) (by simp)]
    rfl
  fa := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, RecAlgebra.Lin.eval_var]
    exact (valOf_keyVal hw (FlockC3.Reps.fa_mem sh) (by simp [accNames]) (by simp) (by simp) (by simp)).symm
  fb := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, RecAlgebra.Lin.eval_var]
    exact (valOf_keyVal hw (FlockC3.Reps.fb_mem sh) (by simp [accNames]) (by simp) (by simp) (by simp)).symm
  fc := hfc

/-- **The lincheck the frames absorb is the decoded one's** (`LcProofOf`). -/
theorem lcProofOf_of (hw : WHolds sh w p T) (hp : ProofSizes sh p) : LcProofOf (decoded sh w comb cons) p.lincheck where
  rounds := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]
    refine eq_map_range hp.lcRounds fun j hj => ?_
    rw [valOf_keyVal hw (FlockC3.Reps.q1_mem hj) (by simp [accNames]) (by simp) (by simp) (by simp),
      valOf_keyVal hw (FlockC3.Reps.qinf_mem hj) (by simp [accNames]) (by simp) (by simp) (by simp)]
    rfl
  zp := by
    simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
      RecAlgebra.Lin.eval_var]
    exact eq_map_range hp.zPartial fun l hl =>
      valOf_keyVal hw (FlockC3.Reps.zp_mem sh hl) (by simp [accNames]) (by simp) (by simp) (by simp)

/-- **The ring switches the frames absorb are the decoded ones.** -/
theorem rs_of (hw : WHolds sh w p T) (hp : ProofSizes sh p) : p.ringSwitches = (decoded sh w comb cons).rs := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var]
  refine eq_map_range hp.rs fun k hk => (eq_map_range (rsEach_get hp hk) fun u hu => ?_).symm
  exact valOf_keyVal hw (FlockC3.Reps.rs_mem hk hu) (by simp [accNames]) (by simp) (by simp) (by simp)

/-- **Ligerito's sumcheck messages the frames absorb are the decoded ones**, at a schedule of `n + 2` levels. -/
theorem msgs_of (hw : WHolds sh w p T) (hp : ProofSizes sh p) {n : ℕ} (hR : sh.sched.levels.size - 1 = n + 1) :
    p.ligerito.messages = (decoded sh w comb cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2) := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var, RecAlgebra.KV.map]
  refine eq_map_range hp.messages fun i hi => ?_
  have e : ∀ a < 2, ∀ b < 2, valOf sh w comb cons (.msg i a b) = keyVal p T (.msg i a b) := fun a ha b hb =>
    valOf_keyVal hw (FlockC3.Reps.mem_wNames_of_lig
      (((FlockC3.Reps.ligRounds_has sh.sched).2.2 n hR).1 i (by omega) a ha b hb))
      (by simp [accNames]) (by simp) (by simp) (by simp)
  rw [e 0 (by omega) 0 (by omega), e 0 (by omega) 1 (by omega), e 1 (by omega) 0 (by omega),
    e 1 (by omega) 1 (by omega)]
  rfl

/-- **Ligerito's out-of-domain values the frames absorb are the decoded ones**, at a schedule of `n + 2` levels. -/
theorem ood_of (hw : WHolds sh w p T) (hp : ProofSizes sh p) {n : ℕ} (hR : sh.sched.levels.size - 1 = n + 1) :
    p.ligerito.oodValues = (decoded sh w comb cons).ood := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var]
  exact eq_map_range hp.oodValues fun i hi =>
    valOf_keyVal hw (FlockC3.Reps.mem_wNames_of_lig (((FlockC3.Reps.ligRounds_has sh.sched).2.2 n hR).2.1 i
      (by omega))) (by simp [accNames]) (by simp) (by simp) (by simp)

/-- **The final `yr` the frames absorb is the decoded one**, at a schedule of `n + 2` levels. -/
theorem yr_of (hw : WHolds sh w p T) (hp : ProofSizes sh p) {n : ℕ} (hR : sh.sched.levels.size - 1 = n + 1) :
    p.ligerito.finalProof.yr = (decoded sh w comb cons).yr := by
  simp only [decoded, RecAlgebra.Messages.map, symbolic, Array.map_map, Function.comp_def,
    RecAlgebra.Lin.eval_var]
  exact eq_map_range (yr_size hp) fun j hj =>
    valOf_keyVal hw (FlockC3.Reps.mem_wNames_of_lig (((FlockC3.Reps.ligRounds_has sh.sched).2.2 n hR).2.2 j hj))
      (by simp [accNames]) (by simp) (by simp) (by simp)

end fields

/-! ## w from the frames' values -/

theorem vals_flatten : ∀ L : List (List Frame), vals L.flatten = L.flatMap vals
  | [] => rfl
  | c :: L => by rw [List.flatten_cons, vals_append, vals_flatten L, List.flatMap_cons]

/-- **A rep's rounds absorb its frames' values**: its chunks' values, in order, are its frames', none pending. -/
theorem chunks_vals (st : Setup) (sch : Schedule) (dom : ByteArray) (n : ℕ) (p : Proof) (T : F128) :
    (chunks (repFrames st sch dom n p T)).1.flatMap vals = vals (repFrames st sch dom n p T) := by
  have h := chunks_flatten (repFrames st sch dom n p T)
  rw [repFrames_chunks_snd, List.append_nil] at h
  rw [← vals_flatten, h]

/-- A w that starts with the rounds' values holds them. -/
theorem wHolds_of_prefix {sh : Shape} {w : Array F128} {p : Proof} {T : F128}
    (h : (rounds sh).flatten.toList.map (keyVal p T) <+: w.toList) : WHolds sh w p T := by
  intro i hi
  obtain ⟨t, ht⟩ := h
  have hw : i < w.size := by
    have := congrArg List.length ht
    simp only [List.length_append, List.length_map, Array.length_toList] at this
    omega
  have e : w.toList[i]? = some (keyVal p T (rounds sh).flatten[i]) := by
    rw [← ht, List.getElem?_append_left (by rw [List.length_map, Array.length_toList]; exact hi),
      List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hi, Option.map_some]
  rw [getElem!_pos w i hw]
  simpa [hw] using e

/-- **A w that starts with a rep's frames' values holds the rounds' values**, at a plain shape of the setup's
dimensions and the decoded proof's sizes. -/
theorem wHolds_of_frames {sh : Shape} {st : Setup} {sch : Schedule} {n : ℕ} {dom : ByteArray} {p : Proof} {T : F128}
    {w : Array F128} (hm : st.m = sh.m) (hk : st.fold.kLog = sh.kLog) (hn : n = sh.claims) (hs : sch = sh.sched)
    (hh : sh.holo = false) (hp : ProofSizes sh p) (h : vals (repFrames st sch dom n p T) <+: w.toList) :
    WHolds sh w p T := by
  refine wHolds_of_prefix ?_
  rwa [← vals_repFrames_rounds dom p T hm hk hn hs hh hp.r1ab hp.r1c hp.zPartial
    (fun k hk => rsEach_get hp (by omega)) (by rw [hs]; exact yr_size hp)]

end FlockDecodeTies
