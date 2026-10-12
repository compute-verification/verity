import Proofs.Pouw.PearlC.TileValues
import Proofs.Pouw.PearlC.Tile4Circuit
import Definitions.Pouw.PearlC.DeviceChainOnly

/-!
# The tile units compute the protocol's words (proofs)

`tile_devAt` and `tile4_devFp4` read a tile unit's outputs as a device record's ticket and useful value on the operands
its inputs hold. Here those operands are the unit's own, so the outputs are the protocol's words:

* at Pearl-C's `pearlCProtocolDev (devAt p 4 pr) sem ρ`, `pearl-c-sm120` v1's among them, on row `i` of `A′` and row `j`
  of `B̃` from `PearlCSem.unitChain` and the bases and noise rows `PearlCSem.unitUsefulDev` reads (`tile_protocol`).
  The rev1 protocols and their kernel-`W_ref` and chain-only readings have its ticket and useful value
  (`pearlCProtocolDevRev1K_words`, `pearlCProtocolDevRev1KChainOnly_words`);
* at Pearl-C4's `pearlCProtocolFp4 devFp4Sm120 sem ρ`, on the rows `Fp4Sem.unitForm` forms and the bases and `Ẽ` rows
  `Fp4Sem.unitUseful` reads (`tile4_protocol`). A formed row's length is the unit's depth wherever its factor rows
  exist (`peelRows4_length`), so the depth is not a hypothesis.

Where the commitment registers the outputs, entry `(i, j)` then meets `TileWordsGood`'s conjunct
(`tile_words_good`, `tile4_words_good`). What the tile unit takes as given is what L3's other parts supply: the row
units put the formed rows and the factor rows on its inputs, the factor rows exist (`peelRowsP`, `peelRows4`), and the
registered words are the outputs (`TileCheck`'s `reg`, outside `Collides`).
-/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP

variable {Q R S : Type} {TM TN : ℕ}

theorem atNat_fin {p q : ℕ} (M : Matrix (Fin p) (Fin q) (Option ℚ)) (i : Fin p) (j : Fin q) :
    atNat M i j = M i j := by
  simp [atNat]

/-! ## Pearl-C -/

section PearlC

variable (p : Pipeline)

/-- `tile_devAt` at a depth given as a term equal to `128·nw`. -/
theorem tile_devAt_of_eq (pr : Prices) {k nw m n : ℕ} (hnw : 0 < nw) (hk : k = 128 * nw)
    (X : (tileSpec p k TM TN).toCircuit.Transcript) (hX : (tileSpec p k TM TN).part.Correct X 0)
    (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code)
    (beA : Matrix (Fin m) (Fin 32) ℕ) (beB : Matrix (Fin n) (Fin 32) ℕ) (i : Fin m) (j : Fin n) {ti tj : ℕ}
    (hti : ti < TM) (htj : tj < TN) (hA : ∀ l : Fin k, val X (ti * k + l) = (Aq i l).val)
    (hB : ∀ l : Fin k, val X (TM * k + tj * k + l) = (Bq j l).val) {c : ℕ} {pa pb : List ℕ}
    (hr : PeelExactProofsP.peelRowsP p 4 Aq Bq FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * k + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * k + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (cGate k TM TN ti tj)) = (devAt p 4 pr).ticket Aq Bq i j ∧
      wordQ (val X (uGate k TM TN ti tj)) = (devAt p 4 pr).useful Aq Bq FA FB beA beB i j := by
  subst hk
  exact tile_devAt p pr hnw X hX Aq Bq FA FB beA beB i j hti htj hA hB hr hpa hpb

/-- **Pearl-C's tile unit computes the protocol's words.** At `pearlCProtocolDev (devAt p 4 pr) sem ρ`, unit `u` of
`U` (depth `k = 128·nw`) with activations `act`: a transcript whose tile unit is correct, holding at entry `(ti, tj)`'s
inputs row `i` of the unit's `A′` and row `j` of its `B̃` (`PearlCSem.unitChain`) and the factor rows `P_A,i`, `P_B,j`
that U's pinned order forms from them at the bases and noise rows of the unit's useful value (`FA FB beA beB`, any with
`hU`; the unit's own have it, `unitUsefulDev_eq`), has on that entry's outputs the protocol's ticket and useful value at
`(i, j)`. -/
theorem tile_protocol (pr : Prices) (sem : PearlCSem Q R S) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) {nw : ℕ} (hnw : 0 < nw) (hk : (U.layout.shape u).k = 128 * nw)
    (X : (tileSpec p (U.layout.shape u).k TM TN).toCircuit.Transcript)
    (hX : (tileSpec p (U.layout.shape u).k TM TN).part.Correct X 0)
    (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l : Fin (U.layout.shape u).k,
      val X (ti * (U.layout.shape u).k + l) = ((sem.unitChain H s U u act).1 i l).val)
    (hB : ∀ l : Fin (U.layout.shape u).k,
      val X (TM * (U.layout.shape u).k + tj * (U.layout.shape u).k + l) = ((sem.unitChain H s U u act).2 j l).val)
    (FA FB : Matrix (Fin (U.layout.shape u).k) (Fin 32) Code) (beA : Matrix (Fin (U.layout.shape u).m) (Fin 32) ℕ)
    (beB : Matrix (Fin (U.layout.shape u).n) (Fin 32) ℕ)
    (hU : sem.unitUsefulDev (devAt p 4 pr) H s U u act =
      (devAt p 4 pr).useful (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 FA FB beA beB)
    {c : ℕ} {pa pb : List ℕ}
    (hr : PeelExactProofsP.peelRowsP p 4 (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 FA FB beA beB
      i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * (U.layout.shape u).k + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * (U.layout.shape u).k + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (cGate (U.layout.shape u).k TM TN ti tj)) =
        (pearlCProtocolDev (devAt p 4 pr) sem ρ).ticket H s U u act i j ∧
      wordQ (val X (uGate (U.layout.shape u).k TM TN ti tj)) =
        (pearlCProtocolDev (devAt p 4 pr) sem ρ).useful H s U u act i j := by
  obtain ⟨h1, h2⟩ := tile_devAt_of_eq p pr hnw hk X hX _ _ FA FB beA beB i j hti htj hA hB hr hpa hpb
  refine ⟨h1.trans ?_, h2.trans ?_⟩
  · simp only [pearlCProtocolDev, atNat_fin]
  · simp only [pearlCProtocolDev, atNat_fin, hU]

/-- **A registered entry of Pearl-C's tile is good**: with the commitment's ticket and useful words at `(i, j)` the
tile unit's outputs, `tile_protocol`'s hypotheses give `TileWordsGood`'s conjunct at `(i, j)`. -/
theorem tile_words_good (pr : Prices) (sem : PearlCSem Q R S) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (cm : Commit) {nw : ℕ} (hnw : 0 < nw) (hk : (U.layout.shape u).k = 128 * nw)
    (X : (tileSpec p (U.layout.shape u).k TM TN).toCircuit.Transcript)
    (hX : (tileSpec p (U.layout.shape u).k TM TN).part.Correct X 0)
    (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l : Fin (U.layout.shape u).k,
      val X (ti * (U.layout.shape u).k + l) = ((sem.unitChain H s U u cm.act).1 i l).val)
    (hB : ∀ l : Fin (U.layout.shape u).k,
      val X (TM * (U.layout.shape u).k + tj * (U.layout.shape u).k + l) = ((sem.unitChain H s U u cm.act).2 j l).val)
    (FA FB : Matrix (Fin (U.layout.shape u).k) (Fin 32) Code) (beA : Matrix (Fin (U.layout.shape u).m) (Fin 32) ℕ)
    (beB : Matrix (Fin (U.layout.shape u).n) (Fin 32) ℕ)
    (hU : sem.unitUsefulDev (devAt p 4 pr) H s U u cm.act =
      (devAt p 4 pr).useful (sem.unitChain H s U u cm.act).1 (sem.unitChain H s U u cm.act).2 FA FB beA beB)
    {c : ℕ} {pa pb : List ℕ}
    (hr : PeelExactProofsP.peelRowsP p 4 (sem.unitChain H s U u cm.act).1 (sem.unitChain H s U u cm.act).2 FA FB beA
      beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * (U.layout.shape u).k + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * (U.layout.shape u).k + 64 * TM + 64 * tj + q) = pb.getD q 0)
    (hct : cm.ticket i j = val X (cGate (U.layout.shape u).k TM TN ti tj))
    (hcu : cm.useful i j = val X (uGate (U.layout.shape u).k TM TN ti tj)) :
    wordQ (cm.ticket i j) = (pearlCProtocolDev (devAt p 4 pr) sem ρ).ticket H s U u cm.act i j ∧
      wordQ (cm.useful i j) = (pearlCProtocolDev (devAt p 4 pr) sem ρ).useful H s U u cm.act i j := by
  rw [hct, hcu]
  exact tile_protocol p pr sem ρ H s U u cm.act hnw hk X hX i j hti htj hA hB FA FB beA beB hU hr hpa hpb

theorem pearlCProtocolDevRev1K_words (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    (pearlCProtocolDevRev1K d sem ρ c).ticket = (pearlCProtocolDev d sem ρ).ticket ∧
      (pearlCProtocolDevRev1K d sem ρ c).useful = (pearlCProtocolDev d sem ρ).useful :=
  ⟨rfl, rfl⟩

theorem pearlCProtocolDevRev1KChainOnly_words (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) :
    (pearlCProtocolDevRev1KChainOnly d sem ρ c).ticket = (pearlCProtocolDev d sem ρ).ticket ∧
      (pearlCProtocolDevRev1KChainOnly d sem ρ c).useful = (pearlCProtocolDev d sem ρ).useful :=
  ⟨rfl, rfl⟩

end PearlC

/-! ## Pearl-C4 -/

section PearlC4

/-- A formed row of a peel that succeeds has `k` codes, the bases' depth: its `A′·F_B` chain runs against `F_B`'s
columns, and a chain checks its operands' lengths. -/
theorem peelRows4_length {m n k : ℕ} {A : Op4 m} {B : Op4 n} {FA FB : Matrix (Fin k) (Fin 32) Nib}
    {beA : Matrix (Fin m) (Fin 32) ℕ} {beB : Matrix (Fin n) (Fin 32) ℕ} {i : Fin m} {j : Fin n} {c : ℕ}
    {pa pb : List ℕ} (hr : PeelFp4Proofs.peelRows4 sm120Nvf4 A B FA FB beA beB i j = some (c, pa, pb)) {x : Row4}
    (hAi : A i = some x) : x.1.length = k := by
  obtain ⟨a, -, af, -, -, hA, -, -, haf, -⟩ := PeelFp4Proofs.peelRows4_spec hr
  rw [hAi] at hA
  cases hA
  obtain ⟨-, hq⟩ := PeelExactProofs.mapM_ofFn _ 0 haf
  obtain ⟨w, hw, -⟩ := Option.bind_eq_some_iff.1 (hq 0)
  unfold cTilde Fp4Atom.chain at hw
  split_ifs at hw with hlen
  simpa [colF] using hlen.1.symm

/-- **Pearl-C4's tile unit computes the protocol's words.** At `pearlCProtocolFp4 devFp4Sm120 sem ρ`, unit `u` of `U`
with activations `act`: a transcript whose tile unit (depth `K`, the unit's) is correct, holding at entry `(ti, tj)`'s
inputs row `i` of the unit's `A′` and row `j` of its `B̃` (`Fp4Sem.unitForm`: codes and scale bytes) and the factor rows
`P_A,i`, `P_B,j` that U's pinned order forms from them at the bases and `Ẽ` rows of the unit's useful value
(`FA FB beA beB`, any with `hU`; the unit's own have it, `unitUseful_eq`), has on that entry's outputs the protocol's
ticket and useful value at `(i, j)`. -/
theorem tile4_protocol (sem : Fp4Sem Q R S) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes)
    {K : ℕ} (hk : (U.layout.shape u).k = K) (hK : 0 < K) (X : (tile4Spec K TM TN).toCircuit.Transcript)
    (hX : (tile4Spec K TM TN).part.Correct X 0) (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n)
    {x y : Row4} (hAi : (sem.unitForm devFp4Sm120 H s U u act).1 i = some x)
    (hBj : (sem.unitForm devFp4Sm120 H s U u act).2 j = some y) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l < K, val X (ti * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val X (TM * K + ti * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val X (TM * row4 K + tj * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val X (TM * row4 K + TN * K + tj * (K / 16) + l) = (y.2.getD l 0).val)
    (FA FB : Matrix (Fin (U.layout.shape u).k) (Fin 32) Nib) (beA : Matrix (Fin (U.layout.shape u).m) (Fin 32) ℕ)
    (beB : Matrix (Fin (U.layout.shape u).n) (Fin 32) ℕ)
    (hU : sem.unitUseful devFp4Sm120 H s U u act =
      devFp4Sm120.useful (sem.unitForm devFp4Sm120 H s U u act).1 (sem.unitForm devFp4Sm120 H s U u act).2 FA FB beA
        beB)
    {c : ℕ} {pa pb : List ℕ}
    (hr : PeelFp4Proofs.peelRows4 sm120Nvf4 (sem.unitForm devFp4Sm120 H s U u act).1
      (sem.unitForm devFp4Sm120 H s U u act).2 FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * TM + 64 * tj + q) = pb.getD q 0) :
    wordQ (val X (c4Gate K TM TN ti tj)) = (pearlCProtocolFp4 devFp4Sm120 sem ρ).ticket H s U u act i j ∧
      wordQ (val X (u4Gate K TM TN ti tj)) = (pearlCProtocolFp4 devFp4Sm120 sem ρ).useful H s U u act i j := by
  have hx : x.1.length = K := hk ▸ peelRows4_length hr hAi
  obtain ⟨h1, h2⟩ := tile4_devFp4 X hX _ _ FA FB beA beB i j hAi hBj hti htj hK hx hA hSA hB hSB hr hpa hpb
  refine ⟨h1.trans ?_, h2.trans ?_⟩
  · simp only [pearlCProtocolFp4, atNat_fin]
  · simp only [pearlCProtocolFp4, atNat_fin, hU]

/-- **A registered entry of Pearl-C4's tile is good**: with the commitment's ticket and useful words at `(i, j)` the
tile unit's outputs, `tile4_protocol`'s hypotheses give `TileWordsGood`'s conjunct at `(i, j)`. -/
theorem tile4_words_good (sem : Fp4Sem Q R S) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (cm : Commit)
    {K : ℕ} (hk : (U.layout.shape u).k = K) (hK : 0 < K) (X : (tile4Spec K TM TN).toCircuit.Transcript)
    (hX : (tile4Spec K TM TN).part.Correct X 0) (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n)
    {x y : Row4} (hAi : (sem.unitForm devFp4Sm120 H s U u cm.act).1 i = some x)
    (hBj : (sem.unitForm devFp4Sm120 H s U u cm.act).2 j = some y) {ti tj : ℕ} (hti : ti < TM) (htj : tj < TN)
    (hA : ∀ l < K, val X (ti * K + l) = (x.1.getD l 0).val)
    (hSA : ∀ l < K / 16, val X (TM * K + ti * (K / 16) + l) = (x.2.getD l 0).val)
    (hB : ∀ l < K, val X (TM * row4 K + tj * K + l) = (y.1.getD l 0).val)
    (hSB : ∀ l < K / 16, val X (TM * row4 K + TN * K + tj * (K / 16) + l) = (y.2.getD l 0).val)
    (FA FB : Matrix (Fin (U.layout.shape u).k) (Fin 32) Nib) (beA : Matrix (Fin (U.layout.shape u).m) (Fin 32) ℕ)
    (beB : Matrix (Fin (U.layout.shape u).n) (Fin 32) ℕ)
    (hU : sem.unitUseful devFp4Sm120 H s U u cm.act =
      devFp4Sm120.useful (sem.unitForm devFp4Sm120 H s U u cm.act).1 (sem.unitForm devFp4Sm120 H s U u cm.act).2 FA
        FB beA beB)
    {c : ℕ} {pa pb : List ℕ}
    (hr : PeelFp4Proofs.peelRows4 sm120Nvf4 (sem.unitForm devFp4Sm120 H s U u cm.act).1
      (sem.unitForm devFp4Sm120 H s U u cm.act).2 FA FB beA beB i j = some (c, pa, pb))
    (hpa : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * ti + q) = pa.getD q 0)
    (hpb : ∀ q < 64, val X ((TM + TN) * row4 K + 64 * TM + 64 * tj + q) = pb.getD q 0)
    (hct : cm.ticket i j = val X (c4Gate K TM TN ti tj)) (hcu : cm.useful i j = val X (u4Gate K TM TN ti tj)) :
    wordQ (cm.ticket i j) = (pearlCProtocolFp4 devFp4Sm120 sem ρ).ticket H s U u cm.act i j ∧
      wordQ (cm.useful i j) = (pearlCProtocolFp4 devFp4Sm120 sem ρ).useful H s U u cm.act i j := by
  rw [hct, hcu]
  exact tile4_protocol sem ρ H s U u cm.act hk hK X hX i j hAi hBj hti htj hA hSA hB hSB FA FB beA beB hU hr hpa hpb

end PearlC4

/-! ## The unit's own bases and noise rows -/

/-- `tile_protocol`'s `hU` at the unit's own bases and noise rows. -/
theorem unitUsefulDev_eq (p : Pipeline) (pr : Prices) (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (act : Codes) :
    let sh := U.layout.shape u
    let z := sem.noise H s U u act
    sem.unitUsefulDev (devAt p 4 pr) H s U u act =
      (devAt p 4 pr).useful (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2
        (fun l q => z.FA l q) (fun l q => z.FB l q)
        (fun i q => codeBf16 (lineCodes (sem.beta sh.k (wordsVal sh.m sh.k act i)) (z.EA i) q))
        (fun j q => codeBf16 (lineCodes (sem.beta sh.k (wordsVal sh.n sh.k (U.weight (U.layout.wid u)) j))
          (z.EB j) q)) :=
  rfl

/-- `tile4_protocol`'s `hU` at the unit's own bases and `Ẽ` rows. -/
theorem unitUseful_eq (sem : Fp4Sem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) :
    let sh := U.layout.shape u
    let z := sem.lines H s U u act
    sem.unitUseful devFp4Sm120 H s U u act =
      devFp4Sm120.useful (sem.unitForm devFp4Sm120 H s U u act).1 (sem.unitForm devFp4Sm120 H s U u act).2 z.FA z.FB
        (fun i => eTilde (z.EA i) (e4m3 (sem.beta sh.k (wordsVal sh.m sh.k act i))))
        (fun j => eTilde (z.EB j) (e4m3 (sem.beta sh.k (wordsVal sh.n sh.k (U.weight (U.layout.wid u)) j)))) :=
  rfl

end Pouw.PearlC.TileCircuit
