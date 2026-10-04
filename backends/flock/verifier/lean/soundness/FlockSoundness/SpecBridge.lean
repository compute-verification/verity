import FlockSoundness.Spec
import FlockSoundness.Discharge.FailClosed.One
import FlockSoundness.Assumptions.Zk

/-!
# The spec's definitions are the proofs'

Each closed-form definition of `FlockSoundness.Spec` is the proofs' definition it restates:

* `accepts_iff`: `FlockSpec.Accepts` is `FailClosed.FlockVerifyAccepts`, with or without `--zk` (`ofExec` carries the
  proofs' inputs over field for field), through `recordDraw_eq`, `tagsOf_zk` and `coinsOf_zk`.
* `prob_ofGame`, `expect_ofGame`, `prCoin_eq`: a proofs' game and strategy as the spec's (`ofGame`, `ofStrategy`) have
  the same probabilities and expectations.
* `sha512CRStrict_iff`, `sha512CRExpected_iff`, `uniformRandomBytes_iff`, `hmRowComputes_iff`, `hm96Hiding_iff`,
  `hashDerivedKey_iff`, `hashDerivedKeyHm96_iff`: each named assumption is the proofs' at the same arguments, the last
  through `keyedHash_eq` and `saltHash_eq` (`ZkView.My`, `ZkView.saltHash`).
-/

namespace FlockSoundness.SpecBridge

open Discharge Discharge.FailClosed

/-! ## What flock-verify accepts -/

/-- The proofs' inputs (`Exec.Inputs`) as the spec's, field for field. -/
def ofExec (I : Exec.Inputs) : FlockSpec.Inputs :=
  { tags := I.tags, circuitFile := I.circuitFile, publicFile := I.publicFile, coins := I.coins, count := I.count,
    tables := I.tables, partition := I.partition, program := I.program, work := I.work, stratified := I.stratified,
    closure := I.closure }

theorem recordDraw_eq (record : String) : FlockSpec.recordDraw record = Exec.recordDraw record := rfl

theorem tagsOf_zk (I : Exec.Inputs) : FlockSpec.tagsOf true (ofExec I) = ZkExec.zkTags I :=
  (ZkExec.zkTags_eq I).symm

theorem coinsOf_zk (I : Exec.Inputs) : FlockSpec.coinsOf true (ofExec I) = ZkExec.zkCoins I := rfl

/-- **`FlockSpec.Accepts` is what the proofs call flock-verify's acceptance** (`FlockVerifyAccepts`). -/
theorem accepts_iff (zk : Bool) (I : Exec.Inputs) (t : Exec.Transcript) :
    FlockSpec.Accepts zk (ofExec I) ⟨t.record, t.proofs⟩ ↔ FlockVerifyAccepts zk I t := by
  cases zk
  · exact And.comm
  · show _ ↔ ZkExec.AcceptsZK I t ∧ Checks (ZkExec.zkTags I) true I
    unfold FlockSpec.Accepts
    rw [tagsOf_zk, coinsOf_zk]
    exact And.comm

/-! ## Games -/

/-- A proofs' game as the spec's. -/
def ofGame {α : Type} : Game α → FlockSpec.Game α
  | .ret a => .ret a
  | .send M k => .send M fun m => ofGame (k m)
  | @Game.coin _ C iF iN k => @FlockSpec.Game.coin _ C iF iN fun c => ofGame (k c)

/-- A proofs' strategy as the spec's, at `ofGame`. -/
def ofStrategy {α : Type} : (g : Game α) → Strategy g → FlockSpec.Strategy (ofGame g)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, ofStrategy (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => ofStrategy (k c) (s c)

theorem prob_ofGame {α : Type} (p : α → Prop) :
    ∀ (g : Game α) (s : Strategy g), FlockSpec.prob p (ofGame g) (ofStrategy g s) = Game.prob p g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_ofGame p (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => Finset.sum_congr rfl fun c _ => congrArg _ (prob_ofGame p (k c) (s c))

theorem expect_ofGame {α : Type} (f : α → ℝ) :
    ∀ (g : Game α) (s : Strategy g), FlockSpec.expect f (ofGame g) (ofStrategy g s) = Game.expect f g s
  | .ret _, _ => rfl
  | .send _ k, s => expect_ofGame f (k s.1) s.2
  | @Game.coin _ _ _ _ k, s =>
    congrArg (· / _) (Finset.sum_congr rfl fun c _ => expect_ofGame f (k c) (s c))

theorem prCoin_eq {C : Type} [Fintype C] (p : C → Prop) : FlockSpec.prCoin p = Game.prCoin p := rfl

/-! ## Named assumptions -/

theorem sha512CRStrict_iff {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) :
    FlockSpec.SHA512CRStrict H (ofGame g) (ofStrategy g s) out cost q ↔ Assumptions.SHA512CRStrict H g s out cost q := by
  unfold FlockSpec.SHA512CRStrict Assumptions.SHA512CRStrict
  rw [prob_ofGame]

theorem sha512CRExpected_iff {D β : Type} (H : List UInt8 → D) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) :
    FlockSpec.SHA512CRExpected H (ofGame g) (ofStrategy g s) out cost ↔ Assumptions.SHA512CRExpected H g s out cost := by
  unfold FlockSpec.SHA512CRExpected Assumptions.SHA512CRExpected
  rw [prob_ofGame, expect_ofGame]

theorem uniformRandomBytes_iff {Ω : Type} [Fintype Ω] {m : ℕ} (L : Fin m → ℕ)
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    FlockSpec.UniformRandomBytes L t ↔ Assumptions.UniformRandomBytes L t := Iff.rfl

theorem hmRowComputes_iff {W : Type} (H : List UInt8 → List UInt8) (saltPrefix : List UInt8)
    (mask : List UInt8 → List UInt8) (sat : W → Prop) (row salt bc : W → List UInt8) :
    FlockSpec.HmRowComputes H saltPrefix mask sat row salt bc ↔
      Assumptions.HmRowComputes H saltPrefix mask sat row salt bc := Iff.rfl

theorem hm96Hiding_iff {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) (δ₁ : ENNReal) :
    FlockSpec.Hm96Hiding My c δ₁ ↔ Assumptions.Hm96Hiding My c δ₁ := Iff.rfl

theorem hashDerivedKey_iff {Y B C : Type} [Fintype Y] [Fintype B] (My : Y → B) (c : Y → C) :
    FlockSpec.HashDerivedKey My c ↔ Assumptions.HashDerivedKey My c := Iff.rfl

theorem keyedHash_eq : FlockSpec.keyedHash = ZkView.My := rfl

theorem saltHash_eq : FlockSpec.saltHash = ZkView.saltHash := rfl

theorem hashDerivedKeyHm96_iff : FlockSpec.HashDerivedKeyHm96 ↔ Assumptions.Zk.HashDerivedKeyHm96 := Iff.rfl

end FlockSoundness.SpecBridge
