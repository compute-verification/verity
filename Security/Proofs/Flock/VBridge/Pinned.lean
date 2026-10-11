import Proofs.Flock.VBridge.Verifier
import VStar.Stage
import Std.Data.String.ToNat

/-!
# A rec-open unit's `(L, H)` is its pinned statement's

V* names the statement of a level's openings `rec-open/l<L>-h<H>` (`VStar.Stmt.name`, `name_recOpen`), which is the
verifier's `RecOpen.unitName L H`. `unit_check` reads `L` and `H` back from the unit's name (`RecOpen.params`), so a
circuit under that name must give back the pinned pair, not another one the file chose.

* `unitName_inj`: the name determines `L` and `H`. A decimal numeral has no `-`, so `l<L>-h<H>` splits at its first `-`,
  and `Nat.repr` is injective.
* `unitName_params`: `params` gives `(L, H)` only for the name `unitName L H`.
* `params_of_check`: when `RecOpen.check` accepted a circuit whose unit is named `unitName L H`, `params` gives `(L, H)`.
  The name starts with `rec-open/` (`startsWith_unitName`), so the check refuses it when `params` gives nothing, and any
  pair `params` gives has the same name. This needs no lemma about `String.splitOn`.
* `unit_check_pinned`: `unit_check` at the pinned name in place of `params`' answer.
-/

namespace FlockVBridge

open Flock.HmNets hiding and
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Hm

theorem toString_str (s : String) : toString s = s := rfl

theorem digits_dash {a b c d : List Char} (ha : ∀ x ∈ a, x.isDigit) (hc : ∀ x ∈ c, x.isDigit)
    (h : a ++ '-' :: b = c ++ '-' :: d) : a = c ∧ b = d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simpa using h
    | cons x c =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      have := hc x (by simp)
      rw [← h.1] at this
      exact absurd this (by decide)
  | cons x a ih =>
    cases c with
    | nil =>
      simp only [List.nil_append, List.cons_append, List.cons.injEq] at h
      have := ha x (by simp)
      rw [h.1] at this
      exact absurd this (by decide)
    | cons y c =>
      simp only [List.cons_append, List.cons.injEq] at h
      obtain ⟨rfl, h⟩ := h
      have := ih (fun z hz => ha z (by simp [hz])) (fun z hz => hc z (by simp [hz])) h
      exact ⟨by rw [this.1], this.2⟩

theorem eq_of_toDigits {m n : ℕ} (h : Nat.toDigits 10 m = Nat.toDigits 10 n) : m = n :=
  Nat.repr_injective (String.toList_injective (by rw [Nat.toList_repr, Nat.toList_repr, h]))

/-- **The name `rec-open/l<L>-h<H>` determines `L` and `H`.** -/
theorem unitName_inj {L H L' H' : ℕ} (h : Flock.RecOpen.unitName L H = Flock.RecOpen.unitName L' H') :
    L = L' ∧ H = H' := by
  have h := congrArg String.toList h
  simp only [Flock.RecOpen.unitName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := digits_dash (dig L) (dig L') (by simpa using h)
  simp only [List.cons.injEq, true_and] at h2
  exact ⟨eq_of_toDigits h1, eq_of_toDigits h2⟩

/-- `params` gives `(L, H)` only for the name `unitName L H`. -/
theorem unitName_params {name : String} {L H : ℕ} (h : Flock.RecOpen.params name = some (L, H)) :
    Flock.RecOpen.unitName L H = name := by
  unfold Flock.RecOpen.params at h
  repeat' split at h
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h; assumption)
    | cases h

theorem startsWith_unitName (L H : ℕ) : (Flock.RecOpen.unitName L H).startsWith "rec-open/" = true := by
  have : Flock.RecOpen.unitName L H = "rec-open/" ++ s!"l{L}-h{H}" := by
    simp [Flock.RecOpen.unitName, ← String.toList_inj, List.append_assoc, toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

/-- **A unit the verifier accepted under the name `unitName L H` gives back `(L, H)`.** -/
theorem params_of_check {c : Flock.Circuit} (h : Flock.RecOpen.check c = .ok ()) {L H : ℕ}
    (hn : c.unit.name = Flock.RecOpen.unitName L H) : Flock.RecOpen.params c.unit.name = some (L, H) := by
  cases hp : Flock.RecOpen.params c.unit.name with
  | some p =>
    obtain ⟨L', H'⟩ := p
    obtain ⟨rfl, rfl⟩ := unitName_inj ((unitName_params hp).trans hn)
    rfl
  | none =>
    unfold Flock.RecOpen.check at h
    rw [hp] at h
    dsimp only at h
    rw [hn, startsWith_unitName] at h
    cases h

/-- V*'s statement of a level's openings is named as the verifier's unit at its `L` and `H`. -/
theorem name_recOpen (level L H n : ℕ) : (VStar.Stmt.recOpen level L H n).name = Flock.RecOpen.unitName L H := rfl

/-- **`unit_check` at the pinned name**: a unit the verifier accepted (`RecOpen.check`) under the name of V*'s statement
at `L` and `H` (`unitName L H`) computes, in a copy in a unit of any partition, the climb and the consistency term at that
`L` and `H` (`params_of_check`). -/
theorem unit_check_pinned {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {c : Flock.Circuit}
    (hB : Built c.unit) {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (X : Fin C.N → Bool) {j : ℕ}
    (hj : j < k) (hone : X (I.wire ((Rows.ofNet hB).stack k).one) = true)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r))
    (hrec : Flock.RecOpen.check c = .ok ()) {LANES H : ℕ} (hn : c.unit.name = Flock.RecOpen.unitName LANES H)
    (hu : c.unit.unitLog ≤ 32)
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hcf : cf.size = 2 * LANES) (hrest : rest.size = 96) (hss : ss.length = H)
    (hrow : ColsHold (copyVal I X hj) 0 (Flock.MerkleScheme.rowBytes r))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧ ColsHold (copyVal I X hj) (128 * LANES + 512 * i) ss[i])
    (hacc : ColsHold (copyVal I X hj) (128 * LANES + 1024 * max 1 ((H + 1) / 2))
      (Flock.MerkleScheme.rowBytes a ++ rest))
    (hcoef : ColsHold (copyVal I X hj) (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024)
      (Flock.MerkleScheme.rowBytes cf))
    (hdir : ∀ h < H, copyVal I X hj (128 * LANES + 1024 * max 1 ((H + 1) / 2) + 1024 + 256 * LANES + h) =
      idx.testBit h) :
    ColsHold (copyVal I X hj)
      (outStart ((Flock.RecOpen.recOpenNet LANES H (Flock.RecOpen.ports LANES H)).run St.empty).2
        (Flock.RecOpen.wordPorts (unitIn LANES H)) (Flock.RecOpen.wordPorts 2048))
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a cf 0, consTerm LANES r a cf 1] ++ rest)) :=
  unit_check hB I X hj hone ms hms hrec (params_of_check hrec hn) hu r a cf rest idx ss hr ha hcf hrest hss hrow haux
    hacc hcoef hdir

end FlockVBridge
