module

public import Proofs.Flock.Claims.DeployInputs
public import Proofs.Flock.Claims.ZK.Reference
public import Proofs.Flock.Recursive.FirewallWire
import all Init.Data.String.Defs

/-! # A statement the deployed `--zk` setup parses

`Loop.not_zkOut_leakL` and the session loop's honest witness take a statement `x` with
`FlockDeploy.setupOf true x = some _`. This file holds what such a statement needs.

* `FlockDeploy.decode_encode_of_fits`: `decode` reads back every `Raw` whose lengths fit their 8-byte words, so a
  statement is the encoding of the `Raw` the setup reads (`Raw.Fits`).
* `Claims.ZKMal.refK_pos`: a statement that sets up has a coin round.
-/

@[expose] public section

namespace FlockDeploy

open Flock

/-! ## `decode` reads back what `Raw.encode` writes -/

theorem run_rdTake {n : ℕ} (l rest : Claims.Msg) (h : l.length = n) :
    (rdTake n).run (l ++ rest) = some (l, rest) := by
  subst h
  simp [StateT.run, rdTake, List.take_left' rfl, List.drop_left' rfl]

theorem length_encNat (n : ℕ) : (encNat n).length = 8 := by simp [encNat]

/-- `k` little-endian bytes of `n` from byte `j` read back as `n`'s digits `j … j + k - 1`. -/
theorem foldr_digits (n : ℕ) : ∀ (k j : ℕ), ((List.range' j k).map fun i => (n / 256 ^ i).toUInt8).foldr
    (fun b acc => b.toNat + 256 * acc) 0 = n / 256 ^ j % 256 ^ k
  | 0, j => by rw [Nat.pow_zero, Nat.mod_one]; rfl
  | k + 1, j => by
    rw [List.range'_succ, List.map_cons, List.foldr_cons, foldr_digits n k (j + 1),
      show n / 256 ^ (j + 1) = n / 256 ^ j / 256 by rw [Nat.pow_succ, Nat.div_div_eq_div_mul], Nat.pow_succ',
      Nat.mod_mul]
    simp

theorem foldr_encNat {n : ℕ} (h : n < 2 ^ 64) : (encNat n).foldr (fun b acc => b.toNat + 256 * acc) 0 = n := by
  rw [encNat, List.range_eq_range', foldr_digits, Nat.pow_zero, Nat.div_one]
  exact Nat.mod_eq_of_lt (by simpa using h)

theorem run_bind_some {α β : Type} {x : Rd α} {f : α → Rd β} {s s' : Claims.Msg} {a : α}
    (h : x.run s = some (a, s')) : (x >>= f).run s = (f a).run s' := by
  rw [StateT.run_bind, h]; rfl

theorem run_pure' {α : Type} (a : α) (s : Claims.Msg) : (pure a : Rd α).run s = some (a, s) := rfl

theorem run_rdNat {n : ℕ} (h : n < 2 ^ 64) (rest : Claims.Msg) : rdNat.run (encNat n ++ rest) = some (n, rest) := by
  rw [rdNat, run_bind_some (run_rdTake (encNat n) rest (length_encNat n)), run_pure', foldr_encNat h]

theorem run_rdBytes {b : ByteArray} (h : b.size < 2 ^ 64) (rest : Claims.Msg) :
    rdBytes.run (encBytes b ++ rest) = some (b, rest) := by
  rw [rdBytes, encBytes, List.append_assoc, run_bind_some (run_rdNat h _),
    run_bind_some (run_rdTake (n := b.size) _ rest rfl), run_pure']

theorem run_rdStr {s : String} (h : s.toUTF8.size < 2 ^ 64) (rest : Claims.Msg) :
    rdStr.run (encStr s ++ rest) = some (s, rest) := by
  rw [rdStr, encStr, run_bind_some (run_rdBytes h rest),
    FlockSoundness.Discharge.Recursion.FirewallWire.fromUTF8?_toUTF8]
  rfl

theorem run_rdBool (b : Bool) (rest : Claims.Msg) : rdBool.run (encBool b ++ rest) = some (b, rest) := by
  rw [rdBool, run_bind_some (run_rdTake (n := 1) (encBool b) rest rfl)]
  cases b <;> rfl

theorem run_rdOpt {α : Type} {r : Rd α} {e : α → Claims.Msg} {o : Option α}
    (hr : ∀ a, o = some a → ∀ rest, r.run (e a ++ rest) = some (a, rest)) (rest : Claims.Msg) :
    (rdOpt r).run (encOpt e o ++ rest) = some (o, rest) := by
  cases o with
  | none =>
    have hb : rdBool.run (encOpt e none ++ rest) = some (false, rest) := run_rdBool false rest
    rw [rdOpt, run_bind_some hb]; rfl
  | some a =>
    have hb : rdBool.run (encOpt e (some a) ++ rest) = some (true, e a ++ rest) := run_rdBool true (e a ++ rest)
    rw [rdOpt, run_bind_some hb]
    show (some <$> r).run (e a ++ rest) = _
    rw [StateT.run_map, hr a rfl rest]; rfl

theorem run_rdListN {α : Type} {r : Rd α} {e : α → Claims.Msg} :
    ∀ (l : List α), (∀ a ∈ l, ∀ rest, r.run (e a ++ rest) = some (a, rest)) → ∀ rest,
      (rdListN r l.length).run (l.flatMap e ++ rest) = some (l, rest)
  | [], _, _ => rfl
  | a :: l, hr, rest => by
    rw [List.length_cons, rdListN, List.flatMap_cons, List.append_assoc,
      run_bind_some (hr a List.mem_cons_self _),
      run_bind_some (run_rdListN l (fun b hb => hr b (List.mem_cons_of_mem a hb)) rest)]
    rfl

theorem run_rdList {α : Type} {r : Rd α} {e : α → Claims.Msg} {l : List α} (hl : l.length < 2 ^ 64)
    (hr : ∀ a ∈ l, ∀ rest, r.run (e a ++ rest) = some (a, rest)) (rest : Claims.Msg) :
    (rdList r).run (encList e l ++ rest) = some (l, rest) := by
  rw [rdList, encList, List.append_assoc, run_bind_some (run_rdNat hl _), run_rdListN l hr]

/-- **Every length of `w` fits its 8-byte word**: its strings', files', lists' and numbers'. -/
structure Raw.Fits (w : Raw) : Prop where
  statement : w.statement.toUTF8.size < 2 ^ 64
  circuit : w.circuit.size < 2 ^ 64
  publicFile : w.publicFile.size < 2 ^ 64
  tables : w.tables.length < 2 ^ 64
  tableItems : ∀ p ∈ w.tables, p.1.toUTF8.size < 2 ^ 64 ∧ p.2.size < 2 ^ 64
  count : w.count < 2 ^ 64
  draw : ∀ b, w.draw = some b → b.size < 2 ^ 64
  partition : ∀ b, w.partition = some b → b.size < 2 ^ 64
  program : ∀ b, w.program = some b → b.size < 2 ^ 64
  work : ∀ n, w.work = some n → n < 2 ^ 64
  workTable : ∀ b, w.workTable = some b → b.size < 2 ^ 64
  stratified : ∀ n, w.stratified = some n → n < 2 ^ 64
  closure : ∀ b, w.closure = some b → b.size < 2 ^ 64
  registered : ∀ b, w.registered = some b → b.size < 2 ^ 64
  publicInputs : ∀ b, w.publicInputs = some b → b.size < 2 ^ 64

theorem run_rdOptBytes {o : Option ByteArray} (h : ∀ b, o = some b → b.size < 2 ^ 64) (rest : Claims.Msg) :
    (rdOpt rdBytes).run (encOpt encBytes o ++ rest) = some (o, rest) :=
  run_rdOpt (fun b hb => run_rdBytes (h b hb)) rest

theorem run_rdOptNat {o : Option ℕ} (h : ∀ n, o = some n → n < 2 ^ 64) (rest : Claims.Msg) :
    (rdOpt rdNat).run (encOpt encNat o ++ rest) = some (o, rest) :=
  run_rdOpt (fun n hn => run_rdNat (h n hn)) rest

theorem run_read {w : Raw} (hw : w.Fits) (rest : Claims.Msg) : Raw.read.run (w.encode ++ rest) = some (w, rest) := by
  have hF : FORMAT.toUTF8.size < 2 ^ 64 := by decide +kernel
  have ht : ∀ rest, (rdList (do let n ← rdStr; let b ← rdBytes; pure (n, b))).run
      (encList (fun p => encStr p.1 ++ encBytes p.2) w.tables ++ rest) = some (w.tables, rest) :=
    run_rdList hw.tables fun p hp rest => by
      obtain ⟨h1, h2⟩ := hw.tableItems p hp
      rw [List.append_assoc, run_bind_some (run_rdStr h1 _), run_bind_some (run_rdBytes h2 _), run_pure']
  simp only [Raw.encode, List.append_assoc]
  rw [Raw.read, run_bind_some (run_rdStr hF _)]
  simp only [ne_eq, not_true_eq_false, ↓reduceIte]
  rw [run_bind_some (run_rdStr hw.statement _), run_bind_some (run_rdBytes hw.circuit _),
    run_bind_some (run_rdBytes hw.publicFile _), run_bind_some (ht _), run_bind_some (run_rdBool _ _),
    run_bind_some (run_rdBool _ _), run_bind_some (run_rdNat hw.count _), run_bind_some (run_rdOptBytes hw.draw _),
    run_bind_some (run_rdOptBytes hw.partition _), run_bind_some (run_rdOptBytes hw.program _),
    run_bind_some (run_rdOptNat hw.work _), run_bind_some (run_rdOptBytes hw.workTable _),
    run_bind_some (run_rdOptNat hw.stratified _), run_bind_some (run_rdOptBytes hw.closure _),
    run_bind_some (run_rdOptBytes hw.registered _), run_bind_some (run_rdOptBytes hw.publicInputs _), run_pure']

/-- **`decode` reads back every statement whose lengths fit their words.** -/
theorem decode_encode_of_fits {w : Raw} (hw : w.Fits) : decode w.encode = some w := by
  have h := run_read hw []
  rw [List.append_nil] at h
  unfold decode
  rw [h]
  simp

end FlockDeploy

namespace Claims.ZKMal

/-- **A statement that sets up has a coin round**: the deployed verifier's `k` is `2 + rounds` there. -/
theorem refK_pos {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) : 0 < refK x := by
  unfold refK
  simp only [FlockDeploy.V, hs]
  omega

end Claims.ZKMal
