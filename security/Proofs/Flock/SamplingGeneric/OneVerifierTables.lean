module

public import Proofs.Flock.SamplingGeneric.MaskedKSTable
public import Proofs.Flock.SamplingGeneric.Deployed
public import Proofs.Flock.Claims.DeployTables
import Mathlib.Tactic.Linarith

/-! One compiled verifier over Flock, complete and zero knowledge at sessions and knowledge sound at tables. -/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Opening KSLinOn Bext Rhon)

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

theorem maxBelow_congr {n : ℕ} {f g : ℕ → ℝ} (h : ∀ u < n, f u = g u) : maxBelow n f = maxBelow n g := by
  unfold maxBelow
  exact congrArg (List.foldr max 0) (List.map_congr_left fun u hu => h u (List.mem_range.1 hu))

section Reruns

variable {D : Claims.Flock.Deploy Coin Col T} {N : Msg → ℕ → ℕ} {ε : Msg → ℕ → ℝ} {names : Msg → ℕ × Msg}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Compiled.es_ofFlockTable (L : Compiled Coin (Arg.ofFlockTable D N ε names)) (x : Msg) (y : ℕ → Msg)
    (hsh : ∀ u < L.nPU x, (Arg.ofFlockTable D N ε names).shape (y u) = L.sh x u) :
    L.es x = maxBelow (L.nPU x) fun u =>
      1 + (D.waits (names (y u)).2 : ℝ) * (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) :=
  maxBelow_congr fun u hu =>
    (congrArg (fun s : (Arg.ofFlockTable D N ε names).Sh =>
      (1 : ℝ) + ((Arg.ofFlockTable D N ε names).waits s : ℝ) * ((Arg.ofFlockTable D N ε names).k s : ℝ))
      (hsh u hu)).symm

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem Compiled.es_ofFlockTable_le (L : Compiled Coin (Arg.ofFlockTable D N ε names)) (x : Msg) (y : ℕ → Msg)
    (hsh : ∀ u < L.nPU x, (Arg.ofFlockTable D N ε names).shape (y u) = L.sh x u) {m k : ℕ}
    (hm : ∀ u < L.nPU x, D.waits (names (y u)).2 ≤ m)
    (hk : ∀ u < L.nPU x, Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 ≤ k) :
    L.es x ≤ 1 + (m : ℝ) * (k : ℝ) := by
  rw [Compiled.es_ofFlockTable L x y hsh]
  have hm0 : (0 : ℝ) ≤ (m : ℝ) := Nat.cast_nonneg m
  have hk0 : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
  have hmk : (0 : ℝ) ≤ (m : ℝ) * (k : ℝ) := mul_nonneg hm0 hk0
  refine maxBelow_le (by linarith) fun u hu => ?_
  have h1 : (D.waits (names (y u)).2 : ℝ) ≤ (m : ℝ) := Nat.cast_le.2 (hm u hu)
  have h2 : (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) ≤ (k : ℝ) := Nat.cast_le.2 (hk u hu)
  have h3 : (D.waits (names (y u)).2 : ℝ) * (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) ≤
      (m : ℝ) * (k : ℝ) := mul_le_mul h1 h2 (Nat.cast_nonneg _) hm0
  show (1 : ℝ) + (D.waits (names (y u)).2 : ℝ) * (Claims.Flock.kTable N (names (y u)).1 (names (y u)).2 : ℝ) ≤
    1 + (m : ℝ) * (k : ℝ)
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
theorem ofFlockTable_verifier_named (D : Claims.Flock.Deploy Coin Col T) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (enc : ℕ → Msg → Msg) (hn : ∀ j xS, names (enc j xS) = (j, xS)) (j : ℕ) (xS : Msg) :
    (Arg.ofFlockTable D N ε names).V.k (enc j xS) = D.V.k xS ∧
      (Arg.ofFlockTable D N ε names).V.next (enc j xS) = D.V.next xS := by
  show D.V.k (names (enc j xS)).2 = D.V.k xS ∧ D.V.next (names (enc j xS)).2 = D.V.next xS
  rw [hn j xS]
  exact ⟨rfl, rfl⟩

end Reruns

section OneVerifier

variable {D : Claims.Flock.Deploy Coin Col T}

/-- Nothing here holds `tab x u` to one of the statement's tables; past them the deployed masked record's error is at
least `1` (`FlockDeploy.KS.one_le_κTable_past`), so the bound of a verifier that names one says nothing. -/
noncomputable def Compiled.atTables (L : Compiled Coin (Arg.ofFlock D)) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ)
    (names : Msg → ℕ × Msg) (enc : ℕ → Msg → Msg) (tab : Msg → ℕ → ℕ) (sh : Msg → ℕ → ℕ × ℕ × ℕ × ℕ)
    (κb γb : Msg → ℕ → ℝ)
    (reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool)) :
    Compiled Coin (Arg.ofFlockTable D N ε names) where
  V := L.V
  parse := L.parse
  scopeCheck := L.scopeCheck
  partition := L.partition
  commit := L.commit
  secondStage := L.secondStage
  checked := L.checked
  sh := sh
  κb := κb
  γb := γb
  rounds := L.rounds
  session x u v := (L.session x u v).map fun p => (enc (tab x u) p.1, p.2)
  skey := L.skey
  nSess := L.nSess
  iroots := L.iroots
  reveal := reveal
  openCalls := L.openCalls
  nOut := L.nOut
  nIn := L.nIn
  Tape := L.Tape
  tapeNonempty := L.tapeNonempty
  P := L.P
  δc := L.δc

variable (L : Compiled Coin (Arg.ofFlock D)) (N : Msg → ℕ → ℕ) (ε : Msg → ℕ → ℝ) (names : Msg → ℕ × Msg)
  (enc : ℕ → Msg → Msg) (tab : Msg → ℕ → ℕ) (sh : Msg → ℕ → ℕ × ℕ × ℕ × ℕ) (κb γb : Msg → ℕ → ℝ)
  (reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool))

omit [Nonempty Coin] [DecidableEq Col] in
theorem Compiled.atTables_complete (H : Msg → Msg) :
    (L.atTables N ε names enc tab sh κb γb reveal).Complete H ↔ L.Complete H := Iff.rfl

omit [DecidableEq Col] in
theorem Compiled.atTables_ks (H : Msg → Msg) :
    (L.atTables N ε names enc tab sh κb γb reveal).KS H ↔
      ∀ δ : ℝ, 0 ≤ δ →
        KSLinOn H L.V L.Scope (Sampling.R H L.commit L.parse L.partition δ)
          (L.atTables N ε names enc tab sh κb γb reveal).E ((L.atTables N ε names enc tab sh κb γb reveal).κ δ)
          (L.atTables N ε names enc tab sh κb γb reveal).γ (fun _ => 1)
          (Bext H L.V (L.atTables N ε names enc tab sh κb γb reveal).E) := Iff.rfl

/-- No compiled verifier is shown to meet the contract at tables (`hS`). -/
theorem Compiled.atTables_ks_complete [Zero Col] (H : Msg → Msg) (hKS : Claims.Flock.KSTable H D N ε)
    (hdec : Claims.Flock.ListWithin D.len D.enc D.θ D.decode) (hB : L.commit.Binds H)
    (hS : (L.atTables N ε names enc tab sh κb γb reveal).Sessions H) (hC : L.Complete H) :
    (L.atTables N ε names enc tab sh κb γb reveal).KS H ∧
      (L.atTables N ε names enc tab sh κb γb reveal).Complete H :=
  ⟨flockKSTable H D N ε names (L.atTables N ε names enc tab sh κb γb reveal) hKS hdec hB hS, hC⟩

end OneVerifier

section Masked

open Claims.Sampling (pathCommit hm96Leaf regPath RowPlaces)

variable (names : Msg → ℕ × Msg) (enc : ℕ → Msg → Msg) (tab : Msg → ℕ → ℕ) (sh : Msg → ℕ → ℕ × ℕ × ℕ × ℕ)
  (κb γb : Msg → ℕ → ℝ)
  (reveal : AuditStatement → ℕ → Seen → Claims.Flock.Wit → List (List Msg × Opening Bool))
  (L : Compiled FlockDeploy.Coin (Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation)))
  {own : Array _root_.Flock.Registered.Port} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
  {row salt π : Opening Bool → Msg} {layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg}

/-- No compiled verifier is shown to meet the contract at tables (`hS`). -/
theorem Compiled.atTables_ks_complete_masked
    (hcommit : L.commit = pathCommit at_ (hm96Leaf row salt) π (regPath own) layer) (hrow : RowPlaces row)
    (hS : (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb
      reveal).Sessions FlockDeploy.sha512)
    (hC : L.Complete FlockDeploy.sha512) :
    (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb
        reveal).KS FlockDeploy.sha512 ∧
      (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb
        reveal).Complete FlockDeploy.sha512 :=
  ⟨ksTable_masked names
      (L.atTables (FlockDeploy.KS.wordLenJ true) (FlockDeploy.KS.tableErrJ true) names enc tab sh κb γb reveal)
      hcommit hrow hS, hC⟩

end Masked

end Claims.SamplingGeneric

namespace FlockDeploy

open Flock

theorem rTable_zero_of_tables_eq_one {zk : Bool} {rel : Readable.Parsed → Readable.Relation}
    {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse zk rel x = some s) (h1 : s.tables = 1)
    (w : Claims.Flock.Wit) : Claims.Flock.RTable (table zk rel) 0 x w ↔ Claims.Flock.RelS s w := by
  unfold Claims.Flock.RTable
  rw [table_zero_of_tables_eq_one hs h1]
  exact ⟨fun ⟨_, ht, hw⟩ => by cases ht; exact hw, fun hw => ⟨s, rfl, hw⟩⟩

end FlockDeploy
