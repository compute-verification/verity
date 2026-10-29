import Proofs.Flock.Claims.Complete.Json.Hex
import Flock.RecCompose

/-!
# What `flock-verify rec-compose` holds V*'s parts to, read back to their files

`RecCompose.Accepts` is over the digests the parts' `VERDICT`s name. `RecShared` is what the compiled verifier's parts
share of their recursive statements (`RecDeploy.RecRaw`'s fields but each part's Flock statement, index and bindings), as
files; `PartView` is one part's run as its `VERDICT` reports it (`PartView.verdict`, what `verify --rec-part` writes,
`recSharedInput`). `composed_views`: where `rec-compose` accepts the parts' verdicts, either two of the parts' shared
files are distinct with one SHA-512 (`Collides`), or every part shares one `RecShared` in the compiled verifier's form at
the composition's inner shape and rows, one part per V* statement, each accepted under `--zk` without an identifier, no
two on one coin key. `Proofs.Flock.Claims.Recursion.ComposeParts` reads it at `RecDeploy.partRaw`.
-/

namespace RecDeploy

open Flock

/-- **What V*'s parts share** of their recursive statements, as files: the verifier's own inner shape, V*'s shared
rows, the inner statement's three files, a record of check 5's coins, the inner session's record, the coin server's
transcript, the circuit registration, and how the inner statement is set up (its id, coins, lookup tables and draw). -/
structure RecShared where
  recInner : Option ByteArray
  recRows : Option ByteArray
  innerCircuit : Option ByteArray
  innerPublic : Option ByteArray
  innerPartition : Option ByteArray
  innerCoins : Option ByteArray
  innerSession : Option ByteArray
  transcript : Option ByteArray
  registration : Option ByteArray
  innerStatement : Option String
  innerStatementCoins : Option Bool
  innerTables : Option (List (String × ByteArray))
  innerDraw : Option ByteArray

/-- Every file of it. -/
def RecShared.files (S : RecShared) : List ByteArray :=
  S.recInner.toList ++ S.recRows.toList ++ S.innerCircuit.toList ++ S.innerPublic.toList ++
    S.innerPartition.toList ++ S.innerCoins.toList ++ S.innerSession.toList ++ S.transcript.toList ++
    S.registration.toList ++ ((S.innerTables.getD []).map (·.2)) ++ S.innerDraw.toList

/-- The digests a `VERDICT`'s `rec_shared` names of it (`RecCompose.Shared.ofFiles`). -/
def RecShared.digests (S : RecShared) : RecCompose.Shared :=
  RecCompose.Shared.ofFiles S.recInner S.innerCircuit S.innerPublic S.innerPartition S.innerCoins S.innerSession
    S.transcript S.registration S.innerStatement S.innerStatementCoins S.innerTables S.innerDraw

/-- **The compiled verifier's form** (`RecDeploy.partRaw`): every input a part reads is held, and no record of check
5's coins. -/
def RecShared.Form (S : RecShared) : Prop :=
  S.recInner.isSome ∧ S.recRows.isSome ∧ S.innerCircuit.isSome ∧ S.innerPublic.isSome ∧ S.innerPartition.isSome ∧
    S.innerCoins = none ∧ S.innerSession.isSome ∧ S.transcript.isSome ∧ S.innerStatement.isSome ∧
    S.innerStatementCoins.isSome ∧ S.innerTables.isSome

/-- **One part's run as `rec-compose` reads it**: the shared inputs its `verify` read, its index (`--rec-part`),
whether it ran `--zk`, its session's verdict and reason, its coin key, and whether `verify` held an identifier. -/
structure PartView where
  shared : RecShared
  part : Option Nat
  zk : Bool
  accepted : Bool
  why : String
  coinKey : Option String
  identifier : Bool

/-- **The `VERDICT` `verify --rec-part` writes of it** (`FlockVerify.verifyCmd`, `recSharedInput`). -/
def PartView.verdict (v : PartView) : RecCompose.Verdict where
  accepted := v.accepted
  why := v.why
  part := v.part
  rows := v.shared.recRows.map RecCompose.digest
  shared := some v.shared.digests
  zk := some v.zk
  coinKey := v.coinKey
  identifier := v.identifier

/-- **A SHA-512 collision among the parts' shared files**: two distinct files of theirs with one hash. -/
def Collides (vs : List PartView) : Prop :=
  ∃ v ∈ vs, ∃ w ∈ vs, ∃ a ∈ v.shared.files, ∃ b ∈ w.shared.files, a ≠ b ∧ Sha512.hash a = Sha512.hash b

theorem digest_hash {a b : ByteArray} (h : RecCompose.digest a = RecCompose.digest b) :
    Sha512.hash a = Sha512.hash b := by
  have ha := FlockComplete.JsonRT.unhex_hexOf (Sha512.hash a)
  unfold RecCompose.digest at h
  rw [h, FlockComplete.JsonRT.unhex_hexOf] at ha
  exact (Except.ok.inj ha).symm

theorem map_digest_eq {o o' : Option ByteArray}
    (hinj : ∀ a b, o = some a → o' = some b → Sha512.hash a = Sha512.hash b → a = b)
    (h : o.map RecCompose.digest = o'.map RecCompose.digest) : o = o' := by
  cases o with
  | none => cases o' with
    | none => rfl
    | some _ => simp only [Option.map_none, Option.map_some, reduceCtorEq] at h
  | some a => cases o' with
    | none => simp only [Option.map_none, Option.map_some, reduceCtorEq] at h
    | some b =>
      simp only [Option.map_some, Option.some.injEq] at h
      rw [hinj a b rfl rfl (digest_hash h)]

theorem map_tables_eq : ∀ {t t' : List (String × ByteArray)},
    (∀ a ∈ t.map (·.2), ∀ b ∈ t'.map (·.2), Sha512.hash a = Sha512.hash b → a = b) →
    t.map (fun x => (x.1, RecCompose.digest x.2)) = t'.map (fun x => (x.1, RecCompose.digest x.2)) → t = t'
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by cases h
  | _ :: _, [], _, h => by cases h
  | (n, a) :: t, (n', b) :: t', hinj, h => by
    simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at h
    obtain ⟨⟨hn, hd⟩, ht⟩ := h
    have hab := hinj a (by simp) b (by simp) (digest_hash hd)
    rw [hn, hab, map_tables_eq (fun x hx y hy => hinj x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy)) ht]

/-- **Two shared sets with one set of digests are one**, unless two of their files collide. -/
theorem shared_eq {S T : RecShared} (hinj : ∀ a ∈ S.files, ∀ b ∈ T.files, Sha512.hash a = Sha512.hash b → a = b)
    (h : S.digests = T.digests) (hr : S.recRows.map RecCompose.digest = T.recRows.map RecCompose.digest) : S = T := by
  obtain ⟨s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13⟩ := S
  obtain ⟨t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13⟩ := T
  simp only [RecShared.digests, RecCompose.Shared.ofFiles, RecCompose.Shared.mk.injEq] at h
  obtain ⟨h1, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h
  dsimp only at hr
  have e1 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h1
  have e2 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) hr
  have e3 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h3
  have e4 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h4
  have e5 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h5
  have e6 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h6
  have e7 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h7
  have e8 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h8
  have e9 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h9
  have e13 := map_digest_eq (fun a b ha hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h13
  have e12 : s12 = t12 := by
    cases s12 with
    | none => cases t12 with
      | none => rfl
      | some _ => simp only [Option.map_none, Option.map_some, reduceCtorEq] at h12
    | some ts => cases t12 with
      | none => simp only [Option.map_none, Option.map_some, reduceCtorEq] at h12
      | some ts' =>
        simp only [Option.map_some, Option.some.injEq] at h12
        rw [map_tables_eq (fun a ha b hb => hinj a (by simp [RecShared.files, ha]) b (by simp [RecShared.files, hb])) h12]
  subst e1 e2 e3 e4 e5 e6 e7 e8 e9 e13 e12 h10 h11
  rfl

/-- **The composition read back to the parts' files**: where `rec-compose` accepts the verdicts the parts' runs wrote
(`RecCompose.Accepts`, which `check_accepts` proves of every composition it accepts), either two of the parts' shared
files are distinct with one SHA-512, or there is one `RecShared` in the compiled verifier's form, at the composition's
own inner shape and rows, that every part read; one part per V* statement, each accepted at one of them under `--zk`
without an identifier, and no two on one coin key. -/
theorem composed_views {n : Nat} {rows inner : String} {vs : List PartView}
    (h : RecCompose.Accepts n rows inner (vs.map PartView.verdict)) :
    Collides vs ∨ ∃ S : RecShared, S.Form ∧ S.recInner.map RecCompose.digest = some inner ∧
      S.recRows.map RecCompose.digest = some rows ∧ vs.length = n ∧
      (∀ j, j < n → (vs.map (·.part)).count (some j) = 1) ∧ (vs.map (·.coinKey)).Nodup ∧
      ∀ v ∈ vs, v.shared = S ∧ v.accepted = true ∧ v.zk = true ∧ v.identifier = false ∧ v.coinKey.isSome ∧
        ∃ j, v.part = some j ∧ j < n := by
  classical
  obtain ⟨hn, hlen, hidx, hall, ⟨sh, hform, hsh⟩, hkeys, hnodup⟩ := h
  by_cases hc : Collides vs
  · exact .inl hc
  right
  obtain _ | ⟨v0, rest⟩ := vs
  · simp at hlen
    omega
  have mem : ∀ {v}, v ∈ v0 :: rest → v.verdict ∈ (v0 :: rest).map PartView.verdict :=
    fun hv => List.mem_map_of_mem hv
  have h0 := mem (List.mem_cons_self (a := v0) (l := rest))
  have hs0 := hsh _ h0
  have hr0 := (hall _ h0).2.2.2.1
  simp only [PartView.verdict, Option.some.injEq] at hs0 hr0
  have heq : ∀ v ∈ v0 :: rest, v.shared = v0.shared := by
    intro v hv
    have h1 := hsh _ (mem hv)
    have h2 := (hall _ (mem hv)).2.2.2.1
    simp only [PartView.verdict, Option.some.injEq] at h1 h2
    refine shared_eq (fun a ha b hb he => ?_) (h1.trans hs0.symm) (h2.trans hr0.symm)
    exact Classical.byContradiction fun hne => hc ⟨v, hv, v0, List.mem_cons_self, a, ha, b, hb, hne, he⟩
  rw [← hs0] at hform
  simp only [RecCompose.Shared.Complete, RecShared.digests, RecCompose.Shared.ofFiles, Option.isSome_map,
    Option.map_eq_none_iff] at hform
  obtain ⟨hi, hc1, hp1, hq1, hk0, hse1, ht1, hst1, hco1, hta1⟩ := hform
  refine ⟨v0.shared, ⟨?_, ?_, hc1, hp1, hq1, hk0, hse1, ht1, hst1, hco1, hta1⟩, hi, hr0, by simpa using hlen, ?_, ?_, ?_⟩
  · cases h : v0.shared.recInner <;> simp_all
  · cases h : v0.shared.recRows <;> simp_all
  · intro j hj
    have h := hidx j hj
    rw [List.map_map] at h
    exact h
  · rw [List.map_map] at hnodup
    exact hnodup
  · intro v hv
    have ⟨ha, hz, hid, _, hj⟩ := hall _ (mem hv)
    simp only [PartView.verdict, Option.some.injEq] at ha hz hid hj
    exact ⟨heq v hv, ha, hz, hid, by simpa [PartView.verdict] using hkeys _ (mem hv), hj⟩

end RecDeploy
