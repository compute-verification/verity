import Proofs.Flock.Claims.Sampling.Bext

/-!
# The block's openings bind the boundary values

The claims block opens a row with `openOC`: it hashes `packBits(vals) ‖ salt` to the leaf, then runs the registered
path check from the leaf to the row's root at the row's position. The extractor runs the same check on every opening it
assembles, as its own calls. Two accepted openings of one row to different values therefore give a collision among
those calls: at the leaf, when the leaves agree (`packBits` is injective at one width), and inside the path checks
otherwise, when the path check binds its leaf (`PathBinds`, a property of the registered scheme that `Deploy.path`
leaves open).

* `packBits_inj`: `packBits` is injective on bit strings of one length.
* `PathBinds`: two accepted path checks at one root and position, of different leaves, make calls that collide.
* `openOC_binds`: two accepted openings of one row to different values make calls that collide.
* `Eopens`, `E_eval_acc`, `Eopens_checked`: the openings `E` assembles on an accepted run of a parsed statement, each one
  accepted by `openOC`, its calls among `E`'s own.
* `E_ownColl_of_disagree`, `KSLin_of_disagree`: **the binding step of the reduction**. Two of them that open one row to
  different values are an own-call collision, so `KS`'s linear term at `Bext` covers every accepted run whose
  assembled openings disagree, and the rest of `KS` is the runs whose openings agree.
-/

namespace Claims.Sampling

open Claims

/-! ## `packBits` is injective at one width -/

theorem ite_toNat (b : Bool) (j : ℕ) : (if b = true then 2 ^ j else 0) = 2 ^ j * b.toNat := by
  cases b <;> simp

theorem bool_eq_of_toNat {a b : Bool} (h : a.toNat = b.toNat) : a = b := by
  cases a <;> cases b <;> simp_all

/-- One byte of `packBits` fixes its eight bits. -/
theorem packBits_byte {bs bs' : List Bool} (i : ℕ)
    (h : UInt8.ofNat ((List.range 8).foldl (fun acc j => acc + if bs.getD (8 * i + j) false then 2 ^ j else 0) 0) =
      UInt8.ofNat ((List.range 8).foldl (fun acc j => acc + if bs'.getD (8 * i + j) false then 2 ^ j else 0) 0))
    (j : ℕ) (hj : j < 8) : bs.getD (8 * i + j) false = bs'.getD (8 * i + j) false := by
  have h' := congrArg UInt8.toNat h
  rw [show List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl] at h'
  simp only [UInt8.toNat_ofNat', List.foldl_cons, List.foldl_nil, ite_toNat] at h'
  have := Bool.toNat_le (bs.getD (8 * i + 0) false)
  have := Bool.toNat_le (bs.getD (8 * i + 1) false)
  have := Bool.toNat_le (bs.getD (8 * i + 2) false)
  have := Bool.toNat_le (bs.getD (8 * i + 3) false)
  have := Bool.toNat_le (bs.getD (8 * i + 4) false)
  have := Bool.toNat_le (bs.getD (8 * i + 5) false)
  have := Bool.toNat_le (bs.getD (8 * i + 6) false)
  have := Bool.toNat_le (bs.getD (8 * i + 7) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 0) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 1) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 2) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 3) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 4) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 5) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 6) false)
  have := Bool.toNat_le (bs'.getD (8 * i + 7) false)
  apply bool_eq_of_toNat
  interval_cases j <;> omega

theorem length_packBits (bs : List Bool) : (packBits bs).length = (bs.length + 7) / 8 := by
  simp [packBits]

/-- **`packBits` is injective on bit strings of one length.** -/
theorem packBits_inj {bs bs' : List Bool} (hl : bs.length = bs'.length) (h : packBits bs = packBits bs') :
    bs = bs' := by
  unfold packBits at h
  rw [hl] at h
  have hb := List.map_inj_left.1 h
  refine List.ext_getElem hl fun n h₁ h₂ => ?_
  rw [List.getElem_eq_getD false, List.getElem_eq_getD false]
  have hn : n = 8 * (n / 8) + n % 8 := (Nat.div_add_mod n 8).symm
  have hi : n / 8 ∈ List.range ((bs'.length + 7) / 8) := List.mem_range.2 (by omega)
  rw [hn]
  exact packBits_byte (n / 8) (hb _ hi) (n % 8) (Nat.mod_lt _ (by norm_num))

/-! ## An opening's check binds its row's values -/

/-- Two different inputs with one output in a log of `H` calls. -/
def LogColl (l : List (Msg × Msg)) : Prop := ∃ p ∈ l, ∃ p' ∈ l, p.1 ≠ p'.1 ∧ p.2 = p'.2

theorem LogColl.mono {l l' : List (Msg × Msg)} (h : LogColl l) (hs : ∀ q ∈ l, q ∈ l') : LogColl l' := by
  obtain ⟨p, hp, p', hp', hne, heq⟩ := h
  exact ⟨p, hs p hp, p', hs p' hp', hne, heq⟩

/-- **The registered path check binds its leaf**: two accepted checks at one root and position, of different leaves,
make `H` calls that collide. What a Merkle path check gives: where the two walks to the root first meet, one hash has
two different inputs. -/
def PathBinds (H : Msg → Msg) (path : Msg → ℕ → Msg → Msg → OC Bool) : Prop :=
  ∀ root pos leaf leaf' π π', leaf ≠ leaf' → ((path root pos leaf π).eval H).1 = true →
    ((path root pos leaf' π').eval H).1 = true →
      LogColl (((path root pos leaf π).eval H).2 ++ ((path root pos leaf' π').eval H).2)

/-- **Two accepted openings of one row to different values make calls that collide**: at the leaf when the leaves
agree, and in the path checks otherwise. -/
theorem openOC_binds {H : Msg → Msg} {path : Msg → ℕ → Msg → Msg → OC Bool} (hpath : PathBinds H path) (s : Stmt)
    (o o' : Opening) (hrow : o.row = o'.row) (h : ((openOC path s o).eval H).1 = true)
    (h' : ((openOC path s o').eval H).1 = true) (hne : o.vals ≠ o'.vals) :
    LogColl (((openOC path s o).eval H).2 ++ ((openOC path s o').eval H).2) := by
  unfold openOC at h h' ⊢
  rw [hrow] at h ⊢
  cases hr : s.rows[o'.row]? with
  | none => simp [hr, OC.eval] at h
  | some row =>
    simp only [hr] at h h' ⊢
    by_cases hw : o.vals.length = row.width
    · by_cases hw' : o'.vals.length = row.width
      · simp only [hw, hw', ↓reduceIte, OC.eval] at h h' ⊢
        by_cases hleaf : H (packBits o.vals ++ o.salt) = H (packBits o'.vals ++ o'.salt)
        · refine ⟨(packBits o.vals ++ o.salt, H (packBits o.vals ++ o.salt)), List.mem_append_left _ (by simp),
            (packBits o'.vals ++ o'.salt, H (packBits o'.vals ++ o'.salt)), List.mem_append_right _ (by simp), ?_,
            hleaf⟩
          intro he
          have hpl : (packBits o.vals).length = (packBits o'.vals).length := by
            rw [length_packBits, length_packBits, hw, hw']
          exact hne (packBits_inj (hw.trans hw'.symm) (List.append_inj he hpl).1)
        · refine (hpath _ _ _ _ _ _ hleaf h h').mono fun q hq => ?_
          rcases List.mem_append.1 hq with hq | hq
          · exact List.mem_append_left _ (List.mem_cons_of_mem _ hq)
          · exact List.mem_append_right _ (List.mem_cons_of_mem _ hq)
      · simp [hw', OC.eval] at h'
    · simp [hw, OC.eval] at h

/-! ## The openings `E` assembles are its own checks' -/

theorem ownLog_ofOC {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ c : OC α, ownLog H run (ofOC c : RP R α) = (c.eval H).2
  | .ret _ => rfl
  | .query q k => by
    simp only [ofOC, ownLog, OC.eval]
    rw [ownLog_ofOC H run (k (H q))]

theorem eval_ofOC {R α : Type} (H : Msg → Msg) (run : R → Trace) :
    ∀ c : OC α, ((ofOC c : RP R α).eval H run).1 = (c.eval H).1
  | .ret _ => rfl
  | .query q k => by
    simp only [ofOC, RP.eval, OC.eval]
    exact eval_ofOC H run (k (H q))

/-- What `flatMapRP` returns comes from one element's program, whose own calls are among the whole program's. -/
theorem flatMapRP_mem {R β γ : Type} (H : Msg → Msg) (run : R → Trace) (f : β → RP R (List γ)) :
    ∀ (l : List β), ∀ c ∈ ((flatMapRP f l).eval H run).1,
      ∃ b ∈ l, c ∈ ((f b).eval H run).1 ∧ ∀ q ∈ ownLog H run (f b), q ∈ ownLog H run (flatMapRP f l)
  | [] => by simp [flatMapRP, RP.eval]
  | b :: bs => by
    intro c hc
    simp only [flatMapRP] at hc ⊢
    rw [(RP.eval_bind H run _ _).1, (RP.eval_bind H run _ _).1] at hc
    simp only [RP.eval, List.mem_append] at hc
    rw [ownLog_bind, ownLog_bind]
    rcases hc with hc | hc
    · exact ⟨b, by simp, hc, fun q hq => List.mem_append_left _ hq⟩
    · obtain ⟨b', hb', hc', hq⟩ := flatMapRP_mem H run f bs c hc
      exact ⟨b', List.mem_cons_of_mem _ hb', hc', fun q h => List.mem_append_right _ (List.mem_append_left _ (hq q h))⟩

theorem checkOpens_mem {Coin Col T R : Type} (H : Msg → Msg) (run : R → Trace) (D : Deploy Coin Col T) (s : Stmt)
    (os : List Opening) :
    ∀ o ∈ ((checkOpens (R := R) D s os).eval H run).1, ((openOC D.path s o).eval H).1 = true ∧
      ∀ q ∈ ((openOC D.path s o).eval H).2, q ∈ ownLog H run (checkOpens (R := R) D s os) := by
  intro o ho
  obtain ⟨o', -, hc, hq⟩ := flatMapRP_mem H run _ os o ho
  rw [(RP.eval_bind H run _ _).1, eval_ofOC] at hc
  rw [ownLog_bind, ownLog_ofOC] at hq
  simp only [RP.eval] at hc
  split_ifs at hc with hok
  · obtain rfl := List.mem_singleton.1 hc
    exact ⟨hok, fun q h => hq q (List.mem_append_left _ h)⟩
  · simp at hc

section E

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem inner_mem (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg) (s : Stmt)
    (r : Fin (D.V.k x) → Coin) (v : Seen) (ρ : Fin (D.innerWaits x) → Fin (D.innerCap x) → Fin (D.V.k x) → Coin)
    (run : (Fin (D.V.k x) → Coin) → Trace) :
    ∀ o ∈ ((inner H D decode x s r v ρ).eval H run).1, ((openOC D.path s o).eval H).1 = true ∧
      ∀ q ∈ ((openOC D.path s o).eval H).2, q ∈ ownLog H run (inner H D decode x s r v ρ) := by
  intro o ho
  unfold inner at ho ⊢
  rcases hsv : D.session v with _ | ⟨xS, vS⟩
  · simp [hsv, RP.eval] at ho
  · simp only [hsv] at ho ⊢
    rw [(RP.eval_bind H run _ _).1] at ho
    rw [ownLog_bind, ownLog_of_noHash H run (NoHash.firstWait _ _ _ _ _), List.nil_append]
    revert ho
    cases ((firstWait (D.innerK x) D.sessionCols (Flock.assemble H D.flock.parse decode D.flock.read xS)
        (Flock.check H D.flock.parse xS) ((List.finRange (D.innerWaits x)).map fun i =>
          (List.finRange (D.innerCap x)).map fun j => atDraw r (ρ i j))).eval H run).1 with
    | none => intro ho; simp [RP.eval] at ho
    | some wF => exact checkOpens_mem H run D s _ o

/-- **The openings `E` assembles** on an accepted judged run of a parsed statement `s`: those its sessions'
extractions recovered and checked. -/
noncomputable def Eopens (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) (x : Msg)
    (s : Stmt) (run : (Fin (D.V.k x) → Coin) → Trace) (c : (E H D decode).C x) : List Opening :=
  ((flatMapRP (fun p => inner H D decode x s p.2.1 p.2.2 (c.2 p.1))
    ((List.finRange (D.k x)).zip ((waitAccC (D.k x) ((List.finRange (D.cap x)).map c.1) []).eval H run).1)).eval
      H run).1

/-- On an accepted judged run of a parsed statement, `E` returns the assembly of `Eopens`. -/
theorem E_eval_acc (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Stmt) (hs : D.parse x = some s) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin)
    (c : (E H D decode).C x) (hacc : (run H P D.V x ω r₀).out = .acc) :
    ((E H D decode).eval H P x ω r₀ c).1 = some (assemble s (Eopens H D decode x s (run H P D.V x ω) c)) := by
  have hseen : (run H P D.V x ω r₀).seen.out = .acc := hacc
  simp only [Extractor.eval, E, RP.eval, hs, hseen, ↓reduceIte]
  rw [(RP.eval_bind _ _ _ _).1, (RP.eval_bind _ _ _ _).1]
  rfl

/-- **Each opening `E` assembles is accepted by `openOC`, and its check's calls are among `E`'s own.** -/
theorem Eopens_checked (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type}
    (P : Prover Ω) (x : Msg) (s : Stmt) (hs : D.parse x = some s) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin)
    (c : (E H D decode).C x) (hacc : (run H P D.V x ω r₀).out = .acc) :
    ∀ o ∈ Eopens H D decode x s (run H P D.V x ω) c, ((openOC D.path s o).eval H).1 = true ∧
      ∀ q ∈ ((openOC D.path s o).eval H).2, q ∈ ownLog H (run H P D.V x ω) ((E H D decode).ext x r₀ c) := by
  intro o ho
  obtain ⟨p, -, hc, hq⟩ := flatMapRP_mem H _ _ _ o ho
  obtain ⟨hok, hlog⟩ := inner_mem H D decode x s _ _ _ _ o hc
  refine ⟨hok, fun q h => ?_⟩
  have hseen : (run H P D.V x ω r₀).seen.out = .acc := hacc
  simp only [E, ownLog, hs, hseen, ↓reduceIte]
  rw [ownLog_bind, ownLog_bind, ownLog_of_noHash H _ (NoHash.waitAccC _ _ _), List.nil_append]
  exact List.mem_append_left _ (hq q (hlog q h))

/-- Two of the openings `E` assembles open one row to different values. -/
def EDisagree (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω)
    (x : Msg) (ω : Ω) (c : (E H D decode).C x) : Prop :=
  ∃ s, D.parse x = some s ∧ ∃ o ∈ Eopens H D decode x s (run H P D.V x ω) c,
    ∃ o' ∈ Eopens H D decode x s (run H P D.V x ω) c, o.row = o'.row ∧ o.vals ≠ o'.vals

/-- **Disagreeing openings are an own-call collision**, when the path check binds. -/
theorem E_ownColl_of_disagree {H : Msg → Msg} {D : Deploy Coin Col T} (hpath : PathBinds H D.path)
    (decode : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (D.V.k x) → Coin)
    (c : (E H D decode).C x) (hacc : (run H P D.V x ω r₀).out = .acc) (hd : EDisagree H D decode P x ω c) :
    OwnColl H (E H D decode) P x ω r₀ c := by
  obtain ⟨s, hs, o, ho, o', ho', hrow, hne⟩ := hd
  obtain ⟨h1, l1⟩ := Eopens_checked H D decode P x s hs ω r₀ c hacc o ho
  obtain ⟨h2, l2⟩ := Eopens_checked H D decode P x s hs ω r₀ c hacc o' ho'
  refine (openOC_binds hpath s o o' hrow h1 h2 hne).mono fun q hq => ?_
  rcases List.mem_append.1 hq with hq | hq
  · exact l1 q hq
  · exact l2 q hq

/-- **The binding step of the reduction**: when the path check binds, `KSLin` at `β = 1` and `Bext` holds for the
block's extractor at every relation once each accepted run whose extraction fails either falls in an event `Q` that
`κ + √(γ · Pr[B])` bounds or has assembled openings that disagree. -/
theorem KSLin_of_disagree {H : Msg → Msg} {D : Deploy Coin Col T} (hpath : PathBinds H D.path)
    (decode : List (ℕ × Col) → List T) (Rl : Rel Wit) (κ γ : Msg → ℝ)
    (Q : ∀ (x : Msg) (Ω : Type), Prover Ω → Ω × (Fin (D.V.k x) → Coin) × (E H D decode).C x → Prop)
    (hsplit : ∀ x (Ω : Type) (P : Prover Ω) (s : Ω × (Fin (D.V.k x) → Coin) × (E H D decode).C x),
      (run H P D.V x s.1 s.2.1).out = .acc → ¬ Rl.ext x ((E H D decode).eval H P x s.1 s.2.1 s.2.2).1 →
        Q x Ω P s ∨ EDisagree H D decode P x s.1 s.2.2)
    (hQ : ∀ x (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (Q x Ω P) ≤ κ x + Real.sqrt (γ x * PrB H D.V P x)) :
    KSLin H D.V Rl (E H D decode) κ γ (fun _ => 1) (Bext H D.V (E H D decode)) :=
  KSLin_of_ownColl H D.V Rl (E H D decode) κ γ Q (fun x Ω P s hacc hR => (hsplit x Ω P s hacc hR).imp_right
    fun hd => E_ownColl_of_disagree hpath decode P x s.1 s.2.1 s.2.2 hacc hd) hQ

end E

end Claims.Sampling
