module

public import Proofs.Flock.RecGeneric.AtFlockReruns

/-! L1 at Flock: `B`'s, `FC`'s and `F0`'s plans at the composite's reruns against `Claims.Recursion`'s finders over `E_O`'s samples. -/

@[expose] public section

namespace Claims.RecGeneric

open Claims Claims.Recursion

section Exec

variable {Coin Ω C : Type} {n : ℕ} (m : RunModel Coin Ω n C) (ω : Ω) (us : ℕ → (Fin n → Coin) × C)

open Classical in
theorem capLoop_exec_pick (j : ℕ) (opens : (ℕ → Coin) → Bool) (g : (Fin n → Coin) × C → Prop)
    (hg : ∀ u, g u ↔ m.acc ω (fork (m.cut j) (us 0).1 u.1, u.2) ∧
      opens (m.coins (fork (m.cut j) (us 0).1 u.1, u.2)) = true) :
    ∀ (N i : ℕ), ∃ e, ((capLoop j opens N i).exec m ω us).2.2 = [.fresh 0, e] ∧
      m.named us e = (fork (m.cut j) (us 0).1 ((firstPick g N fun t : Fin N => us (i + t.val)).getD (us 0)).1,
        ((firstPick g N fun t : Fin N => us (i + t.val)).getD (us 0)).2)
  | 0, _ => ⟨.fresh 0, by simp only [capLoop, Plan.exec],
      by simp only [RunModel.named, firstPick, Option.getD_none, fork_self, Prod.mk.eta]⟩
  | N + 1, i => by
    have htail : Fin.tail (fun t : Fin (N + 1) => us (i + t.val)) = fun t : Fin N => us (i + 1 + t.val) :=
      funext fun t => congrArg us (by simp only [Fin.val_succ]; omega)
    have hp : firstPick g (N + 1) (fun t : Fin (N + 1) => us (i + t.val)) =
        if g (us i) then some (us i) else firstPick g N fun t : Fin N => us (i + 1 + t.val) := by
      simp only [firstPick, Fin.val_zero, Nat.add_zero, htail]
    obtain ⟨e, he, hn⟩ := capLoop_exec_pick j opens g hg N (i + 1)
    by_cases hgi : g (us i)
    · obtain ⟨ha, ho⟩ := (hg _).1 hgi
      refine ⟨.fork j (.fresh 0) (.fresh i), ?_, ?_⟩
      · simp only [capLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ho, ↓reduceIte]
      · simp only [hp, hgi, ↓reduceIte, Option.getD_some, RunModel.named]
    · refine ⟨e, ?_, ?_⟩
      · by_cases ha : m.acc ω (fork (m.cut j) (us 0).1 (us i).1, (us i).2)
        · have ho : ¬ opens (m.coins (fork (m.cut j) (us 0).1 (us i).1, (us i).2)) = true :=
            fun ho => hgi ((hg _).2 ⟨ha, ho⟩)
          simp only [capLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ho, ↓reduceIte]
          exact he
        · simp only [capLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte]
          exact he
      · rw [hp, if_neg hgi]
        exact hn

open Classical in
theorem capPlan_search_pick (Cp : Caps Coin) (M : ℕ) (x : Msg) (q : Fin (Cp.width x)) (u₀ : (Fin n → Coin) × C)
    (hu₀ : us 0 = u₀) (vs : Fin M → (Fin n → Coin) × C) (hvs : ∀ t : Fin M, us (1 + t.val) = vs t) (c : Prop)
    (hc : c ↔ m.acc ω u₀ ∧ Cp.opens x q (m.coins u₀) = true) (g : (Fin n → Coin) × C → Prop)
    (hg : ∀ u, g u ↔ m.acc ω (fork (m.cut (Cp.round x q)) u₀.1 u.1, u.2) ∧
      Cp.opens x q (m.coins (fork (m.cut (Cp.round x q)) u₀.1 u.1, u.2)) = true)
    (φ : (Fin n → Coin) × C → (Fin n → Coin) × C) (hφ : ∀ v, φ v = (fork (m.cut (Cp.round x q)) u₀.1 v.1, v.2)) :
    ((((capPlan Cp M).plan x q).exec m ω us).2.2.flatMap fun e => m.log ω (m.named us e)) =
      if c then m.log ω u₀ ++ m.log ω (φ ((firstPick g M vs).getD u₀)) else [] := by
  subst hu₀
  obtain rfl : (fun t : Fin M => us (1 + t.val)) = vs := funext hvs
  by_cases hc' : c
  · obtain ⟨ha, ho⟩ := hc.1 hc'
    obtain ⟨e, he, hn⟩ := capLoop_exec_pick m ω us (Cp.round x q) (Cp.opens x q) g hg M 1
    have hex : (((capPlan Cp M).plan x q).exec m ω us).2.2 =
        ((capLoop (Cp.round x q) (Cp.opens x q) M 1).exec m ω us).2.2 := by
      simp only [capPlan, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ho, ↓reduceIte]
    rw [if_pos hc', hex, he, hφ]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, hn, RunModel.named]
  · rw [if_neg hc']
    by_cases ha : m.acc ω (us 0)
    · have ho : ¬ Cp.opens x q (m.coins (us 0)) = true := fun ho => hc' (hc.2 ⟨ha, ho⟩)
      simp only [capPlan, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ho, ↓reduceIte, Bool.false_eq_true,
        List.flatMap_nil]
    · simp only [capPlan, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte, List.flatMap_nil]

theorem waitLoop_exec_pick (cont : Option RunExp → Plan Coin) : ∀ (N i : ℕ),
    ((waitLoop cont N i).exec m ω us).2.2 =
      ((cont (firstPick (fun e => m.acc ω (m.named us e)) N fun t : Fin N => RunExp.fresh (i + t.val))).exec
        m ω us).2.2
  | 0, _ => by simp only [waitLoop, firstPick]
  | N + 1, i => by
    have htail : Fin.tail (fun t : Fin (N + 1) => RunExp.fresh (i + t.val)) =
        fun t : Fin N => RunExp.fresh (i + 1 + t.val) :=
      funext fun t => congrArg RunExp.fresh (by simp only [Fin.val_succ]; omega)
    by_cases ha : m.acc ω (us i)
    · simp only [waitLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte, firstPick, Fin.val_zero,
        Nat.add_zero]
    · simp only [waitLoop, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte, firstPick, Fin.val_zero,
        Nat.add_zero, htail]
      exact waitLoop_exec_pick cont N (i + 1)

theorem waitsPlan_exec_pick (R : ℕ) : ∀ (w base : ℕ) (found : List RunExp),
    ((waitsPlan R w base found).exec m ω us).2.2 =
      found ++ (List.finRange w).flatMap fun t =>
        (firstPick (fun e => m.acc ω (m.named us e)) R
          fun r : Fin R => RunExp.fresh (base + R * t.val + r.val)).toList
  | 0, _, found => by simp only [waitsPlan, Plan.exec, List.finRange_zero, List.flatMap_nil, List.append_nil]
  | w + 1, base, found => by
    simp only [waitsPlan, waitLoop_exec_pick, waitsPlan_exec_pick R w (base + R)]
    rw [List.finRange_succ, List.flatMap_cons, List.flatMap_map, List.append_assoc]
    refine congrArg (found ++ ·) (congrArg₂ (· ++ ·) ?_ ?_)
    · refine congrArg (fun f : Fin R → RunExp => (firstPick (fun e => m.acc ω (m.named us e)) R f).toList)
        (funext fun r => ?_)
      simp only [Fin.val_zero, Nat.mul_zero, Nat.add_zero]
    · refine congrArg (fun f => (List.finRange w).flatMap f) (funext fun t => ?_)
      refine congrArg (fun f : Fin R → RunExp => (firstPick (fun e => m.acc ω (m.named us e)) R f).toList)
        (funext fun r => congrArg RunExp.fresh ?_)
      simp only [Fin.val_succ, Nat.mul_add, Nat.mul_one]
      omega

open Classical in
theorem conflictPlan_search_pick (k : Msg → ℕ) (R : ℕ) (x : Msg) (z : Fin 1) (u₀ : (Fin n → Coin) × C)
    (hu₀ : us 0 = u₀) (ws : Fin (k x) → Fin R → (Fin n → Coin) × C)
    (hws : ∀ t r, us (1 + R * t.val + r.val) = ws t r) (c : Prop) (hc : c ↔ m.acc ω u₀)
    (g : (Fin n → Coin) × C → Prop) (hg : ∀ u, g u ↔ m.acc ω u) :
    ((((conflictPlan k R).plan x z).exec m ω us).2.2.flatMap fun e => m.log ω (m.named us e)) =
      if c then m.log ω u₀ ++ (List.finRange (k x)).flatMap fun t => ((firstPick g R (ws t)).map (m.log ω)).getD []
      else [] := by
  subst hu₀
  obtain rfl : m.acc ω = g := funext fun u => propext (hg u).symm
  by_cases hc' : c
  · have ha := hc.1 hc'
    have hex : (((conflictPlan k R).plan x z).exec m ω us).2.2 =
        ((waitsPlan R (k x) 1 [.fresh 0]).exec m ω us).2.2 := by
      simp only [conflictPlan, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte]
    have hmap : ∀ o : Option RunExp,
        (o.map fun e => m.log ω (m.named us e)) = (o.map (m.named us)).map (m.log ω) :=
      fun o => by cases o <;> rfl
    rw [if_pos hc', hex, waitsPlan_exec_pick, List.flatMap_append, flatMap_toList]
    refine congrArg₂ (· ++ ·) ?_ (congrArg (fun f => (List.finRange (k x)).flatMap f) (funext fun t => ?_))
    · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, RunModel.named]
    · rw [hmap, firstPick_map (m.named us) (fun e => m.acc ω (m.named us e)) (m.acc ω) (fun _ => Iff.rfl)]
      exact congrArg (fun f : Fin R → (Fin n → Coin) × C => ((firstPick (m.acc ω) R f).map (m.log ω)).getD [])
        (funext fun r => hws t r)
  · rw [if_neg hc']
    have ha : ¬ m.acc ω (us 0) := fun ha => hc' (hc.2 ha)
    simp only [conflictPlan, Plan.exec, RunModel.named, decide_eq_true_eq, ha, ↓reduceIte, List.flatMap_nil]

end Exec

section Finder

variable {Coin Ω C : Type} [Fintype Coin] [Nonempty Coin] [Fintype Ω] [Nonempty Ω] [Fintype C] [Nonempty C]
  {n : ℕ}

theorem RunFinder.finder_out (F : RunFinder Coin) (m : RunModel Coin Ω n C) (x : Msg) (h : 0 < F.sel x)
    (j : Fin (F.sel x)) (ω : Ω) (sl : Fin (F.slots x + 1) → (Fin n → Coin) × C) :
    (F.finder m x h).out (j, ω, sl) =
      findColl (((F.plan x j).exec m ω (slotRuns sl)).2.2.flatMap fun e => m.log ω (m.named (slotRuns sl) e)) :=
  rfl

end Finder

theorem unifMap_cons {J Ω U : Type} [Fintype J] [Fintype Ω] [Fintype U] (M : ℕ) :
    UnifMap (fun s : J × Ω × U × (Fin M → U) =>
      ((s.1, s.2.1, (Fin.cons s.2.2.1 s.2.2.2 : Fin (M + 1) → U)) : J × Ω × (Fin (M + 1) → U))) :=
  UnifMap.equiv ((Equiv.refl J).prodCongr ((Equiv.refl Ω).prodCongr (Fin.consEquiv fun _ : Fin (M + 1) => U)))

theorem getD_pick {U : Type} {g p : U → Prop} {M : ℕ} {vs : Fin M → U} {u₀ : U} (h₀ : p u₀) (hg : ∀ v, g v → p v) :
    p ((firstPick g M vs).getD u₀) := by
  cases hp : firstPick g M vs with
  | none => exact h₀
  | some v => exact hg v (firstPick_some hp)

section Fork

variable {Coin W Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col']
  (H : Msg → Msg) (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
  (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
  (P : Prover Ω) (x : Msg)

theorem forkPlan_PrAt_le (m : RunModel Coin Ω (D.V.k x) ((EO D I DO decO).C x))
    (hcut : ∀ j, j < I.V.k (xI D x) → m.cut j = j)
    (hlog : ∀ ω u e, e ∈ m.log ω u → e ∈ sampleCalls H D I DO decO P x ω u)
    (hT : ∀ ω u e, e ∈ sampleCalls H D I DO decO P x ω u → H e.1 = e.2) :
    (forkPlan (Coin := Coin) I.V.k).PrAt H m (xI D x) ≤ PrBinnerV H D I DO decO P x := by
  unfold RunFinder.PrAt PrBinnerV
  by_cases hk : 0 < I.V.k (xI D x)
  swap
  · rw [dif_neg (show ¬ 0 < (forkPlan (Coin := Coin) I.V.k).sel (xI D x) from hk), dif_neg hk]
  rw [dif_pos (show 0 < (forkPlan (Coin := Coin) I.V.k).sel (xI D x) from hk), dif_pos hk]
  unfold Finder.Pr
  refine ((UnifMap.equiv (binEquiv (Fin (I.V.k (xI D x))) Ω (Fin (D.V.k x) → Coin)
    ((EO D I DO decO).C x))).Pr _).symm.trans_le (Pr_mono fun s hs => ?_)
  obtain ⟨j, ω, r₀, r₁, c₀, c₁⟩ := s
  obtain ⟨a, b, hout, -⟩ := hs
  have hout' : findColl (m.log ω (r₀, c₀) ++ (m.log ω (fork (m.cut j.val) r₀ r₁, c₁) ++ [])) = some (a, b) := hout
  rw [hcut j.val j.isLt, List.append_nil] at hout'
  have hsub : ∀ e ∈ m.log ω (r₀, c₀) ++ m.log ω (fork j.val r₀ r₁, c₁),
      e ∈ sampleCalls H D I DO decO P x ω (r₀, c₀) ++ sampleCalls H D I DO decO P x ω (fork j.val r₀ r₁, c₁) := by
    intro e he
    rcases List.mem_append.1 he with he | he
    · exact List.mem_append_left _ (hlog ω _ e he)
    · exact List.mem_append_right _ (hlog ω _ e he)
  have hT' : ∀ e ∈ sampleCalls H D I DO decO P x ω (r₀, c₀) ++ sampleCalls H D I DO decO P x ω (fork j.val r₀ r₁, c₁),
      H e.1 = e.2 := fun e he => by
    rcases List.mem_append.1 he with he | he
    · exact hT ω _ e he
    · exact hT ω _ e he
  exact coll_of_isSome hT' (findColl_isSome_of_sub H hT' hsub (Option.isSome_iff_exists.2 ⟨_, hout'⟩))

end Fork

/-! ## The terms at the instance -/

section Instance

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

theorem forkPlansLeBinnerV (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Claims.Flock.Deploy Coin Col' T') : ForkPlansLeBinnerV H D Bd DO := by
  intro decI decO x hT hN Ω _ _ P
  have hcut : ∀ j, j < (Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI (D.record Bd.IVStar) x) →
      min j (Bd.DE.V.k (Claims.Recursion.xI (D.record Bd.IVStar) x)) = j :=
    fun j hj => Nat.min_eq_left hj.le
  have hTd := sampleCalls_true_at D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO (P := P) hT hN
  exact ⟨forkPlan_PrAt_le H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x
      ((compAt D Bd decI DO decO).innerModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x) hcut
      (fun _ _ _ he => rerunCalls_sub D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO he) hTd,
    forkPlan_PrAt_le H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x
      ((compAt D Bd decI DO decO).verModel ((Server.ofBound Bd).IVStar Bd.DE.V Bd.ans) H P x) hcut
      (fun _ _ _ he => verModel_log_sub D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO he) hTd⟩

theorem capForkI_self (D : Claims.Recursion.Deploy Coin) {W : Type} (I : Claims.Recursion.Inner Coin W)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) (x : Msg)
    (q : Fin (Cp.width (Claims.Recursion.xI D x))) (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) :
    capForkI D I DO decO Cp x q d d = d :=
  Prod.ext (fork_self _ _) rfl

theorem capPlan_coll (D : RecAt Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (Cp : Caps Coin) (H : Msg → Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ)
    (hW : 0 < Cp.width (Claims.Recursion.xI (D.record Bd.IVStar) x))
    (hxI : D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x)
    (q : Fin (Cp.width (Claims.Recursion.xI (D.record Bd.IVStar) x))) (ω : Ω)
    (d₀ : (Fin ((D.record Bd.IVStar).V.k x) → Coin) ×
      (EO (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (vs : Fin M → (Fin ((D.record Bd.IVStar).V.k x) → Coin) ×
      (EO (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    (∃ a b, ((capPlan Cp M).finder ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x)
        ((compAt D Bd decI DO decO).xI x) hW).out (q, ω, Fin.cons d₀ vs) = some (a, b) ∧ Collision H a b) ↔
      ∃ a b, (FCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M hW).out (q, ω, d₀, vs) =
        some (a, b) ∧ Collision H a b := by
  have hacc : ∀ d, ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x).acc ω d ↔
      accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω d :=
    innerModel_acc_iff D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO hxI ω
  rw [(RunFinder.finder_out (capPlan Cp M) ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x)
      ((compAt D Bd decI DO decO).xI x) hW q ω (Fin.cons d₀ vs)).trans
    (congrArg findColl (capPlan_search_pick ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x) ω
      (slotRuns (Fin.cons d₀ vs : Fin (M + 1) → _)) Cp M ((compAt D Bd decI DO decO).xI x) q d₀
      (slotRuns_cons_zero d₀ vs) vs (slotRuns_cons_one_add d₀ vs)
      (opensAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x q ω d₀)
      (and_congr (hacc d₀) Iff.rfl).symm
      (fun v => opensAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x q ω
        (capForkI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp x q d₀ v))
      (fun v => (and_congr (hacc _) Iff.rfl).symm)
      (capForkI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp x q d₀) fun _ => rfl))]
  by_cases ho : opensAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x q ω d₀
  · have hw : ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x).acc ω
        (capForkI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp x q d₀
          ((firstPick (fun v => opensAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x q ω
            (capForkI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp x q d₀ v)) M vs).getD d₀)) :=
      getD_pick (p := fun v => ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x).acc ω
          (capForkI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp x q d₀ v))
        (by
          show ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x).acc ω
            (capForkI (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp x q d₀ d₀)
          rw [capForkI_self]
          exact (hacc d₀).2 ho.1)
        fun v hv => (hacc _).2 hv.1
    rw [if_pos ho, innerModel_log_of_acc D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO ((hacc d₀).2 ho.1)]
    erw [innerModel_log_of_acc D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO hw]
    simp only [FCI, ho, ↓reduceIte]
  · rw [if_neg ho]
    simp [FCI, ho, findColl]

theorem capPlanIsFCI (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (Cp : Caps Coin) (hIV : IVStarOfBound Bd) : CapPlanIsFCI D Bd DO Cp := by
  intro hxI H decI decO x Ω _ _ P M
  rw [show (Server.ofBound Bd).IVStar Bd.DE.V Bd.ans = Bd.IVStar from hIV]
  unfold RunFinder.PrAt PrFCI
  by_cases hW : 0 < Cp.width (Claims.Recursion.xI (D.record Bd.IVStar) x)
  swap
  · rw [dif_neg (show ¬ 0 < (capPlan Cp M).sel ((compAt D Bd decI DO decO).xI x) from hW), dif_neg hW]
  rw [dif_pos (show 0 < (capPlan Cp M).sel ((compAt D Bd decI DO decO).xI x) from hW), dif_pos hW]
  unfold Finder.Pr
  refine ((unifMap_cons M).Pr _).symm.trans (Pr_congr fun s => ?_)
  obtain ⟨q, ω, d₀, vs⟩ := s
  exact capPlan_coll D Bd decI DO decO Cp H P x M hW (hxI x) q ω d₀ vs

theorem coll_iff_of_eq {H : Msg → Msg} {l l' : List (Msg × Msg)} (h : l = l') :
    (∃ a b, findColl l = some (a, b) ∧ Collision H a b) ↔ ∃ a b, findColl l' = some (a, b) ∧ Collision H a b := by
  rw [h]

theorem conflictPlan_coll (D : RecAt Coin) (Bd : Bound Coin Col T) (decI : List (ℕ × Col) → List T)
    (DO : Claims.Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') (H : Msg → Msg)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (R : ℕ)
    (hxI : D.xI x = Claims.Recursion.xI (D.record Bd.IVStar) x) (ω : Ω)
    (d₀ : (Fin ((D.record Bd.IVStar).V.k x) → Coin) ×
      (EO (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (ws : Fin (Bd.DE.k (Claims.Recursion.xI (D.record Bd.IVStar) x)) → Fin R →
      (Fin ((D.record Bd.IVStar).V.k x) → Coin) ×
        (EO (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    (∃ a b, ((conflictPlan Bd.DE.k R).finder ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x)
        ((compAt D Bd decI DO decO).xI x) Nat.one_pos).out
          ((default : Fin 1), ω, Fin.cons d₀ fun p => ws (finProdFinEquiv.symm p).1 (finProdFinEquiv.symm p).2) =
        some (a, b) ∧ Collision H a b) ↔
      ∃ a b, (F0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R).out (ω, d₀, ws) =
        some (a, b) ∧ Collision H a b := by
  have hacc : ∀ d, ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x).acc ω d ↔
      accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω d :=
    innerModel_acc_iff D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO hxI ω
  rw [(RunFinder.finder_out (conflictPlan Bd.DE.k R) ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x)
      ((compAt D Bd decI DO decO).xI x) Nat.one_pos (default : Fin 1) ω
      (Fin.cons d₀ fun p => ws (finProdFinEquiv.symm p).1 (finProdFinEquiv.symm p).2)).trans
    (congrArg findColl (conflictPlan_search_pick ((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x) ω
      (slotRuns (Fin.cons d₀ fun p => ws (finProdFinEquiv.symm p).1 (finProdFinEquiv.symm p).2 :
        Fin (Bd.DE.k (Claims.Recursion.xI (D.record Bd.IVStar) x) * R + 1) → _))
      Bd.DE.k R ((compAt D Bd decI DO decO).xI x) default d₀ (slotRuns_cons_zero _ _) ws (slotRuns_wait d₀ ws)
      (accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω d₀) (hacc d₀).symm
      (accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω) fun u => (hacc u).symm))]
  by_cases ha : accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω d₀
  · have hL : ∀ t : Fin (Bd.DE.k ((compAt D Bd decI DO decO).xI x)),
        (firstPick (accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω) R
          (ws t)).map (((compAt D Bd decI DO decO).innerModel Bd.IVStar H P x).log ω) =
        (firstPick (accAtI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω) R (ws t)).map
          fun d => (D.record Bd.IVStar).callsAt H x d.1
            (sampleWits H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω d) :=
      fun t => firstPick_map_congr (fun d hd =>
        innerModel_log_of_acc D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO ((hacc d).2 hd)) R (ws t)
    rw [if_pos ha, innerModel_log_of_acc D Bd.IVStar (Inner.ofFlock Bd.deploy decI) DO decO ((hacc d₀).2 ha)]
    simp only [F0I, ha, ↓reduceIte]
    refine coll_iff_of_eq (congrArg (fun l => (D.record Bd.IVStar).callsAt H x d₀.1
      (sampleWits H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x ω d₀) ++ l) ?_)
    exact congrArg (fun f => (List.finRange (Bd.DE.k ((compAt D Bd decI DO decO).xI x))).flatMap f)
      (funext fun t => congrArg (fun o => Option.getD o []) (hL t))
  · rw [if_neg ha]
    simp [F0I, ha, findColl]

theorem conflictPlanIsF0I (D : RecAt Coin) (Bd : Bound Coin Col T) (DO : Claims.Flock.Deploy Coin Col' T')
    (hIV : IVStarOfBound Bd) : ConflictPlanIsF0I D Bd DO := by
  intro hxI H decI decO x Ω _ _ P R
  rw [show (Server.ofBound Bd).IVStar Bd.DE.V Bd.ans = Bd.IVStar from hIV]
  unfold RunFinder.PrAt PrF0I
  rw [dif_pos (show 0 < (conflictPlan (Coin := Coin) Bd.DE.k R).sel ((compAt D Bd decI DO decO).xI x) from
    Nat.one_pos)]
  unfold Finder.Pr
  refine ((UnifMap.equiv (conflictEquiv Ω _ _ R)).Pr _).symm.trans (Pr_congr fun s => ?_)
  obtain ⟨ω, d₀, ws⟩ := s
  exact conflictPlan_coll D Bd decI DO decO H P x R (hxI x) ω d₀ ws

end Instance

end Claims.RecGeneric
