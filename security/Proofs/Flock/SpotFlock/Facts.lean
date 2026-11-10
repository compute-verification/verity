module

public import Proofs.Flock.SpotFlock.Code
public import Proofs.Flock.Claims.Sampling.KS
public import Proofs.Flock.Bridge.Members

/-!
# SPOT[Flock]'s two-stage code: the facts its order gives

* `stepWith_fst_take`, `run_fst_take`: a run's prover messages through round `m` read only its coins before round `m`;
  the verifier's answer at round `m` may read round `m`'s coin, the prover's message there doesn't.
* `iroots_fst`: the interior roots a run shows are read off its first two prover messages.
* `interiorFixed`: fresh second-stage coins keep them (`Code.InteriorFixed` at `SpotFlock.code h`): the interior
  registration is the prover's message at round 1, before the second stage's coin.
* `topoOk_sound`, `partitionsOk_sound`, `oneStageOk_sound`, `scopeOk_sound`, `filesOk_sound`, `zkOk_sound`: the
  verifier's decisions are sound for what they decide.
* `stepWith_acc_at`, `run_acc_at`: an accepting run passed every round before the verifier could accept.
* `refusesTwo`: an accepting run's statement is in the two-stage scope (`Code.RefusesTwo` at `SpotFlock.code h`), since
  round 2 checks the design evaluates and the design evaluates only in scope.
* `regMatchesTwo`: the registration is the statement's two-stage design (`RegMatchesTwo` at `SpotFlock.code h`), since
  the design evaluates only where the replay layout fits the members' files, whose replay partition has the layout's
  shape (`shape_filesReplay`) and whose cells are the registration's blocks (`Flock.Bridge.Members.cell_filesReplay`).
* `sessionsIn`: every session is in Flock's masked scope with at most the code's tables and the reference statement's
  shapes (`SessionsIn`'s clauses at `SpotFlock.code h`), since the reference statement is in scope (`Held.xS₀_ok`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Finset
open Properties.Spot (slotPartition slotDesign bitsProgs membersPart filesReplay filesBlockOf)

/-- A run appends to its transcript: one more round puts the prover's message and the verifier's answer after `t`'s. -/
theorem stepWith_succ_msgs {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out)
    (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript) :
    ∃ ext, (stepWith H P x ω vnext (n + 1) hist as t).msgs =
      t.msgs ++ (((P.next x hist ω).eval H).1, ((vnext (as ++ [((P.next x hist ω).eval H).1])).eval H).1) :: ext := by
  simp only [stepWith]
  split
  · obtain ⟨ext, he⟩ := stepWith_msgs_prefix H P x ω vnext n _ _ _
    exact ⟨ext, by rw [he]; simp⟩
  · exact ⟨[], by simp⟩
  · exact ⟨[], by simp⟩

/-- **The prover's messages through round `m` read only the verifier's answers before it**: two verifiers that answer
alike on at most `m` prover messages give runs whose first `m + 1` prover messages agree. -/
theorem stepWith_fst_take {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (vnext vnext' : List Msg → OC Out) (m : ℕ) (hv : ∀ as : List Msg, as.length ≤ m → vnext as = vnext' as) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), as.length = t.msgs.length →
      t.msgs.length ≤ m →
      ((stepWith H P x ω vnext n hist as t).msgs.map Prod.fst).take (m + 1) =
        ((stepWith H P x ω vnext' n hist as t).msgs.map Prod.fst).take (m + 1) := by
  intro n
  induction n with
  | zero => intro hist as t _ _; rfl
  | succ n ih =>
    intro hist as t hlen hm
    rcases Nat.lt_or_ge t.msgs.length m with hlt | hge
    · have hv' := hv (as ++ [((P.next x hist ω).eval H).1]) (by simp; omega)
      simp only [stepWith]
      rw [hv']
      split
      · exact ih _ _ _ (by simp [hlen]) (by simp; omega)
      · rfl
      · rfl
    · have hm' : t.msgs.length = m := le_antisymm hm hge
      obtain ⟨e, he⟩ := stepWith_succ_msgs H P x ω vnext n hist as t
      obtain ⟨e', he'⟩ := stepWith_succ_msgs H P x ω vnext' n hist as t
      rw [he, he']
      subst hm'
      simp [List.take_append]

/-- **A run's prover messages through round `m` read only its coins before round `m`.** -/
theorem run_fst_take {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (m : ℕ) (r r' : Fin (V.k x) → Coin) (h : ∀ t : Fin (V.k x), t.val < m → r t = r' t) :
    ((run H P V x ω r).msgs.map Prod.fst).take (m + 1) = ((run H P V x ω r').msgs.map Prod.fst).take (m + 1) := by
  unfold run
  refine stepWith_fst_take H P x ω _ _ m (fun as has => ?_) _ _ _ _ rfl (Nat.zero_le _)
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' : i < m := lt_of_lt_of_le (List.mem_range.1 hi) has
  unfold coinSeq
  split
  · exact h _ hi'
  · rfl

/-- The interior roots of replay block `ρ` the round-1 message states, from a run's prover messages. -/
def irootsAt (ms : List Msg) (ρ : ℕ) : List Msg :=
  match ms with
  | _ :: m₁ :: _ => ((interiorsOf m₁).bind fun it => (it.roots.find? (·.1 == ρ)).map (·.2)).getD []
  | _ => []

/-- **The interior roots are read off the first two prover messages.** -/
theorem iroots_fst (x : Msg) (v : Seen) (ρ : ℕ) :
    iroots x v ρ = irootsAt ((v.msgs.map Prod.fst).take 2) ρ := by
  unfold iroots irootsAt
  rcases v with ⟨ms, out⟩
  match ms with
  | [] => rfl
  | [_] => rfl
  | _ :: _ :: _ => rfl

/-- **Interior fixed** at SPOT[Flock]'s code (`Code.InteriorFixed`): fresh second-stage coins keep the interior roots a
run shows, since the prover states them at round 1, reading only round 0's coin. -/
theorem interiorFixed (h : Held) (H : Msg → Msg) :
    ∀ x (Ω : Type) (P : Prover Ω) ω r d v,
      iroots x (run H P (code h).V x ω (forkOn ((code h).stage2 x) r d)).seen v =
        iroots x (run H P (code h).V x ω r).seen v := by
  intro x Ω P ω r d v
  rw [iroots_fst, iroots_fst]
  congr 1
  refine run_fst_take H P (code h).V x ω 1 _ _ fun t ht => ?_
  have hn : t ∉ (code h).stage2 x := by
    intro hmem
    have hlt : (order h x).first.val < t.val := (mem_filter.1 hmem).2
    have : (order h x).first.val = 0 := rfl
    omega
  simp [forkOn, hn]

/-! ## The decisions are sound -/

theorem topoOk_sound {C : Prog} (hC : topoOk C = true) : C.Topological := by
  unfold topoOk at hC
  simp only [Bool.and_eq_true, List.all_eq_true] at hC
  refine ⟨fun g x hx i hi => ?_, fun o ho => of_decide_eq_true (hC.2 o ho)⟩
  have hg : g < C.gates.length := (List.getElem?_eq_some_iff.1 hx).1
  have h1 : ∀ i ∈ x.reads, decide (i < g) = true := by
    have := hC.1 g (List.mem_range.2 hg)
    rw [hx] at this
    exact this
  exact of_decide_eq_true (h1 i hi)

theorem partitionsOk_sound {P : Partition} {C : Prog} (hP : partitionsOk P C = true) : P.Partitions C := by
  unfold partitionsOk at hP
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at hP
  obtain ⟨⟨⟨⟨hnd, hlt⟩, hcov⟩, hne⟩, hw⟩ := hP
  refine ⟨of_decide_eq_true hnd, fun g hg => of_decide_eq_true (hlt g hg), fun g hg hc hu => ?_,
    fun u hu => ?_, fun u hu => ?_⟩
  · have hg' := hcov g hg
    simp only [Bool.or_eq_true, decide_eq_true_eq] at hg'
    rcases hg' with (hk | hk) | hk
    · exfalso
      unfold constAt at hk
      split at hk
      · exact hc _ ‹_›
      · cases hk
    · exact absurd hk hu
    · simpa using hk
  · have := hne u hu
    simpa using this
  · have := hw u hu
    cases hcl : unitClass C u with
    | none => simp [hcl] at this
    | some w =>
      rw [hcl] at this
      exact ⟨w, by simp [hcl], of_decide_eq_true this⟩

theorem oneStageOk_sound {ns : List ℕ} {o : OneStage} (h : oneStageOk ns o = true) : o.WellFormed ns := by
  cases o <;> simp_all [oneStageOk, OneStage.WellFormed]

theorem scopeOk_sound {C : Prog} {Pt : Partition} {d : Design.Two} (hs : scopeOk C Pt d = true) (s : AuditStatement)
    (hC : s.prog = C) : InScopeTwo Pt d s := by
  unfold scopeOk at hs
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq] at hs
  obtain ⟨⟨⟨⟨⟨⟨⟨htopo, hpart⟩, hnd⟩, hrep⟩, hlen⟩, hrules⟩, hklen⟩, hk⟩ := hs
  subst hC
  exact ⟨topoOk_sound htopo, partitionsOk_sound hpart, of_decide_eq_true hnd, fun v hv => hrep v hv,
    ⟨hlen, fun p hp => oneStageOk_sound (hrules p hp)⟩, hklen, fun k hk' => of_decide_eq_true (hk k hk')⟩

theorem filesOk_sound {g : RegTwo} {ms : Spot.Bits.Members} {l : Spot.Slots.Layout} {k : List ℕ}
    (hc : filesOk g ms l k = true) : g.ns.size = ms.length ∧ (∀ t < ms.length, k.getD t 0 = g.ks.getD t 0) ∧
      Flock.Bridge.Members.FilesFit ms g.ns l ∧ ∀ ρ b, l.blocks[ρ]? = some b → b.unit = some ρ →
        ∀ t < ms.length, ∀ p < g.ns.getD t 0, g.blockOf ρ t p = filesBlockOf ms g.ns ρ t p := by
  unfold filesOk at hc
  rw [Bool.and_eq_true, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.all_eq_true] at hc
  obtain ⟨⟨hT, hk⟩, hc⟩ := hc
  have hslot : ∀ ρ b, l.blocks[ρ]? = some b → ∀ u, b.unit = some u → u = ρ ∧ ∀ t, t < ms.length →
      (ρ + 1) * g.ns.getD t 0 ≤ ((ms[t]?).map (·.2.size)).getD 0 ∧
        ∀ p, p < g.ns.getD t 0 → g.blockOf ρ t p = filesBlockOf ms g.ns ρ t p := by
    intro ρ b hb u hu
    have h1 := hc ρ (List.mem_range.2 (Array.getElem?_eq_some_iff.1 hb).1)
    simpa only [hb, hu, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, List.all_eq_true, List.mem_range] using h1
  refine ⟨hT, fun t ht => by simpa using hk t (List.mem_range.2 ht), ?_,
    fun ρ b hb hbu t ht p hp => ((hslot ρ b hb ρ hbu).2 t ht).2 p hp⟩
  intro ρ b hb u hu
  have hu' := (hslot ρ b hb u hu).1
  refine ⟨hu', fun t ht => ?_⟩
  have := ((hslot ρ b hb u hu).2 t ht).1
  rwa [List.getElem?_eq_getElem ht, Option.map_some, Option.getD_some] at this

theorem zkOk_sound {xS : Msg} (hz : zkOk xS = true) : ∃ s spec sts, FlockDeploy.setupOf true xS = some (s, spec, sts) ∧
    0 < sts.size ∧ sts.size ≤ 200 ∧ spec.points ≤ 2 ∧
    ∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35 ∧ st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64 := by
  unfold zkOk at hz
  cases hs : FlockDeploy.setupOf true xS with
  | none => simp [hs] at hz
  | some p =>
    obtain ⟨s, spec, sts⟩ := p
    simp only [hs, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hz
    obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := hz
    exact ⟨s, spec, sts, by simp [hs], h1, h2, h3, fun st hst => by
      obtain ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩ := h4 st hst
      exact ⟨a, b, c, d, e⟩⟩

theorem ok_of_toOption {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e <;> simp_all [Except.toOption]

/-! ## An accepting run passed round 2 -/

/-- **An accepting run passed round `j`**: where the verifier accepts no `j` or fewer messages, rounds that end
accepting had the verifier answer some `j` messages with other than a refusal. -/
theorem stepWith_acc_at {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out)
    (j : ℕ) (hj : ∀ bs : List Msg, bs.length ≤ j → ((vnext bs).eval H).1 ≠ .acc) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Transcript), as.length < j →
      (stepWith H P x ω vnext n hist as t).out = .acc →
        ∃ bs : List Msg, bs.length = j ∧ ((vnext bs).eval H).1 ≠ .rej := by
  intro n
  induction n with
  | zero => intro hist as t _ h; simp [stepWith] at h
  | succ n ih =>
    intro hist as t hlt h
    simp only [stepWith] at h
    split at h
    · rename_i m hm
      rcases Nat.lt_or_ge (as.length + 1) j with hl | hl
      · exact ih _ _ _ (by simp; omega) h
      · exact ⟨as ++ [((P.next x hist ω).eval H).1], by simp; omega, by simp [hm]⟩
    · rename_i hm
      exact absurd hm (hj _ (by simp; omega))
    · simp at h

theorem run_acc_at {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) (j : ℕ) (hj0 : 0 < j)
    (hj : ∀ as cs, cs.length = as.length → as.length ≤ j → ((V.next x as cs).eval H).1 ≠ .acc)
    (h : (run H P V x ω r).out = .acc) :
    ∃ as cs, as.length = j ∧ cs.length = j ∧ ((V.next x as cs).eval H).1 ≠ .rej := by
  obtain ⟨bs, hbs, hne⟩ := stepWith_acc_at H P x ω (fun as => V.next x as ((List.range as.length).map (coinSeq r))) j
    (fun bs hb => hj _ _ (by simp) hb) (V.k x) [] [] ⟨[], [], [], .rej⟩ hj0 h
  exact ⟨bs, _, hbs, by simp [hbs], hne⟩

variable (h : Held)

/-- The verifier accepts nothing in rounds 0 to 2. -/
theorem next_short (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin)
    (hl : cs.length = as.length) (h3 : as.length ≤ 3) : ((next h x as cs).eval H).1 ≠ .acc := by
  rcases as with _ | ⟨a0, _ | ⟨a1, _ | ⟨a2, _ | ⟨a3, as⟩⟩⟩⟩ <;>
    rcases cs with _ | ⟨c0, _ | ⟨c1, _ | ⟨c2, _ | ⟨c3, cs⟩⟩⟩⟩ <;>
    simp only [List.length_cons, List.length_nil] at hl h3 <;> (try omega) <;>
    unfold next <;> simp only [OC.eval] <;> (try split) <;> simp

/-! ## The statement is in scope -/

theorem parse_prog {x : Msg} {s : AuditStatement} (hs : parse h x = some s) : s.prog = bitsProgs h.ms := by
  unfold parse at hs
  cases hd : decode x with
  | none => simp [hd] at hs
  | some st => rw [hd] at hs; cases hs; rfl

theorem partition_eq {s : AuditStatement} {Pt : Partition} (hP : partition h s = some Pt) : Pt = membersPart h.ms :=
  (Option.some.inj hP).symm

theorem two_eq {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    ∃ l, Spot.Slots.layout h.reg.first.program h.reg.first.query = .ok l ∧ replayDesign h = some d.replay ∧
      d = ⟨filesReplay h.ms h.reg.ns l, d.replay, h.k⟩ ∧ scopeOk (bitsProgs h.ms) (membersPart h.ms) d = true ∧
      filesOk h.reg h.ms l h.k = true := by
  unfold two at hd
  cases hl : (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption with
  | none => simp [hl] at hd
  | some l =>
    cases hr : replayDesign h with
    | none => simp [hl, hr] at hd
    | some dr =>
      simp only [hl, hr] at hd
      split at hd
      · rename_i hc
        cases hd
        rw [Bool.and_eq_true] at hc
        exact ⟨l, ok_of_toOption hl, by simp [hr], rfl, hc.1, hc.2⟩
      · cases hd

/-- **The statement round 2 takes is in scope**: it parses, its partition and design evaluate, and so the two-stage
scope holds. -/
theorem stmtOk_scope {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} (hok : stmtOk h x m₁ xS c₀ c₁ = true) :
    ∃ s Pt d, parse h x = some s ∧ partition h s = some Pt ∧ two h s = some d ∧ InScopeTwo Pt d s := by
  unfold stmtOk at hok
  cases hs : parse h x with
  | none => simp [hs] at hok
  | some s =>
    simp only [hs, Bool.and_eq_true] at hok
    obtain ⟨⟨⟨⟨hP, hd⟩, -⟩, -⟩, -⟩ := hok
    obtain ⟨Pt, hPt⟩ := Option.isSome_iff_exists.1 hP
    obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 hd
    obtain rfl := partition_eq h hPt
    obtain ⟨l, -, -, -, hsc, -⟩ := two_eq h hd
    exact ⟨s, _, d, by simp [hs], hPt, hd, scopeOk_sound hsc s (parse_prog h hs)⟩

/-- **Refuses at two stages** at SPOT[Flock]'s code (`Code.RefusesTwo`): an accepting run passed round 2, whose check
(`stmtOk`) puts its statement in the two-stage scope. -/
theorem refusesTwo (H : Msg → Msg) :
    ∀ x (Ω : Type) (P : Prover Ω) ω r, (run H P (code h).V x ω r).out = .acc →
      ∃ s Pt d, (code h).parse x = some s ∧ (code h).partition s = some Pt ∧ two h s = some d ∧ InScopeTwo Pt d s := by
  intro x Ω P ω r hacc
  obtain ⟨as, cs, has, hcs, hne⟩ :=
    run_acc_at H P (code h).V x ω r 3 (by omega) (fun as cs hl h3 => next_short h H x as cs hl h3) hacc
  obtain ⟨a₀, m₁, xS, rfl⟩ := List.length_eq_three.1 has
  obtain ⟨c₀, c₁, c₂, rfl⟩ := List.length_eq_three.1 hcs
  change ((next h x [a₀, m₁, xS] [c₀, c₁, c₂]).eval H).1 ≠ .rej at hne
  cases hok : stmtOk h x m₁ xS c₀ c₁ with
  | false =>
    unfold next at hne
    simp [OC.eval, hok] at hne
  | true => exact stmtOk_scope h hok

/-- **The members' replay partition has the layout's shape**: its classes' strata are the layout partition's
(`slotPartition`), size for size. -/
theorem shape_filesReplay (ms : Spot.Bits.Members) (ns : Array ℕ) (l : Spot.Slots.Layout) :
    (filesReplay ms ns l).classes.map strataSizes = (slotPartition l).classes.map strataSizes := by
  simp [filesReplay, slotPartition, strataSizes, Function.comp_def]

/-- **The registration is the statement's two-stage design** at SPOT[Flock]'s code (`RegMatchesTwo`): the design
evaluates only where the replay layout fits the members' files (`filesOk`); their replay partition has the layout's
shape, request `ρ`'s cell of template `t` is the registration's blocks `blockOf ρ t` on `[0, n_t)`, and an input slot's
cells are empty. -/
theorem regMatchesTwo :
    ∀ x s Pt d, (code h).parse x = some s → (code h).partition s = some Pt → two h s = some d → InScopeTwo Pt d s →
      ∃ g l, reg h x = some g ∧ Spot.Slots.layout g.first.program g.first.query = .ok l ∧
        d.part.classes.map strataSizes = (slotPartition l).classes.map strataSizes ∧
        slotDesign g.first.rule g.first.program g.first.query g.first.table = .ok d.replay ∧
        g.ns.size = Pt.strata.length ∧ (∀ t < Pt.strata.length, d.k.getD t 0 = g.ks.getD t 0) ∧
        ∀ ρ b, l.blocks[ρ]? = some b →
          (b.unit = some ρ ∧ ∀ t < Pt.strata.length, Set.InjOn (g.blockOf ρ t) (Set.Iio (g.ns.getD t 0)) ∧
              cell Pt d.part ρ t = (range (g.ns.getD t 0)).image (g.blockOf ρ t)) ∨
            (b.unit = none ∧ ∀ t, ∀ v ∈ cell Pt d.part ρ t, g.always v = true) := by
  intro x s Pt d hx hP hd hin
  have hx' : parse h x = some s := hx
  have hP' : partition h s = some Pt := hP
  obtain rfl := partition_eq h hP'
  obtain ⟨l, hl, hr, hdeq, -, hfo⟩ := two_eq h hd
  have hreg : reg h x = some h.reg := by
    unfold parse at hx'
    unfold reg
    cases hdx : decode x with
    | none => simp [hdx] at hx'
    | some _ => rfl
  have hdpart : d.part = filesReplay h.ms h.reg.ns l := congrArg Design.Two.part hdeq
  have hdk : d.k = h.k := congrArg Design.Two.k hdeq
  have hlen : (membersPart h.ms).strata.length = h.ms.length := by
    simp [Flock.Bridge.Members.strata_membersPart]
  obtain ⟨hT, hk, hfit, hblk⟩ := filesOk_sound hfo
  have hnd := hin.2.1.1
  have hne := hin.2.1.2.2.2.1
  refine ⟨h.reg, l, hreg, hl, by rw [hdpart]; exact shape_filesReplay h.ms h.reg.ns l,
    ok_of_toOption hr, by rw [hT, hlen], fun t ht => by rw [hdk]; exact hk t (hlen ▸ ht), fun ρ b hb => ?_⟩
  rcases hbu : b.unit with _ | u
  · refine Or.inr ⟨rfl, fun t v hv => ?_⟩
    rw [hdpart, Flock.Bridge.Members.cell_filesReplay_input h.ms hnd hne hfit hb hbu] at hv
    simp at hv
  · have hu : u = ρ := (hfit ρ b hb u hbu).1
    rw [hu] at hbu
    refine Or.inl ⟨by rw [hu], fun t ht => ?_⟩
    have ht' : t < h.ms.length := hlen ▸ ht
    have heq : Set.EqOn (h.reg.blockOf ρ t) (filesBlockOf h.ms h.reg.ns ρ t) (Set.Iio (h.reg.ns.getD t 0)) :=
      fun p hp => hblk ρ b hb hbu t ht' p hp
    rw [hdpart]
    refine ⟨fun a ha a' ha' he => Flock.Bridge.Members.injective_filesBlockOf h.ms h.reg.ns ρ t ?_, ?_⟩
    · rw [← heq ha, ← heq ha']
      exact he
    · rw [Flock.Bridge.Members.cell_filesReplay h.ms hnd hne hfit hb hbu ht']
      exact Finset.image_congr fun p hp => (heq (Finset.mem_range.1 (Finset.mem_coe.1 hp))).symm

/-! ## The sessions are in scope -/

/-- **Every session is in Flock's masked scope** at SPOT[Flock]'s code (`SessionsIn`'s clauses, `FlockDeploy.ZkIn`
unfolded): with at most the code's tables and the reference statement's shapes, the reference statement being in
scope (`Held.xS₀_ok`). -/
theorem sessionsIn :
    ∀ x u v xS vS, (code h).session x u v = some (xS, vS) →
      (∃ s spec sts, FlockDeploy.setupOf true xS = some (s, spec, sts) ∧ 0 < sts.size ∧ sts.size ≤ 200 ∧
        spec.points ≤ 2 ∧
        (∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35 ∧ st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64) ∧
        sts.size ≤ (code h).tables x) ∧ shapeOf xS = shapeOf h.xS₀ := by
  intro x u v xS vS hv
  have hok : zkOk xS = true ∧ shapeOf xS = shapeOf h.xS₀ := by
    change session h v = some (xS, vS) at hv
    unfold session at hv
    split at hv
    · split at hv
      · rename_i hs
        cases hv
        simp only [sessionOk, Bool.and_eq_true, beq_iff_eq] at hs
        exact hs.1.1
      · cases hv
        exact ⟨h.xS₀_ok, rfl⟩
    · cases hv
      exact ⟨h.xS₀_ok, rfl⟩
  obtain ⟨s, spec, sts, hset, h1, h2, h3, h4⟩ := zkOk_sound hok.1
  refine ⟨⟨s, spec, sts, hset, h1, h2, h3, h4, ?_⟩, hok.2⟩
  change sts.size ≤ (shapeOf h.xS₀).length
  rw [← hok.2]
  unfold shapeOf
  rw [hset]
  simp

end SpotFlock
