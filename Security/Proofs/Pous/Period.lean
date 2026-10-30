import Properties.Pous.Period
import Proofs.Pous.EndToEnd
import Proofs.Properties.Pous.Run
import Proofs.Accounting.Prob
import Proofs.Accounting.Capacity
import Proofs.Accounting.PerRunJoint

/-!
# Proof of PoUS's per-period guarantee, and that it is the joint's capacity guarantee

**The extractor passes** (`extractPasses`). On a certified capacity coin, `Instantiates` gives the memory challenger's
draw and the strategy's answers in the ring model, the challenged blocks of `C`; the certificate's `AnsweredOnTime`
is on time within `Δ + RTT` (`onTime_of`), so `DeviceModel` bounds each round's program by `(D, Q, L)`, and where the
account holds at most `B` bits at every tick, each round's state by `B + S_chip + c < S₀` bits. A state of fewer than
`S₀` bits survives `StateBits.pack` (`unpack_pack`), and a program within the budget runs unclipped (`clip_run`), so
the extractor's server answers round after round as the strategy does (`seqAnswers_clip`), and passes.

**The guarantee.** At a value `o` of the other audits' coins, a certificate `S` with `S · len > 0` and
`B = ⌈S · len⌉₊ − 1`: on a capacity coin where the verifier accepts with no break in the period's data, the period is
certified (`certified_of`, from `runCertifies`), and where the capacity account held fewer than `S · len` bit-ticks, it
held fewer than `S · len`, so at most `B`, bits at every tick of the window (`Accounting.Run.mem_le_held`).
`B < S · len`, so `B + S_chip + c < S₀`, and the extractor's server passes on every such coin; `endToEndRing` bounds its
pass by `δ + ε_max`. With `S · len ≤ 0` the event is empty. A run whose capacity verdict is that acceptance outside a
break then has `CapacitySoundPerRun`, and with it `CapacitySound` (`Accounting.Run.capacitySound_of_perRun`).
-/

namespace Pous.Period.SecurityProofs

open Pous.EndToEnd PousCheck PousCheck.Properties
open scoped ENNReal

set_option exponentiation.threshold 20000

theorem certified_of {ΩW ΩQ : Type} (P : Audits ΩW ΩQ)
    {ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    (ha : P.Accepts ω) (hb : ¬ P.Break ω) : P.Certified ω := by
  obtain ⟨hdec, hpub, haud, hsid, htr, hrun⟩ :=
    Pous.EndToEnd.SecurityProofs.runCertifies P.trusted (P.commitment (ω.1, ω.2.1)) (P.store (ω.1, ω.2.1))
      (P.pub (ω.1, ω.2.1)) (P.digests (ω.1, ω.2.1)) (P.session ω) P.sid (P.device ω) ha
  refine ⟨hdec, hpub, haud, hsid, htr, ?_⟩
  rcases hrun with h | h | h
  · exact h
  · exact absurd (Or.inl h) hb
  · exact absurd (Or.inr h) hb

/-- A certified period's signed record has every answer on time, at a limit within `Δ + RTT`: the session's limit is
the signed commitment's, which `AuditsSetup` bounds. -/
theorem onTime_of {ΩW ΩQ : Type} (P : Audits ΩW ΩQ)
    {ω : ΩW × ΩQ × ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)}
    (h : P.Certified ω) : P.OnTime ω := by
  obtain ⟨_, _, haud, _, _, run, hrun, hc, hot⟩ := h
  have hl : run.limitNs = (P.session ω).commitment.limitNs := congrArg MemoryChallenger.SessionCommitment.limitNs hc
  exact ⟨run, hrun, hc, hl.trans_le haud.2.2.2.2.1, hot⟩

theorem δ_nonneg : (0 : ℝ) ≤ (Params.δ : ℝ) := by
  have : (0 : ℚ) ≤ Params.δ := by norm_num [Params.δ]
  exact_mod_cast this

theorem εMax_ne_top : Params.εMax ≠ ⊤ := by
  unfold Params.εMax
  exact ENNReal.pow_ne_top (ENNReal.inv_ne_top.2 two_ne_zero)

/-! ## The extractor -/

theorem unpack_pack {n : ℕ} (l : List Bool) (hl : l.length < n) : StateBits.unpack (StateBits.pack (n := n) l) = l := by
  have hg : Nat.findGreatest (fun k => ∃ h : k < n, StateBits.pack (n := n) l ⟨k, h⟩ = true) n = l.length := by
    rw [Nat.findGreatest_eq_iff]
    refine ⟨hl.le, fun _ => ⟨hl, by simp [StateBits.pack]⟩, fun k hk hkn ⟨h, hb⟩ => ?_⟩
    simp only [StateBits.pack, show ¬ k < l.length from by omega, dite_eq_right, not_false_eq_true] at hb
    simp at hb
    omega
  unfold StateBits.unpack
  rw [hg]
  apply List.ext_getElem
  · simp; omega
  · intro k h1 h2
    simp only [List.getElem_take, List.getElem_ofFn, StateBits.pack]
    rw [dite_eq_left h2]

theorem clip_run {B w p : ℕ} [NeZero p] {α β : Type} (g : α → β) (d : β)
    (f : (ringRO B w p).Qry → (ringRO B w p).Ans) (P : OracleAlgorithm (ringRO B w p) α) :
    ∀ D Q L, P.mulDepth f ≤ D → P.keyWork f ≤ Q → P.ringWork f ≤ L → (OracleAlgorithm.clip g d D Q L P).run f = g (P.run f) := by
  induction P with
  | ret a => intro D Q L _ _ _; rfl
  | round qs k ih =>
    intro D Q L h1 h2 h3
    simp only [OracleAlgorithm.mulDepth, OracleAlgorithm.keyWork, OracleAlgorithm.ringWork] at h1 h2 h3
    have h : (if qs.any RingQry.isMul then 1 else 0) ≤ D ∧ qs.countP RingQry.isKey ≤ Q ∧
        qs.countP (fun q => !q.isKey) ≤ L := ⟨by omega, by omega, by omega⟩
    rw [OracleAlgorithm.clip, ite_eq_left h]
    exact ih (qs.map f) _ _ _ (by omega) (by omega) (by omega)

/-- **The packed, clipped strategy answers as the strategy does**, on rounds whose states have fewer than `S` bits and
whose programs stay within `(D, Q, L)`. -/
theorem seqAnswers_clip {B w p : ℕ} [NeZero p] {S D Q L ℓ Bn : ℕ} (f : (ringRO B w p).Qry → (ringRO B w p).Ans)
    (d : Bits ℓ × Bits S) (step : ℕ → List Bool → Fin Bn → OracleAlgorithm (ringRO B w p) (Bits ℓ × List Bool)) :
    ∀ (is : List (Fin Bn)) (j : ℕ) (s : List Bool),
      (∀ e ∈ seqRounds f step j is s,
        e.1.length < S ∧ e.2.mulDepth f ≤ D ∧ e.2.keyWork f ≤ Q ∧ e.2.ringWork f ≤ L) →
      seqAnswers f (fun j σ i => OracleAlgorithm.clip (fun out => (out.1, StateBits.pack out.2)) d D Q L
          (step j (StateBits.unpack σ) i)) j is (StateBits.pack s) =
        (seqRounds f step j is s).map fun e => (e.2.run f).1
  | [], _, _, _ => rfl
  | i :: is, j, s, h => by
    have h0 := h _ List.mem_cons_self
    have ht : ∀ e ∈ seqRounds f step (j + 1) is ((step j s i).run f).2, e.1.length < S ∧ e.2.mulDepth f ≤ D ∧
        e.2.keyWork f ≤ Q ∧ e.2.ringWork f ≤ L := fun e he => h e (List.mem_cons_of_mem _ he)
    simp only [seqRounds, List.map_cons, seqAnswers]
    rw [unpack_pack s h0.1, clip_run _ _ f _ D Q L h0.2.1 h0.2.2.1 h0.2.2.2]
    rw [seqAnswers_clip f d step is (j + 1) _ ht]

/-- **The fixed extractor passes where the period is certified.** For every joint run, every period's audit `P` and
every strategy `τ` under the device model (`DeviceModel`) and the instantiation (`Instantiates`), every value `o` of the
other audits' coins and every `B` with `B + S_chip + ∑ cap_i · round < S₀`: the extractor's server `extract τ o` passes
the ring game's session (`Passes`, at the designated id, on the payload `W`) on every capacity coin where the capacity
account holds at most `B` bits of device memory at every tick of the capacity window and the period is certified
(`Certified`). -/
theorem extractPasses (ι α ΩW ΩQ : Type) [Fintype ι] [Fintype α] (d : Accounting.Device ι)
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (τ : Strategy ΩW ΩQ) (D Q L round : ℕ) (hdev : DeviceModel r P τ D Q L round)
    (hinst : Instantiates P τ) (o : ΩW × ΩQ) (B : ℕ) (hB : B + d.chip + ∑ i, d.cap i * round < Pous.EndToEnd.S)
    (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
    (hmem : ∀ t ∈ (r.win .capacity).ticks, (r.schedule (o.1, o.2, x) t).mem (r.acct .capacity) ≤ B)
    (hcert : P.Certified (o.1, o.2, x)) : Passes P.sid (P.W o) (extract τ o D Q L) x := by
  obtain ⟨I, v, hdr, hans⟩ := hinst o x hcert
  obtain ⟨hspeed, hstate⟩ := hdev o x I v hdr
  have hot := hspeed (onTime_of P hcert)
  have hst := hstate B hmem
  refine ⟨I, v, hdr.1, hdr.2, ?_⟩
  rw [← hans]
  exact seqAnswers_clip (S := Pous.EndToEnd.S) (D := D) (Q := Q) (L := L)
    ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1) (fun _ => false, fun _ => false)
    (τ.step o (P.W o) (sch.enc (P.W o) x.1 ()).1) (List.ofFn I) 0
    (τ.pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1)
    fun e he => ⟨(hst e he).trans_lt hB, hot e he⟩

/-! ## The guarantee -/

theorem periodHeld : PeriodHeld := by
  intro D Q L hP ι α ΩW ΩQ _ _ d r P τ round S hdev hinst hS o
  by_cases hpos : 0 < S * (r.win .capacity).len
  · have h1 : 1 ≤ ⌈S * (r.win .capacity).len⌉₊ := Nat.ceil_pos.2 hpos
    have hlt : ((⌈S * (r.win .capacity).len⌉₊ - 1 : ℕ) : ℝ) < S * (r.win .capacity).len := by
      have := Nat.ceil_lt_add_one hpos.le
      rw [Nat.cast_sub h1, Nat.cast_one]
      linarith
    have hn : (⌈S * (r.win .capacity).len⌉₊ - 1) + d.chip + ∑ i, d.cap i * round < Pous.EndToEnd.S := by
      have : (((⌈S * (r.win .capacity).len⌉₊ - 1) + d.chip + ∑ i, d.cap i * round : ℕ) : ℝ) <
          Pous.EndToEnd.S := by
        rw [Nat.cast_add, Nat.cast_add]
        linarith
      exact_mod_cast this
    refine le_trans (Pouw.pr_mono (P' := Passes P.sid (P.W o) (extract τ o D Q L)) ?_) ?_
    · rintro x ⟨ha, hb, hh⟩
      refine extractPasses ι α ΩW ΩQ d r P τ D Q L round hdev hinst o _ hn x (fun t ht => ?_) (certified_of P ha hb)
      exact Nat.le_sub_one_of_lt (Nat.lt_ceil.2 ((Nat.cast_le.2 (r.mem_le_held ht)).trans_lt hh))
    · have hfin : ENNReal.ofReal (Params.δ : ℝ) + Params.εMax ≠ ⊤ :=
        ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩
      calc Pouw.pr (Passes P.sid (P.W o) (extract τ o D Q L))
          = (Pous.pr (Passes P.sid (P.W o) (extract τ o D Q L))).toReal := Accounting.pr_eq_toReal _
        _ ≤ (ENNReal.ofReal (Params.δ : ℝ) + Params.εMax).toReal :=
          ENNReal.toReal_mono hfin
            (Pous.EndToEnd.SecurityProofs.endToEndRing D Q L hP _ _ (extract τ o D Q L))
        _ = _ := by rw [ENNReal.toReal_add ENNReal.ofReal_ne_top εMax_ne_top, ENNReal.toReal_ofReal δ_nonneg]
  · rw [Pouw.pr_eq_zero fun x hx => hpos (lt_of_le_of_lt (Nat.cast_nonneg _) hx.2.2)]
    exact add_nonneg δ_nonneg ENNReal.toReal_nonneg

theorem periodCapacity : PeriodCapacity := by
  intro D Q L hP ι α ΩW ΩQ _ _ _ _ d r P τ round S hdev hinst hS hv
  have h : r.CapacitySoundPerRun S ((Params.δ : ℝ) + Params.εMax.toReal) := fun x q => by
    refine le_trans (Pouw.pr_mono ?_) (periodHeld D Q L hP ι α ΩW ΩQ d r P τ round S hdev hinst hS (x, q))
    rintro s ⟨hacc, hheld⟩
    exact ⟨(hv _ hacc).1, (hv _ hacc).2, hheld⟩
  exact ⟨h, r.capacitySound_of_perRun h⟩

/-! ## The honest and nontrivial lemmas

The device premises, `DeviceModel` and `Instantiates`, hold at the lookup strategy (`lookup`), which keeps all of `C`
and answers each index by looking its block up, with no query, on a run whose capacity account holds all of `C`'s
`FULL` bits at a tick of the capacity window, for every audit whose designated session id has 32 bytes, at every
budget (`periodHeld_honest`); `periodHeldHonest` (`Proofs.Pous.PeriodDeployed`) is it at the deployed honest period. A
verifier accepting every period fails the conclusion: on a run whose account holds nothing, its event holds on every
coin, with probability 1, above `δ + ε_max`. -/

/-- **The lookup strategy**, at any other audits' coins: `C`'s `FULL` bits are what survives setup, and each index is
answered by looking its block up, with no query, keeping the state. -/
noncomputable def lookup (ΩW ΩQ : Type) : Strategy ΩW ΩQ where
  pre _ _ _ C _ := List.ofFn (codeEquiv sch.B sch.ℓ C)
  step _ _ _ _ σ i := .ret ((codeEquiv sch.B sch.ℓ).symm (fun k => σ.getD k.val false) i, σ)

theorem lookup_pre {ΩW ΩQ : Type} (o : ΩW × ΩQ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω) :
    (lookup ΩW ΩQ).pre o W pp C ω = List.ofFn (codeEquiv sch.B sch.ℓ C) := by
  simp only [lookup]

theorem lookup_step {ΩW ΩQ : Type} (o : ΩW × ΩQ) (W : Bits sch.n) (pp : Bits sch.t) (j : ℕ) (σ : List Bool)
    (i : Fin sch.B) :
    (lookup ΩW ΩQ).step o W pp j σ i = .ret ((codeEquiv sch.B sch.ℓ).symm (fun k => σ.getD k.val false) i, σ) := by
  simp only [lookup]

/-- A strategy that keeps its state and answers from it with no query: every round starts from the first state and runs
a `ret`, and its answers are the lookups. -/
theorem seqRounds_ret {M : IdealModel} {σ α ι : Type} (f : M.Qry → M.Ans) (g : σ → ι → α)
    (step : ℕ → σ → ι → OracleAlgorithm M (α × σ)) (hstep : ∀ j s i, step j s i = .ret (g s i, s)) :
    ∀ (j : ℕ) (is : List ι) (s : σ),
      (seqRounds f step j is s).map (fun e => (e.2.run f).1) = is.map (g s) ∧
        ∀ e ∈ seqRounds f step j is s, e.1 = s ∧ ∃ i, e.2 = .ret (g s i, s)
  | _, [], _ => ⟨rfl, fun _ he => by simp [seqRounds] at he⟩
  | j, i :: is, s => by
    have ih := seqRounds_ret f g step hstep (j + 1) is s
    have hs : ((step j s i).run f).2 = s := by simp [hstep, OracleAlgorithm.run]
    refine ⟨?_, fun e he => ?_⟩
    · simp only [seqRounds, List.map_cons, ih.1, hstep, OracleAlgorithm.run]
    · simp only [seqRounds, List.mem_cons, hs] at he
      rcases he with rfl | he
      · exact ⟨rfl, i, hstep j s i⟩
      · exact ih.2 e he

/-- **`periodHeld`'s device premises hold at the honest instance**, at every budget `(D, Q, L)`: on a run whose
capacity account holds all of `C`'s `FULL` bits at a tick of the capacity window on every coin, for every audit whose
designated session id has 32 bytes, as the memory challenger's ids do, the lookup strategy meets the device model
(each answer is a lookup, no query, and its state is `C`'s bits) and the instantiation (the code's draw never fails,
and a lookup returns the challenged block of `C`). -/
theorem periodHeld_honest {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ) (hs : P.sid.size = 32)
    (hfull : ∀ ω, ∃ t ∈ (r.win .capacity).ticks, FULL ≤ (r.schedule ω t).mem (r.acct .capacity)) :
    DeviceModel r P (lookup ΩW ΩQ) D Q L round ∧ Instantiates P (lookup ΩW ΩQ) := by
  have key : ∀ (o : ΩW × ΩQ) (x : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams)
      (I : Fin 105 → Fin (2 ^ 19)),
      (P.answers (lookup ΩW ΩQ) o x I =
          (List.ofFn I).map fun i => (codeEquiv sch.B sch.ℓ).symm (fun k =>
            ((lookup ΩW ΩQ).pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1).getD k.val
              false) i) ∧
        ∀ e ∈ P.trace (lookup ΩW ΩQ) o x I,
          e.1 = (lookup ΩW ΩQ).pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1 ∧
            ∃ i, e.2 = .ret ((codeEquiv sch.B sch.ℓ).symm (fun k =>
              ((lookup ΩW ΩQ).pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1).getD k.val
                false) i,
              (lookup ΩW ΩQ).pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1) :=
    fun o x I => seqRounds_ret ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).answer x.1)
      (fun (σ : List Bool) (i : Fin sch.B) => (codeEquiv sch.B sch.ℓ).symm (fun k => σ.getD k.val false) i)
      ((lookup ΩW ΩQ).step o (P.W o) (sch.enc (P.W o) x.1 ()).1)
      (lookup_step o (P.W o) (sch.enc (P.W o) x.1 ()).1) 0
      (List.ofFn I) ((lookup ΩW ΩQ).pre o (P.W o) (sch.enc (P.W o) x.1 ()).1 (sch.enc (P.W o) x.1 ()).2 x.1)
  refine ⟨fun o x I _ _ => ⟨fun _ e he => ?_, fun B hB e he => ?_⟩, fun o x _ => ?_⟩
  · obtain ⟨-, i, hi⟩ := (key o x I).2 e he
    rw [hi]
    exact ⟨Nat.zero_le _, Nat.zero_le _, Nat.zero_le _⟩
  · obtain ⟨t, ht, hF⟩ := hfull (o.1, o.2, x)
    have h1 : FULL ≤ B := hF.trans (hB t ht)
    have hl : e.1.length = sch.B * sch.ℓ := by
      rw [((key o x I).2 e he).1, lookup_pre, List.length_ofFn]
    have hF' : sch.B * sch.ℓ = FULL := rfl
    omega
  · obtain ⟨v, hv⟩ := Pous.EndToEnd.SecurityProofs.coinView_some P.sid hs x.2.1
      (Pous.EndToEnd.SecurityProofs.splitAll x.2.2).1
    refine ⟨(Pous.EndToEnd.SecurityProofs.splitAll x.2.2).1, v,
      ⟨fun j => Pous.EndToEnd.SecurityProofs.uniformOn_split (x.2.2 j), hv⟩, ?_⟩
    have hfun : (fun k : Fin (sch.B * sch.ℓ) =>
        (List.ofFn (codeEquiv sch.B sch.ℓ (sch.enc (P.W o) x.1 ()).2)).getD k.val false) =
        codeEquiv sch.B sch.ℓ (sch.enc (P.W o) x.1 ()).2 := by
      funext k
      simp
    rw [(key o x _).1, lookup_pre, hfun]
    simp only [Equiv.symm_apply_apply]
    exact List.map_ofFn

theorem bound_toReal_lt_one : (Params.δ : ℝ) + Params.εMax.toReal < 1 := by
  have hfin : ENNReal.ofReal (Params.δ : ℝ) + Params.εMax ≠ ⊤ :=
    ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, εMax_ne_top⟩
  have h := (ENNReal.toReal_lt_toReal hfin ENNReal.one_ne_top).2 Pous.EndToEnd.SecurityProofs.bound_lt_one
  rwa [ENNReal.toReal_add ENNReal.ofReal_ne_top εMax_ne_top, ENNReal.toReal_ofReal δ_nonneg,
    ENNReal.toReal_one] at h

/-- **`periodHeld` is nontrivial**: an event that holds on every capacity coin, a verifier accepting every period on a
run whose account holds nothing, exceeds its bound `δ + ε_max`. -/
theorem periodHeld_nontrivial :
    ¬ Pouw.pr (fun _ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal := by
  rw [Pouw.pr_eq_one fun _ => trivial]
  exact not_le.2 bound_toReal_lt_one

/-- **`periodCapacity` holds at the honest instance**: `periodHeld_honest`, its premises being `periodHeld`'s. -/
theorem periodCapacity_honest {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (r : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ) (hs : P.sid.size = 32)
    (hfull : ∀ ω, ∃ t ∈ (r.win .capacity).ticks, FULL ≤ (r.schedule ω t).mem (r.acct .capacity)) :
    DeviceModel r P (lookup ΩW ΩQ) D Q L round ∧ Instantiates P (lookup ΩW ΩQ) :=
  periodHeld_honest r P D Q L round hs hfull

/-- **`periodCapacity` is nontrivial**: `periodHeld_nontrivial`, its bound being `periodHeld`'s. -/
theorem periodCapacity_nontrivial :
    ¬ Pouw.pr (fun _ : (ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams => True) ≤
      (Params.δ : ℝ) + Params.εMax.toReal :=
  periodHeld_nontrivial

end Pous.Period.SecurityProofs
