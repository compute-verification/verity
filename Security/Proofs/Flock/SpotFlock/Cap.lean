module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SamplingGeneric.TwoCode
public import Proofs.Flock.Claims.Sampling.Select.OS
public import Proofs.Flock.Bridge.Draw

/-!
# SPOT[Flock]'s replay cap

`replayCap`: at SPOT[Flock]'s code, the replay sample of record at outer coins some second-stage fork of which is
accepted has at most `cap h` proof blocks (`Code.ReplayCap`):

* an accepted fork passed round 2 (`stmtOk`), so its second stage accepted the replay sample, which is then not empty:
  the first stage accepted, and the sample of record is the slots `Slots.checked` keeps of the units it proved;
* each population slot `ρ` it keeps is unit `ρ`, whose cell of each template `t` is `blockOf ρ t` on `[0, n_t)`, and
  every proof block of an input slot is always checked (`regMatchesTwo`), so the sample has at most `|proved| · Σ_t n_t`
  proof blocks of population slots (`perBlock`) and always checked ones besides;
* a first stage with no closure map whose law Flock reads as a uniform `K`-subset (`firstK`) proves `K` units.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy Finset
open Properties.Spot (bytesOf membersPart)
open Properties.Spot.SecurityProofs (bind_eq_ok lawObj mem_checked flockLaw_subset deriveLaw_subset)
open Lean (Json)

/-! ## The first stage's count -/

/-- The `K` of the first stage's law over the population, where Flock reads it as a uniform `K`-subset. -/
def firstKOf (g : Reg) : Option ℕ :=
  match Spot.Partition.units g.program g.query with
  | .ok n =>
    match g.rule.lawOver n with
    | .ok named =>
      match Spot.Draw.flockLaw named n with
      | .ok (.subset k) => some k
      | _ => none
    | .error _ => none
  | .error _ => none

/-- **The first stage's count** `K`, where it samples a uniform `K`-subset of the replay blocks: its rule is not
`All`, it has no closure map, and Flock reads its law over the population as `subset K`. -/
def firstK (g : Reg) : Option ℕ :=
  match g.closure, g.rule with
  | none, .law _ => firstKOf g
  | none, .twoStage _ _ _ => firstKOf g
  | _, _ => none

theorem firstK_some {g : Reg} {K : ℕ} (hK : firstK g = some K) :
    g.closure = none ∧ g.rule ≠ .all ∧ ∃ n named, Spot.Partition.units g.program g.query = .ok n ∧
      g.rule.lawOver n = .ok named ∧ Spot.Draw.flockLaw named n = .ok (.subset K) := by
  have h3 : g.closure = none ∧ g.rule ≠ .all ∧ firstKOf g = some K := by
    unfold firstK at hK
    cases hc : g.closure <;> cases hr : g.rule <;> simp [hc, hr] at hK <;> exact ⟨rfl, by simp, hK⟩
  refine ⟨h3.1, h3.2.1, ?_⟩
  have hK' := h3.2.2
  unfold firstKOf at hK'
  cases hn : Spot.Partition.units g.program g.query with
  | error e => simp [hn] at hK'
  | ok n =>
    cases hl : g.rule.lawOver n with
    | error e => simp [hn, hl] at hK'
    | ok named =>
      cases hf : Spot.Draw.flockLaw named n with
      | error e => simp [hn, hl, hf] at hK'
      | ok law =>
        cases law <;> simp [hn, hl, hf] at hK'
        subst hK'
        exact ⟨n, named, rfl, hl, hf⟩

/-- **`Select.select` proves exactly the units it drew where it has no closure map**: `select_ok`'s steps, the units all
of `proved` when `cm` is none. -/
theorem select_proved (rule : Spot.Select.Rule) (d : Spot.Program) (query : Json) (table : Option Json)
    (cm : Option (Array (Int × Array Int))) (src : ByteArray) (draw : Json) (proved : Array ℕ) (read : ℕ)
    (h : Spot.Select.select rule d query table cm src = .ok (draw, proved, read)) :
    ∃ n named derived, Spot.Partition.units d query = .ok n ∧ rule.lawOver n = .ok named ∧
      Spot.Draw.deriveLaw d query named table = .ok derived ∧
        ∃ units : Array ℕ, (cm = none → proved = units) ∧ ((rule = .all ∧ units = Array.range n) ∨
          (rule ≠ .all ∧ ∃ parsed read', Spot.Draw.flockLaw (lawObj named derived) n = .ok parsed ∧
            Spot.Draw.drawUnits (derived.getD parsed) n src = .ok (units, read'))) := by
  unfold Spot.Select.select at h
  obtain ⟨n, hn, h⟩ := bind_eq_ok.1 h
  obtain ⟨named, hnamed, h⟩ := bind_eq_ok.1 h
  obtain ⟨derived, hderived, h⟩ := bind_eq_ok.1 h
  refine ⟨n, named, derived, hn, hnamed, hderived, ?_⟩
  dsimp only at h
  have tail : ∀ (units : Array ℕ) (c : Bool) (e : String) (js : Json) (cl : Array ℕ) (rd : ℕ),
      (if c = true then ((throw e : _root_.Flock.V Unit) >>= fun _ =>
          (pure (js, units ++ cl, rd) : _root_.Flock.V (Json × Array ℕ × ℕ)))
        else pure (js, units ++ cl, rd)) = .ok (draw, proved, read) → proved = units ++ cl := by
    intro units c e js cl rd ht
    split at ht
    · exact absurd ht (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at ht
      exact ht.2.1.symm
  split at h
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind])
    · split at h
      · rw [pure_bind] at h
        exact ⟨_, fun hc => absurd hc (by simp), Or.inl ⟨rfl, rfl⟩⟩
      · rename_i hne
        obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
        obtain ⟨⟨units, read'⟩, hu, h⟩ := bind_eq_ok.1 h
        exact ⟨units, fun hc => absurd hc (by simp), Or.inr ⟨fun e => hne e, parsed, read', hp, hu⟩⟩
  · split at h
    · rw [pure_bind] at h
      exact ⟨_, fun hc => by subst hc; exact (tail _ _ _ _ _ _ h).trans Array.append_empty, Or.inl ⟨rfl, rfl⟩⟩
    · rename_i hne
      obtain ⟨parsed, hp, h⟩ := bind_eq_ok.1 h
      obtain ⟨⟨units, read'⟩, hu, h⟩ := bind_eq_ok.1 h
      exact ⟨units, fun hc => by subst hc; exact (tail _ _ _ _ _ _ h).trans Array.append_empty,
        Or.inr ⟨fun e => hne e, parsed, read', hp, hu⟩⟩

/-- **A first stage of count `K` proves `K` units**: with no closure map it proves the units it drew, `K` of them. -/
theorem select_subset_size {g : Reg} {K : ℕ} (hK : firstK g = some K) {src : ByteArray} {draw : Json}
    {proved : Array ℕ} {read : ℕ}
    (h : Spot.Select.select g.rule g.program g.query g.table g.closure src = .ok (draw, proved, read)) :
    proved.size = K := by
  obtain ⟨hc, hne, n, named, hn, hl, hf⟩ := firstK_some hK
  obtain ⟨n', named', derived, hn', hl', hder, units, hpu, hcase⟩ := select_proved _ _ _ _ _ _ _ _ _ h
  rw [hn] at hn'
  obtain rfl := Except.ok.inj hn'
  rw [hl] at hl'
  obtain rfl := Except.ok.inj hl'
  obtain rfl : derived = none := deriveLaw_subset _ _ _ _ _ (flockLaw_subset _ _ _ hf).1 hder
  rw [hpu hc]
  rcases hcase with ⟨hall, -⟩ | ⟨-, parsed, read', hp, hu⟩
  · exact absurd hall hne
  · have hp' : Spot.Draw.flockLaw named n = .ok parsed := hp
    rw [hf] at hp'
    obtain rfl : parsed = .subset K := (Except.ok.inj hp').symm
    have hu' : Spot.Draw.drawUnits (.subset K) n src = .ok (units, read') := hu
    exact _root_.Flock.Bridge.drawUnits_size (law := .subset K) (ns := [K]) (kd := K) ⟨rfl, by simp⟩ hu'

/-! ## The cap -/

/-- **An accepting run passed round `j` at its own coins**: `run_acc_at`, the verifier's coins the run's. -/
theorem run_acc_coins {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (ω : Ω) (r : Fin (V.k x) → Coin) (j : ℕ) (hj0 : 0 < j)
    (hj : ∀ as cs, cs.length = as.length → as.length ≤ j → ((V.next x as cs).eval H).1 ≠ .acc)
    (h : (run H P V x ω r).out = .acc) :
    ∃ as : List Msg, as.length = j ∧ ((V.next x as ((List.range j).map (coinSeq r))).eval H).1 ≠ .rej := by
  obtain ⟨bs, hbs, hne⟩ := stepWith_acc_at H P x ω
    (fun as => V.next x as ((List.range as.length).map (coinSeq r))) j
    (fun bs hb => hj _ _ (by simp) hb) (V.k x) [] [] ⟨[], [], [], .rej⟩ hj0 h
  subst hbs
  exact ⟨bs, rfl, hne⟩

variable (h : Held)

/-- The proof blocks of the statement's partition, the members' units (`membersPart`). -/
def nBlocks : ℕ := (membersPart h.ms).blocks.length

/-- The proof blocks of a replay block: `n_t` in template `t`'s cell (`RegTwo.ns`), summed over the templates. -/
def perBlock : ℕ := ∑ t ∈ range h.reg.ns.size, h.reg.ns.getD t 0

/-- **The replay cap** at SPOT[Flock]'s code: where the first stage samples a uniform `K`-subset of the replay blocks
(`firstK`), the `K · Σ_t n_t` proof blocks of `K` replay blocks (`perBlock`) and the proof blocks always checked, and
never more than every proof block; every proof block otherwise. -/
def cap (_ : Msg) : ℕ :=
  match firstK h.reg.first with
  | some K => min (nBlocks h) (K * perBlock h + ((range (nBlocks h)).filter fun v => h.reg.always v = true).card)
  | none => nBlocks h

/-- **The replay cap holds** at SPOT[Flock]'s code (`Code.ReplayCap`): an accepted fork's second stage accepted the
first stage's sample, whose replay blocks' proof blocks `regMatchesTwo` counts. -/
theorem replayCap (H : Msg → Msg) : ((base h).atOrder (order h)).ReplayCap H (two h) (rs h) (cap h) := by
  intro x s Pt d hs hPt hd hin Ω P ω r e hacc
  have hs' : parse h x = some s := hs
  have hreg : reg h x = some h.reg := by
    unfold reg
    unfold parse at hs'
    cases hdx : decode x with
    | none => simp [hdx] at hs'
    | some st => rfl
  have hPteq : Pt = membersPart h.ms := partition_eq h (s := s) hPt
  obtain ⟨g, l, hg, hl, -, -, hT, -, hcl⟩ := regMatchesTwo h x s Pt d hs hPt hd hin
  obtain rfl : g = h.reg := Option.some.inj (hg.symm.trans hreg)
  obtain ⟨r', hr'⟩ : ∃ r', r' = forkOn (((base h).atOrder (order h)).stage2 x) r e := ⟨_, rfl⟩
  rw [← hr'] at hacc
  obtain ⟨as, has, hne⟩ := run_acc_coins H P ((base h).atOrder (order h)).V x ω r' 3 (by omega)
    (fun as cs hl h3 => next_short h H x as cs hl h3) hacc
  obtain ⟨a₀, m₁, xS, rfl⟩ := List.length_eq_three.1 has
  have hl3 : (List.range 3).map (coinSeq r') = [coinSeq r' 0, coinSeq r' 1, coinSeq r' 2] := rfl
  rw [hl3] at hne
  change ((next h x [a₀, m₁, xS] [coinSeq r' 0, coinSeq r' 1, coinSeq r' 2]).eval H).1 ≠ .rej at hne
  have hok : stmtOk h x m₁ xS (coinSeq r' 0) (coinSeq r' 1) = true := by
    cases hk : stmtOk h x m₁ xS (coinSeq r' 0) (coinSeq r' 1) with
    | false =>
      unfold next at hne
      simp [OC.eval, hk] at hne
    | true => rfl
  have hc0 : coinSeq r' 0 = r (order h x).first := by
    have h0 : 0 < ((base h).atOrder (order h)).V.k x := by show 0 < 3 + k₀ h; omega
    have e0 : coinSeq r' 0 = r' ⟨0, h0⟩ := dite_eq_left h0
    have hnot : (⟨0, h0⟩ : Fin _) ∉ ((base h).atOrder (order h)).stage2 x := fun hm =>
      lt_irrefl _ (mem_filter.1 hm).2
    rw [e0, hr']
    exact ite_eq_right hnot
  have hr : readsOk h s m₁ xS (coinSeq r' 0) (coinSeq r' 1) = true := by
    simp only [stmtOk, hs', Bool.and_eq_true] at hok
    exact hok.2
  obtain ⟨draw₂, hd₂⟩ : ∃ draw, secondDraw h (coinSeq r' 0) (coinSeq r' 1) = some draw := by
    unfold readsOk at hr
    split at hr
    · rename_i it draw ins spec sts w hit hd hset hdec
      exact ⟨draw, hd⟩
    · simp at hr
  obtain ⟨q, hq⟩ : ∃ q, Spot.Select.selectSecond (h.reg.sample (src (coinSeq r' 0))) h.reg.ns h.reg.ks
      (src (coinSeq r' 1)) = .ok q := by
    unfold secondDraw at hd₂
    cases hq : Spot.Select.selectSecond (h.reg.sample (src (coinSeq r' 0))) h.reg.ns h.reg.ks (src (coinSeq r' 1)) with
    | error e => simp [hq] at hd₂
    | ok q => exact ⟨q, rfl⟩
  have hne0 := (selectSecond_sample hq).1
  obtain ⟨draw₁, proved, read₁, hS⟩ : ∃ draw proved read,
      Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
        h.reg.first.closure (src (coinSeq r' 0)) = .ok (draw, proved, read) := by
    cases hS : Spot.Select.select h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table
        h.reg.first.closure (src (coinSeq r' 0)) with
    | error e => exact absurd (by simp [RegTwo.sample, hS]) hne0
    | ok t => exact ⟨t.1, t.2.1, t.2.2, rfl⟩
  have hrs : rs h x r = (Spot.Slots.checked l proved).toList := by
    show ((reg h x).map fun g => g.first.checked (bytesOf (coinByte (r (order h x).first)))).getD [] = _
    rw [hreg]
    show (drawChecked h.reg.first.rule h.reg.first.program h.reg.first.query h.reg.first.table h.reg.first.closure
      h.reg.first.layout (src (r (order h x).first))).toList = _
    rw [← hc0]
    unfold drawChecked
    simp only [slots_select_of _ _ _ _ _ _ hl hS]
  have hN : nBlocks h = Pt.blocks.length := by
    rw [hPteq]
    rfl
  have hle : (sampleBlocks Pt d.part (rs h x r)).card ≤ nBlocks h := (card_sampleBlocks_le _ _ _).trans hN.ge
  unfold cap
  cases hK : firstK h.reg.first with
  | none => exact hle
  | some K =>
    dsimp only
    refine le_min hle ?_
    have hsize : proved.size = K := select_subset_size hK hS
    have hsub : sampleBlocks Pt d.part (rs h x r) ⊆
        (proved.toList.toFinset.biUnion fun ρ => (range h.reg.ns.size).biUnion fun t =>
          (range (h.reg.ns.getD t 0)).image (h.reg.blockOf ρ t)) ∪
          (range (nBlocks h)).filter fun v => h.reg.always v = true := by
      intro v hv
      obtain ⟨hvN, ρ, hρ, hvρ⟩ := mem_filter.1 hv
      rw [hrs, Array.mem_toList_iff] at hρ
      obtain ⟨hρl, -⟩ := (mem_checked l proved ρ).1 hρ
      have hvc : v ∈ cell Pt d.part ρ (Pt.stratumOf v) := mem_filter.2 ⟨hvN, hvρ, rfl⟩
      rcases hcl ρ l.blocks[ρ] (Array.getElem?_eq_getElem hρl) with ⟨hu, hcell⟩ | ⟨-, halw⟩
      · have ht : Pt.stratumOf v < Pt.strata.length := Claims.Sampling.stratumOf_lt (mem_range.1 hvN)
        refine mem_union_left _ (mem_biUnion.2 ⟨ρ, ?_, mem_biUnion.2 ⟨Pt.stratumOf v, ?_, ?_⟩⟩)
        · simpa using mem_slots_checked hρ (Array.getElem?_eq_getElem hρl) hu
        · rw [mem_range, hT]
          exact ht
        · rw [(hcell _ ht).2] at hvc
          exact hvc
      · exact mem_union_right _ (mem_filter.2 ⟨by rw [hN]; exact hvN, halw _ v hvc⟩)
    have hblock : ∀ ρ, ((range h.reg.ns.size).biUnion fun t =>
        (range (h.reg.ns.getD t 0)).image (h.reg.blockOf ρ t)).card ≤ perBlock h := by
      intro ρ
      unfold perBlock
      exact card_biUnion_le.trans (sum_le_sum fun t _ => card_image_le.trans (card_range _).le)
    calc (sampleBlocks Pt d.part (rs h x r)).card
        ≤ (proved.toList.toFinset.biUnion fun ρ => (range h.reg.ns.size).biUnion fun t =>
              (range (h.reg.ns.getD t 0)).image (h.reg.blockOf ρ t)).card +
            ((range (nBlocks h)).filter fun v => h.reg.always v = true).card :=
          (card_le_card hsub).trans (card_union_le _ _)
      _ ≤ K * perBlock h + ((range (nBlocks h)).filter fun v => h.reg.always v = true).card := by
          gcongr
          calc (proved.toList.toFinset.biUnion fun ρ => (range h.reg.ns.size).biUnion fun t =>
                (range (h.reg.ns.getD t 0)).image (h.reg.blockOf ρ t)).card
              ≤ ∑ ρ ∈ proved.toList.toFinset, ((range h.reg.ns.size).biUnion fun t =>
                  (range (h.reg.ns.getD t 0)).image (h.reg.blockOf ρ t)).card := card_biUnion_le
            _ ≤ ∑ _ρ ∈ proved.toList.toFinset, perBlock h := sum_le_sum fun ρ _ => hblock ρ
            _ = proved.toList.toFinset.card * perBlock h := by rw [sum_const, smul_eq_mul]
            _ ≤ K * perBlock h := by
                gcongr
                calc proved.toList.toFinset.card ≤ proved.toList.length := List.toFinset_card_le _
                  _ = K := by rw [Array.length_toList, hsize]

end SpotFlock
