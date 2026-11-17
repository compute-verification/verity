module

public import Proofs.Properties.Verity.Sampled.Window.Streams
public import Proofs.Properties.Verity.Sampled.RootB

/-! SPOT's coupling over a window: a session's moves are its record's. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger
open FlockSoundness.Refine (roundOf ptsOf)
open FlockSoundness.Discharge.Tables (LiveOutJ SubmitJ addRoundJ)
open Claims.Bridge.CustodyZkP (repRounds)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem _root_.Verity.Game.Pos.Le.through {β W κ : Type} {g : Game β} {σ : Strategy g} {q r : Pos g σ W κ}
    (h : q.Le r) : ∃ y, r.tape = q.back y := by
  obtain ⟨f, hf⟩ := h
  exact ⟨f r.x, hf r.x⟩

theorem stop_modes_tape {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {rd : Reading R} {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {pre L : List Release} {r : Release}
    (hc : CutsAt rd.reveal q pre (CoinTape.run (V.game θ F c) σ z).log)
    (hL : (CoinTape.run (V.game θ F c) σ z).log = pre ++ r :: L) {i : Fin R} (hi : (q.Ks i).isHere = true)
    {w : Claims.Msg} (hw : rd.reveal i (CoinTape.run (V.game θ F c) σ z) = some w)
    (hd : sha512Msg w = r.answers.data.toList) : q.modes i = some w := by
  obtain ⟨y, rfl⟩ := hz
  have hrev : (roundCut V θ F c rd i).cut.revealOn σ (q.back y) = some w := by
    refine (Pos.revealOn_here hq (Pos.Le.refl q) (hc i r L hi hL) y).trans ?_
    unfold digestReveal
    rw [hw]
    simp [Option.filter, hd]
  rw [Pos.modes_eq hq (Pos.Le.refl q) y]
  dsimp only
  rw [ite_eq_left hi]
  by_contra hne
  exact hn.1 i ((DigestCut.missOn_iff _ _ _).2 ⟨w, hrev, hne⟩)

theorem executionPhaseStop_modes {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) (hT : 0 < tables.size)
    {per : ℕ} (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g}
    {q : Pos g σ Claims.Msg (Fin R)} {j : Fin R} {pre : List Release} {fr : Frame}
    (hall : ∀ j', (q.Ks j').isHere = true → (executionPhaseReadingAt tables per reveal record recordDigest).round j' pre fr)
    (hP : (executionPhaseReadingAt tables per reveal record recordDigest).round j pre fr) :
    ∀ j', j' ≠ j → q.modes j' = none := fun j' hj' =>
  Pos.modes_eq_none (Bool.eq_false_iff.2 fun h =>
    hj' (executionPhaseReadingAt_unique ht hT hper reveal record recordDigest (hall j' h) hP))

theorem LoggedAfter.session_link {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup)
    (hT : 0 < tables.size) {per : ℕ} (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g}
    {q : Pos g σ Claims.Msg (Fin R)} {L₀ Lg : List Release} {bits : List Bool} {u : ℕ}
    (h : LoggedAfter (executionPhaseReadingAt tables per reveal record recordDigest).round q L₀ bits Lg)
    (hL₀ : ∀ j : Fin R, PicksIn (executionPhaseReadingAt tables per reveal record recordDigest).round Lg j L₀ →
      slotOf tables.size per (j.val % per) = .link ∧ j.val / per = u) :
    ∃ (j : Fin R) (i s n : ℕ) (b : ByteArray) (L : List Release), (q.Ks j).isHere = true ∧
      (∀ j', j' ≠ j → q.modes j' = none) ∧ slotOf tables.size per (j.val % per) = .link ∧ j.val / per = u ∧
      Lg = L₀ ++ ⟨.opening i s n, b, bits⟩ :: L ∧ i ∉ openedTrees L₀ ∧ (openedTrees L₀).length = u := by
  obtain ⟨j, i, s, n, b, L, hjK, hall, hj, hlog⟩ := h
  obtain ⟨hjl, hju⟩ := hL₀ j ⟨_, L, hlog, hj⟩
  have hj' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) L₀ (.opening i s n) := hj
  rw [hjl, hju] at hj'
  obtain ⟨i', s', n', hfr, -, hni, hlen⟩ := hj'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  exact ⟨j, _, _, _, b, L, hjK, executionPhaseStop_modes ht hT hper reveal record recordDigest hall hj, hjl, hju, hlog,
    hni, hlen⟩

theorem LoggedAfter.session_slot {R J : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) {tables : Array String}
    (ht : tables.toList.Nodup) (hJ : J ≤ tables.size) {per : ℕ} (hper : 0 < per)
    (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {sub : SubmitJ J}
    (hsub : (decodersAt reg fun j : Fin R => slotOf tables.size per (j.val % per)).sub J q.modes = some sub)
    {pre : List Release} {bits : List Bool} {log : List Release}
    (h : LoggedAfter (executionPhaseReadingAt tables per reveal record recordDigest).round q pre bits log) :
    ∃ (ht' : 0 < J) (r : Fin 2) (k : ℕ) (w : Claims.Msg) (j : Fin R) (i s n : ℕ) (b : ByteArray)
      (L' : List Release), slotOf tables.size per (j.val % per) = .rep 0 r k ∧ (q.Ks j).isHere = true ∧
        q.modes j = some w ∧ sub = ⟨⟨0, ht'⟩, r, ⟨w.toArray⟩, tailCount w⟩ ∧ Served pre i s n ∧
        (openedTrees pre)[j.val / per]? = some i ∧
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = tables[0]?.map (repStreamName · r.val) ∧
        openedOf pre i s = k ∧ log = pre ++ ⟨.opening i s n, b, bits⟩ :: L' ∧
        (executionPhaseReadingAt tables per reveal record recordDigest).round j pre (.opening i s n) := by
  obtain ⟨j, i, s, n, b, L', hj, hall, hP, he⟩ := h
  obtain ⟨t, ht', r, k, w, hLj, hw, hs⟩ :=
    sub_of_single reg (fun j : Fin R => slotOf tables.size per (j.val % per)) J (modes := q.modes)
      (executionPhaseStop_modes ht (Nat.lt_of_lt_of_le sub.tab.pos hJ) hper reveal record recordDigest hall hP)
      hsub
  have hLj' : slotOf tables.size per (j.val % per) = .rep t r k := hLj
  have htT : t < tables.size := Nat.lt_of_lt_of_le ht' hJ
  obtain ⟨-, hLa⟩ := slotOf_rep hLj' htT
  obtain ⟨rfl, -⟩ := slotAt_eq_rep hLa
  have hP' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) pre (.opening i s n) := hP
  rw [hLj'] at hP'
  obtain ⟨i', s', n', hfr, hsv, hu, hname, hk⟩ := hP'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  rw [slotStream_lt htT r] at hname
  exact ⟨ht', r, k, w, j, _, _, _, b, L', hLj', hj, hw, hs, hsv, hu, hname, hk, he, hP⟩

theorem executionPhase_covers (T : ℕ) {R per u K : ℕ} (hper : 2 * K + 2 ≤ per) (hu : (u + 1) * per ≤ R) (ρ : Fin 2) :
    ∀ k < K, ∃ j : Fin R, j.val / per = u ∧ slotOf T per (j.val % per) = .rep 0 ρ k := fun k hk => by
  have hρ := ρ.isLt
  have hlt : 2 * k + ρ.val + 1 < per := by omega
  have hne : 2 * k + ρ.val + 1 + 1 ≠ per := by omega
  have hpos : 0 < per := by omega
  have hR : per * u + (2 * k + ρ.val + 1) < R := by
    have e : (u + 1) * per = per * u + per := by rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    omega
  refine ⟨⟨per * u + (2 * k + ρ.val + 1), hR⟩, ?_, ?_⟩
  · show (per * u + (2 * k + ρ.val + 1)) / per = u
    rw [Nat.mul_add_div hpos, Nat.div_eq_of_lt hlt, Nat.add_zero]
  · show slotOf T per ((per * u + (2 * k + ρ.val + 1)) % per) = .rep 0 ρ k
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hlt, slotOf_of_ne T hne]
    exact slotAt_rep k ρ

theorem executionPhase_close (T : ℕ) {R per u : ℕ} (hper : 0 < per) (hu : (u + 1) * per ≤ R) :
    ∃ j : Fin R, j.val / per = u ∧ slotOf T per (j.val % per) = closeSlot T := by
  have hlt : per - 1 < per := by omega
  have hR : per * u + (per - 1) < R := by
    have e : (u + 1) * per = per * u + per := by rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
    omega
  refine ⟨⟨per * u + (per - 1), hR⟩, ?_, ?_⟩
  · show (per * u + (per - 1)) / per = u
    rw [Nat.mul_add_div hper, Nat.div_eq_of_lt hlt, Nat.add_zero]
  · show slotOf T per ((per * u + (per - 1)) % per) = closeSlot T
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
    exact slotOf_last T (by omega)

theorem one_table_pos (tb : String) : 0 < (#[tb] : Array String).size := Nat.zero_lt_one

def sessionRecordAt (u : ℕ) (e : _root_.Spot.OneStageZk.Evidence) (bs : _root_.Core.Blobs) : Option String :=
  e.sessions[u]?.bind fun s => (bs.get s.record s!"session {s.name}'s record").toOption.bind String.fromUTF8?

theorem sessionRecordAt_zero : sessionRecordAt 0 = sessionRecord := rfl

def sentRecordAt (u : ℕ) (s : Verity.Sent) : Option String :=
  (caseOf s).toOption.bind fun x => sessionRecordAt u x.2.2 (_root_.Core.Blobs.ofArray s.blobs)

theorem sentRecordAt_zero : sentRecordAt 0 = sentRecord := rfl

def sessionReveal (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u : ℕ) :
    Slot → Exchange → Option Claims.Msg
  | .link => fun o => (sentRecordAt u o.sent).bind (linkMsg · tables)
  | .rep t r k => fun o =>
    if t = tables.size then evidence u o
    else tables[t]?.bind fun a => (sentRecordAt u o.sent).bind (roundMsg · (repStreamName a r.val) k)

theorem sessionReveal_link (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u : ℕ)
    (o : Exchange) : sessionReveal tables evidence u .link o = (sentRecordAt u o.sent).bind (linkMsg · tables) :=
  rfl

theorem sessionReveal_rep (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u t : ℕ)
    (r : Fin 2) (k : ℕ) (o : Exchange) {a : String} (ha : tables[t]? = some a) :
    sessionReveal tables evidence u (.rep t r k) o =
      (sentRecordAt u o.sent).bind (roundMsg · (repStreamName a r.val) k) := by
  have ha' : tables.toList[t]? = some a := by
    rw [Array.getElem?_toList]
    exact ha
  have hlt : t < tables.size := (List.getElem?_eq_some_iff.1 ha').1
  have hne : ¬ t = tables.size := Nat.ne_of_lt hlt
  show (if t = tables.size then evidence u o
    else tables[t]?.bind fun a => (sentRecordAt u o.sent).bind (roundMsg · (repStreamName a r.val) k)) = _
  rw [if_neg hne, ha, Option.bind_some]

theorem sessionReveal_close (tables : Array String) (evidence : ℕ → Exchange → Option Claims.Msg) (u : ℕ)
    (o : Exchange) : sessionReveal tables evidence u (closeSlot tables.size) o = evidence u o :=
  if_pos rfl

theorem opened_at_or_new {Lg L₁ L₂ : List Release} {x : Release} {i s n u : ℕ} (he : Lg = L₁ ++ x :: L₂)
    (hx : x.frame = .opening i s n) (hτ : (openedTrees Lg)[u]? = some i) :
    (i ∉ openedTrees L₁ ∧ (openedTrees L₁).length = u) ∨ (openedTrees L₁)[u]? = some i := by
  by_cases hmem : i ∈ openedTrees L₁
  · obtain ⟨v, hv⟩ := List.getElem?_of_mem hmem
    have h := openedTrees_getElem?_append hv (x :: L₂)
    rw [← he] at h
    have hvu : v = u := nodup_getElem?_inj (openedTrees_nodup Lg) h hτ
    rw [hvu] at hv
    exact Or.inr hv
  · obtain ⟨e, hE⟩ := openedTrees_new hx hmem L₂
    rw [← he] at hE
    have h : (openedTrees Lg)[(openedTrees L₁).length]? = some i := by
      rw [hE, List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero]
    exact Or.inl ⟨hmem, nodup_getElem?_inj (openedTrees_nodup Lg) h hτ⟩

structure SessionOpens (Lg : List Release) (tables : Array String) (u : ℕ) (text : String) (J : Json)
    (sess : Flock.RunRecord) : Prop where
  parse : Flock.JsonParse.parse text = .ok J
  coins : ∀ name, ∀ k < recordRounds J name, (recordRound J name k).isSome
  link : ∀ L₁ x L₂ i s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n → i ∉ openedTrees L₁ →
    (openedTrees L₁).length = u →
      ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧
        (∃ w, linkMsg text tables = some w ∧ sha512Msg w = x.answers.data.toList) ∧
        ∃ b, recordRound J TREE_SESSION 0 = some b ∧ (Verity.bitsOfBytes b ≠ [] → x.coins = Verity.bitsOfBytes b)
  round : ∀ L₁ x L₂ i s n name, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n →
    (openedTrees L₁)[u]? = some i → ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some name →
    (name == TREE_SESSION) = false →
      openedOf L₁ i s < recordRounds J name ∧
        (∀ st, sess.streams.find? (·.name == name) = some st → openedOf L₁ i s < st.rounds.size →
          x.answers = st.rounds[openedOf L₁ i s]!.sha) ∧
        ∀ b, recordRound J name (openedOf L₁ i s) = some b → Verity.bitsOfBytes b ≠ [] →
          x.coins = Verity.bitsOfBytes b
  full : ∀ i s name, (openedTrees Lg)[u]? = some i →
    ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name → (name == TREE_SESSION) = false →
      recordRounds J name ≤ openedOf Lg i s
  key : ∀ name, 0 < recordRounds J name → ∀ i, (openedTrees Lg)[u]? = some i →
    ∃ s : ℕ, ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name
  tree : ∀ i, (openedTrees Lg)[u]? = some i → (helloSpecs Lg)[i]? = some (Flock.Zk.coinSpec tables)
  linkCount : ∀ L₁ x L₂ i s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n → i ∉ openedTrees L₁ →
    (openedTrees L₁).length = u → ∀ b, recordRound J TREE_SESSION 0 = some b → n = b.size / 16
  roundCount : ∀ L₁ x L₂ i s n name, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n →
    (openedTrees L₁)[u]? = some i → ((helloSpecs L₁)[i]?.bind fun sp => sp.streams[s]?) = some name →
    (name == TREE_SESSION) = false → ∀ st, sess.streams.find? (·.name == name) = some st →
      openedOf L₁ i s < st.rounds.size → n = st.rounds[openedOf L₁ i s]!.n

structure SessionRead (Lg : List Release) (tables : Array String) (u : ℕ) (text : String) (J : Json)
    (sess : Flock.RunRecord) (w : Claims.Msg) : Prop extends SessionOpens Lg tables u text J sess where
  close : ∀ i, (openedTrees Lg)[u]? = some i → ∃ Lc x L₂ s n, Lg = Lc ++ x :: L₂ ∧ x.frame = .opening i s n ∧
    (openedTrees Lc)[u]? = some i ∧
    ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION ∧ openedOf Lc i s = 1 ∧
    sha512Msg w = x.answers.data.toList ∧
    ∀ L₁ y L₃ s' n', Lg = L₁ ++ y :: L₃ → y.frame = .opening i s' n' → L₁.length ≤ Lc.length

theorem SessionRead.openedOf_le {Lg : List Release} {tables : Array String} {u : ℕ} {text : String} {J : Json}
    {sess : Flock.RunRecord} {w : Claims.Msg} (hR : SessionRead Lg tables u text J sess w) (hS : LogServed Lg)
    {i s : ℕ}
    {name : String} (hτ : (openedTrees Lg)[u]? = some i)
    (hname : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name)
    (hnT : (name == TREE_SESSION) = false) : openedOf Lg i s ≤ recordRounds J name := by
  suffices h : ∀ pre post, Lg = pre ++ post → openedOf pre i s ≤ recordRounds J name from
    h Lg [] (List.append_nil Lg).symm
  intro pre
  induction pre using List.reverseRecOn with
  | nil => intro _ _; exact Nat.zero_le _
  | append_singleton pre r ih =>
    intro post he
    rw [List.append_assoc, List.singleton_append] at he
    rw [openedOf_append]
    by_cases hr : r.frame.opensAt i s = true
    · obtain ⟨n, hn⟩ := opening_of_opensAt hr
      rw [ite_eq_left hr]
      obtain ⟨sp, hsp, hs, -⟩ := hS pre r post i s n he hn
      have h1 : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some sp.streams[s] := by
        rw [hsp, Option.bind_some, Array.getElem?_eq_getElem hs]
      have h2 := helloSpecs_bind_append h1 (r :: post)
      rw [← he, hname] at h2
      have h1' : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some name := h1.trans h2.symm
      rcases opened_at_or_new he hn hτ with ⟨hni, hlen⟩ | hu
      · have h3 := (hR.link pre r post i s n he hn hni hlen).1
        rw [h1'] at h3
        rw [Option.some.inj h3] at hnT
        simp at hnT
      · have hlt := (hR.round pre r post i s n name he hn hu h1' hnT).1
        omega
    · rw [ite_eq_right hr]
      exact ih (r :: post) he

theorem session_rootB {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ} {Reg : Type}
    (reg : Option Claims.Msg → Reg) {tb : String} {per : ℕ} (hper : 0 < per)
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {L₀ : List Release} {bits : List Bool}
    (hLA : LoggedAfter rd.round q L₀ bits (CoinTape.run (V.game θ F c) σ z).log)
    (hc : CutsAt rd.reveal q L₀ (CoinTape.run (V.game θ F c) σ z).log) {u : ℕ}
    (hL₀ : ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j L₀ →
      slotOf (#[tb] : Array String).size per (j.val % per) = .link ∧ j.val / per = u)
    {text : String} {J : Json} {spec : Flock.SessionSpec} {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (ht : spec.tables = #[tb]) (hσ : spec.sigma.size = 64) :
    (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)).root q.modes =
      sess.rootB := by
  subst hrd
  obtain ⟨j, i, s, n, b, L, hjK, hmodes, hjl, hju, hlog, hni, hlen⟩ :=
    LoggedAfter.session_link (by simp) (one_table_pos tb) hper (sessionReveal #[tb] ev) record₀ recordDigest hLA hL₀
  obtain ⟨-, ⟨w, hw, hwd⟩, -⟩ := hR.link L₀ _ L i s n hlog rfl hni hlen
  have hrev : (executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest).reveal j
      (CoinTape.run (V.game θ F c) σ z) = some w := by
    show sessionReveal #[tb] ev (j.val / per) (slotOf (#[tb] : Array String).size per (j.val % per))
      (CoinTape.run (V.game θ F c) σ z) = some w
    rw [hjl, hju, sessionReveal_link, hrec, Option.bind_some]
    exact hw
  have hmode := stop_modes_tape hq hz hn hc hlog hjK hrev hwd
  refine root_record reg (fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)) (i := j)
    (by show (slotOf (#[tb] : Array String).size per (j.val % per)).isLink = true; rw [hjl]; rfl) hmode
    (fun j' hj' _ => hmodes j' hj') hd ht hσ ?_
  rw [ht]
  exact hw

theorem session_pts {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) (hT : 0 < tables.size) {per : ℕ}
    (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {L₀ Lg : List Release} {mP : ℕ} {pts : Fin 2 → Fin mP → 𝔽} {u : ℕ}
    (hLA : LoggedAfter (executionPhaseReadingAt tables per reveal record recordDigest).round q L₀ (linkBits fieldEnc pts)
      Lg)
    (hL₀ : ∀ j : Fin R, PicksIn (executionPhaseReadingAt tables per reveal record recordDigest).round Lg j L₀ →
      slotOf tables.size per (j.val % per) = .link ∧ j.val / per = u)
    {tables' : Array String} {text : String} {J : Json} {spec : Flock.SessionSpec} {sess : Flock.RunRecord}
    {w₀ : Claims.Msg} (hR : SessionRead Lg tables' u text J sess w₀) (hd : Flock.Record.decode spec text = .ok sess)
    (hp2 : spec.points = 2) (hpos : 0 < spec.mPts) : ptsOf mP sess.points = pts := by
  obtain ⟨j, i, s, n, b, L, -, -, -, -, hlog, hni, hlen⟩ :=
    LoggedAfter.session_link ht hT hper reveal record recordDigest hLA hL₀
  obtain ⟨-, -, b', hb', hcoins⟩ := hR.link L₀ _ L i s n hlog rfl hni hlen
  have hbits : Verity.bitsOfBytes b' = linkBits fieldEnc pts :=
    (hcoins (link_bits_ne hd hR.parse hp2 hpos hb')).symm
  exact pts_of_record hd hR.parse hp2 hb' hbits

theorem session_stop_msg {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {tb : String} (per : ℕ) (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {j : Fin R} {u : ℕ} {ρ : Fin 2} {k : ℕ}
    (hLj : slotOf (#[tb] : Array String).size per (j.val % per) = .rep 0 ρ k) (hju : j.val / per = u)
    (hj : (q.Ks j).isHere = true)
    {w : Claims.Msg} (hw : q.modes j = some w) {bits : List Bool} {m L' : List Release} {i s n : ℕ}
    {b : ByteArray} (hlog : (CoinTape.run (V.game θ F c) σ z).log = m ++ ⟨.opening i s n, b, bits⟩ :: L')
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log)
    (hτ : (openedTrees m)[u]? = some i) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb ρ.val))
    {text : String} {J : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) (htab : spec.tables = #[tb]) :
    ∃ st, sess.streams.find? (·.name == repStreamName tb ρ.val) = some st ∧ k < st.rounds.size ∧
      roundMsg text (repStreamName tb ρ.val) k = some w := by
  subst hrd hk
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  subst h0
  have hnT := repStreamName_ne spec.tables[0]! ρ.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd ρ
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName spec.tables[0]! ρ.val) = some st := hst
  have hrr : recordRounds J (repStreamName spec.tables[0]! ρ.val) = st.rounds.size :=
    recordRounds_decode hd hR.parse hnT hst'
  obtain ⟨hlt, hans, -⟩ := hR.round m _ L' i s n _ hlog rfl hτ hname hnT
  have hkR : openedOf m i s < st.rounds.size := by
    rw [hrr] at hlt
    exact hlt
  obtain ⟨cs', -, -, hsha⟩ := recordRound_decode hd hR.parse hnT hst' hkR
  obtain ⟨bb, hbb, hmsg⟩ := roundMsg_decode hd hret hst' hkR
  have hdg512 : spec.dg bb = Flock.Sha512.hash bb := by simp [Flock.SessionSpec.dg, h512]
  have hyans : b = st.rounds[openedOf m i s].sha := by
    have h : b = st.rounds[openedOf m i s]!.sha := hans st hst' hkR
    rw [getElem!_pos st.rounds (openedOf m i s) hkR] at h
    exact h
  have hrev : (executionPhaseReadingAt #[spec.tables[0]!] per (sessionReveal #[spec.tables[0]!] ev) record₀
      recordDigest).reveal j (CoinTape.run (V.game θ F c) σ z) = some bb.data.toList := by
    show sessionReveal #[spec.tables[0]!] ev (j.val / per)
      (slotOf (#[spec.tables[0]!] : Array String).size per (j.val % per))
      (CoinTape.run (V.game θ F c) σ z) = some bb.data.toList
    rw [hLj, hju]
    refine (sessionReveal_rep #[spec.tables[0]!] ev u 0 ρ (openedOf m i s) (CoinTape.run (V.game θ F c) σ z)
      (a := spec.tables[0]!) rfl).trans ?_
    rw [hrec, Option.bind_some]
    exact hmsg
  have hmode := stop_modes_tape hq hz hn hcut hlog hj hrev (by
    show sha512Msg bb.data.toList = b.data.toList
    rw [hyans, hsha bb hbb, hdg512]
    all_goals rfl)
  have hwb : w = bb.data.toList := Option.some.inj (hw.symm.trans hmode)
  rw [hwb]
  exact ⟨st, hst', hkR, hmsg⟩

theorem session_stop_round {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R : ℕ}
    {tb : String} (per : ℕ) (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {rd : Reading R}
    (hrd : rd = executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {j : Fin R} {u : ℕ} {ρ : Fin 2} {k : ℕ}
    (hLj : slotOf (#[tb] : Array String).size per (j.val % per) = .rep 0 ρ k) (hju : j.val / per = u)
    (hj : (q.Ks j).isHere = true)
    {w : Claims.Msg} (hw : q.modes j = some w) {cs : Fin (tailCount w) → 𝔽} {m L' : List Release} {i s n : ℕ}
    {b : ByteArray}
    (hlog : (CoinTape.run (V.game θ F c) σ z).log = m ++ ⟨.opening i s n, b, roundBits fieldEnc cs⟩ :: L')
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log)
    (hτ : (openedTrees m)[u]? = some i) (hk : openedOf m i s = k)
    (hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb ρ.val))
    {text : String} {J : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) {dg : ByteArray → ByteArray}
    (hdg : spec.dg = dg) (htab : spec.tables = #[tb]) :
    ∃ st, sess.streams.find? (·.name == repStreamName tb ρ.val) = some st ∧
      ∃ hk : k < st.rounds.size, roundOf dg ⟨w.toArray⟩ (tailCount w) cs = st.rounds[k] := by
  obtain ⟨st, hst', hkR, hmsg⟩ := session_stop_msg per ev record₀ recordDigest hrd hq hz hn hLj hju hj hw hlog hcut
    hτ hk hname hrec hR hd hv hret h512 htab
  subst hk
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  subst h0
  have hnT := repStreamName_ne spec.tables[0]! ρ.val
  have hrr : recordRounds J (repStreamName spec.tables[0]! ρ.val) = st.rounds.size :=
    recordRounds_decode hd hR.parse hnT hst'
  obtain ⟨-, -, hcoin⟩ := hR.round m _ L' i s n _ hlog rfl hτ hname hnT
  obtain ⟨b', hb'⟩ := Option.isSome_iff_exists.1
    (hR.coins (repStreamName spec.tables[0]! ρ.val) (openedOf m i s) (by rw [hrr]; exact hkR))
  obtain ⟨cs', hrr', hf', -⟩ := recordRound_decode hd hR.parse hnT hst' hkR
  have hbits : Verity.bitsOfBytes b' = roundBits fieldEnc cs := by
    by_cases hc : Verity.bitsOfBytes b' = []
    · have hrc : _root_.Spot.Challenger.recordCoins? (.arr cs') = some b' := hrr'.symm.trans hb'
      have hlen := congrArg List.length (bits_of_coins hf' hrc)
      rw [hc, length_chunks, List.length_nil, Array.length_toList] at hlen
      have hmem : st ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hst')
      have hwf := FlockSoundness.Refine.decode_roundsWF hd st hmem (openedOf m i s) hkR
      rw [getElem!_pos st.rounds (openedOf m i s) hkR] at hwf
      have htc := rep_tailCount_accepted hv hd hret ρ hst' hkR hmsg
      have hzero : (roundBits fieldEnc cs).length = 0 := by
        rw [length_roundBits, htc, ← hwf]
        omega
      rw [hc, List.length_eq_zero_iff.1 hzero]
    · exact (hcoin b' hb' hc).symm
  obtain ⟨hB', hS', hN', hsz, hC'⟩ := answered_of_record hd hR.parse hret hnT hst' hkR hmsg hb' hbits
  have hcoins : (Array.ofFn fun a => (cs a).val) = st.rounds[openedOf m i s].coins := by
    refine Array.ext (by rw [Array.size_ofFn, hsz]) fun a h₁ h₂ => ?_
    have ha : a < tailCount w := by simpa using h₁
    rw [Array.getElem_ofFn, hC' ⟨a, ha⟩, getElem!_pos _ a h₂]
  refine ⟨st, hst', hkR, ?_⟩
  have e : st.rounds[openedOf m i s] =
      ({ n := st.rounds[openedOf m i s].n, sha := st.rounds[openedOf m i s].sha,
         coins := st.rounds[openedOf m i s].coins, bytes := st.rounds[openedOf m i s].bytes } : Flock.Round) := rfl
  rw [e, hB', hS', hN', ← hcoins, ← hdg]
  all_goals rfl

theorem session_answered {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R J : ℕ}
    {Reg : Type} (reg : Option Claims.Msg → Reg) (hJ : J ≤ 1) {tb : String} {per : ℕ} (hper : 0 < per)
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    {rd : Reading R} (hrd : rd = executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {q : Pos (V.game θ F c) σ Claims.Msg (Fin R)}
    (hq : q.Inv (roundCuts V θ F c rd)) {z : CoinTape (V.game θ F c) σ} (hz : ∃ y, z = q.back y)
    (hn : NoMiss V θ F c rd σ z) {sub : SubmitJ J} {cs : Fin sub.n → 𝔽} {m : List Release}
    (hsub : (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)).sub J q.modes =
      some sub)
    (hlog : LoggedAfter rd.round q m (roundBits fieldEnc cs) (CoinTape.run (V.game θ F c) σ z).log)
    (hcut : CutsAt rd.reveal q m (CoinTape.run (V.game θ F c) σ z).log) {u : ℕ}
    (hses : ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j m → j.val / per = u)
    {text : String} {J' : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J' sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) {dg : ByteArray → ByteArray}
    (hdg : spec.dg = dg) (htab : spec.tables = #[tb]) :
    ∃ (j : Fin R) (k i s n : ℕ) (x : Release) (L' : List Release), j.val / per = u ∧
      slotOf (#[tb] : Array String).size per (j.val % per) = .rep sub.tab.val sub.rep k ∧ rd.round j m x.frame ∧
      (CoinTape.run (V.game θ F c) σ z).log = m ++ x :: L' ∧ x.frame = .opening i s n ∧ Served m i s n ∧
      (openedTrees m)[u]? = some i ∧
      ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb sub.rep.val) ∧
      ∃ st, sess.streams.find? (·.name == repStreamName tb sub.rep.val) = some st ∧
        ∃ hk : openedOf m i s < st.rounds.size, roundOf dg sub.bytes sub.n cs = st.rounds[openedOf m i s] := by
  subst hrd
  obtain ⟨ht', r, k, w, j, i, s, n, b, L', hLj, hj, hw, hsa, hsv, hu', hname, hk, he, hP⟩ :=
    LoggedAfter.session_slot reg (by simp) hJ hper (sessionReveal #[tb] ev) record₀ recordDigest hsub hlog
  subst hsa
  have hju := hses j ⟨_, L', he, hP⟩
  rw [hju] at hu'
  have hname' : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb r.val) := hname
  obtain ⟨st, hst, hkR, hround⟩ := session_stop_round per ev record₀ recordDigest rfl hq hz hn hLj hju hj hw he
    hcut hu' hk hname' hrec hR hd hv hret h512 hdg htab
  subst hk
  exact ⟨j, _, i, s, n, _, L', hju, hLj, hP, he, rfl, hsv, hu', hname', st, hst, hkR, hround⟩

theorem session_reps {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {R J : ℕ}
    {Reg : Type} (reg : Option Claims.Msg → Reg) (hJ : J ≤ 1) {tb : String} {per : ℕ} (hper : 1026 ≤ per)
    (ev : ℕ → Exchange → Option Claims.Msg) (record₀ : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg)
    {rd : Reading R} (hrd : rd = executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest)
    {σ : Strategy (V.game θ F c)} {z : CoinTape (V.game θ F c) σ} (hn : NoMiss V θ F c rd σ z)
    {as : List (Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))}
    {ms : List (List Release)} {L₀ : List Release} {u τ : ℕ} (hu : (u + 1) * per ≤ R)
    (hF : List.Forall₂ (fun a m =>
      (decodersAt reg fun j : Fin R => slotOf (#[tb] : Array String).size per (j.val % per)).sub J
        a.1.toStop.modes = some a.2.1 ∧
      LoggedAfter rd.round a.1.toStop m (roundBits fieldEnc a.2.2) (CoinTape.run (V.game θ F c) σ z).log ∧
      CutsAt rd.reveal a.1.toStop m (CoinTape.run (V.game θ F c) σ z).log) as ms)
    (hI : ∀ a ∈ as, a.1.toStop.Inv (roundCuts V θ F c rd) ∧ ∃ y, z = a.1.toStop.back y)
    (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hses : ∀ m ∈ ms, ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j m → j.val / per = u)
    (hcov : ∀ (j : Fin R) L₁, j.val / per = u →
      PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j L₁ → L₁ = L₀ ∨ L₁ ∈ ms ∨
        slotOf (#[tb] : Array String).size per (j.val % per) = closeSlot (#[tb] : Array String).size)
    (hL₀ : ∀ j : Fin R, PicksIn rd.round (CoinTape.run (V.game θ F c) σ z).log j L₀ →
      slotOf (#[tb] : Array String).size per (j.val % per) = .link)
    (hτ : (openedTrees (CoinTape.run (V.game θ F c) σ z).log)[u]? = some τ)
    (hnd : ∀ sp ∈ helloSpecs (CoinTape.run (V.game θ F c) σ z).log, sp.streams.toList.Nodup)
    (t : Fin J) (htz : t.val = 0) (ρ : Fin 2)
    {text : String} {J' : Json} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {proofs : Array ByteArray}
    {sess : Flock.RunRecord} {w₀ : Claims.Msg}
    (hrec : sentRecordAt u (CoinTape.run (V.game θ F c) σ z).sent = some text)
    (hR : SessionRead (CoinTape.run (V.game θ F c) σ z).log #[tb] u text J' sess w₀)
    (hd : Flock.Record.decode spec text = .ok sess) (hv : Flock.Zk.verify spec tabs text proofs = .ok ())
    (hret : spec.retainRounds = true) (h512 : spec.sha512Digests = true) {dg : ByteArray → ByteArray}
    (hdg : spec.dg = dg) (htab : spec.tables = #[tb]) :
    as.foldl (fun rs a => addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2))
      (fun _ _ => []) t ρ = repRounds spec sess t.val ρ.val := by
  subst hrd
  have h0 : tb = spec.tables[0]! := by
    rw [htab]
    all_goals rfl
  obtain ⟨tv, htv⟩ := t
  obtain rfl : tv = 0 := htz
  show _ = repRounds spec sess 0 ρ.val
  have hS := logServed_run V θ F c σ z
  have hnT := repStreamName_ne tb ρ.val
  obtain ⟨h0t, sch, tPad, cap, -, hrep⟩ := verifyRep_of_accepted hv hd ρ
  obtain ⟨st, _, _, _, hst, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrep
  have hst' : sess.streams.find? (·.name == repStreamName tb ρ.val) = some st := by
    rw [h0]
    exact hst
  have hrr := recordRounds_decode hd hR.parse hnT hst'
  have hRR : repRounds spec sess 0 ρ.val = st.rounds.toList := by
    show ((sess.streams.find? (·.name == repStreamName spec.tables[0]! ρ.val)).map (·.rounds.toList)).getD [] = _
    rw [← h0, hst']
    all_goals rfl
  have hstop : ∀ a m, (a, m) ∈ as.zip ms → ∃ (j : Fin R) (k i s n : ℕ) (x : Release) (L' : List Release),
      j.val / per = u ∧ slotOf (#[tb] : Array String).size per (j.val % per) = .rep a.2.1.tab.val a.2.1.rep k ∧
      (executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest).round j m x.frame ∧
      (CoinTape.run (V.game θ F c) σ z).log = m ++ x :: L' ∧ x.frame = .opening i s n ∧ Served m i s n ∧
      (openedTrees m)[u]? = some i ∧
      ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = some (repStreamName tb a.2.1.rep.val) ∧
      ∃ st', sess.streams.find? (·.name == repStreamName tb a.2.1.rep.val) = some st' ∧
        ∃ hk : openedOf m i s < st'.rounds.size,
          roundOf dg a.2.1.bytes a.2.1.n a.2.2 = st'.rounds[openedOf m i s] := by
    intro a m hx
    obtain ⟨hsub, hlog, hcut⟩ := List.forall₂_zip hF hx
    obtain ⟨hinv, hzq⟩ := hI a (List.of_mem_zip hx).1
    exact session_answered reg hJ (by omega) ev record₀ recordDigest rfl hinv hzq hn hsub hlog hcut
      (hses m (List.of_mem_zip hx).2) hrec hR hd hv hret h512 hdg htab
  by_cases hpos : 0 < st.rounds.size
  · obtain ⟨s, hs0⟩ := hR.key (repStreamName tb ρ.val) (by rw [hrr]; exact hpos) τ hτ
    have hs : ((helloSpecs (CoinTape.run (V.game θ F c) σ z).log)[τ]?.bind fun sp => sp.streams[s]?) =
        (#[tb] : Array String)[(⟨0, htv⟩ : Fin J).val]?.map (repStreamName · ρ.val) := hs0
    have hF' : List.Forall₂ (fun a m => ∃ (j : Fin R) (k : ℕ) (x : Release) (L' : List Release),
        j.val / per = u ∧ slotOf (#[tb] : Array String).size per (j.val % per) = .rep a.2.1.tab.val a.2.1.rep k ∧
          (executionPhaseReadingAt #[tb] per (sessionReveal #[tb] ev) record₀ recordDigest).round j m x.frame ∧
          (CoinTape.run (V.game θ F c) σ z).log = m ++ x :: L') as ms := by
      refine List.forall₂_iff_zip.2 ⟨hF.length_eq, fun {a m} hx => ?_⟩
      obtain ⟨j, k, _, _, _, x, L', hju, hLj, hP, he, -⟩ := hstop a m hx
      exact ⟨j, k, x, L', hju, hLj, hP, he⟩
    have hfirst : ∀ L₁ x L₂ s' n, (CoinTape.run (V.game θ F c) σ z).log = L₁ ++ x :: L₂ →
        x.frame = .opening τ s' n → τ ∉ openedTrees L₁ →
        ((helloSpecs L₁)[τ]?.bind fun sp => sp.streams[s']?) = some _root_.Spot.Challenger.TREE_SESSION := by
      intro L₁ x L₂ s' n he hx hni
      rcases opened_at_or_new he hx hτ with ⟨-, hlen⟩ | hmem
      · exact (hR.link L₁ x L₂ τ s' n he hx hni hlen).1
      · exact absurd (List.mem_of_getElem? hmem) hni
    have hle : openedOf (CoinTape.run (V.game θ F c) σ z).log τ s ≤ 512 := hS.openedOf_le (hR.tree τ hτ) s
    have hcovL : ∀ k < openedOf (CoinTape.run (V.game θ F c) σ z).log τ s, ∃ j : Fin R,
        j.val / per = u ∧ slotOf (#[tb] : Array String).size per (j.val % per) = .rep (⟨0, htv⟩ : Fin J).val ρ k :=
      fun k hk => executionPhase_covers (#[tb] : Array String).size (K := 512) (by omega) hu ρ k (by omega)
    have hf : ∀ a m, (a, m) ∈ as.zip ms → a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = ρ →
        roundOf dg a.2.1.bytes a.2.1.n a.2.2 =
          st.rounds[openedOf m τ s]! := by
      intro a m hx htr
      obtain ⟨j, k, i, s', n, x, L', -, -, -, he, -, ⟨sp, hsp, hslt, -⟩, hu', hname, st', hst'', hkR, hround⟩ :=
        hstop a m hx
      rw [htr.2] at hname hst''
      have hi : τ = i := by
        have h := openedTrees_getElem?_append hu' (x :: L')
        rw [← he, hτ] at h
        exact Option.some.inj h
      subst hi
      have hss : s = s' := (stream_eq he hnd ⟨sp, hsp, hslt⟩ hname hs0).symm
      subst hss
      have hE : st = st' := Option.some.inj (hst'.symm.trans hst'')
      subst hE
      exact hround.trans (getElem!_pos st.rounds (openedOf m τ s) hkR).symm
    have hlen : openedOf (CoinTape.run (V.game θ F c) σ z).log τ s = st.rounds.size := by
      have h1 := hR.openedOf_le hS hτ hs0 hnT
      have h2 := hR.full τ s _ hτ hs0 hnT
      rw [hrr] at h1 h2
      omega
    have key := session_stream_rounds
      (γ := Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))) (J := J)
      (fun a => a.2.1.tab) (fun a => a.2.1.rep) (fun a => roundOf dg a.2.1.bytes a.2.1.n a.2.2)
      (tables := #[tb]) (by simp) (per := per) (by omega) (sessionReveal #[tb] ev) record₀ recordDigest hτ hF' hpw
      hcov hL₀ hS hnd hfirst ⟨0, htv⟩ (one_table_pos tb) ρ hs hcovL (fun k => st.rounds[k]!) hf
    refine key.trans ?_
    rw [hlen, map_range_getElem!, hRR]
  · have hzero : st.rounds.size = 0 := by omega
    have hnone : ∀ a ∈ as, ¬ (a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = ρ) := by
      intro a ha htr
      have ha' : a ∈ (as.zip ms).map Prod.fst := by
        rw [List.map_fst_zip hF.length_eq.le]
        exact ha
      obtain ⟨⟨a', m⟩, hx, rfl⟩ := List.mem_map.1 ha'
      have hrep : a'.2.1.rep = ρ := htr.2
      obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, st', hst'', hkR, _⟩ := hstop a' m hx
      rw [hrep] at hst''
      obtain rfl : st' = st := Option.some.inj (hst''.symm.trans hst')
      omega
    have hfil : as.filter (fun a => decide (a.2.1.tab = (⟨0, htv⟩ : Fin J) ∧ a.2.1.rep = ρ)) = [] :=
      List.filter_eq_nil_iff.2 fun a ha h => hnone a ha (of_decide_eq_true h)
    have hnil : st.rounds.toList = [] := List.eq_nil_of_length_eq_zero (by simpa using hzero)
    refine (foldl_addRoundJ
      (fun a : Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)) => a.2.1.tab)
      (fun a : Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)) => a.2.1.rep)
      (fun a : Pos (V.game θ F c) σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)) =>
        roundOf dg a.2.1.bytes a.2.1.n a.2.2)
      as (fun _ _ => []) ⟨0, htv⟩ ρ).trans ?_
    rw [hfil, hRR, hnil]
    all_goals rfl

end Verity.Sampled
