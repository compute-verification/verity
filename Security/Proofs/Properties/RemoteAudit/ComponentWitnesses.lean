import Proofs.Accounting.RemoteHeld.Component
import Proofs.Accounting.RemoteHeld.Draw
import Proofs.Pouw.Prob

/-!
# The bandwidth component's properties hold at the honest node and fail at a verifier that accepts everything

Each of `Properties.RemoteAudit.Component`'s three properties, and the memory line
`Properties.RemoteAudit.Line.LineHeld`, is its form over the verifier at the deployed one (`acceptIsVerified_iff`,
`publishedEps_iff`, `lineHeldAtComponent_iff`, `lineHeld_iff`), and each form has its witnesses:

* **AcceptIsVerified** (`AcceptIsVerifiedFor`). Honest: the deployed period's parameters and log decode to its session
  at one window a draw, which the check leaves undecided until its windows verify (`acceptIsVerified_honest`), and the
  check accepts, publishing `certify`'s outputs, wherever `Shared.run g` accepts the session it decodes
  (`acceptIsVerified_complete`). Nontrivial: a check that accepts everything accepts a log that decodes to no session
  (`acceptIsVerified_nontrivial`).
* **PublishedEps** (`PublishedEpsFor`), whose antecedent is the same acceptance, so its honest witnesses are
  `AcceptIsVerified`'s. Nontrivial: a check that accepts every log, certifying a byte read at eps 0, accepts the
  deployed period's log, and 0 is below the draw term at every node of positive δ (`drawTerm_pos`,
  `publishedEps_nontrivial`).
* **LineHeld** (`LineHeldBy`) and **LineHeldAtComponent** (`LineHeldAtComponentBy`), which take the same premises.
  Honest: every premise holds at a node that holds its whole store in memory throughout, every receipt correct, no
  outside device, nothing but the verifier's messages entering it as its certifiers count, and the sampler's fallbacks
  0, on a schedule that runs each pass inside its window with each draw's receipts fixed (`HonestSchedule`), and at its
  correct receipts the distinguishers' advantage and the hiding terms are 0 (`lineHeldAtComponent_honest`); and there
  the event is empty: the certifiers count nothing else entering and every line credits no more bits read than the
  passes read (`lineHeld_honest`). Nontrivial: at a verifier that accepts everything and records one summary for every
  record, the bound fails on a node whose four passes read its one block once (`lineHeld_nontrivial`,
  `lineHeldAtComponent_nontrivial`); and the line credits only what the passes read: on one record and summary its
  claim holds of a trace with the store in memory and fails of the same trace with memory emptied at tick 10
  (`lineHeldAtComponent_bites`).

At the deployed verifier the premises' acceptance is load-bearing and `Plays` is not: the line reads the record
through `Record.summary`, which is some summary only where the Lean verifier accepts, so the event's failed read claim
alone gives `Verified` (`verified_of_not_readBy`) and the bound holds at any acceptance in the event
(`lineHeldAtComponentBy_record`). The nontrivial witnesses therefore replace the verdict and the record with an
accept-all's as well.

The toy instance (`Toy`) is a node whose parameters have no window (`Node.windows` 0) at four windows a draw, so no
draw (`Node.draws` 0), one block of one byte, whose recorded summary has four windows of one block each, one pass a
group, and a line that reads no wrong receipt, so it credits `8 · 1 · (4 · 1 − 0) = 32` bits read.
-/

deriving instance DecidableEq for RemoteAudit.Line.Group

namespace Properties.RemoteAudit.Component

open Finset Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.Line

theorem acceptIsVerified_iff : AcceptIsVerified ↔ AcceptIsVerifiedFor RemoteAudit.Protocol.check := Iff.rfl

theorem publishedEps_iff : PublishedEps ↔ PublishedEpsFor RemoteAudit.Protocol.check := Iff.rfl

theorem lineHeldAtComponent_iff : LineHeldAtComponent ↔ LineHeldAtComponentBy Plays Verified Record.summary :=
  Iff.rfl

theorem lineHeld_iff : LineHeld ↔ LineHeldBy Verified Record.summary := Iff.rfl

/-! ## The premises at an honest node -/

/-- A trace read over its own registered blocks is itself. -/
theorem overBlocks_self {Ω : Type} (T : Trace Ω) : overBlocks T T.blocks = T := by
  have h : ∀ s : Finset (Fin T.blocks),
      (Finset.univ.filter fun j : Fin T.blocks => (j : ℕ) ∈ s.image Fin.val) = s := fun s => by
    ext j
    rw [Finset.mem_filter, Finset.mem_image]
    exact ⟨fun ⟨_, a, ha, e⟩ => Fin.ext e ▸ ha, fun hj => ⟨Finset.mem_univ j, j, hj, rfl⟩⟩
  unfold overBlocks
  simp only [h]
  all_goals rfl

/-- The honest trace is over the node's registered blocks, so its trace of record is it with the Lean verifier's
verdict. -/
theorem ofRecord_honest {Ω : Type} (node : Node) (passes : ℕ) (tick : Ω → ℕ → ℕ) (rec : Ω → Record) :
    ofRecord (honestTrace node passes tick) node rec =
      { honestTrace node passes tick with accepts := fun ω => Verified node (rec ω) } := by
  show { overBlocks (honestTrace node passes tick) (honestTrace node passes tick).blocks with
      accepts := fun ω => Verified node (rec ω) } = _
  rw [overBlocks_self]

theorem covers_univ {Ω : Type} (T : Trace Ω) (h : ∀ ω p, T.plan ω p = univ) : Covers T := fun ω _ p _ => h ω p

theorem covered_noOutside {Ω : Type} (T : Trace Ω) (I : Ingress Ω) : Covered T (noOutside T) I :=
  fun _ _ _ _ _ _ _ _ => ⟨rfl, rfl⟩

theorem ingressCounted_quiet {Ω : Type} [Fintype Ω] (T : Trace Ω) : IngressCounted T (quietIngress Ω) 0 := by
  unfold IngressCounted Trace.Holds
  exact le_of_eq (Pouw.pr_eq_zero fun _ hω => hω.2 fun _ _ _ _ _ _ => rfl)

theorem sketch_zero {Ω : Type} [Fintype Ω] (T : Trace Ω) (O : Outside T)
    (h : ∀ ω p, p < T.passes → ∃ t, T.tick ω p ≤ t ∧ t < T.tick ω (p + 1) ∧ T.present ω t = univ) :
    SketchBoundAt T O 0 := by
  unfold SketchBoundAt Trace.Holds
  refine le_of_eq (Pouw.pr_eq_zero fun ω hω => hω.2 fun p hp j _ => ?_)
  obtain ⟨t, h1, h2, h3⟩ := h ω p hp
  exact ⟨t, h1, h2, mem_union_left _ (h3 ▸ mem_univ j)⟩

theorem counted_of {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m s κ X)
    (T : Trace (Coins n κ X)) (h : ∀ ω p, T.plan ω p ⊆ T.correct ω p) : Counted D T := fun ω _ => by
  have hw : T.wrong ω = 0 := by
    unfold Trace.wrong
    exact Finset.sum_eq_zero fun p _ => by rw [Finset.sdiff_eq_empty_iff_subset.2 (h ω p), card_empty]
  rw [hw]
  exact Nat.zero_le _

theorem getD_map_val_lt {N : ℕ} (hN : 0 < N) : ∀ o : Option (Fin N), (o.map Fin.val).getD 0 < N
  | none => hN
  | some i => i.isLt

theorem offset_lt {N : ℕ} (hN : 0 < N) (b : Pous.Bits 512) : offset N 0 b < N := getD_map_val_lt hN _

theorem offsets_zero_lt {m U k : ℕ} (b : Blocks m) (hk : k ≤ U) (hkm : k ≤ m) :
    ∀ l, l < k → l + offsets U b (fun _ => 0) l < U := by
  intro l hl
  have h : l < m := by omega
  have := offset_lt (N := U - l) (by omega) (b ⟨l, h⟩)
  unfold offsets
  rw [dif_pos h]
  show l + offset (U - l) 0 (b ⟨l, h⟩) < U
  omega

/-- With the sampler's fallbacks 0, every receipt a draw opens is among its receipts, so at correct receipts that
hold all of them nothing is refused. -/
theorem refused_honest {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] (D : Session n m s κ X)
    (hc : ∀ t ω, range (D.size t ω : ℕ) ⊆ D.correct t ω) (prf : Fin n → Pous.Bits κ → Blocks m)
    (T : Trace (Coins n κ X)) (R : Receipts n κ X) : Refused D prf (fun _ _ _ => 0) T R := by
  intro ω _ t p hp
  left
  apply hc t ω
  unfold Session.opened openings at hp
  obtain ⟨q, hq, rfl⟩ := mem_image.1 hp
  have hk : D.draws t ω ≤ D.size t ω := min_le_right _ _
  exact mem_range.2 (Accounting.SecurityProofs.Remote.Draw.arrange_lt hk
    (offsets_zero_lt (prf t (ω.2 t)) hk (Accounting.SecurityProofs.Remote.Draw.draws_le D t ω)) q
    (lt_of_lt_of_le (mem_range.1 hq) hk))

/-- **The premises of `LineHeld` and `LineHeldAtComponent` hold at the honest node.** On any node, at any records of
the prover's whose schedule is honest (`HonestSchedule`), the honest trace (`honestTrace`: the whole store in node
memory at every tick, nothing on a chip, rebuilt or read in, every receipt correct), no outside device (`noOutside`),
nothing but the verifier's messages entering it as its certifiers count (`quietIngress`), correct receipts holding
every receipt a draw can have and the sampler's fallbacks 0, every premise holds: `Paced` at the record's groups,
`Covers`, `SketchBoundAt` at 0, `Covered`, `IngressCounted` at 0, the draw's `Live` at every commitment, `Refused`
and `Counted`; and the bound's terms the prover could raise are 0: each draw's distinguisher has no advantage on its
coin function and each hiding term is 0. -/
theorem lineHeldAtComponent_honest (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X]
    (prf : Fin node.draws → Pous.Bits κ → Blocks receiptMax) (recs : Cm → Coins node.draws κ X → Record)
    (com : Coins node.draws κ X → Cm) (R : Receipts node.draws κ X) (passes : ℕ)
    (tick : Coins node.draws κ X → ℕ → ℕ) (hS : HonestSchedule node passes tick recs com) :
    Paced (ofRecord (honestTrace node passes tick) node (played recs com)) (recordPace node (played recs com)) ∧
    Covers (ofRecord (honestTrace node passes tick) node (played recs com)) ∧
    SketchBoundAt (ofRecord (honestTrace node passes tick) node (played recs com))
      (noOutside (ofRecord (honestTrace node passes tick) node (played recs com))) 0 ∧
    Covered (ofRecord (honestTrace node passes tick) node (played recs com))
      (noOutside (ofRecord (honestTrace node passes tick) node (played recs com))) (quietIngress _) ∧
    IngressCounted (ofRecord (honestTrace node passes tick) node (played recs com)) (quietIngress _) 0 ∧
    (∀ c, Live ((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).view c)) ∧
    Refused ((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play com) prf (fun _ _ _ => 0)
      (ofRecord (honestTrace node passes tick) node (played recs com)) R ∧
    Counted ((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play com)
      (ofRecord (honestTrace node passes tick) node (played recs com)) ∧
    (∀ t, Pous.StreamedWipe.prfAdvantage (prf t)
      (((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play
        (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t) = 0) ∧
    ∀ t, (recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).hide com prf (node.delta : ℝ) t = 0 := by
  obtain ⟨hW, htick, hsize, hgs⟩ := hS
  rw [ofRecord_honest]
  have hc : ∀ D : Session node.draws receiptMax (fun _ => node.params.samples) κ X,
      (∀ t ω, D.correct t ω = Finset.range (receiptMax + 1)) → ∀ t ω, range (D.size t ω : ℕ) ⊆ D.correct t ω :=
    fun D h t ω => by
      rw [h]
      exact Finset.range_subset_range.2 (D.size t ω).isLt.le
  refine ⟨?_, covers_univ _ fun _ _ => rfl,
    sketch_zero _ _ fun ω p _ => ⟨tick ω p, le_refl _, htick ω p, Finset.union_empty _⟩, covered_noOutside _ _,
    ingressCounted_quiet _, fun c t ω ω' _ _ => ⟨Fin.ext (congrArg (min · receiptMax) (hsize c c ω ω' t)), rfl⟩,
    refused_honest _ (hc _ fun _ _ => rfl) prf _ R, counted_of _ _ fun _ _ => Finset.Subset.refl _,
    fun t => Accounting.SecurityProofs.Remote.Draw.prfAdvantage_honest _ (hc _ fun _ _ => rfl) prf _ t,
    fun t => ?_⟩
  · intro ω hv
    obtain ⟨hsz, hpass, hmono, hfp, hrun⟩ := hgs ω hv _ rfl
    refine ⟨monotone_nat_of_le_succ fun p => (htick ω p).le, by simp [recordPace, firstPass], ?_,
      fun g hg => ?_, fun g h hgh => ?_, fun g hg => ?_, fun g hg p h1 h2 => ?_⟩
    · show firstPass _ (node.windows - 1 + 1) ≤ passes
      rw [Nat.sub_add_cancel hW]
      exact hfp
    · change g ≤ node.windows - 1 at hg
      show firstPass _ g < firstPass _ (g + 1)
      rw [firstPass_succ]
      have := (hpass g (by omega)).1
      omega
    · show released _ node.windows g ≤ released _ node.windows h
      unfold released
      exact hmono _ _ (min_le_min_right _ hgh) (by omega)
    · change g ≤ node.windows - 1 at hg
      show released _ node.windows (g + 1) ≤ ended _ g
      unfold released ended
      rw [hsz]
      by_cases h1 : g + 1 < node.windows
      · rw [if_pos h1, min_eq_left (by omega : g + 1 ≤ node.windows - 1)]
        exact le_max_right _ _
      · rw [if_neg h1, min_eq_right (by omega : node.windows - 1 ≤ g + 1), (by omega : node.windows - 1 = g)]
        exact (hpass g (by omega)).2
    · change g ≤ node.windows - 1 at hg
      exact hrun g (by omega) p h1 h2
  · have e : (recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play com =
        (recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play (Committed.junk com t) := by
      unfold Committed.play
      congr 1
      funext t' ω
      exact Fin.ext (congrArg (min · receiptMax) (hsize _ _ ω ω t'))
    unfold Committed.hide
    rw [← e, tsub_self, tsub_self, add_zero]

/-! ## What the passes read -/

/-- A pass that plans every registered block, every receipt correct, and runs a tick or more with every registered
block on the node at its first tick reads every registered block. -/
theorem read_full {Ω : Type} (T : Trace Ω) (ω : Ω) (p : ℕ) (hp : T.plan ω p = univ) (hc : T.correct ω p = univ)
    (ht : T.tick ω p < T.tick ω (p + 1)) (hm : T.present ω (T.tick ω p) = univ) : (T.read ω p).card = T.blocks := by
  have hf : univ ⊆ (Ico (T.tick ω p) (T.tick ω (p + 1))).biUnion (T.present ω) := fun j _ =>
    Finset.mem_biUnion.2 ⟨T.tick ω p, Finset.mem_Ico.2 ⟨le_rfl, ht⟩, hm ▸ Finset.mem_univ j⟩
  unfold Trace.read
  rw [hp, hc, Finset.inter_self, Finset.inter_eq_left.2 hf, Finset.card_univ, Fintype.card_fin]

/-- A pass with no registered block on the node at any of its ticks reads none. -/
theorem read_absent {Ω : Type} (T : Trace Ω) (ω : Ω) (p : ℕ)
    (h : ∀ t, T.tick ω p ≤ t → t < T.tick ω (p + 1) → T.present ω t = ∅) : T.read ω p = ∅ := by
  unfold Trace.read
  refine Finset.subset_empty.1 fun j hj => ?_
  obtain ⟨t, ht, htj⟩ := Finset.mem_biUnion.1 (Finset.mem_inter.1 hj).2
  rwa [h t (Finset.mem_Ico.1 ht).1 (Finset.mem_Ico.1 ht).2] at htj

/-- **`LineHeld`'s event is empty at the honest node.** At the honest trace on a clock whose passes take a tick or more
each, at every outcome the certifiers count nothing but the verifier's messages entering the node (`quietIngress`), and
the line of any record credits no more bits read than the passes read: every pass reads every registered block
(`read_full`). -/
theorem lineHeld_honest (node : Node) {Ω : Type} (passes : ℕ) (tick : Ω → ℕ → ℕ) (rec : Ω → Record)
    (htick : ∀ ω p, tick ω p < tick ω (p + 1)) (ω : Ω) :
    Accounting.Remote.Held.Quiet (ofRecord (honestTrace node passes tick) node rec) (quietIngress Ω).counted ω ∧
      ReadAt (ofRecord (honestTrace node passes tick) node rec) ω node (rec ω) := by
  refine ⟨fun _ _ _ _ _ => rfl, fun l hline => ?_⟩
  obtain ⟨sm, gs, ws, -, -, -, hln⟩ := Properties.RemoteAudit.Line.SecurityProofs.lineOf_ok hline
  have hw : (0 : ℝ) ≤ l.wrong := by exact_mod_cast (lineChecked _ _ _ _ _ _ _ _ _ _ hln).2.2.2.2.2.2.2.2
  have hr : ∀ P, (ofRecord (honestTrace node passes tick) node rec).reads ω P = P * node.blocks := fun P => by
    rw [ofRecord_honest]
    unfold Trace.reads
    rw [Finset.sum_const_nat fun p _ => read_full _ ω p rfl rfl (htick ω p) (Finset.union_empty _),
      Finset.card_range]
    rfl
  rw [hr]
  have hb : (0 : ℝ) ≤ 8 * node.blockBytes := by positivity
  refine mul_le_mul_of_nonneg_left ?_ hb
  linarith

/-! ## At the deployed verifier the event's acceptance is the summary's -/

/-- Where the line's bits-read claim fails under any hypothesis, the Lean verifier accepted the record: its line reads
`Record.summary`, which is some summary only where the session ended accepted. -/
theorem verified_of_not_readBy {Ω : Type} {T : Trace Ω} {ω : Ω} {node : Node} {r : Record} {Q : Prop}
    (h : ¬ (Q → ReadBy Record.summary T ω node r)) : Verified node r := by
  have hne : Record.summary node r ≠ none := fun hn => h fun _ => by simp [ReadBy, lineBy, hn]
  obtain ⟨sm, hs⟩ := Option.ne_none_iff_exists'.1 hne
  unfold Record.summary at hs
  cases hr : r.run node with
  | error _ => rw [hr] at hs; cases hs
  | ok x =>
    obtain ⟨rs, s⟩ := x
    rw [hr] at hs
    refine ⟨rs, s, hr, ?_⟩
    have hs' : s.summary = some sm := hs
    unfold RemoteAudit.Session.State.summary at hs'
    split at hs'
    · cases hs'
    · rename_i hc
      simpa using hc

/-- **The bound holds at any acceptance in the event**: at the deployed verdict and record, whatever `acc` is. -/
theorem lineHeldAtComponentBy_record (acc : Node → Record → ByteArray → ByteArray → ByteArray → Prop) :
    LineHeldAtComponentBy acc Verified Record.summary := by
  intro node κ X Cm _ prf rest recs correct com T R δs δnet a b e r F O I hP hCv hSk hCov hIc hLive hRef hCnt
  exact le_trans (Pouw.pr_mono fun ω h => ⟨verified_of_not_readBy h.2, h.2⟩)
    (Properties.RemoteAudit.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs δnet O I hP hCv
      hSk hCov hIc hLive hRef hCnt)

/-! ## A verifier that accepts everything -/

namespace Toy

def codec : RemoteAudit.Codec.Codec := ⟨RemoteAudit.Codec.P512, 2, false⟩

def reg : RemoteAudit.Registration where
  node := .empty
  blocks := 1
  payloadRoot := .empty
  codecSeed := .empty
  codec := codec
  receipt := .sketch
  rows := 0
  key := ⟨codec, #[], 0, none, .empty⟩

def win : RemoteAudit.Session.WindowSummary where
  lengths := #[1]
  steps := #[]
  post := #[]
  heads := #[]
  stale := #[]
  arrivedNs := #[]
  releasedNs := 0
  deadlineNs := 0
  openedNs := 0
  drawn := 1

def sm : RemoteAudit.Session.Summary where
  windows := #[win, win, win, win]
  blocks := 1
  appended := 0
  released := 0
  lifetime := 1
  pace := none
  appends := none

def seen : Array _root_.RemoteAudit.Line.Seen :=
  #[⟨0, 5, 1, 0, 1, 0⟩, ⟨10, 15, 1, 0, 1, 0⟩, ⟨20, 25, 1, 0, 1, 0⟩, ⟨30, 35, 1, 0, 1, 0⟩]

def gs : Array _root_.RemoteAudit.Line.Group :=
  #[⟨0, 5, 1, 1, 0, 1, 0⟩, ⟨10, 15, 1, 1, 0, 1, 0⟩, ⟨20, 25, 1, 1, 0, 1, 0⟩, ⟨30, 35, 1, 1, 0, 1, 0⟩]

def rates : _root_.RemoteAudit.Line.Rates := ⟨0, 1, 0, 0, 0, 0, 0⟩

def node : Node where
  params := { deadlineNs := 1, windows := 0, samples := 4, minPasses := 1, windowNs := 1, lifetime := 1 }
  groupsPerCoin := 4
  reg := reg
  per := 2
  blockBytes := 1
  chipBytes := 0
  nodeBytes := 1
  rates := rates
  delta := ⟨1 / 2, by norm_num⟩

def rec : Record where
  coins := #[]
  appendCoins := #[]
  notBeforeNs := 0
  events := #[]
  seen := seen

abbrev sum : Node → Record → Option RemoteAudit.Session.Summary := fun _ _ => some sm

/-- One block, held in node memory until tick 10 and not after, one pass over `[0, 10)` and the passes after it
empty. -/
def trace {Ω : Type} : Trace Ω where
  blocks := 1
  passes := 1
  accepts _ := True
  tick _ p := if p = 0 then 0 else 10
  plan _ _ := univ
  correct _ _ := univ
  mem _ t := if t < 10 then univ else ∅
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

/-- One block in node memory at every tick, four passes of ten ticks each. -/
def steady : Trace Unit where
  blocks := 1
  passes := 4
  accepts _ := True
  tick _ p := 10 * p
  plan _ _ := univ
  correct _ _ := univ
  mem _ _ := univ
  chip _ _ := ∅
  rebuilt _ _ := 0
  disk _ _ := 0
  link _ _ := 0

theorem draws_zero : node.draws = 0 := by decide

instance instIsEmptyFinDrawsNode : IsEmpty (Fin node.draws) := by
  rw [draws_zero]
  infer_instance

def receipts : Receipts node.draws 0 Unit :=
  ⟨fun _ _ => 0, fun t => isEmptyElim t, fun t => isEmptyElim t, fun t => isEmptyElim t, fun t => isEmptyElim t⟩

end Toy

open _root_.RemoteAudit.Line in
theorem toy_ofs : ofSummaryAt 4 Toy.sm 4 Toy.seen = .ok (Toy.gs, #[(4, 4)]) := by
  have h : (ofSummaryAt 4 Toy.sm 4 Toy.seen).toOption = some (Toy.gs, #[(4, 4)]) := by decide +kernel
  revert h
  cases ofSummaryAt 4 Toy.sm 4 Toy.seen with
  | error _ => intro h; cases h
  | ok x => intro h; cases h; rfl

theorem toy_pool : ∃ b, RemoteAudit.Pooled.pooled #[(4, 4)] (Toy.node.delta : ℚ) = .ok b ∧ b.wrong = 0 := by
  have h : (RemoteAudit.Pooled.pooled #[(4, 4)] (Toy.node.delta : ℚ)).toOption.map (·.wrong) = some 0 := by
    decide +kernel
  revert h
  cases RemoteAudit.Pooled.pooled #[(4, 4)] (Toy.node.delta : ℚ) with
  | error _ => intro h; cases h
  | ok b => intro h; exact ⟨b, rfl, Option.some.inj h⟩

/-- The wrong receipts' greedy fill (`stretches`) over groups `l` from `st`. -/
def greedy (v cap : ℕ → ℚ) : List ℕ → ℚ × ℚ → ℚ × ℚ
  | [], st => st
  | g :: l, st => if st.1 ≤ 0 then st else greedy v cap l (st.1 - min st.1 (cap g), st.2 + min st.1 (cap g) * v g)

theorem forIn_greedy (v cap : ℕ → ℚ) (l : List ℕ) (st : ℚ × ℚ) :
    forIn (m := Except String) l st (fun g s => if s.1 ≤ 0 then pure (ForInStep.done (s.1, s.2))
      else pure (ForInStep.yield (s.1 - min s.1 (cap g), s.2 + min s.1 (cap g) * v g))) =
      pure (greedy v cap l st) := by
  induction l generalizing st with
  | nil => rfl
  | cons g l ih =>
    rw [List.forIn_cons]
    by_cases h : st.1 ≤ 0
    · simp [h, greedy]
    · simp [h, greedy, ih]

/-- With no wrong receipt to place, the fill places none, whatever the groups' order (`mergeSort`'s, which the kernel
does not evaluate). -/
theorem forIn_fill_nonpos (v cap : ℕ → ℚ) (l : List ℕ) (st : ℚ × ℚ) (h : st.1 ≤ 0) :
    forIn (m := Except String) l st (fun g s => if s.1 ≤ 0 then pure (ForInStep.done (s.1, s.2))
      else pure (ForInStep.yield (s.1 - min s.1 (cap g), s.2 + min s.1 (cap g) * v g))) = pure st := by
  rw [forIn_greedy]
  cases l <;> simp [greedy, h]

/-- Whether the code computes a line, reading no wrong receipt. -/
def probe (r : Except String _root_.RemoteAudit.Line.Line) : Bool :=
  match r with
  | .ok l => l.wrong == 0
  | .error _ => false

theorem probe_extract {r : Except String _root_.RemoteAudit.Line.Line} (h : probe r = true) :
    ∃ l, r = .ok l ∧ l.wrong = 0 := by
  cases r with
  | error _ => cases h
  | ok l => exact ⟨l, rfl, by simpa [probe] using h⟩

open _root_.RemoteAudit.Line in
theorem toy_probe : probe (lineOfRecordAt 4 Toy.sm 4 Toy.seen 2 1 0 1 (Toy.node.delta : ℚ) Toy.rates) = true := by
  obtain ⟨b, hp, hb⟩ := toy_pool
  have h0 : ((0 : ℚ), (0 : ℚ)).1 ≤ 0 := le_refl _
  simp only [lineOfRecordAt, toy_ofs, line, hp, hb, bind, Except.bind, stretches,
    Std.Legacy.Range.forIn_eq_forIn_range', forIn_fill_nonpos _ _ _ _ h0]
  decide +kernel

theorem toy_line : ∃ l, lineBy Toy.sum Toy.node Toy.rec = .ok l ∧ l.wrong = 0 := by
  have e : lineBy Toy.sum Toy.node Toy.rec =
      _root_.RemoteAudit.Line.lineOfRecordAt 4 Toy.sm 4 Toy.seen 2 1 0 1 (Toy.node.delta : ℚ) Toy.rates := rfl
  rw [e]
  exact probe_extract toy_probe

theorem toy_groups : groupsBy Toy.sum Toy.node Toy.rec = Toy.gs := by decide +kernel

theorem toy_passes : firstPass Toy.gs Toy.gs.size = 4 := by decide +kernel

/-- Where the passes after the first read nothing of the one block and the first reads it once, the toy line's claim
of `4 · 1` blocks read, no wrong receipt, fails. -/
theorem toy_not_read {Ω : Type} (T : Trace Ω) (ω : Ω) (hr : T.reads ω 4 ≤ 3) :
    ¬ ReadBy Toy.sum T ω Toy.node Toy.rec := fun h => by
  obtain ⟨l, hl, hw⟩ := toy_line
  have h1 := h l hl
  rw [toy_groups, toy_passes, hw] at h1
  have h2 : (T.reads ω 4 : ℝ) ≤ 3 := by exact_mod_cast hr
  have hb : (Toy.node.blockBytes : ℝ) = 1 := by norm_num [Toy.node]
  have hn : Toy.node.blocks = 1 := rfl
  rw [hb, hn] at h1
  push_cast at h1
  linarith

/-- The toy trace's four passes read its one block once: the first over `[0, 10)`, the rest over no tick. -/
theorem toy_reads {Ω : Type} (ω : Ω) : (Toy.trace (Ω := Ω)).reads ω 4 = 1 := by
  have h0 : ((Toy.trace (Ω := Ω)).read ω 0).card = 1 :=
    read_full _ ω 0 rfl rfl (by show (if 0 = 0 then 0 else 10) < (if 0 + 1 = 0 then 0 else 10); simp)
      (Finset.eq_univ_of_forall fun j => Finset.mem_union_left _ (by
        show j ∈ (if (0 : ℕ) < 10 then Finset.univ else ∅)
        rw [if_pos (by norm_num : (0 : ℕ) < 10)]
        exact Finset.mem_univ j))
  have hp : ∀ p, 1 ≤ p → (Toy.trace (Ω := Ω)).read ω p = ∅ := fun p hp => read_absent _ ω p fun t h1 h2 => by
    change (if p = 0 then 0 else 10) ≤ t at h1
    change t < (if p + 1 = 0 then 0 else 10) at h2
    rw [if_neg (by omega)] at h1
    rw [if_neg (by omega)] at h2
    omega
  simp [Trace.reads, Finset.sum_range_succ, h0, hp 1 le_rfl, hp 2 (by norm_num), hp 3 (by norm_num)]

/-- The toy trace is over the toy node's one registered block, so at any verdict its trace of record is it with that
verdict. -/
theorem traceBy_toy {Ω : Type} (ver : Node → Record → Prop) :
    traceBy (Toy.trace (Ω := Ω)) Toy.node ver (fun _ => Toy.rec) =
      { Toy.trace (Ω := Ω) with accepts := fun _ => ver Toy.node Toy.rec } := by
  show { overBlocks (Toy.trace (Ω := Ω)) (Toy.trace (Ω := Ω)).blocks with
      accepts := fun _ => ver Toy.node Toy.rec } = _
  rw [overBlocks_self]

theorem toy_covers {Ω : Type} (ver : Node → Record → Prop) :
    Covers (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) := by
  rw [traceBy_toy]
  exact covers_univ _ fun _ _ => rfl

theorem toy_reads_by {Ω : Type} (ver : Node → Record → Prop) (ω : Ω) :
    (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec).reads ω 4 ≤ 3 := by
  rw [traceBy_toy]
  exact (toy_reads ω).trans_le (by norm_num)

theorem toy_paced {Ω : Type} (ver : Node → Record → Prop) :
    Paced (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) (paceBy Toy.sum Toy.node fun _ => Toy.rec) := by
  rw [traceBy_toy]
  intro ω _
  refine ⟨fun a b hab => ?_, ?_, ?_, fun g hg => ?_, fun a b _ => ?_, fun g hg => ?_, fun g hg p h1 h2 => ?_⟩
  · show (if a = 0 then 0 else 10) ≤ (if b = 0 then 0 else 10)
    split_ifs <;> omega
  · show firstPass (groupsBy Toy.sum Toy.node Toy.rec) 0 = 0
    simp [firstPass]
  · show firstPass (groupsBy Toy.sum Toy.node Toy.rec) (Toy.node.windows - 1 + 1) ≤ 1
    rw [toy_groups]
    decide +kernel
  · change g ≤ 0 at hg
    obtain rfl := Nat.le_zero.1 hg
    show firstPass (groupsBy Toy.sum Toy.node Toy.rec) 0 < firstPass (groupsBy Toy.sum Toy.node Toy.rec) (0 + 1)
    rw [toy_groups]
    decide +kernel
  · show released (groupsBy Toy.sum Toy.node Toy.rec) Toy.node.windows a ≤
      released (groupsBy Toy.sum Toy.node Toy.rec) Toy.node.windows b
    unfold released
    rw [show Toy.node.windows - 1 = 0 from rfl, Nat.min_zero, Nat.min_zero]
  · change g ≤ 0 at hg
    obtain rfl := Nat.le_zero.1 hg
    show released (groupsBy Toy.sum Toy.node Toy.rec) Toy.node.windows (0 + 1) ≤
      ended (groupsBy Toy.sum Toy.node Toy.rec) 0
    rw [toy_groups]
    decide +kernel
  · change g ≤ 0 at hg
    obtain rfl := Nat.le_zero.1 hg
    change p < firstPass (groupsBy Toy.sum Toy.node Toy.rec) (0 + 1) at h2
    rw [toy_groups, show firstPass Toy.gs (0 + 1) = 1 by decide +kernel] at h2
    obtain rfl : p = 0 := by omega
    show released (groupsBy Toy.sum Toy.node Toy.rec) Toy.node.windows 0 ≤ (if 0 = 0 then 0 else 10) ∧
      (if 0 + 1 = 0 then 0 else 10) ≤ ended (groupsBy Toy.sum Toy.node Toy.rec) 0
    rw [toy_groups]
    decide +kernel

theorem toy_sketch {Ω : Type} [Fintype Ω] (ver : Node → Record → Prop)
    (O : Outside (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec)) :
    SketchBoundAt (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) O 0 := by
  revert O
  rw [traceBy_toy]
  exact fun O => sketch_zero _ O fun _ p hp => by
    change p < 1 at hp
    obtain rfl : p = 0 := by omega
    refine ⟨0, ?_, ?_, ?_⟩
    · show (if 0 = 0 then 0 else 10) ≤ 0
      simp
    · show 0 < (if 0 + 1 = 0 then 0 else 10)
      simp
    · refine Finset.eq_univ_of_forall fun j => Finset.mem_union_left _ ?_
      show j ∈ (if (0 : ℕ) < 10 then Finset.univ else ∅)
      rw [if_pos (by norm_num)]
      exact Finset.mem_univ j

theorem binding_empty {n κ : ℕ} {X : Type} [Fintype X] [IsEmpty (Fin n)] {m : ℕ} {s : Fin n → ℕ}
    (D : Session n m s κ X) (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) : R.binding D prf rest = 0 := by
  simp [Receipts.binding, Pous.pr]

theorem toy_delta : (Toy.node.delta : ℝ) = 1 / 2 := by
  rw [← Rat.cast_nnratCast]
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

/-- **`LineHeld` fails at a verifier that accepts everything.** With a verdict that holds of every record and one
summary recorded for every record, the toy node (no draw, one block, held in node memory until tick 10 only, its four
passes reading it once) meets every premise at `δs = δnet = 0`, while its line credits `4 · 1` blocks read at every
outcome where the certifiers count nothing else entering, which is every outcome: the event has probability 1 and the
bound is `δ = 1/2`. -/
theorem lineHeld_nontrivial : ∃ sm : RemoteAudit.Session.Summary,
    ¬ LineHeldBy (fun _ _ => True) (fun _ _ => some sm) := by
  refine ⟨Toy.sm, fun h => ?_⟩
  have hb := h Toy.node 0 Unit Unit (fun t => isEmptyElim t) (fun _ _ _ => 0) (fun _ _ => Toy.rec)
    (fun _ _ _ => ∅) (fun _ => ()) Toy.trace Toy.receipts 0 0 (noOutside _) (quietIngress _) (toy_paced _)
    (toy_covers _) (toy_sketch _ _) (covered_noOutside _ _) (ingressCounted_quiet _)
    (fun _ t => isEmptyElim t) (fun _ _ t => isEmptyElim t) (counted_of _ _ fun _ _ => Finset.Subset.refl _)
  refine absurd hb (not_le.2 ?_)
  rw [Pouw.pr_eq_one]
  · norm_num [binding_empty, toy_delta]
  · exact fun ω => ⟨trivial, fun hq => toy_not_read _ ω (toy_reads_by _ ω) (hq fun _ _ _ _ _ => rfl)⟩

/-- **`LineHeldAtComponent` fails at a verifier that accepts everything.** With an acceptance and a verdict that hold
of every record and one summary recorded for every record, the toy node meets every premise at `δs = δnet = 0`, while
its line credits `4 · 1` blocks read where its passes read one: the event has probability 1 and the bound is
`δ = 1/2`. -/
theorem lineHeldAtComponent_nontrivial : ∃ sm : RemoteAudit.Session.Summary,
    ¬ LineHeldAtComponentBy (fun _ _ _ _ _ => True) (fun _ _ => True) (fun _ _ => some sm) := by
  refine ⟨Toy.sm, fun h => ?_⟩
  have hb := h Toy.node 0 Unit Unit (fun t => isEmptyElim t) (fun _ _ _ => 0) (fun _ _ => Toy.rec)
    (fun _ _ _ => ∅) (fun _ => ()) Toy.trace Toy.receipts 0 0 (fun _ => .empty) (fun _ => .empty) (fun _ => .empty)
    (noOutside _) (quietIngress _) (toy_paced _) (toy_covers _) (toy_sketch _ _)
    (covered_noOutside _ _) (ingressCounted_quiet _) (fun _ t => isEmptyElim t) (fun _ _ t => isEmptyElim t)
    (counted_of _ _ fun _ _ => Finset.Subset.refl _)
  refine absurd hb (not_le.2 ?_)
  rw [Pouw.pr_eq_one]
  · norm_num [binding_empty, toy_delta]
  · exact fun ω => ⟨trivial, fun hq => toy_not_read _ ω (toy_reads_by _ ω) (hq fun _ _ _ _ _ => rfl)⟩

/-- **The line credits only what the passes read**: on one node, record and summary, the line's bits-read claim holds
of a trace whose four passes each read the store in node memory, and fails of the same trace with node memory emptied
at tick 10. -/
theorem lineHeldAtComponent_bites : ∃ (node : Node) (r : Record) (sm : RemoteAudit.Session.Summary) (T : Trace Unit),
    (∃ l, lineBy (fun _ _ => some sm) node r = .ok l) ∧ ReadBy (fun _ _ => some sm) T () node r ∧
    ¬ ReadBy (fun _ _ => some sm) { T with mem := fun _ t => if t < 10 then Finset.univ else ∅ } () node r := by
  obtain ⟨l, hl, hw⟩ := toy_line
  have hfull : ∀ p, (Toy.steady.read () p).card = 1 := fun p =>
    read_full _ () p rfl rfl (by show 10 * p < 10 * (p + 1); omega) (Finset.union_empty _)
  have hsteady : Toy.steady.reads () 4 = 4 := by simp [Trace.reads, hfull]
  have h0 : (({ Toy.steady with mem := fun _ t => if t < 10 then Finset.univ else ∅ } : Trace Unit).read () 0).card
      = 1 := read_full _ () 0 rfl rfl (by show 10 * 0 < 10 * (0 + 1); omega)
      (Finset.eq_univ_of_forall fun j => Finset.mem_union_left _ (by
        simp [Toy.steady]
        exact Finset.mem_univ j))
  have hp : ∀ p, 1 ≤ p →
      ({ Toy.steady with mem := fun _ t => if t < 10 then Finset.univ else ∅ } : Trace Unit).read () p = ∅ :=
    fun p hp => read_absent _ () p fun t h1 _ => by
      change 10 * p ≤ t at h1
      show (if t < 10 then Finset.univ else ∅) ∪ ∅ = (∅ : Finset (Fin 1))
      rw [if_neg (by omega)]
      simp
  have hemptied : ({ Toy.steady with mem := fun _ t => if t < 10 then Finset.univ else ∅ } : Trace Unit).reads () 4
      ≤ 3 := by
    simp [Trace.reads, Finset.sum_range_succ, h0, hp 1 le_rfl, hp 2 (by norm_num), hp 3 (by norm_num)]
  refine ⟨Toy.node, Toy.rec, Toy.sm, Toy.steady, ⟨l, hl⟩, fun l' hl' => ?_, toy_not_read _ () hemptied⟩
  obtain rfl : l' = l := Except.ok.inj (hl'.symm.trans hl)
  rw [toy_groups, toy_passes, hsteady, hw]
  norm_num [Toy.node, Node.blocks, Toy.reg]

/-! ## `AcceptIsVerified` at a check that accepts everything, and at the deployed check -/

/-- **`AcceptIsVerified` fails at a check that accepts everything**: it accepts a log that decodes to no session. -/
theorem acceptIsVerified_nontrivial : ¬ AcceptIsVerifiedFor (fun _ _ _ => .accept ⟨⟨0, 1⟩, 0⟩ Lean.Json.null) := by
  intro h
  obtain ⟨_, _, _, _, _, _, _, _, _, hs, -⟩ :=
    h ⟨.null, .null⟩ Toy.reg ⟨.null, .null, #[]⟩ ⟨⟨0, 1⟩, 0⟩ .null rfl
  have hb : (match RemoteAudit.Protocol.sessionOf ⟨.null, .null⟩ ⟨.null, .null, #[]⟩ with
      | .ok _ => true
      | .error _ => false) = false := by decide +kernel
  rw [hs] at hb
  simp at hb

theorem honest_decodes : (match RemoteAudit.Protocol.sessionOf honestParams honestEvidence with
    | .ok (params, g, _, appendCoins, notBeforeNs, events) =>
      params.windows == 2 && g == 1 && appendCoins.size == 0 && notBeforeNs == 0 && events.size == 0
    | .error _ => false) = true := by decide +kernel

/-- **The deployed check at the honest period's log**: the deployed period's parameters and its log before the first
event decode to a session of two windows at one window a draw, with no append coin, clock floor 0 and no event, which
the check leaves undecided: no window has verified yet. -/
theorem acceptIsVerified_honest (reg : RemoteAudit.Registration) :
    (∃ (params : RemoteAudit.Session.Params) (coins : Array ByteArray),
      RemoteAudit.Protocol.sessionOf honestParams honestEvidence = .ok (params, 1, coins, #[], 0, #[]) ∧
        params.windows = 2) ∧
    RemoteAudit.Protocol.check honestParams reg honestEvidence =
      .undecided "the session has a window not yet verified" := by
  have hd := honest_decodes
  revert hd
  cases hs : RemoteAudit.Protocol.sessionOf honestParams honestEvidence with
  | error _ => intro hd; cases hd
  | ok x =>
    obtain ⟨params, g, coins, ac, nb, events⟩ := x
    intro hd
    simp only [Bool.and_eq_true, beq_iff_eq] at hd
    obtain ⟨⟨⟨⟨hw, rfl⟩, hac⟩, rfl⟩, hev⟩ := hd
    obtain rfl := Array.eq_empty_of_size_eq_zero hac
    obtain rfl := Array.eq_empty_of_size_eq_zero hev
    refine ⟨⟨params, coins, rfl, hw⟩, ?_⟩
    simp [RemoteAudit.Protocol.check, RemoteAudit.Protocol.replay, hs, bind, Except.bind, RemoteAudit.Shared.run,
      pure, Except.pure, RemoteAudit.Session.State.outcome, RemoteAudit.Session.State.init, hw]

/-- **The deployed check accepts whatever `Shared.run g` accepts**: where the parameters and the log decode to a
session that `Shared.run g` runs to an accepted state with a record `sm`, the check accepts, publishing `certify`'s
outputs of `sm` at the session's parameters and g. -/
theorem acceptIsVerified_complete {p : RemoteAudit.Protocol.Params} {reg : RemoteAudit.Registration}
    {e : RemoteAudit.Protocol.Evidence} {params : RemoteAudit.Session.Params} {g : ℕ}
    {coins appendCoins : Array ByteArray} {notBeforeNs : ℕ} {events : Array RemoteAudit.Session.Event}
    {rs : Array RemoteAudit.Session.Result} {s : RemoteAudit.Session.State} {sm : RemoteAudit.Session.Summary}
    (hs : RemoteAudit.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events))
    (hr : RemoteAudit.Shared.run g (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events =
      .ok (rs, s))
    (ho : s.outcome = some "accepted") (hsm : s.summary = some sm) :
    RemoteAudit.Protocol.check p reg e =
      .accept (RemoteAudit.Protocol.certify params g sm).1 (RemoteAudit.Protocol.certify params g sm).2 := by
  have hrep : RemoteAudit.Protocol.replay p reg e = .ok (s, params, g) := by
    simp only [RemoteAudit.Protocol.replay, hs, hr, bind, Except.bind, pure, Except.pure]
  unfold RemoteAudit.Protocol.check
  rw [hrep]
  simp only [ho, hsm]

/-! ## `PublishedEps` at a check that publishes reads at eps 0 -/

/-- A node's draw term is positive at a positive δ. -/
theorem drawTerm_pos (node : Node) (hδ : 0 < (node.delta : ℚ)) :
    0 < (node.delta : ℚ) + ((∑ _c : Fin node.draws, node.params.samples : ℕ) : ℚ) / 2 ^ 64 :=
  add_pos_of_pos_of_nonneg hδ (by positivity)

/-- **`PublishedEps` fails at a check that certifies reads at eps 0**: a check that accepts every log, publishing one
byte read at eps 0, accepts the deployed period's log, and 0 is below the draw term at every node of positive δ. -/
theorem publishedEps_nontrivial : ¬ PublishedEpsFor (fun _ _ _ => .accept ⟨⟨0, 1⟩, 1⟩ Lean.Json.null) := by
  intro h
  obtain ⟨params, g, _, _, _, _, -, hd⟩ := h honestParams Toy.reg honestEvidence ⟨⟨0, 1⟩, 1⟩ .null rfl
  have hpos : (0 : ℚ) < RemoteAudit.Protocol.DELTA := by
    unfold RemoteAudit.Protocol.DELTA
    positivity
  rcases hd with ⟨-, hq⟩ | hn
  · exact absurd hq (by decide)
  · obtain ⟨node, hp, hg, hδ⟩ : ∃ node : Node, node.params = params ∧ node.groupsPerCoin = g ∧
        (node.delta : ℚ) = RemoteAudit.Protocol.DELTA :=
      ⟨{ Toy.node with params := params, groupsPerCoin := g, delta := ⟨RemoteAudit.Protocol.DELTA, hpos.le⟩ },
        rfl, rfl, rfl⟩
    have h0 := lt_of_lt_of_eq (drawTerm_pos node (by rw [hδ]; exact hpos)) (hn node hp hg hδ).symm
    norm_num at h0

end Properties.RemoteAudit.Component
