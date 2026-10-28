import Proofs.Accounting.RemoteHeld.Component
import Proofs.Accounting.RemoteHeld.Draw
import Proofs.Pouw.Prob

/-!
# The bandwidth component's properties hold at the honest node and fail at a verifier that accepts everything

Each of `Properties.RemoteAudit.Component`'s two properties is its form over the verifier at the deployed one
(`acceptIsVerified_iff`, `lineHeldAtComponent_iff`), and each form has its two witnesses:

* **AcceptIsVerified** (`AcceptIsVerifiedFor`). Honest: the deployed period's parameters and log decode to its session
  at one window a draw, which the check leaves undecided until its windows verify (`acceptIsVerified_honest`), and the
  check accepts, publishing `certify`'s outputs, wherever `Shared.run g` accepts the session it decodes
  (`acceptIsVerified_complete`). Nontrivial: a check that accepts everything accepts a log that decodes to no session
  (`acceptIsVerified_nontrivial`).
* **LineHeldAtComponent** (`LineHeldAtComponentBy`). Honest: every premise holds at a node that holds its whole store
  in memory throughout, every receipt correct, no outside device and the sampler's fallbacks 0, on a schedule that runs
  each pass inside its window with each draw's receipts fixed (`HonestSchedule`) and the line's nonnegative rates,
  and at its correct receipts the distinguishers' advantage and the hiding terms are 0
  (`lineHeldAtComponent_honest`). Nontrivial: at a verifier that accepts everything and records one summary for every
  record, the bound fails on a node that holds nothing after tick 10 (`lineHeldAtComponent_nontrivial`); and the line
  credits only what the trace held: on one record and summary it holds of a trace with the store in memory and fails
  of the same trace with memory emptied at tick 10 (`lineHeldAtComponent_bites`).

At the deployed verifier the premises' acceptance is load-bearing and `Plays` is not: the line reads the record
through `Record.summary`, which is some summary only where the Lean verifier accepts, so the event's `¬ HeldAt` alone
gives `Verified` (`verified_of_not_heldBy`) and the bound holds at any acceptance in the event
(`lineHeldAtComponentBy_record`). The nontrivial witness therefore replaces the verdict and the record with an
accept-all's as well.

The toy instance (`Toy`) is a node whose parameters have no window (`Node.windows` 0) at four windows a draw, so no
draw (`Node.draws` 0), one block and two groups a stretch, whose recorded summary has four windows of one block each and
one link-charged stretch of span 10 credited `10 · (1 − 0)` blocks held.
-/

deriving instance DecidableEq for RemoteAudit.Line.Group

namespace Properties.RemoteAudit.Component

open Finset Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions Properties.RemoteAudit.Line

theorem acceptIsVerified_iff : AcceptIsVerified ↔ AcceptIsVerifiedFor RemoteAudit.Protocol.check := Iff.rfl

theorem lineHeldAtComponent_iff : LineHeldAtComponent ↔ LineHeldAtComponentBy Plays Verified Record.summary :=
  Iff.rfl

/-! ## The premises at an honest node -/

theorem bucket_zero {Ω : Type} (ω : Ω) (u v : ℕ) {β B : ℝ} (hβ : 0 ≤ β) (hB : 0 ≤ B) :
    ((Accounting.Remote.Held.over (fun _ _ => 0) ω u v : ℕ) : ℝ) ≤ β * (v - u : ℕ) + B := by
  simp only [Accounting.Remote.Held.over, Finset.sum_const_zero, Nat.cast_zero]
  exact add_nonneg (mul_nonneg hβ (Nat.cast_nonneg _)) hB

theorem ns_pos : (0 : ℝ) < ((_root_.RemoteAudit.Line.ns : ℚ) : ℝ) := by
  rw [show _root_.RemoteAudit.Line.ns = 1000000000 from rfl]
  norm_num

theorem link_nonneg (n : Node) (h : 0 ≤ n.rates.linkBytesPerS) : 0 ≤ n.link := by
  unfold Node.link
  exact div_nonneg (div_nonneg (Rat.cast_nonneg.2 h) (Nat.cast_nonneg _)) ns_pos.le

theorem disk_nonneg (n : Node) (h : 0 ≤ n.rates.diskBytesPerS) : 0 ≤ n.disk := by
  unfold Node.disk
  exact div_nonneg (div_nonneg (Rat.cast_nonneg.2 h) (Nat.cast_nonneg _)) ns_pos.le

theorem squarings_nonneg (n : Node) (h : 0 ≤ n.rates.squaringsPerS) : 0 ≤ n.squarings / n.encode := by
  unfold Node.squarings Node.encode
  exact div_nonneg (div_nonneg (Rat.cast_nonneg.2 h) ns_pos.le) (Nat.cast_nonneg _)

theorem burst_nonneg {q : ℚ} (h : 0 ≤ q) (b : ℕ) : (0 : ℝ) ≤ (q : ℝ) / b :=
  div_nonneg (Rat.cast_nonneg.2 h) (Nat.cast_nonneg _)

theorem links_zero {Ω : Type} (T : Trace Ω) (h : T.link = fun _ _ => 0) {β B : ℝ} (hβ : 0 ≤ β) (hB : 0 ≤ B) :
    Links T β B := fun ω u v _ => by
  rw [h]
  exact bucket_zero ω u v hβ hB

theorem squarings_zero {Ω : Type} (T : Trace Ω) (h : T.rebuilt = fun _ _ => 0) {s E B : ℝ} (hs : 0 ≤ s / E)
    (hB : 0 ≤ B) : SquaringCeiling T s E B := fun ω u v _ => by
  rw [h]
  exact bucket_zero ω u v hs hB

theorem disks_zero {Ω : Type} (T : Trace Ω) (h : T.disk = fun _ _ => 0) {β B : ℝ} (hβ : 0 ≤ β) (hB : 0 ≤ B) :
    Disks T β B := fun ω u v _ => by
  rw [h]
  exact bucket_zero ω u v hβ hB

/-- `overBlocks`' reading of a set of registered blocks below `M` as indices below `N`. -/
def reIdx (M N : ℕ) (s : Finset (Fin M)) : Finset (Fin N) := univ.filter fun j => (j : ℕ) ∈ s.image Fin.val

theorem reIdx_empty (M N : ℕ) : reIdx M N ∅ = ∅ := by
  unfold reIdx
  exact filter_false_of_mem fun j _ => by simp

theorem reIdx_univ {M N : ℕ} (h : N ≤ M) : reIdx M N univ = univ := by
  unfold reIdx
  exact filter_true_of_mem fun j _ => mem_image.2 ⟨⟨j, lt_of_lt_of_le j.isLt h⟩, mem_univ _, rfl⟩

theorem reIdx_mono {M N : ℕ} {s s' : Finset (Fin M)} (h : s ⊆ s') : reIdx M N s ⊆ reIdx M N s' := by
  intro j hj
  unfold reIdx at hj ⊢
  rw [mem_filter] at hj ⊢
  exact ⟨hj.1, image_subset_image h hj.2⟩

theorem present_full {M N : ℕ} (h : N ≤ M) : reIdx M N univ ∪ reIdx M N ∅ = univ := by
  rw [reIdx_univ h, reIdx_empty, union_empty]

theorem chip_empty {Ω : Type} (T : Trace Ω) (h : ∀ ω t, T.chip ω t = ∅) (S : ℕ) : ChipCeiling T S :=
  fun ω t => by
    rw [h ω t, card_empty]
    exact Nat.zero_le _

theorem routed_of_antitone {Ω : Type} (T : Trace Ω) (h : ∀ ω t, T.present ω (t + 1) ⊆ T.present ω t) :
    Routed T := fun ω t => by
  unfold Trace.entered
  rw [Finset.sdiff_eq_empty_iff_subset.2 (h ω t), card_empty]
  exact Nat.zero_le _

theorem covers_univ {Ω : Type} (T : Trace Ω) (h : ∀ ω p, T.plan ω p = univ) : Covers T := fun ω _ p _ => h ω p

theorem noOutside_ok {Ω : Type} (T : Trace Ω) : NoOutsideCompute T (noOutside T) :=
  fun _ _ _ _ _ _ _ => ⟨rfl, rfl⟩

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

/-- **`LineHeldAtComponent`'s premises hold at the honest node.** On any node with the line's nonnegative rates, at
any records of the prover's whose schedule is honest (`HonestSchedule`), the honest trace (`honestTrace`: the whole
store in node memory at every tick, nothing on a chip, rebuilt or read in, every receipt correct), no outside device
(`noOutside`), correct receipts holding every receipt a draw can have and the sampler's fallbacks 0, every premise
holds of its trace of record (`ofRecord`, the honest trace over the registration's blocks): the three ceilings, the
chips', `Routed`, `Paced` at the record's groups, `Covers`, `SketchBoundAt` at 0, `NoOutsideCompute`, the draw's
`Live` at every commitment, `Refused` and `Counted`; and the bound's terms the prover could raise are 0: each draw's
distinguisher has no advantage on its coin function and each hiding term is 0. -/
theorem lineHeldAtComponent_honest (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X]
    (prf : Fin node.draws → Pous.Bits κ → Blocks receiptMax) (recs : Cm → Coins node.draws κ X → Record)
    (com : Coins node.draws κ X → Cm) (R : Receipts node.draws κ X) (passes : ℕ)
    (tick : Coins node.draws κ X → ℕ → ℕ)
    (hRates : 0 ≤ node.rates.squaringsPerS ∧ 0 ≤ node.rates.diskBytesPerS ∧ 0 ≤ node.rates.linkBytesPerS ∧
      0 ≤ node.rates.rebuildBurstBytes ∧ 0 ≤ node.rates.diskBurstBytes ∧ 0 ≤ node.rates.linkBurstBytes)
    (hS : HonestSchedule node passes tick recs com) :
    Links (ofRecord (honestTrace node passes tick) node (played recs com)) node.link node.linkBurst ∧
    SquaringCeiling (ofRecord (honestTrace node passes tick) node (played recs com)) node.squarings node.encode
      node.rebuildBurst ∧
    Disks (ofRecord (honestTrace node passes tick) node (played recs com)) node.disk node.diskBurst ∧
    ChipCeiling (ofRecord (honestTrace node passes tick) node (played recs com)) node.chipBlocks ∧
    Routed (ofRecord (honestTrace node passes tick) node (played recs com)) ∧
    Paced (ofRecord (honestTrace node passes tick) node (played recs com)) (recordPace node (played recs com)) ∧
    Covers (ofRecord (honestTrace node passes tick) node (played recs com)) ∧
    SketchBoundAt (ofRecord (honestTrace node passes tick) node (played recs com))
      (noOutside (ofRecord (honestTrace node passes tick) node (played recs com))) 0 ∧
    NoOutsideCompute (ofRecord (honestTrace node passes tick) node (played recs com))
      (noOutside (ofRecord (honestTrace node passes tick) node (played recs com))) ∧
    (∀ c, Live ((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).view c)) ∧
    Refused ((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play com) prf (fun _ _ _ => 0)
      (ofRecord (honestTrace node passes tick) node (played recs com)) R ∧
    Counted ((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play com)
      (ofRecord (honestTrace node passes tick) node (played recs com)) ∧
    (∀ t, Pous.StreamedWipe.prfAdvantage (prf t)
      (((recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).play
        (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t) = 0) ∧
    ∀ t, (recordCommitted node recs fun _ _ _ => Finset.range (receiptMax + 1)).hide com prf (node.delta : ℝ) t = 0 := by
  obtain ⟨hsq, hd, hl, hrb, hdb, hlb⟩ := hRates
  obtain ⟨hW, htick, hsize, hgs⟩ := hS
  have hc : ∀ D : Session node.draws receiptMax (fun _ => node.params.samples) κ X,
      (∀ t ω, D.correct t ω = Finset.range (receiptMax + 1)) → ∀ t ω, range (D.size t ω : ℕ) ⊆ D.correct t ω :=
    fun D h t ω => by
      rw [h]
      exact Finset.range_subset_range.2 (D.size t ω).isLt.le
  refine ⟨links_zero _ rfl (link_nonneg node hl) (burst_nonneg hlb _),
    squarings_zero _ rfl (squarings_nonneg node hsq) (burst_nonneg hrb _),
    disks_zero _ rfl (disk_nonneg node hd) (burst_nonneg hdb _), chip_empty _ (fun _ _ => reIdx_empty _ _) _,
    routed_of_antitone _ fun _ _ => Finset.Subset.refl _, ?_, covers_univ _ fun _ _ => reIdx_univ le_rfl,
    sketch_zero _ _ fun ω p _ => ⟨tick ω p, le_refl _, htick ω p, present_full le_rfl⟩, noOutside_ok _,
    fun c t ω ω' _ _ => ⟨Fin.ext (congrArg (min · receiptMax) (hsize c c ω ω' t)), rfl⟩,
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

/-! ## At the deployed verifier the event's acceptance is the summary's -/

/-- Where a stretch of the line fails to credit what the trace held, the Lean verifier accepted the record: its line
reads `Record.summary`, which is some summary only where the session ended accepted. -/
theorem verified_of_not_heldBy {Ω : Type} {T : Trace Ω} {ω : Ω} {node : Node} {r : Record} {form : ℕ}
    (h : ¬ HeldBy Record.summary T ω node r form) : Verified node r := by
  have hne : Record.summary node r ≠ none := fun hn => h (by simp [HeldBy, lineBy, hn])
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
  intro node κ X Cm _ prf rest recs correct com T R δs a b e r F O hL hS hD hC hR hP hCv hSk hN hLive hRef hCnt
  exact le_trans (Pouw.pr_mono fun ω h => ⟨verified_of_not_heldBy h.2, h.2⟩)
    (Properties.RemoteAudit.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs O hL hS hD hC
      hR hP hCv hSk hN hLive hRef hCnt)

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

/-- One block, held in node memory until tick 10 and not after, one pass over `[0, 10)`. -/
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

/-- The same with the block in node memory at every tick. -/
def full : Trace Unit := { trace with mem := fun _ _ => univ }

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

/-- Whether a line's link-charged stretches are some, each the first stretch, of span 10, with nothing lost and no
weights credited. -/
def probe (r : Except String _root_.RemoteAudit.Line.Line) : Bool :=
  match r with
  | .ok l => match l.forms[1]? with
    | some f => !f.stretches.isEmpty && f.stretches.all fun x =>
        x.stretch == 0 && x.spanNs == 10 && x.time.lost == 0 && x.time.weights == 0
    | none => false
  | .error _ => false

theorem probe_extract {r : Except String _root_.RemoteAudit.Line.Line} (h : probe r = true) :
    ∃ l f, r = .ok l ∧ l.forms[1]? = some f ∧ f.stretches ≠ #[] ∧
      ∀ x ∈ f.stretches, x.stretch = 0 ∧ x.spanNs = 10 ∧ x.time.lost = 0 ∧ x.time.weights = 0 := by
  cases r with
  | error _ => cases h
  | ok l =>
    cases hf : l.forms[1]? with
    | none => simp [probe, hf] at h
    | some f =>
      simp only [probe, hf, Bool.and_eq_true, Bool.not_eq_true'] at h
      refine ⟨l, f, rfl, hf, fun h3 => by rw [h3] at h; exact absurd h.1 (by decide), fun x hx => ?_⟩
      have h4 := Array.all_eq_true_iff_forall_mem.1 h.2 x hx
      simp only [Bool.and_eq_true, beq_iff_eq] at h4
      exact ⟨h4.1.1.1, h4.1.1.2, h4.1.2, h4.2⟩

open _root_.RemoteAudit.Line in
theorem toy_probe : probe (lineOfRecordAt 4 Toy.sm 4 Toy.seen 2 1 0 1 (Toy.node.delta : ℚ) Toy.rates) = true := by
  obtain ⟨b, hp, hb⟩ := toy_pool
  have h0 : ((0 : ℚ), (0 : ℚ)).1 ≤ 0 := le_refl _
  simp only [lineOfRecordAt, toy_ofs, line, hp, hb, bind, Except.bind, stretches,
    Std.Legacy.Range.forIn_eq_forIn_range', forIn_fill_nonpos _ _ _ _ h0]
  decide +kernel

theorem toy_line : ∃ l f, lineBy Toy.sum Toy.node Toy.rec = .ok l ∧ l.forms[1]? = some f ∧ f.stretches ≠ #[] ∧
    ∀ x ∈ f.stretches, x.stretch = 0 ∧ x.spanNs = 10 ∧ x.time.lost = 0 ∧ x.time.weights = 0 := by
  have e : lineBy Toy.sum Toy.node Toy.rec =
      _root_.RemoteAudit.Line.lineOfRecordAt 4 Toy.sm 4 Toy.seen 2 1 0 1 (Toy.node.delta : ℚ) Toy.rates := rfl
  rw [e]
  exact probe_extract toy_probe

theorem toy_groups : groupsBy Toy.sum Toy.node Toy.rec = Toy.gs := by decide +kernel

theorem toy_start : spanStart (groupsBy Toy.sum Toy.node Toy.rec) Toy.node.per 0 = 10 := by
  rw [toy_groups]
  decide +kernel

theorem toy_stop : spanStop (groupsBy Toy.sum Toy.node Toy.rec) Toy.node.per 0 = 20 := by
  rw [toy_groups]
  decide +kernel

/-- On a trace of the one block with nothing in node memory from tick 10, the toy line's stretch over `[10, 20)`
credits more than the trace held. -/
theorem toy_not_held {Ω : Type} (T : Trace Ω) (hb : T.blocks = 1) (hm : ∀ ω t, 10 ≤ t → T.mem ω t = ∅) (ω : Ω) :
    ¬ HeldBy Toy.sum T ω Toy.node Toy.rec 1 := fun h => by
  obtain ⟨l, f, hl, hf, hne, hx⟩ := toy_line
  obtain ⟨x, hxm⟩ := Array.exists_mem_of_ne_empty _ hne
  obtain ⟨hs0, hsp, hl0, -⟩ := hx x hxm
  obtain ⟨-, h2, -⟩ := h l hl f hf x hxm
  have h0 : T.heldOver ω 10 20 = 0 := by
    unfold Trace.heldOver
    exact Finset.sum_eq_zero fun t ht => by rw [hm ω t (Finset.mem_Ico.1 ht).1, Finset.card_empty]
  rw [hs0, toy_start, toy_stop, h0, hsp, hl0, hb] at h2
  norm_num at h2

theorem toy_paced {Ω : Type} (ver : Node → Record → Prop) :
    Paced (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) (paceBy Toy.sum Toy.node fun _ => Toy.rec) := by
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

theorem toy_mem_anti {Ω : Type} (ω : Ω) (t : ℕ) :
    (Toy.trace (Ω := Ω)).mem ω (t + 1) ⊆ (Toy.trace (Ω := Ω)).mem ω t := by
  show (if t + 1 < 10 then Finset.univ else ∅) ⊆ (if t < 10 then Finset.univ else ∅)
  split_ifs <;> first | (exfalso; omega) | exact Finset.Subset.refl _ | exact Finset.empty_subset _

theorem toy_late {Ω : Type} (ver : Node → Record → Prop) (ω : Ω) (t : ℕ) (ht : 10 ≤ t) :
    (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec).mem ω t = ∅ := by
  show reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).mem ω t) = ∅
  rw [show (Toy.trace (Ω := Ω)).mem ω t = ∅ from if_neg (by omega)]
  exact reIdx_empty _ _

theorem toy_routed {Ω : Type} (ver : Node → Record → Prop) :
    Routed (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) :=
  routed_of_antitone _ fun ω t => by
    show reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).mem ω (t + 1)) ∪
        reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).chip ω (t + 1)) ⊆
      reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).mem ω t) ∪ reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).chip ω t)
    exact union_subset_union (reIdx_mono (toy_mem_anti ω t)) (Finset.Subset.refl _)

theorem toy_sketch {Ω : Type} [Fintype Ω] (ver : Node → Record → Prop)
    (O : Outside (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec)) :
    SketchBoundAt (traceBy (Toy.trace (Ω := Ω)) Toy.node ver fun _ => Toy.rec) O 0 :=
  sketch_zero _ O fun ω p hp => by
    change p < 1 at hp
    obtain rfl : p = 0 := by omega
    refine ⟨0, ?_, ?_, ?_⟩
    · show (if 0 = 0 then 0 else 10) ≤ 0
      simp
    · show 0 < (if 0 + 1 = 0 then 0 else 10)
      simp
    · show reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).mem ω 0) ∪
          reIdx _ Toy.node.blocks ((Toy.trace (Ω := Ω)).chip ω 0) = Finset.univ
      have h0 : (Toy.trace (Ω := Ω)).mem ω 0 = Finset.univ := rfl
      rw [h0]
      exact present_full le_rfl

theorem binding_empty {n κ : ℕ} {X : Type} [Fintype X] [IsEmpty (Fin n)] {m : ℕ} {s : Fin n → ℕ}
    (D : Session n m s κ X) (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) : R.binding D prf rest = 0 := by
  simp [Receipts.binding, Pous.pr]

theorem toy_delta : (Toy.node.delta : ℝ) = 1 / 2 := by
  rw [← Rat.cast_nnratCast]
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

/-- **`LineHeldAtComponent` fails at a verifier that accepts everything.** With an acceptance and a verdict that hold
of every record and one summary recorded for every record, the toy node (no draw, one block, held in node memory until
tick 10 only) meets every premise at `δs = 0`, while its line's stretch over `[10, 20)` credits a block held there at
every outcome: the event has probability 1 and the bound is `δ = 1/2`. -/
theorem lineHeldAtComponent_nontrivial : ∃ sm : RemoteAudit.Session.Summary,
    ¬ LineHeldAtComponentBy (fun _ _ _ _ _ => True) (fun _ _ => True) (fun _ _ => some sm) := by
  refine ⟨Toy.sm, fun h => ?_⟩
  have hb := h Toy.node 0 Unit Unit (fun t => isEmptyElim t) (fun _ _ _ => 0) (fun _ _ => Toy.rec)
    (fun _ _ _ => ∅) (fun _ => ()) Toy.trace Toy.receipts 0 (fun _ => .empty) (fun _ => .empty) (fun _ => .empty)
    (noOutside _) (links_zero _ rfl (link_nonneg _ (le_refl _)) (burst_nonneg (le_refl _) _))
    (squarings_zero _ rfl (squarings_nonneg _ (le_refl _)) (burst_nonneg (le_refl _) _))
    (disks_zero _ rfl (disk_nonneg _ (le_refl _)) (burst_nonneg (le_refl _) _))
    (chip_empty _ (fun _ _ => reIdx_empty _ _) _) (toy_routed _) (toy_paced _)
    (covers_univ _ fun _ _ => reIdx_univ le_rfl) (toy_sketch _ _) (noOutside_ok _)
    (fun _ t => isEmptyElim t) (fun _ _ t => isEmptyElim t) (counted_of _ _ fun _ _ => Finset.Subset.refl _)
  refine absurd hb (not_le.2 ?_)
  rw [Pouw.pr_eq_one]
  · simp [binding_empty, toy_delta]
    norm_num
  · exact fun ω => ⟨trivial, toy_not_held _ rfl (fun ω' t ht => toy_late _ ω' t ht) ω⟩

/-- **The line credits only what the trace held**: on one node, record and summary, every link-charged stretch
credits what a trace with the store in node memory held, and not what the same trace held with node memory emptied
at tick 10. -/
theorem lineHeldAtComponent_bites : ∃ (node : Node) (r : Record) (sm : RemoteAudit.Session.Summary) (T : Trace Unit),
    (∃ l, lineBy (fun _ _ => some sm) node r = .ok l) ∧ HeldBy (fun _ _ => some sm) T () node r 1 ∧
    ¬ HeldBy (fun _ _ => some sm) { T with mem := fun _ t => if t < 10 then Finset.univ else ∅ } () node r 1 := by
  obtain ⟨l, f, hl, hf, -, hx⟩ := toy_line
  refine ⟨Toy.node, Toy.rec, Toy.sm, Toy.full, ⟨l, hl⟩, ?_,
    toy_not_held _ rfl (fun _ t ht => if_neg (by omega)) ()⟩
  intro l' hl' f' hf' x hxm
  obtain rfl : l' = l := Except.ok.inj (hl'.symm.trans hl)
  obtain rfl : f' = f := Option.some.inj (hf'.symm.trans hf)
  obtain ⟨hs0, hsp, hl0, hw0⟩ := hx x hxm
  have hh : Toy.full.heldOver () 10 20 = 10 := by
    unfold Trace.heldOver
    simp [Toy.full]
    rfl
  refine ⟨?_, ?_, fun R _ => ?_⟩
  · rw [hs0, hsp, toy_start, toy_stop]
  · rw [hs0, toy_start, toy_stop, hh, hsp, hl0]
    norm_num [Toy.full, Toy.trace]
  · rw [hw0, Rat.cast_zero, mul_zero]
    exact mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)

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
outputs of `sm` at g. -/
theorem acceptIsVerified_complete {p : RemoteAudit.Protocol.Params} {reg : RemoteAudit.Registration}
    {e : RemoteAudit.Protocol.Evidence} {params : RemoteAudit.Session.Params} {g : ℕ}
    {coins appendCoins : Array ByteArray} {notBeforeNs : ℕ} {events : Array RemoteAudit.Session.Event}
    {rs : Array RemoteAudit.Session.Result} {s : RemoteAudit.Session.State} {sm : RemoteAudit.Session.Summary}
    (hs : RemoteAudit.Protocol.sessionOf p e = .ok (params, g, coins, appendCoins, notBeforeNs, events))
    (hr : RemoteAudit.Shared.run g (RemoteAudit.Session.State.init params reg coins appendCoins notBeforeNs) events =
      .ok (rs, s))
    (ho : s.outcome = some "accepted") (hsm : s.summary = some sm) :
    RemoteAudit.Protocol.check p reg e =
      .accept (RemoteAudit.Protocol.certify sm g).1 (RemoteAudit.Protocol.certify sm g).2 := by
  have hrep : RemoteAudit.Protocol.replay p reg e = .ok (s, g) := by
    simp only [RemoteAudit.Protocol.replay, hs, hr, bind, Except.bind, pure, Except.pure]
  unfold RemoteAudit.Protocol.check
  rw [hrep]
  simp only [ho, hsm]

end Properties.RemoteAudit.Component
