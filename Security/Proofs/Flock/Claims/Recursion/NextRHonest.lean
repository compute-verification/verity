import Proofs.Flock.Claims.Recursion.NextR

/-!
# An honest prover's rounds read back at `nextR`

`nextR` reads each rep's proof from its opened rounds (`readRep`, `FitsRep`), so `RecursiveKS` at `nextR` is
non-vacuous only if an honest prover's rounds read back: on the messages the honest recursive prover sends, some proof
fits every rep, and the proof `readPs` reads is the prover's. This file defines what the honest prover sends and states
that.

* **What the honest prover sends** (`honestRows`, `repRows`, `capRows`, `honestMsgs`): round 0 carries each table's
  level-0 cap nodes, in table order. Round `j` of rep `k`, along the rep's schedule (`RecDeploy.repSchedAt`,
  `rec_live.rep_rounds`), carries the nodes of the tops the round commits, its proof's next recursive caps in order
  (`rec_live.top_rows`), then, where the round absorbs, one row of the values the `j`-th chunk of the rep's frames
  absorbs (`message_rows`, `vals`). Each round's rows are joined (`joinR`). The last message is any that the reader `rd`
  reads as the proofs.
* **The honest proofs** (`HonestRep`, `SchedOf`): each rep's absorbed fields have the lengths its frames read
  (`AbsLens`, which `ProofShaped` at the rep's file shape gives: `FlockComplete`'s `repShaped`), every nonce is `0` (the
  deployed provers', D20), and each recursive cap has its level's `2^cap` nodes. Both reps of a table carry its level-0
  cap, as `sessionP` commits each table once. The rep's schedule is `RecVStar.repRounds` at its frames' `m`, `k_log` and
  claims.
* **The statements**: `RepFits` (the honest proof fits the honest rows) and `FitUnique` (rows fix a fitting proof's
  absorbed fields and its `T`) give `RepReadsBack`, that `readRep` reads the honest proof back (`repReadsBack_of`). That
  gives `ReadsBack`, that `decR` decides on the honest messages and reads exactly the honest proofs
  (`readsBack_of_rep`).

Open: `RepFits` and `FitUnique`, which match `RecVStar.repRounds` with `repFrames`' chunks round by round.
-/

namespace FlockDeploy

open Flock
open FlockSoundness.Refine (Frame colBytes)

/-! ## What the honest prover sends -/

/-- A row's bytes, as a message. -/
def rowMsg (b : ByteArray) : Claims.Msg := b.data.toList

/-- **The honest prover's rows at a rep's rounds**, along its schedule: each round's top nodes (the next of `caps`, in
order), then, where the round absorbs, one row of the values its frames `ch` absorb. -/
def honestRows : List RecVStar.SchedRound → List (List Frame) → List (Array ByteArray) → List (List ByteArray)
  | s :: ss, ch :: chs, caps =>
    ((caps.take s.2.1.size).flatMap Array.toList ++
        (if s.2.2.2 then [(⟨(colBytes (vals ch)).toArray⟩ : ByteArray)] else [])) ::
      honestRows ss chs (caps.drop s.2.1.size)
  | _, _, _ => []

/-- **What the honest prover commits at a rep's rounds**, for its proof `p` at the batched value `T`. -/
def repRows (c : RepCtx) (p : Flock.Proof) (T : F128) : List (List ByteArray) :=
  honestRows c.sched (chunksOf c p T) p.ligerito.recursiveCaps.toList

/-- Round 0's rows at the honest prover: each table's level-0 cap nodes, in table order. -/
def capRows (caps : Array (Array ByteArray)) : List ByteArray := (caps.toList.map Array.toList).flatten

/-- **The honest prover's element-level messages**: round 0's rows (`capRows`), then each rep's rounds' rows, rep by
rep, each round joined (`joinR`); then the last message. -/
def honestMsgs (caps : Array (Array ByteArray)) (reps : List (List (List ByteArray))) (last : Claims.Msg) :
    List Claims.Msg :=
  (joinR ((capRows caps).map rowMsg) :: reps.flatten.map fun rs => joinR (rs.map rowMsg)) ++ [last]

/-- **An honest rep's proof**: its absorbed fields of the lengths its frames read (`AbsLens`), every nonce `0`, and one
recursive cap per level after the first, each of its level's `2^cap` nodes. -/
structure HonestRep (c : RepCtx) (p : Flock.Proof) : Prop where
  lens : AbsLens c p
  nonces : (∀ n ∈ p.ligerito.queryNonces, n = 0) ∧ (∀ n ∈ p.ligerito.claimNonces, n = 0) ∧
    ∀ n ∈ p.ligerito.consistencyNonces, n = 0
  caps : p.ligerito.recursiveCaps.size = c.sch.levels.size - 1
  capNodes : ∀ (i : ℕ) (h : i < p.ligerito.recursiveCaps.size),
    p.ligerito.recursiveCaps[i].size = 2 ^ (c.sch.levels.getD (i + 1) default).cap

/-- **A rep's schedule is its frames'**: `RecVStar.repRounds` at the frames' `m`, `k_log` and claim count, at the
frames' `fast100` schedule. -/
def SchedOf (c : RepCtx) : Prop :=
  fast100 c.st.m = some c.sch ∧ ∃ rs, RecVStar.repRounds c.st.m c.st.fold.kLog c.n = some rs ∧ c.sched = rs.toList

/-- Rows one message can join (`rowsOf_joinR`): fewer than `2^64`, each shorter than `2^64` bytes. -/
def RowsFit (rs : List ByteArray) : Prop := rs.length < 2 ^ 64 ∧ ∀ r ∈ rs, r.size < 2 ^ 64

/-! ## The statements -/

/-- **The honest proof fits the honest rows**, at any `T`. -/
def RepFits : Prop :=
  ∀ (c : RepCtx) (p : Flock.Proof) (T : F128), SchedOf c → HonestRep c p → p.cap.toList = c.cap0 →
    FitsRep c (repRows c p T) p T

/-- **Rows fix a fitting proof's absorbed fields and its `T`.** -/
def FitUnique : Prop :=
  ∀ (c : RepCtx) (rows : List (List ByteArray)) (p p' : Flock.Proof) (T T' : F128),
    FitsRep c rows p T → FitsRep c rows p' T' → ∀ q, absorbedFrom p' q = absorbedFrom p q ∧ T' = T

/-- **One honest rep reads back**: from the honest rows, with the honest proof as the last message's, `readRep` reads
the honest proof and its round messages. -/
def RepReadsBack : Prop :=
  ∀ (c : RepCtx) (p : Flock.Proof) (T : F128), SchedOf c → HonestRep c p → p.cap.toList = c.cap0 →
    readRep c (repRows c p T) p = some (p, framesOf c p T)

/-- **An honest prover's rounds read back.** Take a statement that sets up as `(s, spec, sts)` and coins whose first is
`c₁`. The honest prover sends round 0 from each table's level-0 cap `caps[i]` (of its `2^cap₀` nodes), and each rep's
rounds from its proof `ps[k]` (honest, at its table's cap) at its batched value `Ts k`, with a last message the reader
reads as `fl`, whose proofs are `ps`. On those messages `decR` decides. It reads round 0 as the link of the caps, each
table's `root_B` its cap's nodes joined, with no public words; every rep's proof as `ps[k]`; and the round messages as
the honest proofs' framed bytes. -/
def ReadsBack : Prop :=
  ∀ (rd : Claims.Msg → Option FinalP) (x last : Claims.Msg) (cs : List Coin) (s : Ins) (spec : SessionSpec)
    (sts : Array Stmt) (c₁ : Coin) (caps : Array (Array ByteArray)) (ps : Array Flock.Proof) (Ts : ℕ → F128)
    (fl : FinalP),
    setupOf false x = some (s, spec, sts) → cs.head? = some c₁ →
    caps.size = sts.size → (∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) →
    ps.size = 2 * sts.size →
    (∀ k < ps.size, SchedOf (repCtx spec sts c₁ (capRows caps) k) ∧
      HonestRep (repCtx spec sts c₁ (capRows caps) k) ps[k]! ∧ ps[k]!.cap = caps[k / 2]!) →
    RowsFit (capRows caps) →
    (∀ k < ps.size, ∀ rs ∈ repRows (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k), RowsFit rs) →
    rd last = some fl → fl.ps = ps →
    decR rd x (honestMsgs caps ((List.range ps.size).map fun k =>
        repRows (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)) last) cs =
      some ⟨s, spec, sts,
        ((caps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
          sts.toList.map fun _ => ByteArray.empty),
        c₁, ((List.range ps.size).map fun k => framesOf (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)).flatten.zip
          cs.tail.dropLast, ⟨fl.stmt, fl.j, ps⟩⟩

/-! ## From one rep to the session -/

theorem absorbedFrom_self (p : Flock.Proof) : absorbedFrom p p = p := by
  rcases p with ⟨cap, params, ⟨a, b, r, fa, fb, fc⟩, lc, rs, ⟨ic, ip, rc, rp, ⟨yr, rows, sib, salts⟩, ms, qn, ood, cbn,
    cn⟩⟩
  rfl

/-- **One rep reads back where the honest proof fits and fits uniquely.** -/
theorem repReadsBack_of (hE : RepFits) (hU : FitUnique) : RepReadsBack := by
  intro c p T hs hp hc
  have hf := hE c p T hs hp hc
  have hx : ∃ pT : Flock.Proof × F128, FitsRep c (repRows c p T) pT.1 pT.2 := ⟨(p, T), hf⟩
  obtain ⟨h1, h2⟩ := hU c _ p _ T _ hf (Classical.choose_spec hx) p
  unfold readRep
  rw [dif_pos hx, h1, absorbedFrom_self, h2]

theorem rowsOf_rows {rs : List ByteArray} (h : RowsFit rs) : rowsOf (joinR (rs.map rowMsg)) = some rs := by
  rw [rowsOf_joinR (by rw [List.length_map]; exact h.1) (by
    intro r hr
    obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hr
    show b.data.toList.length < 2 ^ 64
    rw [Array.length_toList]
    exact h.2 b hb)]
  have e : ((fun r : Claims.Msg => (⟨r.toArray⟩ : ByteArray)) ∘ rowMsg) = id := funext fun _ => rfl
  rw [List.map_map, e, List.map_id]

theorem mapM_rowsOf : ∀ {l : List (List ByteArray)}, (∀ rs ∈ l, RowsFit rs) →
    (l.map fun rs => joinR (rs.map rowMsg)).mapM rowsOf = some l
  | [], _ => rfl
  | rs :: l, h => by
    rw [List.map_cons, List.mapM_cons, rowsOf_rows (h rs (List.mem_cons_self ..)),
      mapM_rowsOf fun r hr => h r (List.mem_cons_of_mem _ hr)]
    rfl

theorem mapM_eq_some {α β : Type} {f : α → Option β} {g : α → β} :
    ∀ {l : List α}, (∀ a ∈ l, f a = some (g a)) → l.mapM f = some (l.map g)
  | [], _ => rfl
  | a :: l, h => by
    rw [List.mapM_cons, h a (List.mem_cons_self ..), mapM_eq_some fun b hb => h b (List.mem_cons_of_mem _ hb)]
    rfl

/-- **Blocks cut at their own lengths are the blocks.** -/
theorem cutLens_flatten_eq {α : Type} : ∀ (ls : List (List α)), cutLens (ls.map List.length) ls.flatten = ls
  | [] => rfl
  | l :: ls => by
    simp only [List.map_cons, List.flatten_cons, cutLens, List.take_left, List.drop_left, cutLens_flatten_eq ls]

theorem capLens {sts : Array Stmt} {caps : Array (Array ByteArray)} (hs : caps.size = sts.size)
    (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) :
    (caps.toList.map Array.toList).map List.length = level0Lens sts := by
  refine List.ext_getElem (by simp [level0Lens, hs]) fun i h₁ h₂ => ?_
  have hi : i < caps.size := by simpa using h₁
  have hi' : i < sts.size := hs ▸ hi
  simp only [List.map_map, List.getElem_map, Array.getElem_toList, level0Lens, Function.comp_apply,
    Array.length_toList]
  rw [← getElem!_pos caps i hi, hc i hi, getElem!_pos sts i hi']

theorem cutLens_capRows {sts : Array Stmt} {caps : Array (Array ByteArray)} (hs : caps.size = sts.size)
    (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) :
    cutLens (level0Lens sts) (capRows caps) = caps.toList.map Array.toList := by
  rw [← capLens hs hc, capRows, cutLens_flatten_eq]

/-- **The honest link**: each table's `root_B` its cap's nodes joined, and no public words. -/
theorem linkR_capRows {sts : Array Stmt} {caps : Array (Array ByteArray)} (hs : caps.size = sts.size)
    (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) :
    linkR sts (capRows caps) =
      some ((caps.toList.map Array.toList).map (·.foldl (· ++ ·) ByteArray.empty),
        sts.toList.map fun _ => ByteArray.empty) := by
  have hsum : (level0Lens sts).sum = (capRows caps).length := by
    rw [← capLens hs hc, capRows, List.length_flatten]
  unfold linkR
  rw [if_pos hsum, cutLens_capRows hs hc]

/-- **Rep `k` reads its table's level-0 cap among the honest round 0's rows.** -/
theorem cap0_capRows {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {caps : Array (Array ByteArray)}
    (hs : caps.size = sts.size) (hc : ∀ i < caps.size, caps[i]!.size = RecDeploy.level0Nodes sts[i]!) {k : ℕ}
    (hk : k / 2 < caps.size) : (repCtx spec sts c₁ (capRows caps) k).cap0 = caps[k / 2]!.toList := by
  show (cutLens (level0Lens sts) (capRows caps)).getD (k / 2) [] = _
  rw [cutLens_capRows hs hc]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hk,
    getElem!_pos caps (k / 2) hk, Option.map_some, Option.getD_some]

theorem range_map_getElem! {α : Type} [Inhabited α] (a : Array α) :
    ((List.range a.size).map fun k => a[k]!).toArray = a := by
  refine Array.ext (by simp) fun i h₁ h₂ => ?_
  simp only [List.getElem_toArray, List.getElem_map, List.getElem_range]
  exact getElem!_pos a i h₂

/-- **The session reads back where each rep does.** -/
theorem readsBack_of_rep (hR : RepReadsBack) : ReadsBack := by
  intro rd x last cs s spec sts c₁ caps ps Ts fl hx hc1 hs hcs hps hrep h0 hrows hrd hfl
  set reps := (List.range ps.size).map fun k => repRows (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)
    with hreps
  have hk2 : ∀ k < ps.size, k / 2 < caps.size := fun k hk => by omega
  have hone : ∀ k < ps.size, readRep (repCtx spec sts c₁ (capRows caps) k)
      (repRows (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)) ps[k]! =
      some (ps[k]!, framesOf (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)) := fun k hk => by
    obtain ⟨h1, h2, h3⟩ := hrep k hk
    refine hR _ _ _ h1 h2 ?_
    rw [cap0_capRows hs hcs (hk2 k hk), h3]
  have hlen : ∀ k < ps.size, (repRows (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)).length =
      (RecDeploy.repSchedAt spec sts[k / 2]!).length := fun k hk => by
    obtain ⟨T, hT, -⟩ := readRep_fits (hone k hk)
    exact hT.sched.symm
  have hlens : reps.map List.length = repLens spec sts := by
    rw [hreps, List.map_map, repLens, ← hps]
    exact List.map_congr_left fun k hk => hlen k (List.mem_range.1 hk)
  have hm : ((List.range ps.size).mapM fun k =>
      readRep (repCtx spec sts c₁ (capRows caps) k) (reps.getD k []) fl.ps[k]!) =
      some ((List.range ps.size).map fun k =>
        (ps[k]!, framesOf (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k))) := by
    refine mapM_eq_some fun k hk => ?_
    have hk' := List.mem_range.1 hk
    rw [hreps, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hk', Option.map_some,
      Option.getD_some, hfl]
    exact hone k hk'
  have hif : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = reps.flatten.length :=
    ⟨by rw [hfl, hps], by rw [← hlens, List.length_flatten]⟩
  have hpf : readPs spec sts c₁ (capRows caps) reps.flatten fl =
      some (ps, ((List.range ps.size).map fun k =>
        framesOf (repCtx spec sts c₁ (capRows caps) k) ps[k]! (Ts k)).flatten) := by
    unfold readPs
    rw [if_pos hif, ← hlens, cutLens_flatten_eq, ← hps, hm]
    simp only [Option.map_some, List.map_map, Function.comp_def, range_map_getElem!]
  have hhead : (honestMsgs caps reps last).head? = some (joinR ((capRows caps).map rowMsg)) := rfl
  have hlast : (honestMsgs caps reps last).getLast? = some last := List.getLast?_concat
  have hmid : (honestMsgs caps reps last).tail.dropLast.mapM rowsOf = some reps.flatten := by
    rw [show (honestMsgs caps reps last).tail.dropLast = reps.flatten.map fun rs => joinR (rs.map rowMsg) from
      List.dropLast_concat]
    refine mapM_rowsOf fun rs hr => ?_
    obtain ⟨l, hl, hrl⟩ := List.mem_flatten.1 hr
    rw [hreps] at hl
    obtain ⟨k, hk, rfl⟩ := List.mem_map.1 hl
    exact hrows k (List.mem_range.1 hk) rs hrl
  simp only [decR, hx, hlast, hhead, hc1, hrd, rowsOf_rows h0, hmid, linkR_capRows hs hcs, hpf,
    Option.bind_eq_bind, Option.bind_some, Option.pure_def]

end FlockDeploy
