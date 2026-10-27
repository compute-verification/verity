import Proofs.Flock.Claims.Recursion.RestRead
import Proofs.Flock.Claims.Recursion.DeployedR

/-!
# `ReadsAt` at the deployed readers: the last step reads the opened messages

`hrest_read` takes `ReadsAt rd final fin`: wherever `final` reads V\*'s last message, `nextR rd`'s last step reads the
opened messages (`ReadOkR`). This file splits that reading into its parts at any reader (`readOkR_of`): the plain
verifier answers no coin at a last message not tagged `0` (`act_not_coin`), `decR` reads (the statement sets up, the
rows read back, round 0 carries the link's nodes, the rounds after it the reps' blocks, a fit at every rep), and the
record value `decR` builds reads as custody's (`DecLive`).

The fits are the one part that reads the rows' contents. `FitsLift` is what they need: a fit at some rows gives a fit
at every rows of the same byte sizes, at a setup that folds at its own `k_log` (`RestFitsLift.fitsLift` proves it).
The honest proof fits the honest rows (`RepFits`), so every rows the layout reads at the honest rows' sizes fit.
-/

namespace RecDeploy.RestRead

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP DecR decR actR ReadOkR FitsRep repCtx repLens level0Lens cutLens rowsOf liveOfJ coinSubJ)

/-- **The plain verifier answers no coin** at a last message not tagged `0`. -/
theorem act_not_coin {x : Msg} {ms : List Msg} {cs : List Coin} {a : Msg} (hlast : ms.getLast? = some a)
    (ha : ∀ t, a ≠ 0 :: t) (c : Coin) : FlockDeploy.act false x ms cs ≠ .coin c := by
  intro hc
  unfold FlockDeploy.act at hc
  repeat' (split at hc)
  all_goals first | exact absurd hlast.symm (ha _) | (simp_all <;> exact absurd hlast.symm (ha _))

/-- **The last step decides** where the plain verifier answers no coin and `decR` reads. -/
theorem actR_of_decR {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin} {a : Msg} {d : DecR}
    (hlast : ms.getLast? = some a) (ha : ∀ t, a ≠ 0 :: t) (hd : decR rd x ms cs = some d) :
    actR rd x ms cs = .decide d.s d.spec d.sts d.link d.c₁ d.mids d.f := by
  unfold actR
  split
  · rename_i c hc
    exact absurd hc (act_not_coin hlast ha c)
  · rw [hd]

theorem mapM_some_of {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, ∃ b, f a = some b) → ∃ bs, l.mapM f = some bs
  | [], _ => ⟨[], rfl⟩
  | a :: l, h => by
    obtain ⟨b, hb⟩ := h a List.mem_cons_self
    obtain ⟨bs, hbs⟩ := mapM_some_of (l := l) fun a' ha' => h a' (List.mem_cons_of_mem a ha')
    exact ⟨b :: bs, by simp [List.mapM_cons, hb, hbs]⟩

/-- **The record value `decR` builds reads as custody's**, at reader `rd`, at every decision with a coin a message
whose statement passes its scope check. -/
def DecLive (rd : Msg → Option FinalP) : Prop :=
  ∀ x ms cs d, ms.length = cs.length → decR rd x ms cs = some d → d.s.read.check = .ok () →
    ∃ lv, liveOfJ d.spec (coinSubJ d.spec d.c₁ d.mids d.f.j) = .ok lv

/-- **The last step reads**, at any reader: the statement sets up, the last message isn't tagged `0` and `rd` reads
it with two proofs a table, round 0's rows read back as the link's nodes, the rounds after it as the reps' blocks, every
rep's block has a fit, and the record value reads as custody's where there is a coin a message and the statement
passes its scope check. -/
theorem readOkR_of {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin}
    {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt} {a : Msg} {fl : FinalP} {c₁ : Coin}
    {r₀ : List ByteArray} {rss : List (List ByteArray)}
    (hv : FlockDeploy.setupOf false x = some v) (hlast : ms.getLast? = some a) (ha : ∀ t, a ≠ 0 :: t)
    (hrd : rd a = some fl) (hst : fl.stmt = x) (hc : cs.head? = some c₁)
    (h0 : (rowsOf (ms.headD [])).map (List.map FlockDeploy.nodeOf) = some r₀)
    (hr₀ : (level0Lens v.2.2).sum = r₀.length) (hmid : ms.tail.dropLast.mapM rowsOf = some rss)
    (hps : fl.ps.size = 2 * v.2.2.size) (hrss : (repLens v.2.1 v.2.2).sum = rss.length)
    (hfit : ∀ k < 2 * v.2.2.size, ∃ pT : Flock.Proof × F128,
      FitsRep (repCtx v.2.1 v.2.2 c₁ r₀ cs.tail.dropLast k) ((cutLens (repLens v.2.1 v.2.2) rss).getD k []) pT.1 pT.2)
    (hshape : ∀ pf, FlockDeploy.readPs v.2.1 v.2.2 c₁ r₀ rss cs.tail.dropLast fl = some pf →
      FlockDeploy.ShapedAt v.2.1 v.2.2 c₁ pf.1)
    (hlen : ms.length = cs.length) (hck : v.1.read.check = .ok ()) (hlive : DecLive rd) : ReadOkR rd x ms cs := by
  obtain ⟨d, hd⟩ : ∃ d, decR rd x ms cs = some d := by
    have hm : ∃ a₁, ms.head? = some a₁ := by
      cases ms with
      | nil => cases hlast
      | cons b l => exact ⟨b, rfl⟩
    obtain ⟨a₁, ha₁⟩ := hm
    have h0' : (rowsOf a₁).map (List.map FlockDeploy.nodeOf) = some r₀ := by
      cases ms with
      | nil => cases ha₁
      | cons b l => cases Option.some.inj ha₁; exact h0
    have hl : FlockDeploy.linkR v.2.2 r₀ = some ((cutLens (level0Lens v.2.2) r₀).map (·.foldl (· ++ ·) ByteArray.empty),
        v.2.2.toList.map fun _ => ByteArray.empty) := by
      unfold FlockDeploy.linkR
      rw [if_pos hr₀]
    obtain ⟨pf, hpf⟩ : ∃ pf, FlockDeploy.readPs v.2.1 v.2.2 c₁ r₀ rss cs.tail.dropLast fl = some pf := by
      unfold FlockDeploy.readPs
      rw [if_pos ⟨hps, hrss⟩]
      have hk : ∀ k ∈ List.range (2 * v.2.2.size), ∃ y, FlockDeploy.readRep (repCtx v.2.1 v.2.2 c₁ r₀ cs.tail.dropLast k)
          ((cutLens (repLens v.2.1 v.2.2) rss).getD k []) fl.ps[k]! = some y := by
        intro k hk
        obtain ⟨pT, hpT⟩ := hfit k (List.mem_range.1 hk)
        unfold FlockDeploy.readRep
        rw [dif_pos ⟨pT, hpT⟩]
        exact ⟨_, rfl⟩
      obtain ⟨ys, hys⟩ := mapM_some_of hk
      rw [hys]
      exact ⟨_, rfl⟩
    have hfl : (rd a).filter (·.stmt == x) = some fl := Option.filter_eq_some_iff.2 ⟨hrd, by simp [hst]⟩
    have hpf' : (FlockDeploy.readPs v.2.1 v.2.2 c₁ r₀ rss cs.tail.dropLast fl).filter (fun pf =>
        @decide (FlockDeploy.ShapedAt v.2.1 v.2.2 c₁ pf.1) (Classical.propDecidable _)) = some pf :=
      Option.filter_eq_some_iff.2 ⟨hpf, @decide_eq_true _ (Classical.propDecidable _) (hshape pf hpf)⟩
    simp only [decR, Option.bind_eq_bind, Option.pure_def, hv, hlast, ha₁, hc, hfl, h0', hmid, hl, hpf',
      Option.bind_some, Option.some.injEq, exists_eq']
  have ha := actR_of_decR hlast ha hd
  unfold ReadOkR
  rw [ha]
  obtain ⟨v', _, _, _, _, _, _, hv', -, -, -, -, -, hdd⟩ := FlockDeploy.decR_some hd
  have e : v' = v := Option.some.inj (hv'.symm.trans hv)
  subst e
  exact hlive x ms cs d hlen hd (by rw [hdd]; exact hck)

/-- **A fit lifts to every rows of its rows' byte sizes**, at a setup that folds at its own `k_log`: rows the same
sizes as fitting ones have a fit. -/
def FitsLift : Prop :=
  ∀ (c : FlockDeploy.RepCtx), c.st.fold.kLog = c.st.kLog → ∀ (rows rows' : List (List ByteArray)) (p' : Flock.Proof)
    (T' : F128),
    FitsRep c rows' p' T' → rows.map (·.map ByteArray.size) = rows'.map (·.map ByteArray.size) →
      ∃ pT : Flock.Proof × F128, FitsRep c rows pT.1 pT.2

end RecDeploy.RestRead
