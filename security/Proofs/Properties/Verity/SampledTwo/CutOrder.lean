module

public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Challenger.Window
public import Proofs.Properties.Challenger.CanonLine
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-! The cut where the device's ledger orders the two sessions (`cutBound_order`); its premise
`PrintsInOrder` is a fact of the device that no check gives, no named assumption, and no theorem here proves it. -/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

set_option mvcgen.warning false

open private strOf from Spot.ChallengerSession

namespace Verity.SampledTwo

open Lean (Json)
open Std.Do
open Properties.Challenger Spot.Challenger
open FlockDeploy (JsonEquiv)
open Properties.Pouw.LiveSalt.Coin (taggedBytes)
open Properties.Pouw.LiveSalt.Ledger (of_triple of_triple_opt throw_spec none_spec failure_spec toOption_spec
  ledgerTag_spec nat_spec some_pure_spec)
open Properties.Challenger.ChallengePhase (ledgerBytes recordBytes)
open Properties.Challenger.CanonRead (getD_equiv sorted_getD equiv_str)
open scoped ENNReal

def openedRoot (e : LedgerEntry) : String := strOf (Properties.Pouw.LiveSalt.openRoot e)

theorem strOf_equiv {x y : Json} (h : JsonEquiv x y) : strOf x = strOf y := by
  by_cases hx : ∃ s, x = .str s
  · obtain ⟨s, rfl⟩ := hx
    rw [equiv_str h.symm]
  · by_cases hy : ∃ s, y = .str s
    · obtain ⟨s, rfl⟩ := hy
      exact absurd ⟨s, equiv_str h⟩ hx
    · have e1 : strOf x = "" := by
        cases x with
        | str s => exact absurd ⟨s, rfl⟩ hx
        | _ => rfl
      have e2 : strOf y = "" := by
        cases y with
        | str s => exact absurd ⟨s, rfl⟩ hy
        | _ => rfl
      rw [e1, e2]

-- `mvcgen` splits `Core.Record.ofJson` into one goal per field it parses
set_option maxHeartbeats 4000000 in
theorem recordOfJson_seq {j : Json} {r : _root_.Core.Record} (h : _root_.Core.Record.ofJson j = .ok r) :
    ∃ n : Lean.JsonNumber, Spot.getD j "seq" = .num n ∧ r.seq = n.mantissa.toNat := by
  refine of_triple (Q := fun r => ∃ n : Lean.JsonNumber, Spot.getD j "seq" = .num n ∧ r.seq = n.mantissa.toNat) ?_ h
  clear h
  mvcgen [_root_.Core.Record.ofJson, -Spec.throw_Except, throw_spec]
  all_goals first
    | exact ⟨_, by assumption, rfl⟩
    | (refine ⟨_, ?_, rfl⟩; assumption)

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
theorem ledgerEntry_more {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    ∃ r, (_root_.Core.Record.ofJson e.entry).toOption = some r ∧ r.seq = e.seq ∧
      (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → r.session = some e.session) := by
  refine of_triple_opt (Q := fun e => ∃ r, (_root_.Core.Record.ofJson e.entry).toOption = some r ∧
      r.seq = e.seq ∧ (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → r.session = some e.session)) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Spec.throw_Option, none_spec, failure_spec, toOption_spec, ledgerTag_spec, nat_spec,
    some_pure_spec]
  all_goals refine ⟨_, by assumption, rfl, fun ho => ?_⟩
  all_goals first | assumption | (exfalso; simp_all)

theorem entry_seq {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    (∃ n : Lean.JsonNumber, Spot.getD e.entry "seq" = .num n ∧ e.seq = n.mantissa.toNat) ∧
      (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → Spot.getD e.entry "session" = .str e.session) := by
  obtain ⟨r, hr, hq, hses⟩ := ledgerEntry_more h
  have hr' := Properties.Challenger.ChallengePhase.toOption_some hr
  obtain ⟨n, hn, hrn⟩ := recordOfJson_seq hr'
  obtain ⟨-, -, h3⟩ := Properties.Challenger.Window.recordOfJson_fields hr'
  exact ⟨⟨n, hn, hq.symm.trans hrn⟩, fun hk => h3 _ (hses hk)⟩

theorem same_fields {l l' : Json} {e e' : LedgerEntry} (he : ledgerEntry? l = some e)
    (he' : ledgerEntry? l' = some e') {s : String}
    (hs : Flock.canonUtf8 (_root_.Core.Record.unsigned e.entry) = .ok s)
    (hs' : Flock.canonUtf8 (_root_.Core.Record.unsigned e'.entry) = .ok s) :
    e'.kind = e.kind ∧ e'.seq = e.seq ∧ openedRoot e' = openedRoot e ∧
      (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → e'.session = e.session) := by
  obtain ⟨sO, objO, kindO, schemaO, bodyO, -, -⟩ := Properties.Challenger.Window.ledgerEntry_fields he
  obtain ⟨sE, objE, kindE, schemaE, bodyE, -, -⟩ := Properties.Challenger.Window.ledgerEntry_fields he'
  obtain ⟨⟨n, hn, hq⟩, hses⟩ := entry_seq he
  obtain ⟨⟨n', hn', hq'⟩, hses'⟩ := entry_seq he'
  have F : ∀ k, k ≠ "sig" → JsonEquiv (Spot.getD e.entry k) (Spot.getD e'.entry k) := fun k hk =>
    Properties.Challenger.Window.entry_fields_equiv sO sE objO objE hs hs' hk
  have hK := F "kind" (by decide)
  rw [kindO, kindE] at hK
  have hk : e'.kind = e.kind := Json.str.inj (equiv_str hK.symm)
  have hQ := F "seq" (by decide)
  rw [hn, hn'] at hQ
  have hnn : n' = n := Json.num.inj (Properties.Challenger.CanonLine.equiv_num hQ.symm)
  have hseq : e'.seq = e.seq := by rw [hq, hq', hnn]
  have hS := F "schema" (by decide)
  rw [schemaO, schemaE] at hS
  have hsch : e'.schema = e.schema := Json.str.inj (equiv_str hS.symm)
  have hB := F "body" (by decide)
  have sBo := sorted_getD sO "body"
  have sBe := sorted_getD sE "body"
  rw [bodyO, bodyE] at hB
  rw [bodyO] at sBo
  rw [bodyE] at sBe
  have hW := getD_equiv sBo sBe hB "window"
  have hR : JsonEquiv (Properties.Pouw.LiveSalt.openRoot e) (Properties.Pouw.LiveSalt.openRoot e') := by
    unfold Properties.Pouw.LiveSalt.openRoot
    rw [hsch]
    exact getD_equiv (sorted_getD sBo "window") (sorted_getD sBe "window") hW _
  have hroot : openedRoot e' = openedRoot e := (strOf_equiv hR).symm
  refine ⟨hk, hseq, hroot, fun hkind => ?_⟩
  have hkind' : e'.kind = "open" ∨ e'.kind = "end" ∨ e'.kind = "reserve" := by
    rw [hk]
    exact hkind
  have hI := F "session" (by decide)
  rw [hses hkind, hses' hkind'] at hI
  exact Json.str.inj (equiv_str hI.symm)

section Lines

variable {dev : History} {key : ChallengerKey} {label : String} {coins : Json}

theorem line_device {ls : Array Json} {l : Json} {e : LedgerEntry} {m : ByteArray}
    (hc : Spot.getD coins "ledger" = .arr ls) (hl : l ∈ ls.toList) (hle : ledgerEntry? l = some e)
    (hm : ledgerMessage (ident key.pub) e.entry = .ok m) (hv : LmsVerifies key m e.sig) :
    (∃ line ∈ dev.ledger key.pub, ∃ e' s, ledgerEntry? line = some e' ∧
        Flock.canonUtf8 (_root_.Core.Record.unsigned e.entry) = .ok s ∧
        Flock.canonUtf8 (_root_.Core.Record.unsigned e'.entry) = .ok s) ∨ Break dev key label coins := by
  rcases signed_or_break (dev := dev) (label := label)
      (Properties.Challenger.Window.ledger_mem_signatures hc hl hle hm) hv with hsigned | b
  · obtain ⟨tag, s1, ht1, hc1, hm1, hU1, hT1⟩ := Properties.Challenger.Window.ledger_hashed hle hm
    have hU : ledgerBytes (ident key.pub) (Flock.Sha512.hash (taggedBytes tag s1)) ∈
        readHashed key label coins ++ deviceHashed dev key.pub :=
      List.mem_append_left _ (Properties.Challenger.Window.mem_readHashed_ledger hc hl hU1)
    unfold Signed signedBy at hsigned
    rcases List.mem_append.1 hsigned with hrun | hled
    · obtain ⟨r, hr, hms⟩ := List.mem_flatMap.1 hrun
      cases hcs : r.commitment with
      | none => rw [hcs] at hms; cases hms
      | some cs =>
        rw [hcs] at hms
        rcases List.mem_append.1 hms with hrec | hcl
        · cases hrm : recordMessage (ident key.pub) r.host cs with
          | error _ => rw [hrm] at hrec; cases hrec
          | ok m' =>
            rw [hrm] at hrec
            have hmm : m = m' := List.mem_singleton.1 hrec
            obtain ⟨d', hd', hm''⟩ := Properties.Challenger.ChallengePhase.recordMessage_ok hrm
            have hV : recordBytes (ident key.pub) r.host d' ∈
                readHashed key label coins ++ deviceHashed dev key.pub :=
              List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_run hr hcs
                (.inl (Properties.Challenger.ChallengePhase.record_mem_closeHashed
                  (Properties.Challenger.ChallengePhase.record_mem_recordHashed hd'))))
            rcases eq_or_break hU hV (hm1.symm.trans (hmm.trans hm'')) with hEq | b
            · exact absurd hEq.symm (Properties.Challenger.Window.recordBytes_ne_ledger _ _ _ _ _)
            · exact .inr b
        · cases hcm : closeMessage (ident key.pub) r.host cs r.id r.log.length (logChain r.log) r.ended with
          | error _ => rw [hcm] at hcl; cases hcl
          | ok m' =>
            rw [hcm] at hcl
            have hmm : m = m' := List.mem_singleton.1 hcl
            obtain ⟨rs, hrs, hm''⟩ := Properties.Challenger.ChallengePhase.closeMessage_ok hcm
            have hV : Properties.Challenger.ChallengePhase.closeBytes (ident key.pub) rs r.id r.log.length
                (logChain r.log) r.ended ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
              List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_run hr hcs
                (.inl (Properties.Challenger.ChallengePhase.close_mem_closeHashed hrs)))
            rcases eq_or_break hU hV (hm1.symm.trans (hmm.trans hm'')) with hEq | b
            · exact absurd hEq.symm (Properties.Challenger.ChallengePhase.closeBytes_ne_ledger _ _ _ _ _ _ _ _)
            · exact .inr b
    · obtain ⟨line, hline, hlm⟩ := List.mem_filterMap.1 hled
      obtain ⟨e', he', hem⟩ := Option.bind_eq_some_iff.1 hlm
      obtain ⟨tag', s2, ht2, hc2, hm2, hU2, hT2⟩ :=
        Properties.Challenger.Window.ledger_hashed he' (Properties.Challenger.ChallengePhase.toOption_some hem)
      have hV : ledgerBytes (ident key.pub) (Flock.Sha512.hash (taggedBytes tag' s2)) ∈
          readHashed key label coins ++ deviceHashed dev key.pub :=
        List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_ledger hline hU2)
      rcases eq_or_break hU hV (hm1.symm.trans hm2) with hEq | b
      · have hTr : taggedBytes tag s1 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_left _ (Properties.Challenger.Window.mem_readHashed_ledger hc hl hT1)
        have hTd : taggedBytes tag' s2 ∈ readHashed key label coins ++ deviceHashed dev key.pub :=
          List.mem_append_right _ (Properties.Challenger.ChallengePhase.mem_deviceHashed_ledger hline hT2)
        rcases eq_or_break hTr hTd (Properties.Challenger.Window.ledgerBytes_inj hEq) with hEq2 | b
        · obtain ⟨rfl, rfl⟩ := Properties.Challenger.Window.taggedBytes_inj2
            (Properties.Challenger.Window.ledgerTag_size ht1 ht2) hEq2
          exact .inl ⟨line, hline, e', s1, he', hc1, hc2⟩
        · exact .inr b
      · exact .inr b
  · exact .inr b

theorem lines_signed {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ)
    {e : LedgerEntry} (he : e ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger")) :
    ∃ ls l m, Spot.getD coins "ledger" = .arr ls ∧ l ∈ ls.toList ∧ ledgerEntry? l = some e ∧
      ledgerMessage (ident key.pub) e.entry = .ok m ∧ LmsVerifies key m e.sig := by
  obtain ⟨c, roots, ht, -⟩ := Properties.Challenger.ChallengePhase.drawCoins_ok h
  obtain ⟨a, hp, -⟩ := Properties.Challenger.ChallengePhase.trustedChallengePhaseOf_ok ht
  obtain ⟨ob, hao⟩ := Properties.Pouw.LiveSalt.Ledger.pinnedChallengePhase_ok hp
  obtain ⟨wp, win, en, hch, -⟩ := Properties.Pouw.LiveSalt.Ledger.auditedOpen_ok hao
  obtain ⟨es, hes, hvl⟩ := Properties.Pouw.LiveSalt.Ledger.checked_ok hch
  have hmem : e ∈ es.toList := by
    rw [← hes]
    exact he
  obtain ⟨m, hm, hv⟩ := Properties.Challenger.Window.verifyLedger_sig hvl e hmem
  obtain ⟨ls, hc, l, hl, hle⟩ : ∃ ls : Array Json, Spot.getD coins "ledger" = .arr ls ∧
      ∃ l ∈ ls.toList, ledgerEntry? l = some e := by
    unfold Properties.Pouw.LiveSalt.entries at he
    split at he
    · rename_i ls hc
      obtain ⟨l, hl, he⟩ := List.mem_filterMap.1 he
      exact ⟨ls, hc, l, hl, he⟩
    · simp at he
  exact ⟨ls, l, m, hc, hl, hle, hm, Properties.Challenger.ChallengePhase.lms_of_check hv⟩

theorem entry_device {w : String} {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ)
    {e : LedgerEntry} (he : e ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger")) :
    (∃ line ∈ dev.ledger key.pub, ∃ e', ledgerEntry? line = some e' ∧ e'.kind = e.kind ∧ e'.seq = e.seq ∧
        openedRoot e' = openedRoot e ∧
        (e.kind = "open" ∨ e.kind = "end" ∨ e.kind = "reserve" → e'.session = e.session)) ∨
      Break dev key label coins := by
  obtain ⟨ls, l, m, hc, hl, hle, hm, hv⟩ := lines_signed h he
  rcases line_device (dev := dev) (label := label) hc hl hle hm hv with ⟨line, hline, e', s, he', hs, hs'⟩ | b
  · obtain ⟨hk, hq, hr, hses⟩ := same_fields hle he' hs hs'
    exact .inl ⟨line, hline, e', he', hk, hq, hr, hses⟩
  · exact .inr b

end Lines

def LedgerOpens (ledger : List Json) (w a : String) : Prop :=
  ∃ line ∈ ledger, ∃ e, ledgerEntry? line = some e ∧ e.kind = "open" ∧ e.session = a ∧ openedRoot e = w

def EndsBefore (ledger : List Json) (a b : String) : Prop :=
  ∃ x ∈ ledger, ∃ y ∈ ledger, ∃ ex ey, ledgerEntry? x = some ex ∧ ledgerEntry? y = some ey ∧
    ex.kind = "end" ∧ ex.session = a ∧ ey.kind = "reserve" ∧ ey.session = b ∧ ex.seq < ey.seq

/-- On two windows, where the device's own ledger opens the first window for `a` and the second for `b`
and ends `a` before it reserves `b`, the log is the first window's runs' releases then the second's; a fact of the
device alone, which no check gives. -/
def PrintsInOrder (dev : History) (w₁ w₂ : String) (log : List Release) : Prop :=
  if w₁ = w₂ then PrintsOn dev w₁ log
  else ∀ pk a b, LedgerOpens (dev.ledger pk) w₁ a → LedgerOpens (dev.ledger pk) w₂ b → EndsBefore (dev.ledger pk) a b →
    ∃ log₁ log₂, log = log₁ ++ log₂ ∧ PrintsOn dev w₁ log₁ ∧ PrintsOn dev w₂ log₂

def OrderShown (θ : ByteArray) : Prop :=
  ∀ s : Verity.Sent, s.params = θ → Accepts s → ∀ p c e, caseOf s = .ok (p, c, e) →
    ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      ∃ o₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
      ∃ e₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
      ∃ rv ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
      ∃ o₂ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD e.interior.base.coins "ledger"),
        o₁.kind = "open" ∧ openedRoot o₁ = w₁ ∧ e₁.kind = "end" ∧ e₁.session = o₁.session ∧
        o₂.kind = "open" ∧ openedRoot o₂ = w₂ ∧ rv.kind = "reserve" ∧ rv.session = o₂.session ∧ e₁.seq < rv.seq

theorem printsTwo_of_order {dev : History} {key : ChallengerKey} {label : String} {coins : Json} {w : String}
    {after ξ : ByteArray} (h : sampleCoins key w label after coins = .ok ξ) {w₁ w₂ : String} (hne : w₁ ≠ w₂)
    {log : List Release} (hP : PrintsInOrder dev w₁ w₂ log) {o₁ e₁ rv o₂ : LedgerEntry}
    (ho₁ : o₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (he₁ : e₁ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (hrv : rv ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (ho₂ : o₂ ∈ Properties.Pouw.LiveSalt.entries (Spot.getD coins "ledger"))
    (k₁ : o₁.kind = "open") (r₁ : openedRoot o₁ = w₁) (ke : e₁.kind = "end") (se : e₁.session = o₁.session)
    (k₂ : o₂.kind = "open") (r₂ : openedRoot o₂ = w₂) (kr : rv.kind = "reserve") (sr : rv.session = o₂.session)
    (hlt : e₁.seq < rv.seq) : PrintsTwo dev w₁ w₂ log ∨ Break dev key label coins := by
  rcases entry_device (dev := dev) h ho₁ with ⟨la, hla, oa, hoa, hka, -, hra, hsa⟩ | b
  · rcases entry_device (dev := dev) h he₁ with ⟨lx, hlx, ex, hex, hkx, hqx, -, hsx⟩ | b
    · rcases entry_device (dev := dev) h hrv with ⟨ly, hly, ey, hey, hky, hqy, -, hsy⟩ | b
      · rcases entry_device (dev := dev) h ho₂ with ⟨lb, hlb, ob, hob, hkb, -, hrb, hsb⟩ | b
        · left
          unfold PrintsInOrder at hP
          rw [if_neg hne] at hP
          unfold PrintsTwo
          rw [if_neg hne]
          refine hP key.pub o₁.session o₂.session ?_ ?_ ?_
          · exact ⟨la, hla, oa, hoa, hka.trans k₁, hsa (.inl k₁), hra.trans r₁⟩
          · exact ⟨lb, hlb, ob, hob, hkb.trans k₂, hsb (.inl k₂), hrb.trans r₂⟩
          · refine ⟨lx, hlx, ly, hly, ex, ey, hex, hey, hkx.trans ke, (hsx (.inr (.inl ke))).trans se,
              hky.trans kr, (hsy (.inr (.inr kr))).trans sr, ?_⟩
            rw [hqx, hqy]
            exact hlt
        · exact .inr b
      · exact .inr b
    · exact .inr b
  · exact .inr b

open Classical in
noncomputable def onlyAt (o : Exchange) (h : History) : Exchange → History :=
  fun o' => if o' = o then h else ⟨fun _ => [], fun _ => []⟩

theorem onlyAt_self (o : Exchange) (h : History) : onlyAt o h o = h := by
  unfold onlyAt
  exact if_pos rfl

theorem printsOn_nil (w : String) (log : List Release) : PrintsOn ⟨fun _ => [], fun _ => []⟩ w log := by
  intro pk r hr
  first
    | cases hr
    | exact absurd hr List.not_mem_nil

theorem printsTwo_onlyAt {o : Exchange} {h : History} {θ : ByteArray}
    (hP : ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → PrintsTwo h w₁ w₂ o.log) :
    ∀ o' : Exchange, o'.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) →
      PrintsTwo (onlyAt o h o') w₁ w₂ o'.log := by
  intro o' _ w₁ w₂ hw
  by_cases ho : o' = o
  · rw [ho, onlyAt_self]
    exact hP w₁ w₂ hw
  · have hn : onlyAt o h o' = ⟨fun _ => [], fun _ => []⟩ := by
      unfold onlyAt
      exact if_neg ho
    rw [hn]
    unfold PrintsTwo
    by_cases hww : w₁ = w₂
    · rw [if_pos hww]
      exact printsOn_nil _ _
    · rw [if_neg hww]
      exact ⟨o'.log, [], (List.append_nil _).symm, printsOn_nil _ _, printsOn_nil _ _⟩

theorem binds_at (dev : Exchange → History) (θ : ByteArray) (hF : FirstReads θ) {o : Exchange} (hθ : o.params = θ)
    (ha : Accepts o.sent) (hP : ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → PrintsTwo (dev o) w₁ w₂ o.log)
    (hg : ¬ ReadIsDevice o) :
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ l ∈ opensReads o.sent, ∃ u v, Verity.Sampled.Lines.openCollision l o.log = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v := by
  have hP' := printsTwo_onlyAt (o := o) (h := dev o) (θ := θ) hP
  have hd := onlyAt_self o (dev o)
  rcases reads_or_break (onlyAt o (dev o)) θ hF hP' hθ ha with hr | hb
  · have hb := binds_of_accepts (onlyAt o (dev o)) θ hP' o hθ ha fun hl => hg ⟨hr, hl⟩
    rw [hd] at hb
    exact hb
  · rw [hd] at hb
    exact .inl hb

theorem order_binds (dev : Exchange → History) (θ : ByteArray) (hF : FirstReads θ) (hO : OrderShown θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) →
      PrintsInOrder (dev o) w₁ w₂ o.log)
    {o : Exchange} (hθ : o.params = θ) (ha : Accepts o.sent) (hg : ¬ ReadIsDevice o) :
    (∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2) ∨
      ∃ l ∈ opensReads o.sent, ∃ u v, Verity.Sampled.Lines.openCollision l o.log = some (u, v) ∧ u ≠ v ∧
        Flock.Sha512.hash u = Flock.Sha512.hash v := by
  have ha' := ha
  obtain ⟨p, c, e, pub, prof, h1, hacc⟩ := ha'
  obtain ⟨key, d₁, _, hK, h3, _⟩ := hF o.sent hθ ha p c e h1
  obtain ⟨-, c₂, ξ₂, r, i, _run, hin, -, -, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨⟨sj, hcanon, hcom⟩, h7, _, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have h5 : interiorOf e = .ok c₂ := by
    unfold interiorOf
    rw [hcanon]
    first
      | exact hcom
      | simpa [bind, Except.bind] using hcom
  obtain ⟨d₂, hd₂⟩ := oneStageZk_source h7
  have h6 : secondSamples p e = some d₂ := by
    unfold secondSamples
    rw [h5]
    first
      | exact hd₂
      | simpa [Except.toOption] using hd₂
  have h2 : keyOf o.sent.params = some key := by
    have h := hK
    rw [← hθ] at h
    exact h
  have hk₂ : d₂.key = key := key_second h1 h2 h5 h6
  have hm₂ : (d₂, e.interior.base.coins) ∈ records o.sent := by rw [records_eq h1 h2, h3, h6]; simp
  have hw : executionPhasesOf θ = some (d₁.window, d₂.window) := by rw [← hθ]; exact executionPhasesOf_eq h1 h3 h5 h6
  obtain ⟨raw, hraw, -⟩ := oneStageZk_reads p.audit c₂ d₂ hd₂ e.interior ξ₂ h7
  have hPT : PrintsTwo (dev o) d₁.window d₂.window o.log ∨
      Break (dev o) d₂.key d₂.label e.interior.base.coins := by
    have hPO := hP o hθ d₁.window d₂.window hw
    by_cases hww : d₁.window = d₂.window
    · left
      unfold PrintsInOrder at hPO
      rw [if_pos hww] at hPO
      unfold PrintsTwo
      rw [if_pos hww]
      exact hPO
    · obtain ⟨o₁, ho₁, e₁, he₁, rv, hrv, o₂, ho₂, k₁, r₁, ke, se, k₂, r₂, kr, sr, hlt⟩ :=
        hO o.sent hθ ha p c e h1 d₁.window d₂.window hw hww
      exact printsTwo_of_order hraw hww hPO ho₁ he₁ hrv ho₂ k₁ r₁ ke se k₂ r₂ kr sr hlt
  rcases hPT with hPT | b
  · refine binds_at dev θ hF hθ ha (fun w₁ w₂ hw' => ?_) hg
    rw [hw] at hw'
    have hpair := Option.some.inj hw'
    simp only [Prod.mk.injEq] at hpair
    obtain ⟨rfl, rfl⟩ := hpair
    exact hPT
  · exact .inl ⟨(d₂, e.interior.base.coins), hm₂, by rw [hK, ← hk₂], b⟩

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem cutBound_order (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hw : Wired V k θ) (hF : FirstReads θ) (hO : OrderShown θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) →
      PrintsInOrder (dev o) w₁ w₂ o.log)
    (hs : SignsPlay V dev θ F c) : CutBound V k θ F c σ ReadIsDevice (terms V dev θ F c σ) :=
  cutBound_of V k dev θ F c σ ReadIsDevice hw (fun _ hθ ha hg => order_binds dev θ hF hO hP hθ ha hg) hs

end Verity.SampledTwo
