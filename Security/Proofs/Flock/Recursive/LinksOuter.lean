import Proofs.Flock.Recursive.LinksInner
import Proofs.Flock.Claims.Recursion.KSLinks
import Proofs.Flock.Claims.Recursion.InnerAccepts
import Proofs.Flock.Claims.Recursion.ResLowerDeployed
import Proofs.Flock.Claims.Recursion.MoreSlots
import Proofs.Flock.VBridge.RecLink
import Proofs.Flock.VBridge.Pinned
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# The link part on the outer side: what an accepted run's link part fixes

The links bind (`Claims.Recursion.LinkBindsAt`) when an accepted run whose extracted witness fails the link clause gives
a collision among what the link finder reads. This file holds the outer side's pieces that chain reads:

* `Claims.Recursion.EO_judged_of_E`: where `E*` extracts, `E_O` gave outer witnesses at its judged draw (else the inner
  view there is rejected and the inner `E^rep` stops), so the link finder reads them off the first draw.
* `collision_of_collides`, `commit_bind`: the verifier's `Collides` pair is a `Collision` of the deployed SHA-512, and two
  equal commit strings are one row or a collision of the row's or the salt's evaluations.
* `linkName_inj`, `linkName_params`, `linkParams_of_check`: the name `rec-link/u<UB>-p<PB>` determines `UB` and `PB`
  (as `FlockVBridge.unitName_inj` does a level's), so `RecLinkPart.check` at a part so named ran `checkLinks` at them
  (`linkCheck_staged`).
* `stagedLink`: a staged part whose V\* statement is `.link UB PB N` has one table, its circuit the part's, named
  `linkName UB PB`, its rows and leaves V\*'s, and its population `N`.
-/

namespace Claims.Recursion

open Claims

section Judged

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

private theorem evalBindFst {R α β : Type} (H : Msg → Msg) (run : R → Trace) (p : RP R α) (f : α → RP R β) :
    ((p.bind f).eval H run).1 = ((f (p.eval H run).1).eval H run).1 :=
  (RP.eval_bind H run p f).1

/-- **Where `E*` extracts, `E_O` gave outer witnesses at its judged draw**: else `dec` reads a rejected inner view there
and the inner `E^rep` stops at its judged draw. -/
theorem EO_judged_of_E (H : Msg → Msg) (A : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (decO : List (ℕ × Col) → List T) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (r₀ : Fin (A.V.k x) → Coin)
    (c : (E H A I DO decO).C x) {w : W} (h : ((E H A I DO decO).eval H P x ω r₀ c).1 = some w) :
    ∃ ws, ((EO A I DO decO).eval H P x ω r₀ (judgedC c)).1 = some ws := by
  cases hO : (((EO A I DO decO).ext x r₀ c.1).eval H (run H P A.V x ω)).1 with
  | some ws => exact ⟨ws, hO⟩
  | none =>
    exfalso
    unfold Extractor.eval at h
    simp only [E, served, erep, serve, outerStage, evalBindFst, hO, dec, RP.eval, reduceCtorEq, ↓reduceIte] at h

end Judged

end Claims.Recursion

namespace RecDeploy.Links

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB

/-! ## Collisions -/

/-- **The verifier's colliding pair is a collision of the deployed SHA-512.** -/
theorem collision_of_collides {a b : ByteArray} (h : RecLinkBind.Collides (a, b)) :
    Claims.Collision FlockDeploy.sha512 a.data.toList b.data.toList := by
  refine ⟨fun he => h.1 ?_, ?_⟩
  · have := congrArg Claims.ofMsg he
    rwa [show a.data.toList = Claims.toMsg a from rfl, show b.data.toList = Claims.toMsg b from rfl,
      Claims.ofMsg_toMsg, Claims.ofMsg_toMsg] at this
  · show Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg a))) =
      Claims.toMsg (Sha512.hash (Claims.ofMsg (Claims.toMsg b)))
    rw [Claims.ofMsg_toMsg, Claims.ofMsg_toMsg, h.2]

/-- **Two equal commit strings are one row, or a collision**: of the rows' evaluations under one salt, or of the
salts' (`commit_tie` without the finder's list). -/
theorem commit_bind {A B y σ : Msg}
    (heq : FlockSoundness.Binding.commitString A y = FlockSoundness.Binding.commitString B σ) :
    A = B ∨ Claims.Collision FlockDeploy.sha512 A B ∨
      Claims.Collision FlockDeploy.sha512 (FlockSoundness.Refine.K512.saltPrefix ++ y)
        (FlockSoundness.Refine.K512.saltPrefix ++ σ) := by
  unfold FlockSoundness.Binding.commitString at heq
  obtain ⟨h1, h2⟩ := List.append_inj heq ((length_xorMask A y).trans (length_xorMask B σ).symm)
  by_cases hs : y = σ
  · subst hs
    by_cases hAB : A = B
    · exact Or.inl hAB
    · refine Or.inr (Or.inl ⟨hAB, ?_⟩)
      rw [sha512_eq, sha512_eq]
      have hl : ∀ r : Msg, (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 r)).length =
          (FlockSoundness.Refine.K512.mask y).length := fun r => by
        rw [FlockSoundness.Discharge.Hm.length_digBytes, FlockSoundness.Refine.K512.mask_len]; rfl
      exact FlockSoundness.Merkle.xorBytes_inj (hl A) (hl B) h1
  · refine Or.inr (Or.inr ⟨fun he => hs (List.append_cancel_left he), ?_⟩)
    rw [sha512_eq, sha512_eq]
    exact h2

/-! ## The part's name -/

/-- **The name `rec-link/u<UB>-p<PB>` determines `UB` and `PB`**: a decimal numeral has no `-`. -/
theorem linkName_inj {UB PB UB' PB' : ℕ} (h : VStar.linkName UB PB = VStar.linkName UB' PB') :
    UB = UB' ∧ PB = PB' := by
  have h := congrArg String.toList h
  simp only [VStar.linkName, String.toList_append, toString, Nat.toList_repr, List.append_assoc] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  obtain ⟨h1, h2⟩ := FlockVBridge.digits_dash (dig UB) (dig UB') (by simpa using h)
  simp only [List.cons.injEq, true_and] at h2
  exact ⟨FlockVBridge.eq_of_toDigits h1, FlockVBridge.eq_of_toDigits h2⟩

/-- `linkParams` gives `(UB, PB)` only for the name `linkName UB PB`. -/
theorem linkName_params {name : String} {UB PB : ℕ} (h : VStar.linkParams name = some (UB, PB)) :
    VStar.linkName UB PB = name := by
  unfold VStar.linkParams at h
  repeat' split at h
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h; exact beq_iff_eq.1 ‹_›)
    | cases h

theorem startsWith_linkName (UB PB : ℕ) : (VStar.linkName UB PB).startsWith "rec-link/" = true := by
  have : VStar.linkName UB PB = "rec-link/" ++ s!"u{UB}-p{PB}" := by
    simp [VStar.linkName, ← String.toList_inj, List.append_assoc, FlockVBridge.toString_str]
  rw [this]
  simp [String.startsWith_string_iff]

/-- **A part the verifier accepted under the name `linkName UB PB` gives back `(UB, PB)`**: the name starts with
`rec-link/`, so `check` refuses it when `linkParams` gives nothing, and any pair `linkParams` gives has that name. -/
theorem linkParams_of_check {st : Stmt} {own : Option (Array Registered.Port)} {files : Option RecCoins.Source}
    {links : Option (ByteArray × ℕ)} (h : RecLinkPart.check st own files links = .ok ()) {UB PB : ℕ}
    (hn : st.c.unit.name = VStar.linkName UB PB) : VStar.linkParams st.c.unit.name = some (UB, PB) := by
  cases hp : VStar.linkParams st.c.unit.name with
  | some p =>
    obtain ⟨UB', PB'⟩ := p
    obtain ⟨rfl, rfl⟩ := linkName_inj ((linkName_params hp).trans hn)
    rfl
  | none =>
    unfold RecLinkPart.check at h
    dsimp only at h
    rw [hp] at h
    dsimp only at h
    rw [hn, startsWith_linkName] at h
    cases h

/-- **The recursive setup ran the link part's checks at a table named `linkName UB PB`**: `checkLinks` at `UB`, `PB`,
V\*'s unit and V\*'s output layout. -/
theorem linkCheck_staged {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hsu : setupO xO = some (i, spec, sts)) {t : ℕ} (ht : t < sts.size) {UB PB : ℕ}
    (hn : sts[t].c.unit.name = VStar.linkName UB PB) :
    RecLinkPart.checkLinks sts[t] i.base.registered i.src i.links UB PB = .ok () ∧
      RecLinkPart.unitNet UB PB = .ok sts[t].c.unit ∧ RecLinkPart.outsOk sts[t].c = true := by
  obtain ⟨-, hset⟩ := setupO_some hsu
  have h := Flock.Readable.array_forM_ok (RecIns.setup_linkPart hset) t ht
  exact RecLinkPart.check_link h (linkParams_of_check h hn)

/-! ## The staged link part -/

/-- **A staged link part**: at a part whose V\* statement is `.link UB PB N`, the outer statement has one table, its
circuit the part's, named `linkName UB PB`, its output net its unit's, its rows and leaves V\*'s `RecLink{UB, PB}`'s,
no `v2` port, untyped, without public inputs, and its population `N`. -/
theorem stagedLink {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : Flock.RecVStar.statements r.shape = .ok ss) {UB PB N : ℕ}
    (hsv : ss[s]? = some (.link UB PB N)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c) :
    p.sts.size = 1 ∧ p.sts[0].c = c ∧ (∃ tags, FlockSoundness.ParseFacts tags c) ∧
      c.unit.name = VStar.linkName UB PB ∧ c.outNet = c.unitNet ∧
      (∃ u, VStar.linkUnit UB PB = .ok u ∧ c.upv = 1 ∧
        c.leavesIn.getD 0 #[] = (Array.range u.inWords).map Int.ofNat ∧
        c.ports.toList.map (fun q => (q.name, q.words, q.bits)) =
          Flock.RecVStar.vstarRows (VStar.linkSub UB PB) u) ∧
      (∀ q ∈ c.ports.toList, q.v2 = false) ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
      Flock.RecVStar.population p.sts[0] = N := by
  obtain ⟨p', r', sv, hx', ht, hpi, hr', ⟨ss', hss', hsv'⟩, -, htab⟩ := stagedTables hS
  have e1 : p = p' := Option.some.inj (hx.symm.trans hx')
  subst e1
  have e2 : r = r' := Option.some.inj (hr.symm.trans hr')
  subst e2
  have e3 : ss = ss' := Except.ok.inj (hss.symm.trans hss')
  subst e3
  have e4 : VStar.Stmt.link UB PB N = sv := Option.some.inj (hsv.symm.trans hsv')
  subst e4
  have hc0 := partCircuit_staged hS hx ht hpi hr hc h0
  have h1 : p.sts.size = 1 := by
    obtain ⟨-, i, spec, sts, -, hsu, -⟩ := id hS
    have e : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
    rw [e]
    exact (oneFull_staged hS hsu).1
  obtain ⟨tags, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  rw [hc0] at P
  have hn : c.unit.name = VStar.linkName UB PB := by rw [← hc0]; exact (htab 0 h0).1
  obtain ⟨hsc, r2, ss2, sv2, hr2, hss2, hsv2, hpop⟩ := staged_scope hS hx h0
  have e5 : r2 = r := Option.some.inj (hr2.symm.trans hr)
  subst e5
  have e6 : ss2 = ss := Except.ok.inj (hss2.symm.trans hss)
  subst e6
  have e7 : sv2 = VStar.Stmt.link UB PB N := Option.some.inj (hsv2.symm.trans hsv)
  subst e7
  rw [hc0, FlockSoundness.Discharge.FailClosed.scopeOk_eq] at hsc
  have ho : c.outNet = c.unitNet := by
    have hs := hsc
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  obtain ⟨u, hu, hupv, -, hleaves, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  rw [hc0] at hupv hleaves hrows
  exact ⟨h1, hc0, ⟨tags, P⟩, hn, ho, ⟨u, hu, hupv, hleaves, hrows⟩,
    fun q hq => (FlockSoundness.Discharge.Layout.scopeOk_ports hsc q hq).1, ht, hpi, hpop⟩

end RecDeploy.Links
