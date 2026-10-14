import Proofs.Flock.Recursive.PartTie
import Proofs.Flock.C3.RegLevels.LevelsChecked
import Proofs.Flock.C3.RegLevels.AccReads
import Proofs.Flock.Recursive.HPortTie

/-!
# The parts' value ties at C3's family: `OwnNamesV`, `VRowsAre` and `ZeroReads`' bit, off `ValueColl`

L's ties at one read (`Recursive/PartTie.lean`), at every accepted outcome of C3's family off a bad event `Bad` that
holds at `ValueColl` (`hbr_honest_reg`'s, whose `RepsReadOff` they enter), at a session whose unit is a `rec-residuals`
part. The family's facts they read: check 5 at the verifier's source and candidate 3's inputs (`CoinsAcceptedAt`, the
checked family's by `coinsAcceptedAt_checked`), each registered read the reads file's (`ReadsOwnAt`), and the
statement checks (`ChecksAccepted`), whose `checkAcc` names each port once (`names_of_checks`); no registered read is
of a segmented row (`readsUnsegC3`).

* `ResUnit`: a unit named as part `i` of `rec-residuals` and as none of check 5's other units;
* `source_of_check`, `prog_of_res`, `resCheck_of_accepted`: check 5 at such a unit holds only at a source, there its
  circuit has a `program_sha512` (`checkValue`'s, `checkRows_root`), and it holds at the session's first statement,
  whose circuit is the session's (`setup_zero_c`);
* `ownVOut_of`: the reads file names `v` and `out` (`own_v_out`), `OwnNamesV`;
* `vRowsOff_of`: the registrant's bits at a `v` read are `vOf`'s at the rep's rounds and comb (`vRows_tie`),
  `VRowsAre`;
* `zeroOff_of`: an `out` read is of a zero row and the registrant's bits there are 0 (`zero_tie`), `ZeroReads`' bit.
-/

namespace FlockC3.RegLevels

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

/-- **A `rec-residuals` part's unit**: named part `i` at `(m, k, cl)`, and none of check 5's other units. -/
def ResUnit (c : Flock.Circuit) (m k cl i : ℕ) : Prop :=
  Flock.RecCoins.stackedParams c.unit.name = none ∧ Flock.RecOpening.params c.unit.name = none ∧
    Flock.RecSparse.params c.unit.name = none ∧ Flock.RecOpen.params c.unit.name = none ∧
    Flock.RecResiduals.params c.unit.name = some (m, k, cl, i)

/-- **Check 5 at a part's unit holds only at a source**: without one it refuses (`--rec-inner-session`). -/
theorem source_of_check {sj : Flock.Stmt} {own : Array Flock.Registered.Port}
    {coins : Option Flock.RecCoins.Source} {c3 : Option Flock.RecCoins.C3}
    (h : Flock.RecCoins.check sj (some own) coins c3 = .ok ()) {m k cl i : ℕ} (hu : ResUnit sj.c m k cl i) :
    ∃ f, coins = some f := by
  obtain ⟨hso, hop, hsp, ho, hres⟩ := hu
  cases coins with
  | some f => exact ⟨f, rfl⟩
  | none =>
    exfalso
    revert h
    unfold Flock.RecCoins.check
    simp [hso, hop, hsp, ho, hres]

/-- **A part's circuit check 5 accepted has a `program_sha512`**: `checkValue`'s at `v` (`checkRows_root`). -/
theorem prog_of_res {sj : Flock.Stmt} {own : Array Flock.Registered.Port} {f : Flock.RecCoins.Source}
    {c3 : Option Flock.RecCoins.C3} (hck : Flock.RecCoins.check sj (some own) (some f) c3 = .ok ()) {m k cl i : ℕ}
    (hu : ResUnit sj.c m k cl i) : ∃ prog, sj.c.metaJ.getObjValAs? String "program_sha512" = .ok prog := by
  obtain ⟨hso, hop, hsp, ho, hres⟩ := hu
  obtain ⟨inn, -, hck'⟩ := residualsCheck_of_check hck hso hop hsp ho hres
  obtain ⟨rows, what, -, hcv, -⟩ := residualsBoth_of_check hck'
  obtain ⟨reg, prog, -, -, -, hprog, -⟩ :=
    checkRows_root (salt := fun p => Flock.RecCoins.verifierSalt p Flock.RecCoins.V_VALUE) hcv
  exact ⟨prog, hprog⟩

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Where a session accepts, a transcript it accepts passes check 5 at the history's source `src` and candidate 3's
`c3`** (`CoinsAccepted` at those). -/
def CoinsAcceptedAt (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂)
    (src : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source)
    (c3 : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.C3) : Prop :=
  ∀ R ω t R₂ j, (∃ tr', AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr') →
    ∃ tr', CoinsChecked (c3 t) (src t) (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr'

/-- **The checked family passes check 5 at `valueSrcAll`'s source and candidate 3's inputs** (`ChecksC3`'s third
conjunct, as `coinsAccepted_checked`). -/
theorem coinsAcceptedAt_checked {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} :
    CoinsAcceptedAt (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
      (fun t => coins (wr t).2) (fun t => Flock.Guarantees.c3Of ish (Flock.Guarantees.innerHeld wr st reg t)) := by
  intro R ω t R₂ j hex
  obtain ⟨tr', -, hc⟩ := FlockC3.PubRows.checks_of_keepOwn (z := Flock.Guarantees.custodied zr hCust h R ω t R₂ j)
    (P := fun J o tr' => (ℐ).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (innerCommitsAt (wr t).1 (wr t).2 st reg).toOption (coins (wr t).2) j (bind j)
        rows J o tr')
    (Pc := fun J o tr' =>
      CoinsChecked (Flock.Guarantees.c3Of ish (Flock.Guarantees.innerHeld wr st reg t)) (coins (wr t).2) J o tr')
    (fun J o tr' hJ => by
      obtain ⟨-, spec, sts, hs, -, hco, -⟩ := hJ
      exact ⟨spec, sts, hs, hco⟩) hex
  exact ⟨tr', hc⟩

/-- **An accepted session's reads file names each port once**, where the verifier checked its statements
(`ChecksAccepted`): the first statement's check ran `checkAcc` (`regs_of_checkStmt`), which refuses a name twice
(`names_of_checkAcc`). -/
theorem names_of_checks {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂}
    (j : Fin (vstarCount F.ish)) (hex : ∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) :
    ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p' := by
  obtain ⟨_, -, -, _, _, hs, hfm⟩ := hca R ω t R₂ j hex
  obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hs
  obtain ⟨_, _, -, -, hacc, -, -⟩ := regs_of_checkStmt (array_forM_ok hfm 0 h0)
  exact names_of_checkAcc hacc

/-- **Check 5 at an accepted session whose unit is a part's**: at its first statement, of the session's circuit, at a
source `src` gives. -/
theorem resCheck_of_accepted {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {src : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.C3}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hco : CoinsAcceptedAt F src c3) (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o)
    (j : Fin (vstarCount F.ish)) {m kr cl i : ℕ} (hu : ResUnit (F.Vs j).c m kr cl i) :
    ∃ (sj : Flock.Stmt) (f : Flock.RecCoins.Source), sj.c = (F.Vs j).c ∧ src t = some f ∧
      Flock.RecCoins.check sj (some (F.zs R ω t R₂ j).own) (some f) (c3 t) = .ok () := by
  obtain ⟨_, _, _, hs, hfm⟩ := hco R ω t R₂ j ⟨_, (h.1 j).1⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hs
  have hk := array_forM_ok hfm 0 h0
  obtain ⟨f, hf⟩ := source_of_check hk (m := m) (k := kr) (cl := cl) (i := i) (by rw [hc0]; exact hu)
  rw [hf] at hk
  exact ⟨_, f, hc0, hf, hk⟩

/-- **A registered read's port is the reads file's one entry of its name, read at the unit's position**
(`ReadsOwnAt`, names once: `names_of_checks`), `vRows_tie`'s and `zero_tie`'s `hone` and `hpos`. -/
theorem readOne_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hro : FlockC3.HPort.ReadsOwnAt F dg N k Rw) {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : FlockC3.PubRows.ChecksAccepted F rows wr reg)
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish)) {u : Fin (F.Vs j).nv}
    {q : Fin (F.Vs j).c.ports.size} (hr : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))) :
    (∀ reg ∈ (F.zs R ω t R₂ j).own, reg.name = ((F.Vs j).c.ports[q.val]!).name →
      reg = (F.zs R ω t R₂ j).rd.port (.inl (u, q))) ∧
    ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).reads[u.val]? =
      some (some ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q)))) := by
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o h j u q hr
  refine ⟨fun reg hreg hn => names_of_checks hchk j ⟨_, (h.1 j).1⟩ reg hreg _ hmem ?_, hpos⟩
  rw [hn, hname]
  exact congrArg (·.name) (getElem!_pos (F.Vs j).c.ports q.val q.isLt)

/-- **`OwnNamesV` at an accepted part session**: its reads file names `v` and `out` (`own_v_out`). -/
theorem ownVOut_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {src : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.C3}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hco : CoinsAcceptedAt F src c3) (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o)
    (j : Fin (vstarCount F.ish)) {m kr cl i : ℕ} (hu : ResUnit (F.Vs j).c m kr cl i) :
    (∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = "v") ∧ ∃ pt ∈ (F.zs R ω t R₂ j).own, pt.name = "out" := by
  obtain ⟨sj, f, hc, -, hck⟩ := resCheck_of_accepted hco h j hu
  rw [← hc] at hu
  obtain ⟨hso, hop, hsp, ho, hres⟩ := hu
  exact own_v_out (c3 := c3) (t := t) hck hso hop hsp ho hres

/-- **`VRowsAre` at an accepted part session, off `Bad`**: where `Bad` holds at `ValueColl` (`hbad`), the registrant's
bit `b` at a `v` read `(u, q)` is `wbit` of the part's elements `vs` of `vOf` at the rep's rounds and comb (`hrep`, the
inner session's as the source loads it), `vRows_tie` at the session's check 5. -/
theorem vRowsOff_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {src : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.C3}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hco : CoinsAcceptedAt F src c3) (hro : FlockC3.HPort.ReadsOwnAt F dg N k Rw)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : FlockC3.PubRows.ChecksAccepted F rows wr reg) {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop}
    (hbad : ∀ R ω t R₂, ValueColl F.zs (valueSrcAll src c3) R ω t R₂ → Bad R ω t R₂)
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (hb : ¬ Bad R ω t R₂) (j : Fin (vstarCount F.ish))
    {m kr cl i : ℕ} (hu : ResUnit (F.Vs j).c m kr cl i) {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size}
    (hname : ((F.Vs j).c.ports[q.val]!).name = "v") (hr : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    {f : Flock.RecCoins.Source} (hf : src t = some f) {inn : Flock.RecCoins.Inner}
    (hl : f.load (F.Vs j).c.unit.name = .ok inn) {sh : Flock.Shape} {extras : Array Flock.Claim}
    (halg : inn.algebra = .ok (sh, extras)) {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hgs : residualGroups sh = .ok gs) {a b' vs : Array ℕ} (hgi : gs[i]? = some (a, b', vs))
    {rounds : Array (Array Flock.F128)} {comb : Array Flock.F128}
    (hrep : ∀ rep, inn.rep sh extras u.val = .ok rep →
      rep.rounds = rounds ∧ (vs.any (· ≥ rep.v.size - 64) = true → inn.comb u.val rep = .ok comb)) :
    ∀ b < 128 * vs.size, ∀ hb : b < FlockSoundness.Discharge.Layout.portRowW (F.Vs j).c q,
      (FlockVBridge.Decode.regVal (F.zs R ω t R₂ j) (.inl (u, q))).2
          ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
        FlockVBridge.wbit (FlockVBridge.Decode.sub vs (FlockVBridge.Decode.vOf sh rounds extras comb)) b := by
  obtain ⟨sj, f', hc, hf', hck⟩ := resCheck_of_accepted hco h j hu
  obtain rfl : f' = f := Option.some.inj (hf'.symm.trans hf)
  obtain ⟨prog, hprog⟩ := prog_of_res hck (m := m) (k := kr) (cl := cl) (i := i) (by rw [hc]; exact hu)
  rw [hc] at hprog
  obtain ⟨hone, hpos⟩ := readOne_of hro hchk h j hr
  obtain ⟨hso, hop, hsp, ho, hres⟩ := hu
  exact vRows_tie (zs := F.zs) (src := src) (c3 := c3) hc hck hf hso hop hsp ho hres hname hprog hr hone hpos
    (FlockC3.HPort.readsUnsegC3 F dg N k Rw R ω t R₂ τ o h j u q hr) hl halg hgs hgi hrep fun hv => hb (hbad R ω t R₂ hv)

/-- **`ZeroReads`' bit at an accepted part session, off `Bad`**: where `Bad` holds at `ValueColl` (`hbad`), an `out`
read `(u, q)` is of one of the two zero rows and every bit the registrant holds there is 0 (`zero_tie`). -/
theorem zeroOff_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {src : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {c3 : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.C3}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hco : CoinsAcceptedAt F src c3) (hro : FlockC3.HPort.ReadsOwnAt F dg N k Rw)
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : FlockC3.PubRows.ChecksAccepted F rows wr reg) {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop}
    (hbad : ∀ R ω t R₂, ValueColl F.zs (valueSrcAll src c3) R ω t R₂ → Bad R ω t R₂)
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (hb : ¬ Bad R ω t R₂) (j : Fin (vstarCount F.ish))
    {m kr cl i : ℕ} (hu : ResUnit (F.Vs j).c m kr cl i) {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size}
    (hname : ((F.Vs j).c.ports[q.val]!).name = "out") (hr : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q))) :
    u.val < 2 ∧ ∀ (b : ℕ) (hb : b < FlockSoundness.Discharge.Layout.portRowW (F.Vs j).c q),
      (FlockVBridge.Decode.regVal (F.zs R ω t R₂ j) (.inl (u, q))).2
          ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ = false := by
  obtain ⟨sj, f, hc, hf, hck⟩ := resCheck_of_accepted hco h j hu
  obtain ⟨prog, hprog⟩ := prog_of_res hck (m := m) (k := kr) (cl := cl) (i := i) (by rw [hc]; exact hu)
  rw [hc] at hprog
  obtain ⟨hone, hpos⟩ := readOne_of hro hchk h j hr
  obtain ⟨hso, hop, hsp, ho, hres⟩ := hu
  obtain ⟨hu2, hor⟩ := zero_tie (zs := F.zs) (src := src) (c3 := c3) hc hck hf hso hop hsp ho hres hname hprog hr
    hone hpos (FlockC3.HPort.readsUnsegC3 F dg N k Rw R ω t R₂ τ o h j u q hr)
  exact ⟨hu2, hor.resolve_right fun hv => hb (hbad R ω t R₂ hv)⟩

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.names_of_checks
#print axioms FlockC3.RegLevels.vRowsOff_of
#print axioms FlockC3.RegLevels.zeroOff_of
