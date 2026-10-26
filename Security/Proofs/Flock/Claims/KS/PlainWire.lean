import Proofs.Flock.Claims.KS.PlainSimW
import Proofs.Flock.Claims.Recursion.PlainK

/-!
# The plain wire, and the lockstep a simulator owes

`plainWire` is the live session's wire at the deployed plain verifier's message formats: the roots as the link message
carries them (`linkOf`, as `plainRoot_of_first` reads them), the proof files as the last message does (`finalOf`), and a
round as tag `2`, its rep, its count in three bytes and its retained bytes (`plainSub`), the count at most a coin's
`65536` field elements. The deployed verifier's round message is tag `0` and its retained bytes only, the rep and the
count being the record's, so a simulator `P'` at this wire sends each of `P`'s round messages' tails as a round of the
rep and count the setup at `x` gives that round. The verifier's replies carry the link points and a round's coins in
any injective code (`finCode`), read back on its image (`finDecode_code`): `P'` reads them back and merges them with
its tape into the deployed coins it feeds `P`.

`PlainSimRun` at any wire is the live outcome's four parts at the view's (`plainSimRun_of_parts`): the roots
(`PlainSimRoot`), the link's points at the first coin (`PlainSimPts`), each rep's rounds, the lockstep through `P`'s
round messages and the coins' used parts (`PlainSimReps`), and the proof files (`PlainSimProofs`). The roots hold at
the plain wire wherever `P'` sends `P`'s first message, since every round after `Commit` keeps them
(`liveOut_rootB`, `plainSimRoot_of_first`). The link's points are the first live coin's at any prover, so they ask
only that the deployed link coin's points be the live coin's (`liveOut_pts`, `plainSimPts_of_coins`). The reps'
rounds and the proof files are the remaining named Props. The abort, the live event
failing at `P'`'s outcome wherever `P`'s run is rejected (`PlainSimAbort`), gives the acceptance `PlainRawClashRun`
asks for (`plain_acc_of_abort`, `plainRawClashRun_of_abort`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.Tables Claims.Bridge.Live

/-- **An injective code of a finite type's values**: as many zero bytes as the value's index. -/
noncomputable def finCode {α : Type} [Fintype α] (a : α) : Claims.Msg := List.replicate (Fintype.equivFin α a).val 0

theorem finCode_inj {α : Type} [Fintype α] : Function.Injective (finCode (α := α)) := fun a b h => by
  have e := congrArg List.length h
  simp only [finCode, List.length_replicate] at e
  exact (Fintype.equivFin α).injective (Fin.ext e)

/-- **The code read back**: the value whose code a message is, where it is one. -/
noncomputable def finDecode {α : Type} [Fintype α] [Nonempty α] (m : Claims.Msg) : α := Function.invFun finCode m

theorem finDecode_code {α : Type} [Fintype α] [Nonempty α] (a : α) : finDecode (finCode a) = a :=
  Function.leftInverse_invFun finCode_inj a

/-- **A round on the plain wire**: tag `2`, the rep, the count in three bytes (low first) and the retained bytes; a rep
past `1` or a count past `65536` is a stop, as is any other message. -/
def plainSub : Claims.Msg → Option (SubmitJ 1)
  | 2 :: r :: n₀ :: n₁ :: n₂ :: rest =>
    if h : r.toNat < 2 ∧ n₀.toNat + 256 * n₁.toNat + 65536 * n₂.toNat ≤ 65536 then
      some { tab := 0, rep := ⟨r.toNat, h.1⟩, bytes := ⟨rest.toArray⟩,
             n := n₀.toNat + 256 * n₁.toNat + 65536 * n₂.toNat }
    else none
  | _ => none

theorem plainSub_le (a : Claims.Msg) (s : SubmitJ 1) (h : plainSub a = some s) : s.n ≤ 65536 := by
  unfold plainSub at h
  split at h
  · split at h
    · next hc =>
      cases h
      exact hc.2
    · cases h
  · cases h

/-- **The plain wire** at `mPts` link coordinates. -/
noncomputable def plainWire (mPts : ℕ) (h : 2 * mPts ≤ 65536) : Wire 65536 1 mPts where
  root a := ((linkOf a).getD default).1.toArray
  sub := plainSub
  sub_le := plainSub_le
  proofs a := ((finalOf a).map Final.proofs).getD #[]
  pts_le := h
  encPts := finCode
  decPts := finDecode
  decPts_enc := finDecode_code
  encRd _ := finCode
  decRd _ := finDecode
  decRd_enc _ := finDecode_code

/-- **The plain wire reads roots as the deployed link does**: `plainRoot_of_first`'s premise on the wire. -/
theorem plainWire_root (mPts : ℕ) (h : 2 * mPts ≤ 65536) (a : Claims.Msg) :
    (plainWire mPts h).root a = ((linkOf a).getD default).1.toArray := rfl

theorem liveOutJ_ext' {J mPts : ℕ} {o o' : LiveOutJ J mPts} (h₁ : o.rootB = o'.rootB) (h₂ : o.pts = o'.pts)
    (h₃ : o.reps = o'.reps) (h₄ : o.proofs = o'.proofs) : o = o' := by
  cases o
  cases o'
  simp only [LiveOutJ.mk.injEq]
  exact ⟨h₁, h₂, h₃, h₄⟩

/-- **`P`'s run at the coins `Φ (u, r')` accepts, deciding** at the setup `s, spec, sts`, the link `link` and its coin
`c₁`, the round messages `mids` and the last message `f`: where `PlainSimRun` asks for the view. -/
def PlainDecides {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {nMax : ℕ} {N : ℕ} {τ : Type}
    (Φ : τ × (Fin (N + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) (u : τ) (r' : Fin (N + 2) → LCoin nMax)
    (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Prop :=
  (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc ∧
    act false x ((Claims.run sha512 P (V false) x ω (Φ (u, r'))).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V false) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
          (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f

/-- **`P`'s run at the coins `Φ (u, r')` accepts, deciding** at the setup `s, spec, sts`, the link `link` and its coin
`c₁`, the round messages `mids` and the last message `f`: where `PlainSimRun` asks for the view. -/
def PlainDecidesK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) {nMax : ℕ} {N : ℕ} {τ : Type}
    (Φ : τ × (Fin (N + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) (u : τ) (r' : Fin (N + 2) → LCoin nMax)
    (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Prop :=
  (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out = .acc ∧
    act false x ((Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).msgs.map Prod.fst).length).map
          (Claims.coinSeq (Φ (u, r')))) = .decide s spec sts link c₁ mids f

/-- **The roots**: where `P`'s run decides, `P'`'s live outcome has the link's roots. -/
def PlainSimRoot {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecides P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').rootB =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).rootB

/-- **The roots**: where `P`'s run decides, `P'`'s live outcome has the link's roots. -/
def PlainSimRootK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecidesK keep P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').rootB =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).rootB

/-- **The link's points**: where `P`'s run decides, `P'`'s live outcome has the points of the link's coin `c₁`. -/
def PlainSimPts {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecides P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').pts =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).pts

/-- **The link's points**: where `P`'s run decides, `P'`'s live outcome has the points of the link's coin `c₁`. -/
def PlainSimPtsK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecidesK keep P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').pts =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).pts

/-- **The lockstep**: where `P`'s run decides, each rep's rounds in `P'`'s live outcome are the record's, round by
round `P`'s round message's retained bytes with the used part of its coin, which the live coin is. -/
def PlainSimReps {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecides P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').reps =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).reps

/-- **The lockstep**: where `P`'s run decides, each rep's rounds in `P'`'s live outcome are the record's, round by
round `P`'s round message's retained bytes with the used part of its coin, which the live coin is. -/
def PlainSimRepsK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecidesK keep P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').reps =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).reps

/-- **The proof files**: where `P`'s run decides, `P'`'s live outcome has the last message's proof files. -/
def PlainSimProofs {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecides P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').proofs =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).proofs

/-- **The proof files**: where `P`'s run decides, `P'`'s live outcome has the last message's proof files. -/
def PlainSimProofsK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin) : Prop :=
  ∀ u r' s spec sts link c₁ mids f, PlainDecidesK keep P x ω Φ u r' s spec sts link c₁ mids f →
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r').proofs =
      (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).proofs

/-- **The simulation from its four parts**: the live outcome is the view where each part is. -/
theorem plainSimRun_of_partsK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts nMax : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire nMax 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin}
    (h₁ : PlainSimRootK keep P x ω mPts N dg w P' Φ) (h₂ : PlainSimPtsK keep P x ω mPts N dg w P' Φ)
    (h₃ : PlainSimRepsK keep P x ω mPts N dg w P' Φ) (h₄ : PlainSimProofsK keep P x ω mPts N dg w P' Φ) :
    PlainSimRunK keep P x ω mPts N dg w P' Φ := fun u r' hacc s spec sts link c₁ mids f hact =>
  liveOutJ_ext' (h₁ u r' s spec sts link c₁ mids f ⟨hacc, hact⟩) (h₂ u r' s spec sts link c₁ mids f ⟨hacc, hact⟩)
    (h₃ u r' s spec sts link c₁ mids f ⟨hacc, hact⟩) (h₄ u r' s spec sts link c₁ mids f ⟨hacc, hact⟩)

/-- `plainSimRun_of_partsK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimRun_of_parts {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts nMax : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire nMax 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin}
    (h₁ : PlainSimRoot P x ω mPts N dg w P' Φ) (h₂ : PlainSimPts P x ω mPts N dg w P' Φ)
    (h₃ : PlainSimReps P x ω mPts N dg w P' Φ) (h₄ : PlainSimProofs P x ω mPts N dg w P' Φ) :
    PlainSimRun P x ω mPts N dg w P' Φ :=
  plainSimRun_of_partsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (nMax := nMax) (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (h₁ := h₁) (h₂ := h₂) (h₃ := h₃) (h₄ := h₄)

/-! ## The roots -/

/-- **Every outcome of a game has `p`.** -/
def AllRet {β : Type} (p : β → Prop) : Game β → Prop
  | .ret b => p b
  | .send _ k => ∀ m, AllRet p (k m)
  | @Game.coin _ _ _ _ k => ∀ c, AllRet p (k c)

theorem playOn_allRet {β Coin : Type} [Fintype Coin] {p : β → Prop} :
    ∀ (g : Game β) (I : Claims.Bridge.Iface Coin g) (s : Claims.Bridge.RStrat Coin g I) (o : Option Coin)
      (r : ℕ → Coin), AllRet p g → p (Claims.Bridge.playOn g I s o r)
  | .ret _, _, _, _, _, h => h
  | .send _ k, I, s, _, _, h => playOn_allRet (k (I.1 s.1)) _ _ _ _ (h _)
  | @Game.coin _ _ _ _ k, I, _, some u, _, h => playOn_allRet (k (I.1.proj u)) _ _ _ _ (h _)
  | @Game.coin _ _ _ _ k, I, _, none, r, h => playOn_allRet (k (I.1.proj (r 0))) _ _ _ _ (h _)

/-- **The rounds keep the roots**: every outcome of the rounds after `Commit` has its roots. -/
theorem allRet_roundsK (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round),
      AllRet (fun o : LiveOutJ J mPts => o.rootB = rootB) (roundsK dg rootB pts N rs)
  | 0, rs => by
    simp only [roundsK, KLJ, AllRet]
    intro _
    trivial
  | N + 1, rs => by
    simp only [roundsK, AllRet]
    intro o
    cases o with
    | none =>
      simp only [branch, KLJ, AllRet]
      intro _
      trivial
    | some s =>
      simp only [branch, AllRet]
      intro _
      exact allRet_roundsK dg rootB pts N _

/-- **After a send, every outcome has the property of the message sent.** -/
theorem playOn_send_allRet {β Coin : Type} [Fintype Coin] {M : Type} {k : M → Game β} {p : M → β → Prop}
    (hk : ∀ m, AllRet (p m) (k m)) (I : Claims.Bridge.Iface Coin (.send M k))
    (s : Claims.Bridge.RStrat Coin (.send M k) I) (o : Option Coin) (r : ℕ → Coin) :
    p (I.1 s.1) (Claims.Bridge.playOn (.send M k) I s o r) :=
  playOn_allRet (k (I.1 s.1)) (I.2 s.1) s.2 _ _ (hk _)

/-- **The live outcome's roots are the wire's reading of the prover's first message.** -/
theorem liveOut_rootB {nMax mPts : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Claims.Msg → Wire nMax 1 mPts) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin (N x + 2) → LCoin nMax) :
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P x ω r).rootB = (w x).root ((P.next x [] ω).eval sha512).1 :=
  playOn_send_allRet (M := Array ByteArray)
    (k := fun rootB => Game.coin (Fin 2 → Fin mPts → FlockLevel3.GF128) fun pts =>
      roundsK dg rootB pts (N x) fun _ _ => [])
    (p := fun v o => o.rootB = v)
    (by
      intro v
      simp only [AllRet]
      intro pts
      exact allRet_roundsK dg v pts (N x) _)
    (sessionIf (w x) dg (N x)) (Claims.Bridge.stratOf sha512 P x ω _ (sessionIf (w x) dg (N x)) [] none) none
    (Claims.coinSeq r)

/-- **The roots, where `P'` sends `P`'s first message**: at the plain wire, the live outcome's roots are the deployed
link's (`liveOut_rootB`, `act_decide_link`). -/
theorem plainSimRoot_of_firstK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    (hm : 2 * mPts ≤ 65536) {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    (h1 : ∀ u, ((P'.next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1) :
    PlainSimRootK keep P x ω mPts N dg (fun _ => plainWire mPts hm) P' Φ := by
  intro u r' s spec sts link c₁ mids f hd
  obtain ⟨hacc, hact⟩ := hd
  rw [liveOut_rootB, h1]
  obtain ⟨rest, hhead⟩ := run_head sha512 P (VK keep) x ω (Φ (u, r')) hacc
  obtain ⟨_, _, hlink⟩ := act_decide_link hact
  rw [hhead] at hlink
  have hl : linkOf ((P.next x [] ω).eval sha512).1 = some link := hlink
  show ((linkOf ((P.next x [] ω).eval sha512).1).getD default).1.toArray = link.1.toArray
  rw [hl]
  rfl

/-- `plainSimRoot_of_firstK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimRoot_of_first {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    (hm : 2 * mPts ≤ 65536) {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    (h1 : ∀ u, ((P'.next x [] (ω, u)).eval sha512).1 = ((P.next x [] ω).eval sha512).1) :
    PlainSimRoot P x ω mPts N dg (fun _ => plainWire mPts hm) P' Φ :=
  plainSimRoot_of_firstK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (hm := hm) (N := N) (dg := dg) (τ := τ) (P' := P') (Φ := Φ) (h1 := h1)

/-! ## The link's points -/

/-- **The rounds keep the link's points.** -/
theorem allRet_roundsK_pts (dg : ByteArray → ByteArray) {J mPts : ℕ} (rootB : Array ByteArray)
    (pts : Fin 2 → Fin mPts → FlockLevel3.GF128) :
    ∀ (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round),
      AllRet (fun o : LiveOutJ J mPts => o.pts = pts) (roundsK dg rootB pts N rs)
  | 0, rs => by
    simp only [roundsK, KLJ, AllRet]
    intro _
    trivial
  | N + 1, rs => by
    simp only [roundsK, AllRet]
    intro o
    cases o with
    | none =>
      simp only [branch, KLJ, AllRet]
      intro _
      trivial
    | some s =>
      simp only [branch, AllRet]
      intro _
      exact allRet_roundsK_pts dg rootB pts N _

/-- **At a coin read off a pending coin, every outcome has the property of the value read.** -/
theorem playOn_coin_allRet {β Coin : Type} [Fintype Coin] {C : Type} [Fintype C] [Nonempty C] {k : C → Game β}
    {p : C → β → Prop} (hk : ∀ c, AllRet (p c) (k c)) (I : Claims.Bridge.Iface Coin (.coin C k))
    (s : Claims.Bridge.RStrat Coin (.coin C k) I) (u : Coin) (r : ℕ → Coin) :
    p (I.1.proj u) (Claims.Bridge.playOn (.coin C k) I s (some u) r) :=
  playOn_allRet (k (I.1.proj u)) _ _ _ _ (hk _)

/-- **The live outcome's link points are the first live coin's**, at any prover. -/
theorem liveOut_pts {nMax mPts : ℕ} (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray)
    (w : Claims.Msg → Wire nMax 1 mPts) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin (N x + 2) → LCoin nMax) :
    (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P x ω r).pts = (ptsIf (w x)).proj (Claims.coinSeq r 0) :=
  playOn_coin_allRet (C := Fin 2 → Fin mPts → FlockLevel3.GF128)
    (k := fun pts => roundsK dg ((w x).root ((P.next x [] ω).eval sha512).1) pts (N x) fun _ _ => [])
    (p := fun c o => o.pts = c) (fun c => allRet_roundsK_pts dg _ c (N x) _)
    ((sessionIf (w x) dg (N x)).2 ((P.next x [] ω).eval sha512).1)
    (Claims.Bridge.stratOf sha512 P x ω _ (sessionIf (w x) dg (N x)) [] none).2 (Claims.coinSeq r 0)
    fun i => Claims.coinSeq r (i + 1)

/-- **The link's points from the coins alone**: where `P`'s run decides, the first live coin's points are the
deployed link coin's, whatever `P'` sends (`liveOut_pts`). -/
theorem plainSimPts_of_coinsK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts nMax : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire nMax 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin}
    (h : ∀ u r' s spec sts link c₁ mids f, PlainDecidesK keep P x ω Φ u r' s spec sts link c₁ mids f →
      (ptsIf (w x)).proj (Claims.coinSeq r' 0) = (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).pts) :
    PlainSimPtsK keep P x ω mPts N dg w P' Φ := fun u r' s spec sts link c₁ mids f hd => by
  rw [liveOut_pts]
  exact h u r' s spec sts link c₁ mids f hd

/-- `plainSimPts_of_coinsK` at the deployed plain verifier (`keep := id`). -/
theorem plainSimPts_of_coins {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts nMax : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire nMax 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin}
    (h : ∀ u r' s spec sts link c₁ mids f, PlainDecides P x ω Φ u r' s spec sts link c₁ mids f →
      (ptsIf (w x)).proj (Claims.coinSeq r' 0) = (Claims.Bridge.Custody.viewOf 1 mPts spec link c₁ f).pts) :
    PlainSimPts P x ω mPts N dg w P' Φ :=
  plainSimPts_of_coinsK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (nMax := nMax) (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (h := h)

/-! ## The abort -/

/-- **`P'` aborts where `P`'s run is rejected**: at every tape and live coins at which `V false` does not accept `P` at
`Φ`, the live event `pL` fails at `P'`'s outcome. -/
def PlainSimAbort {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin)
    (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out ≠ .acc →
    ¬ pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r')

/-- **`P'` aborts where `P`'s run is rejected**: at every tape and live coins at which `VK keep` does not accept `P` at
`Φ`, the live event `pL` fails at `P'`'s outcome. -/
def PlainSimAbortK (keep : Bool → Bool) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) {nMax : ℕ}
    (N : Claims.Msg → ℕ) (dg : ByteArray → ByteArray) (w : Claims.Msg → Wire nMax 1 mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin)
    (pL : LiveOutJ 1 mPts → Prop) : Prop :=
  ∀ u r', (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out ≠ .acc →
    ¬ pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r')

/-- **The live event at an aborting simulator gives an accepting run.** -/
theorem plain_acc_of_abortK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts nMax : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire nMax 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainSimAbortK keep P x ω mPts N dg w P' Φ pL) {u : τ}
    {r' : Fin (N x + 2) → LCoin nMax} (hp : pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r')) :
    (Claims.run sha512 P (VK keep) x ω (Φ (u, r'))).out = .acc := by
  by_contra hn
  exact h u r' hn hp

/-- `plain_acc_of_abortK` at the deployed plain verifier (`keep := id`). -/
theorem plain_acc_of_abort {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts nMax : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire nMax 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin nMax) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (h : PlainSimAbort P x ω mPts N dg w P' Φ pL) {u : τ}
    {r' : Fin (N x + 2) → LCoin nMax} (hp : pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r')) :
    (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc :=
  plain_acc_of_abortK id (Ω := Ω) (P := P) (x := x) (ω := ω) (mPts := mPts) (nMax := nMax) (N := N) (dg := dg) (w := w) (τ := τ) (P' := P') (Φ := Φ) (pL := pL) (h := h) (u := u) (r' := r') (hp := hp)

/-- **The raw clash from the abort and the accepting runs' clash**: `PlainRawClashRun` where `P'` aborts on rejected
runs and, at the accepting ones, the live event gives two different columns at one position or a collision. -/
theorem plainRawClashRun_of_abort {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : Claims.Msg → Wire 65536 1 mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V false).k x) → Coin}
    {pL : LiveOutJ 1 mPts → Prop} (habort : PlainSimAbort P x ω mPts N dg w P' Φ pL)
    (hclash : ∀ u r', pL (liveOut (fun _ => 1) (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
      (Claims.run sha512 P (V false) x ω (Φ (u, r'))).out = .acc →
      (∃ (q : ℕ) (a b : List FlockLevel3.GF128),
        (q, a) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧
        (q, b) ∈ plainRawCols (Claims.run sha512 P (V false) x ω (Φ (u, r'))).seen ∧ a ≠ b) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V false) x ω (Φ (u, r'))).vLog) :
    PlainRawClashRun P x ω mPts N dg w P' Φ pL := fun u r' hp =>
  ⟨plain_acc_of_abort habort hp, hclash u r' hp (plain_acc_of_abort habort hp)⟩

end FlockDeploy.KS
