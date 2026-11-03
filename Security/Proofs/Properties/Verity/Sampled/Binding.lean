import Proofs.Properties.Verity.Sampled
import Proofs.Properties.Challenger
import Proofs.Core.Game.Tape
import Proofs.Properties.Challenger.Break

/-!
# The log the sampled check read is the device's, or a break: the read finder and the forger

The read finder's and the forger's terms of `Verity.Sampled.CouplingTarget`. On a play of the period's game whose
sampled verdict accepts, the device's log as v2's check read it is the play's, or the read finder outputs a SHA-512
collision, or the forger outputs a forgery of the device's LMS signature.

* **The log a check read** (`Read`, `Logged`): each release's frame and the bytes it answered, and its coins where the
  check's evidence shows them. v2's evidence shows the draw's segment's coins and each coin tree's root and key, and of
  a tree opening only its coins' SHA-512. `Logged reads` is `Vocab.Logged` at reads that show every release's coins
  (`vocab_logged_iff`), so it binds as much wherever the evidence shows the coins, and it can hold on an honest play
  where the evidence doesn't. v2's read (`readsV2`) is `Verity.Sampled.coinsRead`'s, each release's coins where v2
  read any, and `Vocab.Logged` at it gives `Logged readsV2` (`logged_v2_of_vocab`); the reduction takes any `reads`.
* **v2's records in what the prover sent** (`records`): the components' parameters and cases the composition decodes
  that decode as sampled proofs' one-stage form, v2's, under parameters naming a challenger and a window, each with what
  its coins are read under (`Properties.Challenger.oneStageZkSource`) and the record its evidence carries. The
  parameters' challenger key (`keyOf`) is that of their first component of sampled proofs' id that names one.
* **An accepted v2 case is one of them** (`accepted_case`): a composite names each component's id once (`ids_nodup`), so
  on parameters naming the one-stage form, the case the sampled component accepts is v2's (`caseOf`), its record is one
  of `records` under `keyOf`'s key, and `drawCoins` reads from it the coins v2's check read.
* **The constructions** at a strategy, on a uniform tape of its play (`Verity.Game.Tape`), at the device's history at
  each play (`dev`). The read finder (`readFinder`) outputs the first collision `findBreak` finds in a record, and the
  forger (`forger`) the first forgery `findBreak` finds in a record under the parameters' key. Each outputs only what
  `findBreak` outputs (`findBreak_valid`): two different strings with one digest, or a signature that verifies over a
  message the device didn't sign (`Properties.Challenger.Signed`). The game releases no device signature, so the forger
  saw none (`signed := []`). Each runs the play once and `findBreak` once per record: its SHA-512 calls are the
  strategy's, one per string `findBreak` hashes and `lmsCalls` per signature it verifies (`breakCalls`).
  `withBinding` puts them in a `Constructions`.
* **The reduction** (`logged_le`). Suppose every accepted run that isn't `Logged` carries a break under the
  parameters' key in one of its records (`Binds`). Then an accepted play that isn't `Logged` is at most as likely as
  the read finder's collision plus the forger's forgery, at a key at the deployed LMS parameters (`lmsVerify`'s).
-/

namespace Verity.Sampled.Binding

open Lean (Json)
open Properties.Challenger SampledProofs.Challenger
open scoped ENNReal

local instance lawfulBEqByteArrayBinding : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## The log a check read -/

/-- **One release as a check read it**: its frame, the bytes it answered, and its coins where the check's evidence
shows them (`none` where it shows only their digest, or nothing of them). -/
structure Read where
  frame : Frame
  answers : ByteArray
  coins : Option (List Bool)

/-- The read is of the release `x`: its frame and the bytes it answered, and its coins where it shows them. -/
def Read.Of (r : Read) (x : Release) : Prop :=
  r.frame = x.frame ∧ r.answers = x.answers ∧ ∀ cs ∈ r.coins, cs = x.coins

/-- The read that shows every coin of the release `x`. -/
def Read.ofRelease (x : Release) : Read := ⟨x.frame, x.answers, some x.coins⟩

/-- **The log the check read is the device's**: each log `reads` decodes from what the prover sent reads the device's
releases, every one in order and no other. -/
def Logged (reads : Sent → Option (List Read)) (o : Exchange) : Prop :=
  ∀ log ∈ reads o.sent, List.Forall₂ Read.Of log o.log

theorem ofRelease_of_iff (x y : Release) : (Read.ofRelease x).Of y ↔ x = y := by
  cases x; cases y
  simp [Read.Of, Read.ofRelease, Option.mem_def]

theorem forall₂_ofRelease_iff : ∀ l l' : List Release, List.Forall₂ Read.Of (l.map Read.ofRelease) l' ↔ l = l'
  | [], [] => ⟨fun _ => rfl, fun _ => .nil⟩
  | [], _ :: _ => ⟨nofun, nofun⟩
  | _ :: _, [] => ⟨nofun, nofun⟩
  | x :: xs, y :: ys => by
    constructor
    · intro h
      cases h with
      | cons hxy hs => rw [(ofRelease_of_iff x y).1 hxy, (forall₂_ofRelease_iff xs ys).1 hs]
    · intro h
      cases h
      exact .cons ((ofRelease_of_iff _ _).2 rfl) ((forall₂_ofRelease_iff _ _).2 rfl)

/-- **`Vocab.Logged` is `Logged` at reads that show every coin.** -/
theorem vocab_logged_iff {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α)
    (a : Subprotocol) (o : Exchange) :
    V.Logged a o ↔ Logged (fun s => (V.coinsRead a s).map (·.map Read.ofRelease)) o := by
  unfold Vocab.Logged Logged
  constructor
  · intro h log hlog
    obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.1 (Option.mem_def.1 hlog)
    exact (forall₂_ofRelease_iff l o.log).2 (h l (Option.mem_def.2 hl))
  · intro h l hl
    exact (forall₂_ofRelease_iff l o.log).1 (h _ (Option.mem_map_of_mem _ hl))

/-- **A release as v2's check reads it** (`Verity.Sampled.coinsRead`): its frame, the bytes it answered, and its coins
where it read them, a release whose coins it read none of having `[]`. -/
def Read.ofV2 (x : Release) : Read := ⟨x.frame, x.answers, if x.coins.isEmpty then none else some x.coins⟩

/-- **The log v2's check read** from what the prover sent (`Verity.Sampled.coinsRead`). -/
def readsV2 (s : Sent) : Option (List Read) := (Verity.Sampled.coinsRead s).map (·.map Read.ofV2)

theorem ofV2_of (x : Release) : (Read.ofV2 x).Of x := by
  refine ⟨rfl, rfl, fun cs hcs => ?_⟩
  change cs ∈ (if x.coins.isEmpty then none else some x.coins) at hcs
  split at hcs
  · cases Option.mem_def.1 hcs
  · exact (Option.some.inj (Option.mem_def.1 hcs)).symm

theorem forall₂_ofV2 : ∀ l : List Release, List.Forall₂ Read.Of (l.map Read.ofV2) l
  | [] => .nil
  | x :: l => .cons (ofV2_of x) (forall₂_ofV2 l)

/-- **`Vocab.Logged` at v2's read gives `Logged readsV2`**: where the vocabulary's log at `.sampled` is v2's
(`Deployed.coinsRead`), a run whose whole log the check read is the device's has the log v2 read the device's. -/
theorem logged_v2_of_vocab {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] {V : Vocab d α}
    (hV : ∀ s, V.coinsRead .sampled s = Verity.Sampled.coinsRead s) {o : Exchange} (h : V.Logged .sampled o) :
    Logged readsV2 o := by
  intro log hlog
  obtain ⟨l, hl, rfl⟩ := Option.map_eq_some_iff.1 (Option.mem_def.1 hlog)
  obtain rfl : l = o.log := h l (by rw [hV]; exact Option.mem_def.2 hl)
  exact forall₂_ofV2 _

/-! ## v2's records in what the prover sent -/

/-- **A component's parameters and case as a v2 record**: where they decode as sampled proofs' one-stage form, v2's,
under parameters naming a challenger and a window, what its coins are read under (`oneStageZkSource`) and the record its
evidence carries. -/
def record? (n : Core.Composition.Named) (x : Core.Composition.Case) : Option (Draws × Json) :=
  match Core.canonicalJson n.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson,
      SampledProofs.OneStage.commitmentOf x.commitment, VStar.Recursive.Sampled.evidenceOf x.evidence with
  | .ok (.inl p), .ok c, .ok (.inl e) => (oneStageZkSource p c).map fun d => (d, e.base.coins)
  | _, _, _ => none

/-- **v2's records in what the prover sent**: the components' parameters and cases the composition decodes, each read
as v2's (`record?`). -/
def records (s : Sent) : List (Draws × Json) :=
  match Core.canonicalJson s.params >>= Core.Composition.decodeParams, Core.Composition.decodeEvidence s.evidence with
  | .ok ps, .ok ev => (ps.zip ev).filterMap fun x => record? x.1 x.2
  | _, _ => []

/-- A component's challenger key: where it is sampled proofs' and its parameters name the one-stage form and a
challenger, that challenger's key. -/
def keyAt (n : Core.Composition.Named) : Option ChallengerKey :=
  if n.id = VStar.Recursive.Sampled.PROTOCOL then
    match Core.canonicalJson n.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson with
    | .ok (.inl p) => p.base.challenger.map (·.1)
    | _ => none
  else none

/-- **The parameters' challenger key**: that of their first component of sampled proofs' id whose parameters name the
one-stage form and a challenger (`keyAt`). -/
def keyOf (θ : ByteArray) : Option ChallengerKey :=
  match Core.canonicalJson θ >>= Core.Composition.decodeParams with
  | .ok ps => ps.findSome? keyAt
  | .error _ => none

/-! ## An accepted v2 case is one of the records, under the parameters' key -/

/-- In a composition whose parameters and evidence name its components' ids in order, each case's parameters and
evidence name its component's id. -/
theorem triples_ids : ∀ {cs : List _root_.Core.Protocol} {ps : List Core.Composition.Named}
    {ev : List Core.Composition.Case}, ps.map (·.id) = cs.map (·.id) → ev.map (·.id) = cs.map (·.id) →
    ∀ x ∈ Properties.Composition.triples cs ps ev, x.2.1.id = x.1.id ∧ x.2.2.id = x.1.id
  | [], _, _, _, _, x, hx => by simp [Properties.Composition.triples] at hx
  | _ :: _, [], _, h, _, _, _ => by simp at h
  | _ :: _, _ :: _, [], _, h, _, _ => by simp at h
  | c :: cs, p :: ps, e :: ev, hp, he, x, hx => by
    simp only [List.map_cons, List.cons.injEq] at hp he
    simp only [Properties.Composition.triples, List.zip_cons_cons, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ⟨hp.1, he.1⟩
    · exact triples_ids (cs := cs) (ps := ps) (ev := ev) hp.2 he.2 x hx

/-- Each composite names each of its components' ids once. -/
theorem ids_nodup (k : Verity.Registration.Composite) : ((componentsOf k).map (·.id)).Nodup := by
  cases k <;> decide

/-- `findSome?` at the one element of a list where the function is defined. -/
theorem findSome_eq {β γ : Type} {f : β → Option γ} :
    ∀ {l : List β} {a : β} {b : γ}, a ∈ l → f a = some b → (∀ a' ∈ l, (f a').isSome → a' = a) →
      l.findSome? f = some b
  | [], _, _, ha, _, _ => by cases ha
  | a' :: l, a, b, ha, hf, hu => by
    cases hfa : f a' with
    | some b' =>
      obtain rfl : a' = a := hu a' (List.mem_cons.2 (.inl rfl)) (by simp [hfa])
      rw [List.findSome?_cons_of_isSome (by simp [hfa])]
      exact hf
    | none =>
      rw [List.findSome?_cons_of_isNone (by simp [hfa])]
      refine findSome_eq ?_ hf fun a'' h => hu a'' (List.mem_cons_of_mem _ h)
      rcases List.mem_cons.1 ha with rfl | h
      · rw [hf] at hfa
        cases hfa
      · exact h

/-- `find?` at the one element of a list that satisfies the test. -/
theorem find_eq {β : Type} {q : β → Bool} : ∀ {l : List β} {a : β}, a ∈ l → q a = true →
    (∀ a' ∈ l, q a' = true → a' = a) → l.find? q = some a
  | [], _, ha, _, _ => by cases ha
  | a' :: l, a, ha, hq, hu => by
    by_cases hq' : q a' = true
    · rw [List.find?_cons_of_pos hq', hu a' (List.mem_cons.2 (.inl rfl)) hq']
    · rw [List.find?_cons_of_neg hq']
      refine find_eq ?_ hq fun a'' h => hu a'' (List.mem_cons_of_mem _ h)
      rcases List.mem_cons.1 ha with rfl | h
      · exact absurd hq hq'
      · exact h

/-- The challenger's key is the key of the draws v2's coins are read under. -/
theorem key_of_source {p : SampledProofs.OneStageZk.Params} {c : SampledProofs.OneStage.Commitment} {d : Draws}
    (h : oneStageZkSource p c = some d) : p.base.challenger.map (·.1) = some d.key := by
  unfold oneStageZkSource oneStageSource at h
  rcases hch : p.base.challenger with _ | ⟨key, label⟩ <;> rcases hw : p.base.verifier.windowRoot with _ | w <;>
    simp only [hch, hw] at h <;> cases h
  simp

/-- `Except.ok`'s bind is the continuation at its value. -/
theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- **An accepted v2 case is one of the records, under the parameters' key** (P1): at a composite that reads sampled
proofs at v2 (`ReadsV2`), on parameters that name its one-stage form (`ParamsAt`), a run accepted as a `same`
composition whose sampled component accepts its own case has that case as v2's (`caseOf`), read under a challenger
and a window its parameters name (`oneStageZkSource`); its record is one of the run's v2 records (`records`) under
the parameters' challenger key (`keyOf`), and v2's coins are what `drawCoins` reads from it. -/
theorem accepted_case {k : Verity.Registration.Composite} (hk : ReadsV2 k) {s : Sent}
    {p : SampledProofs.OneStageZk.Params} (hp : ParamsAt s.params p) (hs : SameAccepts k s)
    (ha : ComponentAccepts k .sampled s) :
    ∃ c e d ξ, caseOf s = .ok (p, c, e) ∧ oneStageZkSource p c = some d ∧ (d, e.base.coins) ∈ records s ∧
      keyOf s.params = some d.key ∧ SampledProofs.OneStageZk.coinsOf p c e = .ok ξ ∧
      drawCoins d.key d.window d.label d.after e.base.coins = .ok ξ := by
  obtain ⟨P, hP, ps, b, ev, ⟨hps, -, hev⟩, x, hx, hx1, hacc⟩ := ha
  obtain rfl : P = VStar.Recursive.Sampled.protocol := Option.some.inj (hP.symm.trans hk)
  obtain ⟨ps', b', ev', ⟨hps', -, hev'⟩, -, -, ⟨⟨⟨hpi, hei, -, -⟩, -, -⟩, -⟩, -⟩ := hs
  have e1 : ps' = ps := Except.ok.inj (hps'.symm.trans hps)
  have e2 : ev' = ev := Except.ok.inj (hev'.symm.trans hev)
  subst ps' ev'
  obtain ⟨ps₀, hps₀, n, hn, hnid, hnp⟩ := hp
  have e3 : ps₀ = ps := Except.ok.inj (hps₀.symm.trans hps)
  subst ps₀
  have hcs : (s.components k).map (·.id) = (componentsOf k).map (·.id) := by
    unfold Sent.components
    rw [List.map_map]
    rfl
  have hnd : (ps.map (·.id)).Nodup := by rw [hpi, hcs]; exact ids_nodup k
  have hnde : (ev.map (·.id)).Nodup := by rw [hei, hcs]; exact ids_nodup k
  obtain ⟨hid1, hid2⟩ := triples_ids hpi hei x hx
  rw [hx1] at hid1 hid2 hacc
  have hid1' : x.2.1.id = VStar.Recursive.Sampled.PROTOCOL := hid1
  have hid2' : x.2.2.id = VStar.Recursive.Sampled.PROTOCOL := hid2
  have hmem := List.of_mem_zip (a := x.1) (b := x.2) hx
  have hmem2 := List.of_mem_zip (a := x.2.1) (b := x.2.2) hmem.2
  obtain rfl : n = x.2.1 := List.inj_on_of_nodup_map hnd hn hmem2.1 (hnid.trans hid1'.symm)
  obtain ⟨p', c, e', hp', hc, he, pub, prof, hch⟩ := hacc
  obtain rfl : p' = Sum.inl p := Except.ok.inj (hp'.symm.trans hnp)
  rcases e' with e | e
  swap
  · cases hch
  have hc' : SampledProofs.OneStage.commitmentOf x.2.2.commitment = .ok c := hc
  have he' : VStar.Recursive.Sampled.evidenceOf x.2.2.evidence = .ok (Sum.inl e) := he
  have hch' : SampledProofs.OneStageZk.checkBlobs p c e (_root_.Core.Blobs.ofArray s.blobs) = .accept pub prof := hch
  cases hξ : SampledProofs.OneStageZk.coinsOf p c e with
  | error r =>
    unfold SampledProofs.OneStageZk.checkBlobs at hch'
    rw [hξ] at hch'
    cases hch'
  | ok ξ =>
    obtain ⟨d, hd⟩ := oneStageZk_source (p := p) (c := c) (e := e) hξ
    obtain ⟨raw, hraw, hrx⟩ := oneStageZk_reads p c d hd e ξ hξ
    subst raw
    have hrec : record? x.2.1 x.2.2 = some (d, e.base.coins) := by
      simp only [record?, hnp, hc', he', hd, Option.map_some]
    have hkeyAt : keyAt x.2.1 = some d.key := by
      unfold keyAt
      rw [ite_eq_left hid1', hnp]
      exact key_of_source hd
    have hf1 : ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.1 :=
      find_eq hmem2.1 (beq_iff_eq.2 hid1') fun a' ha' h =>
        List.inj_on_of_nodup_map hnd ha' hmem2.1 ((beq_iff_eq.1 h).trans hid1'.symm)
    have hf2 : ev.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x.2.2 :=
      find_eq hmem2.2 (beq_iff_eq.2 hid2') fun a' ha' h =>
        List.inj_on_of_nodup_map hnde ha' hmem2.2 ((beq_iff_eq.1 h).trans hid2'.symm)
    refine ⟨c, e, d, ξ, ?_, hd, ?_, ?_, hξ, hraw⟩
    · simp only [caseOf, hps, ok_bind, hf1, hev, hf2, hnp, hc', he']
      rfl
    · unfold records
      rw [hps, hev]
      exact List.mem_filterMap.2 ⟨x.2, hmem.2, hrec⟩
    · unfold keyOf
      rw [hps]
      refine findSome_eq hmem2.1 hkeyAt fun a' ha' hsome => ?_
      by_cases hid : a'.id = VStar.Recursive.Sampled.PROTOCOL
      · exact List.inj_on_of_nodup_map hnd ha' hmem2.1 (hid.trans hid1'.symm)
      · unfold keyAt at hsome
        rw [ite_eq_right hid] at hsome
        cases hsome

/-! ## The read finder and the forger -/

/-- **One LMS verification's SHA-512 calls at the parameters `q`**, at most: the message's digest, each of the `P`
chains' `2^W − 1` steps, the one-time key, the leaf and the `H` path nodes (`Lms.rootOf`). -/
def lmsCalls (q : Lms.Params) : ℕ := q.P * (2 ^ q.W - 1) + q.H + 3

/-- **`findBreak`'s SHA-512 calls on the records of `s`** at the device's history `dev`: on each, one per string it
hashes (those the reader hashes and those the device hashes under the record's key) and `lmsCalls` per signature it
verifies. -/
def breakCalls (dev : History) (s : Sent) : ℕ :=
  ((records s).map fun x => (signatures x.1.key x.2).length * lmsCalls x.1.key.params +
    (readHashed x.1.key x.1.label x.2).length + (deviceHashed dev x.1.key.pub).length).sum

/-- A finding's collision, as two messages. -/
def collisionOf : Finding → Option (Claims.Msg × Claims.Msg)
  | .collision x y => some (x.data.toList, y.data.toList)
  | .forgery _ _ => none

/-- A finding's forgery, as a message and a signature's bytes. -/
def forgeryOf : Finding → Option (Hash.Msg × Hash.Msg)
  | .forgery m sig => some (m.data.toList, sig.data.toList)
  | .collision _ _ => none

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The read finder at a strategy**, at the device's history at each play `dev`: on a uniform tape of the play, the
first collision `findBreak` finds in one of the play's records. -/
noncomputable def readFinder (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Claims.Finder where
  S := Verity.Game.Tape (V.game θ F c) σ
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    ((records o.sent).filterMap fun x => findBreak (dev o) x.1.key x.1.label x.2).findSome? collisionOf
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent

open Classical in
/-- **The forger at a strategy**, against the parameters' challenger key (`keyOf`), at the device's history at each
play `dev`: on a uniform tape of the play, the first forgery `findBreak` finds in one of the play's records under that
key. The game releases no device signature, so it saw none. -/
noncomputable def forger (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) : Hash.Assumptions.LmsForger where
  S := Verity.Game.Tape (V.game θ F c) σ
  key := ((keyOf θ).map fun k => k.pub.data.toList).getD []
  signed := []
  out t :=
    let o := Verity.Game.Tape.out _ σ t
    ((records o.sent).filterMap fun x =>
      if keyOf θ = some x.1.key then findBreak (dev o) x.1.key x.1.label x.2 else none).findSome? forgeryOf
  cost t :=
    let o := Verity.Game.Tape.out _ σ t
    V.cost θ F c σ + breakCalls (dev o) o.sent

/-- **The leg's constructions with this read finder and forger**, at the device's history at each play `dev`: `K`'s
projection and two-continuation finders, and `readFinder` and `forger`. -/
noncomputable def withBinding {V : Vocab d α} (K : Constructions V) (dev : Exchange → History) : Constructions V :=
  { K with readFinder := readFinder V dev, forger := forger V dev }

/-! ## What the finders output -/

theorem bytes_ofList (b : ByteArray) : ({ data := b.data.toList.toArray } : ByteArray) = b := by
  cases b
  simp

/-- A collision of the compiled SHA-512 is one of `sha512Msg`. -/
theorem collision_of_valid {x y : ByteArray} (h : x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) :
    Claims.Collision Verity.sha512Msg x.data.toList y.data.toList := by
  refine ⟨fun e => h.1 (ByteArray.ext (Array.ext' e)), ?_⟩
  unfold Verity.sha512Msg
  rw [bytes_ofList, bytes_ofList, h.2]

/-- **A signature that verifies under a key at the deployed LMS parameters verifies under `lmsVerify`.** -/
theorem lmsVerify_of_verifies {key : ChallengerKey} (hk : key.params = Lms.deployed) {m sig : ByteArray}
    (h : LmsVerifies key m sig) : Verity.lmsVerify key.pub.data.toList m.data.toList sig.data.toList = true := by
  unfold LmsVerifies lmsVerifies at h
  unfold Verity.lmsVerify
  rw [hk] at h
  rw [bytes_ofList, bytes_ofList, bytes_ofList]
  cases hd : Lms.Signature.decode Lms.deployed sig with
  | error e => simp [hd] at h
  | ok s =>
    cases hv : Lms.verify Lms.deployed key.pub m s with
    | error e => simp [hd, hv] at h
    | ok b =>
      simp only [hd, hv] at h
      simp [hv, bind, Except.bind, h]

/-- The first collision among findings that are each valid where they are collisions is a collision of
`sha512Msg`, and there is one where a finding is a collision. -/
theorem findSome_collisionOf {l : List Finding}
    (hl : ∀ x y, Finding.collision x y ∈ l → x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) {x y : ByteArray}
    (hxy : Finding.collision x y ∈ l) :
    ∃ u v, l.findSome? collisionOf = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v := by
  cases hf : l.findSome? collisionOf with
  | none => exact absurd (List.findSome?_eq_none_iff.1 hf _ hxy) (by simp [collisionOf])
  | some uv =>
    obtain ⟨w, hw, hc⟩ := List.exists_of_findSome?_eq_some hf
    cases w with
    | forgery _ _ => simp [collisionOf] at hc
    | collision a b =>
      simp only [collisionOf, Option.some.injEq] at hc
      subst hc
      exact ⟨_, _, rfl, collision_of_valid (hl a b hw)⟩

/-- The first forgery among findings that each verify under `key` where they are forgeries verifies under
`lmsVerify`, at a key at the deployed LMS parameters, and there is one where a finding is a forgery. -/
theorem findSome_forgeryOf {key : ChallengerKey} (hk : key.params = Lms.deployed) {l : List Finding}
    (hl : ∀ m sig, Finding.forgery m sig ∈ l → LmsVerifies key m sig) {m sig : ByteArray}
    (hms : Finding.forgery m sig ∈ l) :
    ∃ u v, l.findSome? forgeryOf = some (u, v) ∧ Verity.lmsVerify key.pub.data.toList u v = true := by
  cases hf : l.findSome? forgeryOf with
  | none => exact absurd (List.findSome?_eq_none_iff.1 hf _ hms) (by simp [forgeryOf])
  | some uv =>
    obtain ⟨w, hw, hc⟩ := List.exists_of_findSome?_eq_some hf
    cases w with
    | collision _ _ => simp [forgeryOf] at hc
    | forgery a b =>
      simp only [forgeryOf, Option.some.injEq] at hc
      subst hc
      exact ⟨_, _, rfl, lmsVerify_of_verifies hk (hl a b hw)⟩

/-! ## The reduction -/

/-- A run of the play after `θ`, `F` and `c` ends with them. -/
theorem play_reaches {fs : List (String × ℕ)} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {o : Exchange}
    (h : Verity.Game.Reaches (play fs θ F c) o) : o.params = θ ∧ o.files = F ∧ o.commitment = c := by
  obtain ⟨log, -, e, he⟩ := Verity.Game.reaches_bind h
  cases he
  exact ⟨rfl, rfl, rfl⟩

/-- **The binding at the parameters `θ`, pointwise**: on every run of the period's game after `θ` whose sampled
verdict at `k` accepts and whose log, as `reads` reads it, isn't the device's, one of its v2 records under the
parameters' key carries a break at the device's history at the run (`Properties.Challenger.Break`). -/
def Binds (V : Vocab d α) (k : Verity.Registration.Composite) (reads : Sent → Option (List Read))
    (dev : Exchange → History) (θ : ByteArray) : Prop :=
  ∀ o : Exchange, o.params = θ → V.Verdict k .sampled o → ¬ Logged reads o →
    ∃ x ∈ records o.sent, keyOf θ = some x.1.key ∧ Break (dev o) x.1.key x.1.label x.2

/-- **A run with a break under the parameters' key in one of its records is one where the read finder or the
forger succeeds**, at a key at the deployed LMS parameters. -/
theorem finders_of_break (V : Vocab d α) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hkey : ∀ key ∈ keyOf θ, key.params = Lms.deployed)
    (t : Verity.Game.Tape (V.game θ F c) σ) {x : Draws × Json}
    (hx : x ∈ records (Verity.Game.Tape.out _ σ t).sent) (hk : keyOf θ = some x.1.key)
    (hb : Break (dev (Verity.Game.Tape.out _ σ t)) x.1.key x.1.label x.2) :
    (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
      (forger V dev θ F c σ).Forges Verity.lmsVerify t := by
  classical
  obtain ⟨w, hw, -⟩ := hb
  cases w with
  | collision a b =>
    left
    refine findSome_collisionOf (fun a' b' h => ?_) (List.mem_filterMap.2 ⟨x, hx, hw⟩)
    obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 h
    exact findBreak_valid hx'
  | forgery m sig =>
    right
    have hd : x.1.key.params = Lms.deployed := hkey _ hk
    let o := Verity.Game.Tape.out _ σ t
    let l := (records o.sent).filterMap fun x' =>
      if keyOf θ = some x'.1.key then findBreak (dev o) x'.1.key x'.1.label x'.2 else none
    have hfx : (if keyOf θ = some x.1.key then findBreak (dev o) x.1.key x.1.label x.2 else none) =
        some (Finding.forgery m sig) := by
      rw [ite_eq_left hk]
      exact hw
    have hmem : Finding.forgery m sig ∈ l := List.mem_filterMap.2 ⟨x, hx, hfx⟩
    have hl : ∀ m' sig', Finding.forgery m' sig' ∈ l → LmsVerifies x.1.key m' sig' := by
      intro m' sig' h
      obtain ⟨x', -, hx'⟩ := List.mem_filterMap.1 h
      have hx'' : (if keyOf θ = some x'.1.key then findBreak (dev o) x'.1.key x'.1.label x'.2 else none) =
          some (Finding.forgery m' sig') := hx'
      by_cases hk' : keyOf θ = some x'.1.key
      · rw [ite_eq_left hk'] at hx''
        have e : x'.1.key = x.1.key := Option.some.inj (hk'.symm.trans hk)
        have hv : LmsVerifies x'.1.key m' sig' ∧ ¬ Signed (dev o) x'.1.key.pub m' := findBreak_valid hx''
        exact e ▸ hv.1
      · rw [ite_eq_right hk'] at hx''
        cases hx''
    obtain ⟨u, v, hf, hv⟩ := findSome_forgeryOf hd hl hmem
    refine ⟨u, v, hf, (fun h => by cases h), ?_⟩
    show Verity.lmsVerify (((keyOf θ).map fun k => k.pub.data.toList).getD []) u v = true
    rw [hk]
    exact hv

/-- **(a): an accepted play whose log the check read isn't the device's is at most as likely as the read finder's
collision plus the forger's forgery**, where each accepted run that isn't `Logged` carries a break under the
parameters' key in one of its records (`Binds`) and that key is at the deployed LMS parameters. -/
theorem logged_le (V : Vocab d α) (k : Verity.Registration.Composite) (reads : Sent → Option (List Read))
    (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (hb : Binds V k reads dev θ)
    (hkey : ∀ key ∈ keyOf θ, key.params = Lms.deployed) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Logged reads o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((readFinder V dev θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal ((forger V dev θ F c σ).Pr Verity.lmsVerify) := by
  rw [Verity.Game.prob_eq_prCoin]
  calc Verity.Game.prCoin (fun t => V.Verdict k .sampled (Verity.Game.Tape.out _ σ t) ∧
          ¬ Logged reads (Verity.Game.Tape.out _ σ t))
      ≤ Verity.Game.prCoin (fun t =>
          (∃ u v, (readFinder V dev θ F c σ).out t = some (u, v) ∧ Claims.Collision Verity.sha512Msg u v) ∨
            (forger V dev θ F c σ).Forges Verity.lmsVerify t) := by
        refine Verity.Game.prCoin_mono fun t ⟨hv, hl⟩ => ?_
        have hθ := (play_reaches (Verity.Game.Tape.reaches_out _ σ t)).1
        obtain ⟨x, hx, hk, hbr⟩ := hb _ hθ hv hl
        exact finders_of_break V dev θ F c σ hkey t hx hk hbr
    _ ≤ _ := Verity.Game.prCoin_or_le _ _
    _ = _ := congrArg₂ (· + ·) (Verity.Game.prCoin_eq_ofReal_pr _) (Verity.Game.prCoin_eq_ofReal_pr _)

/-- **(a) at the leg's constructions with this read finder and forger** (`withBinding`). -/
theorem logged_le_withBinding {V : Vocab d α} (K : Constructions V) (k : Verity.Registration.Composite)
    (reads : Sent → Option (List Read)) (dev : Exchange → History) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hb : Binds V k reads dev θ)
    (hkey : ∀ key ∈ keyOf θ, key.params = Lms.deployed) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Logged reads o) (V.game θ F c) σ ≤
      ENNReal.ofReal (((withBinding K dev).readFinder θ F c σ).Pr Verity.sha512Msg) +
        ENNReal.ofReal (((withBinding K dev).forger θ F c σ).Pr Verity.lmsVerify) :=
  logged_le V k reads dev θ F c σ hb hkey

end Verity.Sampled.Binding
