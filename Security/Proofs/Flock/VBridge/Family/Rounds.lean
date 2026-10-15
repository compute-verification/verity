import Proofs.Flock.VBridge.Family.Commits

/-!
# The family's reads files, round map and round commitments

`skOf`'s `own` and `sel` and `cmtOf`'s `bc` at V*'s staging (`rec_vstage.stage`, `rec_vstage.py:385-399`):

* `ownOf`: V*'s algebra part `p` reads message round `k`'s elements through its port `m<k>` (`Algebra.spec`,
  `Algebra.lean:386`), which its reads file registers under `rec-c<k>`, instance `r` (rep `r`) reading row `r`
  (`rec_vstage.messages`, `rec_vstage.py:156-168`; `rec_residuals.write`, `rec_residuals.py:452-454`). The verifier reads
  that file as `ZkOuter.own` (`Registered.ofJson`, `FlockVerify.lean:231-242`); `HOwn` is that its message-round ports
  are `ownOf`'s (`msgPorts`), and `skOf_zs`, `checkFull_of_msgPorts` carry `skOf` and U4 (`checkFull`) from it;
* `streamTags`: what each stream round of a rep absorbs, one round per coin of the rep's spine `repShape` (`RepTape`'s
  coin steps are the stream's rounds, `InnerAccepts.lean:107-111`; `coins_repShape`); its rounds that absorb elements
  are `RecAlgebra.rounds`' (`rec_algebra.rounds`, `rec_algebra.py:749-774`), in order (`eltRounds`);
* `selOf`: row `r` of `rec-c<k>` holds inner round `r · n₀ + kc k`: the inner transcript is rep 0's tape, then rep 1's
  (`TableFiles.Valid`, `Stages.lean:74-75`; `tableTape`, `InnerAccepts.lean:139-140`), each `n₀` rounds, and message
  round `k` is a rep's stream round `kc k`. `selRounds_selOf`: `SelRounds` by definition;
* `bc`: an inner round is one rep's stream round, whose one commitment `c_k` is the `b ‖ c` of that rep's row of
  `rec-c<k>` (`rec_residuals.commits_entry` takes the `c_k` as the rows' `b ‖ c`, `rec_residuals.py:397-406`).

At the family, `VStarFamily.own` is `ownOf` at each statement and `VStarFamily.sel` is `selOf` at the verifier's shape;
`vbridgeRead_of_rounds` is `vbridgeRead_of_family` at them and `bc`.

The tops (`rec-top-L<l>`) are no commit strings: a RecOpen instance reads its level's tops at the row
`first[rep] + (pos >> (d − c))` (`TopValue.position`, `rec_outer.py:150-151,224,236`) of the query position `pos`, which
the verifier's query coins draw after the cap's round, so no digest the round's commitment fixes is registered at a top
row and `VCommits` cannot hold there.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

/-! ## A rep's stream rounds -/

/-- What a rep's stream round absorbs before its coins: message elements (a round of `RecAlgebra.rounds`), or none (a
cap is absorbed with its level's last folding message). -/
inductive Absorb
  | elts
  | none
  deriving DecidableEq, Repr

/-- A spine step is a coin. -/
def isCoin : Exec.Kind → Bool
  | .coin _ => true
  | .msg _ => false

/-- The folding rounds' (`foldShape`): the first after a coin, the rest each after its folding message. -/
def foldTags : ℕ → List Absorb
  | 0 => []
  | k + 1 => .none :: List.replicate k .elts

/-- The recursive levels' (`recShape`): a level's folding rounds; then its last folding message and the next level's cap,
the out-of-domain samples' two rounds, and the queries', or at the last level its last folding message with `yr`. -/
def recTags : List Accounting.Level → List Absorb
  | [] => []
  | [lv] => foldTags lv.lanes ++ [.elts, .none, .none]
  | lv :: lv' :: rest => foldTags lv.lanes ++ [.elts, .elts, .none, .elts] ++ [.none, .none, .elts] ++ recTags (lv' :: rest)

/-- Compiled Ligerito's (`ligShape`): `T` (which the stream absorbs as Ligerito starts, `rec_algebra.py:751-752`), the
first out-of-domain value, level 0's folding messages, then as `recTags`. -/
def ligTags (sch : Accounting.Schedule) : List Absorb :=
  match sch.levels with
  | [] => []
  | _ :: rest => .elts :: .elts :: (List.replicate sch.k0 .elts ++
      match rest with
      | [] => []
      | _ :: _ => [.elts, .elts, .none, .elts] ++ [.none, .none, .elts] ++ recTags rest)

/-- **A rep's stream rounds**, in order (`repShape`'s coins): the zerocheck's, the lincheck's, the `claims` ring switches'
and the opening's coin, then compiled Ligerito's. -/
def streamTags (m kLog claims : ℕ) (sch : Accounting.Schedule) : List Absorb :=
  ([.none, .none, .elts] ++ List.replicate (m - 6) .elts) ++
    (([.elts, .none] ++ List.replicate (kLog - 6) .elts ++ [.elts]) ++
      ((List.replicate claims .elts ++ [.none]) ++ ligTags sch))

/-- The stream rounds that absorb elements, in order: message round `k` is the `k`-th. -/
def eltRounds (tags : List Absorb) : List ℕ := (List.range tags.length).filter fun s => tags[s]? == some .elts

section coins

variable {F K D : Type}

theorem coins_pairRounds : ∀ n, (Exec.pairRounds F n).countP isCoin = n
  | 0 => rfl
  | n + 1 => by simp [Exec.pairRounds, isCoin, List.countP_cons, coins_pairRounds n]

theorem coins_switches : ∀ n, (Exec.switches F n).countP isCoin = n
  | 0 => rfl
  | n + 1 => by simp [Exec.switches, isCoin, List.countP_cons, coins_switches n]

theorem coins_foldShape : ∀ k, (Exec.foldShape F K k).countP isCoin = k
  | 0 => rfl
  | k + 1 => by simp [Exec.foldShape, isCoin, List.countP_cons, coins_foldShape k]

theorem length_foldTags (k : ℕ) : (foldTags k).length = k := by
  cases k <;> simp [foldTags]

theorem coins_recShape (ext : ℕ) : ∀ levels, (Exec.recShape F K D ext levels).countP isCoin = (recTags levels).length
  | [] => rfl
  | [lv] => by
    simp [Exec.recShape, recTags, Exec.qbShape, isCoin, List.countP_append, List.countP_cons, coins_foldShape, length_foldTags]
  | lv :: lv' :: rest => by
    rw [Exec.recShape, recTags]
    simp [Exec.oodShape, Exec.qbShape, isCoin, List.countP_append, List.countP_cons, coins_foldShape, length_foldTags,
      coins_recShape ext (lv' :: rest)]

/-- **The stream has one round per coin of the rep's spine**: `streamTags` lists `repShape`'s coins. -/
theorem coins_repShape (m kLog nExtra : ℕ) (sch : Accounting.Schedule) :
    (Exec.repShape F K D m kLog nExtra sch).countP isCoin = (streamTags m kLog (nExtra + 2) sch).length := by
  obtain ⟨m', k0, ls, ext⟩ := sch
  rcases ls with _ | ⟨lv₀, _ | ⟨lv₁, rest⟩⟩ <;>
    simp [Exec.repShape, streamTags, Exec.zcShape, Exec.lcShape, Exec.opShape, Exec.ligShape, ligTags, Exec.oodShape, Exec.qbShape, isCoin,
      List.countP_append, List.countP_cons, coins_pairRounds, coins_switches, coins_foldShape, coins_recShape]

end coins

/-! ## The round map and the reads files -/

/-- `rec-c<k>`'s `k`, as `Registered.roundCommitment` reads the name. -/
def cIndex (v : String) : Option ℕ := if v.startsWith "rec-c" then (v.drop 5).toString.toNat? else none

/-- **The inner round a registered row holds**: row `x` of `rec-c<k>` holds rep `x`'s message round `k`, inner round
`x · n₀ + kc k` (rep 0's `n₀` stream rounds come first); a round's commitment by `Registered.roundCommitment`'s test. -/
def selOf (n₀ : ℕ) (kc : ℕ → ℕ) (i : ℕ) (v : String) (x : ℕ) : Prop :=
  Registered.roundCommitment v = true ∧ ∃ k, cIndex v = some k ∧ x < 2 ∧ i = x * n₀ + kc k

theorem selRounds_selOf (n₀ : ℕ) (kc : ℕ → ℕ) : VStarFamily.SelRounds (selOf n₀ kc) := fun _ _ _ h => h.1

/-- Message round `k`'s port of a reads file: `m<k>` under `rec-c<k>`, instance `r` reading row `r` (the fields
`skOf` and U4 do not read are the empty file's). -/
def mPort (name : String) (k : ℕ) : Registered.Port :=
  { name, value := s!"rec-c{k}", root := ByteArray.empty, leaves := 2, words := none, wordBits := 16,
    domain := ByteArray.empty, reads := #[some 0, some 1] }

/-- `m<k>`'s `k`. -/
def mIndex (nm : String) : Option ℕ := if nm.startsWith "m" then (nm.drop 1).toString.toNat? else none

/-- **A statement's message-round ports**: an algebra part's ports `m<k>`, none of a RecOpen's, nor of candidate 3's
statements, which no session of the circuit table's family is. -/
def ownOf : VStar.Stmt → Array Registered.Port
  | .inner _ _ p => (p.S.ports.filterMap fun (nm, _) => (mIndex nm).map (mPort nm)).toArray
  | .recOpen .. | .recOpenStacked .. | .sparse .. | .opening .. => #[]

/-- A message-round port: one named `m<k>` or registering a `rec-c<k>`. -/
def isMsg (p : Registered.Port) : Bool := (mIndex p.name).isSome || (cIndex p.value).isSome

/-- The message-round ports of a reads file: name, value and reads of each. -/
def msgPorts (own : Array Registered.Port) : List (String × String × Array (Option ℕ)) :=
  (own.toList.filter isMsg).map fun p => (p.name, p.value, p.reads)

theorem mem_msgPorts {own : Array Registered.Port} {q : String × String × Array (Option ℕ)} :
    q ∈ msgPorts own ↔ ∃ p ∈ own, isMsg p = true ∧ (p.name, p.value, p.reads) = q := by
  simp [msgPorts, List.mem_filter, Array.mem_toList_iff, and_assoc]

/-- Two reads files with the same message-round ports, in any order (`Registered.ofJson` lists a file's ports in its
object's key order). -/
def SameMsg (own own' : Array Registered.Port) : Prop := ∀ q, q ∈ msgPorts own ↔ q ∈ msgPorts own'

theorem isMsg_ownOf {st : VStar.Stmt} {p : Registered.Port} (hp : p ∈ ownOf st) : isMsg p = true := by
  cases st with
  | recOpen | recOpenStacked | sparse | opening => simp [ownOf] at hp
  | inner _ _ part =>
    simp only [ownOf, List.mem_toArray, List.mem_filterMap, Option.map_eq_some_iff] at hp
    obtain ⟨⟨nm, _⟩, -, k, hk, rfl⟩ := hp
    simp [isMsg, mPort, hk]

/-- A selector of `rec-c<k>` rows reads only a reads file's message-round ports. -/
def SelMsg (sel : ℕ → String → ℕ → Prop) : Prop := ∀ i v x, sel i v x → (cIndex v).isSome = true

theorem selMsg_selOf (n₀ : ℕ) (kc : ℕ → ℕ) : SelMsg (selOf n₀ kc) := by
  rintro i v x ⟨-, k, hk, -⟩
  rw [hk]
  rfl

/-- **`skOf` reads only the message-round ports**: reads files with the same `msgPorts` select the same rows. -/
theorem skOf_congr {S : ℕ} (Vs : Fin S → VStmt) {own own' : Fin S → Array Registered.Port}
    {sel : ℕ → String → ℕ → Prop} (hsel : SelMsg sel) (h : ∀ j, SameMsg (own j) (own' j)) :
    skOf Vs own sel = skOf Vs own' sel := by
  have key : ∀ (o o' : Array Registered.Port), SameMsg o o' → ∀ (nm : String) (u : ℕ) (i : ℕ),
      (∃ p ∈ o, p.name = nm ∧ ∃ x, p.reads[u]? = some (some x) ∧ sel i p.value x) →
      (∃ p ∈ o', p.name = nm ∧ ∃ x, p.reads[u]? = some (some x) ∧ sel i p.value x) := by
    rintro o o' ho nm u i ⟨p, hp, rfl, x, hx, hs⟩
    have hq : (p.name, p.value, p.reads) ∈ msgPorts o' :=
      (ho _).1 (mem_msgPorts.2 ⟨p, hp, by simp [isMsg, hsel _ _ _ hs], rfl⟩)
    obtain ⟨p', hp', -, he⟩ := mem_msgPorts.1 hq
    simp only [Prod.mk.injEq] at he
    obtain ⟨h1, h2, h3⟩ := he
    exact ⟨p', hp', h1, x, h3 ▸ hx, h2 ▸ hs⟩
  funext i j
  ext pos
  rcases pos with ⟨u, q⟩ | g
  · simp only [mem_skOf_inl]
    exact ⟨key _ _ (h j) _ _ _, key _ _ (fun q => (h j q).symm) _ _ _⟩
  · exact iff_of_false (inr_not_mem_skOf g) (inr_not_mem_skOf g)

/-- **U4's pass, port by port**: `fullMiss` finds nothing where every unit reading a round's commitment is drawn in
full. -/
theorem fullMiss_eq_none {own : Array Registered.Port} {d : Flock.Draw.UnitDraw} :
    Registered.fullMiss own d = none ↔
      ∀ p ∈ own, Registered.roundCommitment p.value = true → ∀ u < p.reads.size,
        p.reads[u]!.isSome = true → Registered.drawsInFull d u = true := by
  unfold Registered.fullMiss
  rw [Array.findSome?_eq_none_iff]
  refine forall₂_congr fun p _ => ?_
  by_cases hc : Registered.roundCommitment p.value = true
  · simp only [hc, ite_true, Option.map_eq_none_iff, List.find?_eq_none, List.mem_range, Bool.and_eq_true,
      Bool.not_eq_true', not_and, Bool.not_eq_false, true_implies]
  · simp [hc]

/-- **U4's pass on a parsed draw**: `checkFull` passes where `fullMiss` finds nothing. -/
theorem checkFull_ok_iff {own : Array Registered.Port} {dj : Lean.Json} {d : Flock.Draw.UnitDraw}
    (hd : Flock.Draw.UnitDraw.ofJson dj = .ok d) :
    Registered.checkFull (some own) (some dj) = .ok () ↔ Registered.fullMiss own d = none := by
  unfold Registered.checkFull
  dsimp only
  rw [hd]
  have e : ∀ f : Flock.Draw.UnitDraw → Except String Unit, (Except.ok d >>= f) = f d := fun _ => rfl
  rw [e]
  cases Registered.fullMiss own d with
  | none => exact iff_of_true rfl rfl
  | some _ => exact iff_of_false (fun h => by cases h) (fun h => by cases h)

/-- **U4 at `ownOf` from the verifier's U4**: where a session's reads file has `ownOf`'s message-round ports, a draw
that passes U4 on it passes U4 on `ownOf`'s, whose ports are all message-round ports. -/
theorem checkFull_of_msgPorts {own own' : Array Registered.Port} {dj : Lean.Json} {d : Flock.Draw.UnitDraw}
    (hd : Flock.Draw.UnitDraw.ofJson dj = .ok d) (hm : ∀ p ∈ own', isMsg p = true)
    (h : SameMsg own own') (hU4 : Registered.checkFull (some own) (some dj) = .ok ()) :
    Registered.checkFull (some own') (some dj) = .ok () := by
  have hn := fullMiss_eq_none.1 ((checkFull_ok_iff hd).1 hU4)
  refine (checkFull_ok_iff hd).2 (fullMiss_eq_none.2 fun p' hp' hc u hu hs => ?_)
  have hq : (p'.name, p'.value, p'.reads) ∈ msgPorts own := (h _).2 (mem_msgPorts.2 ⟨p', hp', hm p' hp', rfl⟩)
  obtain ⟨p, hp, -, he⟩ := mem_msgPorts.1 hq
  simp only [Prod.mk.injEq] at he
  obtain ⟨-, h2, h3⟩ := he
  exact hn p hp (h2 ▸ hc) u (h3 ▸ hu) (h3 ▸ hs)

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **Each session's message-round ports**, a definition: `ownOf` at the session's statement. -/
def own (j : Fin F.stmts.size) : Array Registered.Port := ownOf F.stmts[j]

/-- A rep's stream rounds at the verifier's shape `sh`. -/
def tags : List Absorb := streamTags F.sh.m F.sh.kLog F.sh.claims (schedOf F.sh.sched)

/-- A rep's stream rounds' count: the coins of its spine (`coins_repShape`). -/
def n₀ : ℕ := F.tags.length

/-- Message round `k`'s stream round. -/
def kc (k : ℕ) : ℕ := (eltRounds F.tags).getD k 0

/-- **The round map**, a definition: `selOf` at the verifier's shape. -/
def sel : ℕ → String → ℕ → Prop := selOf F.n₀ F.kc

/-- **The rows' `b ‖ c`**, a definition: an inner round is one rep's stream round, whose one commitment's `b ‖ c`
(`cOf`) every row of its commit strings registers (`rec_residuals.commits_entry`, `rec_residuals.py:397-406`). -/
def bc (cOf : Cm → List UInt8) (j : Fin F.stmts.size) (e : Cm) (_ : PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv) :
    List UInt8 :=
  cOf e

theorem selRounds_sel : SelRounds F.sel := selRounds_selOf F.n₀ F.kc

/-- Obligation: discharged by the verifier's check 7 (`cursor/rec-vstar-checks-741b`): the registered reads'
message-round ports are the statement's.
**The verifier's reads files are V*'s**: at every history, the reads file the verifier reads for session `j`
(`ZkOuter.own`, `registeredInput`, `FlockVerify.lean:228-242`) has `own j`'s message-round ports (`SameMsg`). V*'s
staging writes them (`rec_vstage.messages`, `rec_vstage.py:156-168`; `rec_residuals.write`, `rec_residuals.py:452-454`);
no check of the verifier's compares a file's port names with the statement's. The check that establishes it: at
`flock-verify --registered`, the file's ports named `m<k>` or registering `rec-c<k>` are the statement's ports `m<k>`,
each under `rec-c<k>` with reads `[0, 1]` (`msgPorts` against `ownOf`). -/
def HOwn : Prop := ∀ R ω t R₂ j, SameMsg (F.zs R ω t R₂ j).own (F.own j)

/-- **The commit strings the verifier's reads files select are `own`'s**, at every history. -/
theorem skOf_zs (hown : F.HOwn) (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) :
    skOf F.Vs (fun j => (F.zs R ω t R₂ j).own) F.sel = skOf F.Vs F.own F.sel :=
  skOf_congr F.Vs (selMsg_selOf F.n₀ F.kc) (hown R ω t R₂)

/-- **`VBridgeRead` at `own`, `sel` and `bc`**: `vbridgeRead_of_family` where the family's commit strings are
`skOf`'s at `own` and `sel`, its digests `cmtOf`'s at `bc cOf`, and its messages `dec`'s; `HCommits` from `PubCommits`
(`hCommits_of`), `VDecodesRead` from `dec` (`vDecodesRead_of`), and `VFullRead` from V*'s full draw
(`vFullRead_of`). -/
theorem vbridgeRead_of_rounds (dg : ByteArray → ByteArray) (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs) (hsh : F.HShape dg N k Rw GS)
    (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw) (hsch : F.HSch)
    (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw) (cOf : Cm → List UInt8)
    (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg) (hSk : F.Sk = skOf F.Vs F.own F.sel)
    (hcmt : F.cmt = cmtOf F.Vs (F.bc cOf)) (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec)
    (hm : F.msg = msgOf F.Vs F.Sk dec) (hpub : F.PubCommits (F.bc cOf)) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  vbridgeRead_of_family F dg N k Rw hst hsh hres hu hrd hsch hfiles hruns
    (F.hCommits_of F.own F.sel (F.bc cOf) hSk hcmt hpub) (F.vDecodesRead_of dec hw hm)
    F.vFullRead_of

end VStarFamily

end family

end FlockVBridge.Family
