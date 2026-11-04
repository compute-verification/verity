import Proofs.Flock.Claims.Recursion.RepFramesLig
import Proofs.Flock.Claims.Recursion.RepOpenings
import Proofs.Flock.Claims.KS.PlainLevels

/-!
# Two reps accepted on one stream are one proof, or their openings collide

`rep_bind_lig` gives two proofs `verifyRep` accepts as one rep on one retained stream every field but `params`,
`zerocheck.finalC`, `ligerito.initialCap` and the openings. The rest:

* `params` is the pinned fast100 parameters (`verifyRep`'s R7/F2 check), and `initialCap` the proof's cap (Ligerito's
  first check);
* `finalC` is zerocheck's evaluation of `round1c` at its coin `z`, round 2's first coin (`bindAndZerocheck_finalC`);
* every level's openings pass their Merkle checks against one cap at the positions of one round's coins
  (`plainQueryRound_verifyRepWith`, `plainLevels_verifyRepWith`), so they are one, or two of them collide
  (`merkleCheck_coll`);
* a decoded proof has `R − 1` recursive proofs (`decodeProof_rproofs`), so one opening per level is the whole of them.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Reader
open FlockSoundness.Refine (ebind_ok)

/-- A loop whose every step pushes one element returns its start with one element per index. -/
theorem yields_forIn_size {α : Type} (body : ℕ → Array α → Reader.M (ForInStep (Array α)))
    (hb : ∀ i s, Post (body i s) fun st => ∃ x, st = .yield (s.push x)) :
    ∀ (l : List ℕ) (acc out : Array α), Yields (forIn l acc body) out → out.size = acc.size + l.length
  | [], acc, out, h => by
    rw [List.forIn_nil] at h
    rw [yields_pure h]
    rfl
  | i :: l, acc, out, h => by
    rw [List.forIn_cons] at h
    obtain ⟨st, hst, h⟩ := yields_bind h
    obtain ⟨x, rfl⟩ := hb i acc st hst
    have := yields_forIn_size body hb l _ out h
    rw [this, Array.size_push, List.length_cons]
    omega

/-- **A decoded proof has `R − 1` recursive proofs**, `R + 1` its schedule's levels. -/
theorem decodeProof_rproofs {sh : Shape} {bytes : ByteArray} {p : Proof} (h : decodeProof sh bytes = .ok p) :
    p.ligerito.recursiveProofs.size = sh.sched.levels.size - 1 - 1 := by
  replace h := (Flock.decodeProof_read h).1
  unfold decodeRead at h
  obtain ⟨r', hrun, -⟩ := Flock.Reader.runPlain_ok h
  refine (?_ : Post _ fun q : Proof => q.ligerito.recursiveProofs.size = sh.sched.levels.size - 1 - 1) p
    ⟨_, _, hrun⟩
  dsimp only
  refine post_bind_vecBytes fun cap hcap hcapE => ?_
  refine post_bind fun m _ => post_bind fun lir _ => post_bind fun lbs _ => post_bind fun profile _ => ?_
  refine post_bind fun tag _ => ?_
  split
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun numLanes _ => post_bind fun mh _ => ?_
    refine post_bind_vec fun r1ab hr1ab => post_bind_vec fun r1c hr1c => post_bind_vec fun rounds hrounds => ?_
    refine post_bind fun fa _ => post_bind fun fb _ => post_bind fun fc _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun lr hlr => post_bind_vec fun zp hzp => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_switches fun rs hrs hrsE => post_bind fun _ _ => ?_
    refine post_bind_vecBytes fun icap _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun n _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind_caps fun rcaps hrc hrcE => ?_
    refine post_bind fun np _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun rproofs hrp => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_pure ?_
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hrp
    have := yields_forIn_size _ (fun i s => ?_) _ _ _ hrp
    · simpa [Std.Legacy.Range.size] using this
    · exact post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_pure ⟨_, rfl⟩

end FlockSoundness.Discharge.DecodeZL

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

local instance lawfulBEqByteArrayRB : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := by show (a.data == a.data) = true; exact beq_self_eq_true _

/-! ## What `verifyRep` pins outside the frames -/

theorem pcs_eq_of_beq {x y : PcsParams} (h : (x == y) = true) : x = y := by
  obtain ⟨a, b, c, d, e, f⟩ := x
  obtain ⟨a', b', c', d', e', f'⟩ := y
  have h' : (a == a' && (b == b' && (c == c' && (d == d' && (e == e' && f == f'))))) = true := h
  simp only [Bool.and_eq_true, beq_iff_eq] at h'
  obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ := h'
  rfl

/-- **Zerocheck pins `finalC`**: its evaluation of `round1c` at `z`, the first coin of the round two on. -/
theorem bindAndZerocheck_finalC {m : ℕ} (hm : 14 ≤ m) {digest cap : ByteArray} {p : ZerocheckProof}
    {t t' : Transcript} {zc : ZcOut} (h : bindAndZerocheck m digest cap p t = .ok (zc, t')) :
    p.finalC = dot (lagrange nodesL den6Inv ((t.rounds[t.k + 2]!).coins.getD 0 0)) p.round1c := by
  unfold bindAndZerocheck at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨-, -, t₅, r₅, kk₅, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨-, -, t₆, r₆, kk₆, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₇, k₇, h⟩ := absorbSlice_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, -, t₉, -, -, h⟩ := sample_bind h
  obtain ⟨hpc, -⟩ := check_bind h
  have hr : t₈.rounds = t.rounds := by rw [k₈.2, k₇.2, r₆, r₅, k₄.2, k₃.2, k₂.2, k₁.2]
  have hk : t₈.k = t.k + 2 := by rw [k₈.1, k₇.1, kk₆, kk₅, k₄.1, k₃.1, k₂.1, k₁.1]
  rw [hr, hk] at hpc
  exact (beq_iff_eq.1 hpc).symm

/-- **Ligerito pins `initialCap`** to the commitment's cap. -/
theorem ligerito_initialCap {ms : MerkleScheme} {sch : Flock.Schedule} {T : F128} {cap : Array ByteArray}
    {p : LigeritoProof} {basisAt : Array F256 → ℕ → Array F256} {t t' : Transcript}
    (h : Flock.ligerito ms sch T cap p basisAt t = .ok ((), t')) : p.initialCap = cap := by
  unfold Flock.ligerito at h
  obtain ⟨hcap, -⟩ := check_bind h
  exact beq_iff_eq.1 hcap

/-- **What an accepted rep pins outside its frames**: the pinned parameters, the initial cap, and `finalC` at round
2's coin. -/
theorem verifyRep_pins {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule} {sess : RunRecord} {i r : ℕ}
    {bytes cap : ByteArray} (hm : 14 ≤ st.m) (h : verifyRep spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) =
        some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      p.params = pcsParams st.merkle sch ∧ p.ligerito.initialCap = p.cap ∧
      p.zerocheck.finalC = dot (lagrange nodesL den6Inv ((stream.rounds[2]!).coins.getD 0 0)) p.zerocheck.round1c := by
  unfold verifyRep at h
  dsimp only at h
  cases hs : Array.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r)
      sess.streams with
  | none => rw [hs] at h; cases h
  | some stream =>
    rw [hs] at h
    obtain ⟨extras, hex, h⟩ := ebind_ok h
    obtain ⟨p, hp, h⟩ := ebind_ok h
    by_cases c1 : (p.params != pcsParams st.merkle sch) = true
    · rw [ite_eq_left c1] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
    rw [ite_eq_right c1] at h
    obtain ⟨⟨u, t⟩, hrun, -⟩ := ebind_ok h
    rw [StateT.run] at hrun
    obtain ⟨zc, t₁, h₁, hrun⟩ := bind_ok hrun
    obtain ⟨_, _, _, hrun⟩ := bind_ok hrun
    obtain ⟨⟨_, _⟩, _, _, hlig⟩ := bind_ok hrun
    have hpar : (p.params == pcsParams st.merkle sch) = true := by
      cases hb : (p.params == pcsParams st.merkle sch)
      · exact absurd (by simp [bne, hb]) c1
      · rfl
    refine ⟨stream, extras, p, rfl, hex, hp, pcs_eq_of_beq hpar, ligerito_initialCap hlig, ?_⟩
    have hz := bindAndZerocheck_finalC hm h₁
    simp only [Transcript.init, Nat.zero_add] at hz
    exact hz

/-! ## Every level's Merkle check -/

/-- The round of level `ℓ`'s query coins in a plain rep's stream, `nExtra` the statement's extra claims. -/
abbrev repRound (st : Setup) (sch : Flock.Schedule) (nExtra ℓ : ℕ) : ℕ :=
  plainLevelRound st.m st.kLog sch.k0 (2 + nExtra) (sch.levels.size - 1) ℓ

/-- **Query `j`'s opening at level `ℓ`**, as its SHA-512 inputs: the proof's level-`ℓ` row, salt and siblings at the
position of the level's query coins in `rounds` (round `rd ℓ`). -/
def levelIns (ms : MerkleScheme) (sch : Flock.Schedule) (rounds : Array Round) (rd : ℕ → ℕ) (p : Proof) (ℓ j : ℕ) :
    List ByteArray :=
  opensInsOf ms (openAt p.ligerito (sch.levels.size - 1) ℓ).1[j]!
    ((openAt p.ligerito (sch.levels.size - 1) ℓ).2.2.getD j ByteArray.empty)
    (positions (sch.levels.getD ℓ default) (rounds[rd ℓ]!).coins)[j]!
    (sibsOf (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
      ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) j)

theorem named_of {ms : MerkleScheme}
    (hms : ms = MerkleScheme.hm96Sha512 ∨ ms = MerkleScheme.sha512 "flock-leaf/sha512-unsalted") : ms.Named := by
  rcases hms with rfl | rfl
  · intro s hs
    have e : MerkleScheme.find? MerkleScheme.hm96Sha512.id = some MerkleScheme.hm96Sha512 := rfl
    rw [e] at hs
    exact (Option.some.inj hs).symm
  · intro s hs
    have e : MerkleScheme.find? (MerkleScheme.sha512 "flock-leaf/sha512-unsalted").id =
        some (MerkleScheme.sha512 "flock-leaf/sha512-unsalted") := rfl
    rw [e] at hs
    exact (Option.some.inj hs).symm

/-- **An accepted plain rep passed every level's Merkle check** against that level's cap (`capAt`), on its opened
rows, siblings and salts (`openAt`), at the positions of round `repRound`'s coins. -/
theorem rep_levels_check {spec : SessionSpec} {st : Setup} {sch : Flock.Schedule} {sess : RunRecord} {i r : ℕ}
    {bytes cap : ByteArray} (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog) (named : st.merkle.Named) (h : verifyRep spec st sch sess i r bytes = .ok cap) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      ∀ ℓ < sch.levels.size, merkleCheck st.merkle (capAt p.cap p.ligerito ℓ) (sch.levels.getD ℓ default).cap
        (sch.levels.getD ℓ default).d
        (positions (sch.levels.getD ℓ default) (stream.rounds[repRound st sch extras.size ℓ]!).coins)
        (openAt p.ligerito (sch.levels.size - 1) ℓ).1 (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
        (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2 (2 ^ (sch.levels.getD ℓ default).lanes) = .ok () := by
  have hstd : verifyRepWith With.std spec st sch sess i r bytes = .ok cap := by
    rw [With.verifyRepWith_std _ _ named]
    exact h
  have h2 : 2 ≤ sch.levels.size := (fast100_dims _ _ hsch).1.2.1
  obtain ⟨stream, extras, p, hstr, hex, hp, -, -, hmc0, -⟩ :=
    plainRepColsOf_verifyRepWith With.std spec st sch sess i r bytes cap hsch hf hkl hstd
  obtain ⟨stream₁, extras₁, p₁, hstr₁, hex₁, hp₁, hlev⟩ :=
    plainLevels_verifyRepWith With.std spec st sch sess i r bytes cap hsch hf hkl hstd
  rw [hstr] at hstr₁
  cases hstr₁
  rw [hex] at hex₁
  cases hex₁
  rw [hp] at hp₁
  cases hp₁
  refine ⟨stream, extras, p, hstr, hex, hp, fun ℓ hℓ => ?_⟩
  rcases Nat.eq_zero_or_pos ℓ with rfl | h1
  · rw [With.opsWith_std _ named, With.ofScheme_merkleCheck] at hmc0
    have e : repRound st sch extras.size 0 = Claims.Flock.plainQueryRound st.m st.kLog sch.k0 (2 + extras.size) := by
      simp only [repRound, plainLevelRound, levRound, Claims.Flock.plainQueryRound,
        show 0 < sch.levels.size - 1 by omega, ↓reduceIte]
    rw [e]
    exact hmc0
  · have hmc := (hlev ℓ h1 hℓ).2
    rw [With.opsWith_std _ named, With.ofScheme_merkleCheck] at hmc
    have e : capAt p.cap p.ligerito ℓ = p.ligerito.recursiveCaps[ℓ - 1]! :=
      ite_eq_right_iff.2 fun h0 => absurd h0 (by omega)
    rw [e]
    exact hmc

/-! ## Two reps on one stream -/

theorem levelProof_eq {a b : LevelProof} (h : (a.rows, a.siblings, a.salts) = (b.rows, b.siblings, b.salts)) :
    a = b := by
  cases a
  cases b
  simp only [Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  rfl

theorem finalProof_eq {a b : FinalProof} (hy : a.yr = b.yr)
    (h : (a.rows, a.siblings, a.salts) = (b.rows, b.siblings, b.salts)) : a = b := by
  cases a
  cases b
  simp only [Prod.mk.injEq] at h hy
  obtain ⟨rfl, rfl, rfl⟩ := h
  subst hy
  rfl

theorem zerocheckProof_eq {a b : ZerocheckProof} (h1 : a.round1ab = b.round1ab) (h2 : a.round1c = b.round1c)
    (h3 : a.rounds = b.rounds) (h4 : a.finalA = b.finalA) (h5 : a.finalB = b.finalB) (h6 : a.finalC = b.finalC) :
    a = b := by
  cases a
  cases b
  simp only at h1 h2 h3 h4 h5 h6
  subst h1 h2 h3 h4 h5 h6
  rfl

theorem ligeritoProof_eq {a b : LigeritoProof} (h1 : a.initialCap = b.initialCap)
    (h2 : a.initialProof = b.initialProof) (h3 : a.recursiveCaps = b.recursiveCaps)
    (h4 : a.recursiveProofs = b.recursiveProofs) (h5 : a.finalProof = b.finalProof) (h6 : a.messages = b.messages)
    (h7 : a.queryNonces = b.queryNonces) (h8 : a.oodValues = b.oodValues) (h9 : a.claimNonces = b.claimNonces)
    (h10 : a.consistencyNonces = b.consistencyNonces) : a = b := by
  cases a
  cases b
  simp only at h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  subst h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  rfl

theorem proof_eq {a b : Proof} (h1 : a.cap = b.cap) (h2 : a.params = b.params) (h3 : a.zerocheck = b.zerocheck)
    (h4 : a.lincheck = b.lincheck) (h5 : a.ringSwitches = b.ringSwitches) (h6 : a.ligerito = b.ligerito) :
    a = b := by
  cases a
  cases b
  simp only at h1 h2 h3 h4 h5 h6
  subst h1 h2 h3 h4 h5 h6
  rfl

/-- **Two proofs accepted as one rep on one retained stream are one proof, or two of their openings collide**: at
some level, one query's two openings, at the position of that level's query coins, collide among their SHA-512
inputs. -/
theorem rep_bind_proof (spec : SessionSpec) (st : Setup) (sch : Flock.Schedule) (sess : RunRecord) (i r : ℕ)
    (bytes bytes' cap cap' : ByteArray) (hsch : fast100 st.m = some sch)
    (hf : ∀ a b lw rho ts v, st.fold.folded a b lw rho ts = .ok v → rho.size + 6 = st.fold.kLog)
    (hkl : st.fold.kLog = st.kLog)
    (hms : st.merkle = MerkleScheme.hm96Sha512 ∨ st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (h : verifyRep spec st sch sess i r bytes = .ok cap) (h' : verifyRep spec st sch sess i r bytes' = .ok cap')
    (hret : ∀ stream : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString r) =
        some stream → AllRetained stream.rounds) :
    ∃ (stream : StreamRec) (extras : Array Flock.Claim) (p p' : Proof),
      sess.streams.find? (fun x => x.name == s!"{spec.tables[i]!}/rep{r}") = some stream ∧
      st.extra sess.points = .ok extras ∧ decodeProof (shapeOf st sch extras.size) bytes = .ok p ∧
      decodeProof (shapeOf st sch extras.size) bytes' = .ok p' ∧
      (p = p' ∨ ∃ ℓ < sch.levels.size, ∃ j < (positions (sch.levels.getD ℓ default)
          (stream.rounds[repRound st sch extras.size ℓ]!).coins).size,
        CollIn (levelIns st.merkle sch stream.rounds (repRound st sch extras.size) p ℓ j)
          (levelIns st.merkle sch stream.rounds (repRound st sch extras.size) p' ℓ j)) := by
  have h22 := exec_fast100_ge hsch
  have hdim := (fast100_dims _ _ hsch).1
  have h2 : 2 ≤ sch.levels.size := hdim.2.1
  have hlv : ∀ j < sch.levels.size, LevelSeam (sch.levels.getD j default) := fun j hj =>
    fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hj]; exact Array.getElem_mem hj)
  have named := named_of hms
  obtain ⟨ex₀, p₀, p₀', hex₀, hd₀, hd₀', hc, ha, hb, hr, hA, hB, hl, hs, hrc, hmsg, hqn, hood, hcn, hcon, hyr⟩ :=
    FlockSoundness.Discharge.Exec.rep_bind_lig spec st sch sess i r bytes bytes' cap cap' (by omega) hdim hlv hkl h h'
      hret
  obtain ⟨stream, extras, p, hstr, hex, hp, hmc⟩ := rep_levels_check hsch hf hkl named h
  obtain ⟨stream', extras', p', hstr', hex', hp', hmc'⟩ := rep_levels_check hsch hf hkl named h'
  obtain ⟨stream₁, extras₁, p₁, hstr₁, hex₁, hp₁, hpar, hic, hfc⟩ := verifyRep_pins (by omega) h
  obtain ⟨stream₁', extras₁', p₁', hstr₁', hex₁', hp₁', hpar', hic', hfc'⟩ := verifyRep_pins (by omega) h'
  have e₁ : stream' = stream := Option.some.inj (hstr'.symm.trans hstr)
  have e₂ : stream₁ = stream := Option.some.inj (hstr₁.symm.trans hstr)
  have e₃ : stream₁' = stream := Option.some.inj (hstr₁'.symm.trans hstr)
  have e₄ : extras' = extras := Except.ok.inj (hex'.symm.trans hex)
  have e₅ : ex₀ = extras := Except.ok.inj (hex₀.symm.trans hex)
  have e₆ : extras₁ = extras := Except.ok.inj (hex₁.symm.trans hex)
  have e₇ : extras₁' = extras := Except.ok.inj (hex₁'.symm.trans hex)
  subst stream' stream₁ stream₁' extras' ex₀ extras₁ extras₁'
  have e₈ : p₀ = p := Except.ok.inj (hd₀.symm.trans hp)
  have e₉ : p₁ = p := Except.ok.inj (hp₁.symm.trans hp)
  have e₁₀ : p₀' = p' := Except.ok.inj (hd₀'.symm.trans hp')
  have e₁₁ : p₁' = p' := Except.ok.inj (hp₁'.symm.trans hp')
  subst p₀ p₁ p₀' p₁'
  refine ⟨stream, extras, p, p', hstr, hex, hp, hp', ?_⟩
  by_cases hall : ∀ ℓ < sch.levels.size,
      openAt p.ligerito (sch.levels.size - 1) ℓ = openAt p'.ligerito (sch.levels.size - 1) ℓ
  · left
    have h0 := hall 0 (by omega)
    simp only [openAt, ↓reduceIte] at h0
    have hF := hall (sch.levels.size - 1) (by omega)
    simp only [openAt, show sch.levels.size - 1 ≠ 0 by omega, lt_irrefl, ↓reduceIte] at hF
    have hrp := FlockSoundness.Discharge.DecodeZL.decodeProof_rproofs hp
    have hrp' := FlockSoundness.Discharge.DecodeZL.decodeProof_rproofs hp'
    simp only [shapeOf] at hrp hrp'
    have hRP : p.ligerito.recursiveProofs = p'.ligerito.recursiveProofs := by
      apply Array.ext (by rw [hrp, hrp'])
      intro k hk hk'
      have e := hall (k + 1) (by omega)
      simp only [openAt, show k + 1 ≠ 0 by omega, show k + 1 < sch.levels.size - 1 by omega, ↓reduceIte,
        Nat.add_sub_cancel] at e
      rw [getElem!_pos _ k hk, getElem!_pos _ k hk'] at e
      exact levelProof_eq e
    refine proof_eq hc (hpar.trans hpar'.symm)
      (zerocheckProof_eq ha hb hr hA hB (by rw [hfc, hfc', hb])) hl hs
      (ligeritoProof_eq (by rw [hic, hic', hc]) (levelProof_eq h0) hrc hRP (finalProof_eq hyr hF) hmsg hqn hood
        hcn hcon)
  · right
    push Not at hall
    obtain ⟨ℓ, hℓ, hne⟩ := hall
    have hcapE : capAt p.cap p.ligerito ℓ = capAt p'.cap p'.ligerito ℓ := by
      unfold capAt
      rw [hc, hrc]
    have e1 := hmc ℓ hℓ
    rw [hcapE] at e1
    rcases merkleCheck_coll hms e1 (hmc' ℓ hℓ) with ⟨hr1, hs1, hy1⟩ | ⟨j, hj, hcoll⟩
    · exact absurd (Prod.ext hr1 (Prod.ext hs1 hy1)) hne
    · exact ⟨ℓ, hℓ, j, hj, hcoll⟩

end FlockDeploy.KS
