import Proofs.Flock.Recursive.TopFinder

/-!
# The held salts' finder: a registrant's read of R's level-0 top against the salts its nodes were committed under

R's level-0 top is committed node by node: node `i`'s commit string is the hm96 commit string of its row
(`Firewall.topRows`) under `salt i`, `salt` the salts the prover's workers committed R's level-0 top rows under, which
they hold (`registrant-holds-openings`; the guarantee is applied at the held salts, `TopOf`). A session reading node `i` opens the
port's root at the leaf of its own row bytes and salt; where that leaf is node `i`'s (`treeLeaf` of its commit string)
and the bytes are not node `i`'s row's, the two hm96 leaves' preimages collide (`hm96Pair`), a pair computable only with
`salt i` in hand.

* `rowInV`: the bytes `Firewall.rowDigest` hashes for `RecCoins.commitRow`'s row; `capRow`, `capRows`: a node's row and
  `cap`'s rows;
* `ReadsX.SaltClash`, `SaltColl`: some session's registered read whose hm96 pair against row `readAt` of `rows` under
  `salt` collides;
* `saltFinder`, `prob_saltColl_le`: the finder that plays the recursive audit once, with `salt` its input, and its
  `qS²/2^513`; `prob_regTopSalt_le`: its union with `regFinder` and `topFinder`.
-/

namespace FlockC3.RegLevels

open Flock

/-- A row of u16 words' bytes, low byte first. -/
def wordBytes (ws : List ℕ) : List UInt8 := ws.flatMap fun w => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]

/-- What `Firewall.rowDigest` hashes for `commitRow`'s row: `sha512/row/v1`'s first block, then the row's words
zero-padded to whole blocks. -/
def rowInV (row : Array ℕ) : List UInt8 :=
  let ws := RecCoins.padTo row (Registered.rowWords row.size 16)
  (HmRow.rowPrefix ws.size).data.toList ++ wordBytes ws.toList

/-- A tree top's node as its row (`Firewall.topRows`): 32 little-endian u16 words, then 32 zero words. -/
def capRow (top : ByteArray) : Array ℕ := (Firewall.topRows top).headD #[]

/-- The rows of `cap`'s `2^capDepth lv` nodes. -/
def capRows (lv : FlockSoundness.Accounting.Level) (cap : FlockSoundness.Model.Cap FlockSoundness.Refine.Dig) :
    Array (Array ℕ) :=
  (Array.range (2 ^ FlockSoundness.Model.capDepth lv)).map fun i => capRow (cap i).1

end FlockC3.RegLevels

namespace FlockSoundness.Registered

open Game Refine Binding Audit

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}

/-- **A registered read clashes with `rows` under `salt`**: at some registered read `p`, the hm96 pair of the
registrant's row bytes and salt against row `readAt p` of `rows` (`rowInV`) under `salt (readAt p)` is a SHA-512
collision. -/
def ReadsX.SaltClash (rd : ReadsX lay rs) (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) : Prop :=
  ∃ p, rd.isRead p ∧ Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p)
    (FlockC3.RegLevels.rowInV rows[rd.readAt p]!) (salt (rd.readAt p)).data.toList)

theorem ReadsX.SaltClash.collision {rd : ReadsX lay rs} {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    (h : rd.SaltClash rows salt) : Merkle.Collision H512 := by
  obtain ⟨_, -, h1, h2⟩ := h
  exact ⟨_, _, h1, h2⟩

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting StrictCR
open scoped ENNReal

variable {n : ℕ} {L : Law n} {Reg Cm Reg₂ : Type} {I : Inner}
variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}

/-- **A session's read clashes with R's level-0 top under the held salts**: at the history `(R, ω, t)` and registration
`R₂`, some session's registrant record clashes with `cap`'s rows under `salt` (`ReadsX.SaltClash`). -/
def SaltColl (lv : Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg') (R : Reg) (ω : L.Ω)
    (t : List (Cm × I.Coin)) (R₂ : Reg₂) : Prop :=
  ∃ j, (zs R ω t R₂ j).rd.SaltClash (FlockC3.RegLevels.capRows lv cap) salt

theorem SaltColl.collision {lv : Level} {cap : Cap Refine.Dig} {salt : ℕ → ByteArray}
    {zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {R : Reg} {ω : L.Ω}
    {t : List (Cm × I.Coin)} {R₂ : Reg₂} (h : SaltColl lv cap salt zs R ω t R₂) : Merkle.Collision H512 := by
  obtain ⟨_, h⟩ := h
  exact h.collision

/-- **The held salts' finder**: one run of the recursive audit against the prover `σ`, output a session's hm96 pair
against R's level-0 top rows under `salt` where it collides (`SaltColl`), else nothing. `salt` is its input, the salts
the prover's workers committed R's level-0 top rows under, which they hold (`registrant-holds-openings`). An honest
`Finder.CR` budget counts the audit run and, at each registered read of each session, `hm96Pair`'s four SHA-512
evaluations over both salts: the read's row bytes and its salt digest under the registrant's salt, and node `readAt`'s
row bytes (`rowInV`, 256 bytes) and its salt digest under `salt`. -/
noncomputable def saltFinder (lv : Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N)) : Finder Pair :=
  oneRun _ σ (onePick H512 (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
    SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) fun _ h => SaltColl.collision h)

/-- **A session's read clashes with the held salts at most `qS²/2^513` likely** under `cr/sha-512` for `saltFinder` at
the budget `qS`: once per audit, whatever the number of sessions and reads. -/
theorem prob_saltColl_le (lv : Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qS : ℝ} (hS : (saltFinder lv cap salt zs σ).CR H512 qS) :
    prob (fun o => SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
  have hev : (prob (fun o => SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ).toReal ≤ qS ^ 2 / 2 ^ (513 : ℝ) := by
    rw [prob_toReal]
    exact (expect_one_le H512 _ σ (fun o : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) =>
      SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1) fun _ h => SaltColl.collision h).trans hS.le
  exact (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (by positivity)).2 hev

/-- **The registrant, top and held-salt finders union-bound their three one-run bad events** (the value tie's, at the
values the verifier registers itself, is a fourth term when it lands). -/
theorem prob_regTopSalt_le
    (wr : List (Cm × I.Coin) → Lean.Json × String) (st : Flock.Stmt) (reg : Lean.Json)
    (bcs' : Array ByteArray) (lv : Level) (cap : Cap Refine.Dig) (salt : ℕ → ByteArray)
    (hstk : StackedTopAt lv st reg cap salt bcs')
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    {qX qT qS : ℝ} (hX : (regFinder zs dg N σ).CR H512 qX)
    (hT : (topFinder wr st reg bcs' lv cap salt hstk zs σ).CR H512 qT)
    (hS : (saltFinder lv cap salt zs σ).CR H512 qS) :
    prob (fun o => (RegColl zs o.1 o.2.1 o.2.2.1 o.2.2.2.1 ∨
        (AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ TopColl wr st reg bcs' o.2.2.1)) ∨
        SaltColl lv cap salt zs o.1 o.2.1 o.2.2.1 o.2.2.2.1)
      (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
      ENNReal.ofReal (qX ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) +
        ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) :=
  (prob_or_le _ _ _ _).trans (add_le_add (prob_regOrTop_le wr st reg bcs' lv cap salt hstk zs σ hX hT)
    (prob_saltColl_le lv cap salt zs σ hS))

end FlockSoundness.Discharge.Recursion

#print axioms FlockSoundness.Registered.ReadsX.SaltClash.collision
#print axioms FlockSoundness.Discharge.Recursion.SaltColl.collision
#print axioms FlockSoundness.Discharge.Recursion.prob_saltColl_le
#print axioms FlockSoundness.Discharge.Recursion.prob_regTopSalt_le
