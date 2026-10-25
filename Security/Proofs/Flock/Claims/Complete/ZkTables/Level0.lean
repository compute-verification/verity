import Proofs.Flock.Claims.Complete.Zk.LigeritoProverPos
import Proofs.Flock.Claims.Complete.Replay.LigeritoCons
import Proofs.Flock.Claims.Complete.Replay.Steps
import Proofs.Flock.Soundness.Discharge.ZkView.Wire

/-!
# Level 0's masked Ligerito: the premises the table-level forms leave open

The masked prover's Ligerito (`ligeritoPZk`) and the verifier's (`Zk.ligeritoWith`) differ from the plain ones at level
0 (live PROTOCOL.md §7, VEIL Figs. 8–9): the commitment's rows carry four extra lanes and every lane's padding
(`commitPZk`), the out-of-domain value and the first `k_0` sumcheck messages are masked, the pair's sums, `ρ`, the
blinded claim and the padding's combination follow the last lane challenge, and the level-0 rows are checked against
them. These are the facts about those two functions, stated once, that `RepShapedT`, `ProverRunsT`, `LigeritoT` and
`InnerT` need beyond what the plain proofs transfer (`ZkTables.Parts`):

* `ProverRuns`: the masked prover's Ligerito returns wherever its pair coins are independent (its one refusal).
* `ProverShape`: its proof has the lengths `Zk.decode` checks of a Ligerito proof (`LgShaped`, `ZkView.Shaped`'s
  Ligerito conjuncts) and `t_pad` padding coefficients.
* `Lanes`: its lane phase in closed form (`LaneForm`): its record, its masked out-of-domain value and lane messages, the
  pair's sums and the blinded claim, over the folds of its table and weights at the lane coins.
* `Verifier`: the verifier's masked Ligerito returns the prover's record on the honest masked proof, from rounds that
  carry the prover's and the prover's position and pending bytes, at a claim `Σ tb·b = T` and a basis that extends `b`,
  and ends where the prover's did (plain's `Lg.Honest` and `ligeritoWith_of` together).

Each is stated over the functions' own inputs, with every hypothesis a rep supplies: SHA-512 digests of 64 bytes, the
`--zk` schedule of a table with `25 ≤ m ≤ 35` (`Zk.schedule`, `t_pad = v.2`), hm96-SHA-512, tables of `2^n` entries,
level 0's commitment of the table (`commitPZk`), and the coins as the replies to the sends (`asksS resp`). Levels `≥ 1`
run plain's code inside both functions; plain's lemmas for them (`ligeritoP_tot`, `ligeritoP_shape`, `ligeritoWith_of`,
`Lg.honest`) are stated from plain's level-0 start, so here they sit inside the whole-function statements.
-/

namespace FlockComplete.ZkTables.Level0

open Flock Transcript FlockComplete FlockComplete.Replay FlockSoundness.Discharge.ZkView

/-! ## Shapes and closed forms -/

/-- **A masked Ligerito proof's lengths at a shape**: `ZkView.Shaped`'s Ligerito conjuncts, level 0's rows
`2^lanes + EXTRA_LANES` wide, every grinding nonce `0`. -/
def LgShaped (sh : Shape) (lg : LigeritoProof) : Prop :=
  let s := sh.sched
  let R := s.levels.size - 1
  let lvl := fun (i : Nat) => s.levels.getD i default
  CapOk sh (lvl 0) lg.initialCap ∧ LevelOk sh (lvl 0) (2 ^ (lvl 0).lanes + Zk.EXTRA_LANES) lg.initialProof ∧
    lg.recursiveCaps.size = R ∧ (∀ i (h : i < lg.recursiveCaps.size), CapOk sh (lvl (i + 1)) lg.recursiveCaps[i]) ∧
    lg.recursiveProofs.size = R - 1 ∧
    (∀ i (h : i < lg.recursiveProofs.size), LevelOk sh (lvl (i + 1)) 16 lg.recursiveProofs[i]) ∧
    lg.finalProof.yr.size = 2 ^ (s.ext + 1) ∧
    LevelOk sh (lvl R) 16 ⟨lg.finalProof.rows, lg.finalProof.siblings, lg.finalProof.salts⟩ ∧
    lg.messages.size = s.k0 + 8 + 7 * (R - 1) ∧ lg.queryNonces.size = R + 1 ∧ lg.oodValues.size = 1 + 2 * R ∧
    lg.claimNonces.size = 3 * R + 2 ∧ lg.consistencyNonces.size = R + 1 ∧ NoncesZero lg

/-- Level 0's out-of-domain point: the reply to send `L`'s first `n` coins. -/
def lanePoint (resp : ℕ → Claims.Msg) (n L : ℕ) : Array F128 := (Array.range n).map (coinOf (resp L))

/-- The lane coins: the replies to sends `L + 2` to `L + 1 + k_0`, as K elements. -/
def laneCh (resp : ℕ → Claims.Msg) (k0 L : ℕ) : List F256 := (List.range k0).map fun j => coinKOf (resp (L + 2 + j))

/-- The pair's coins `ρ`: the replies to the two sends after the lane coins. -/
def laneRho (resp : ℕ → Claims.Msg) (k0 L : ℕ) : F256 × F256 :=
  (coinKOf (resp (L + 2 + k0)), coinKOf (resp (L + 3 + k0)))

/-- The lane phase's weights before its first fold: `b + β·eq(z)`, `β` the reply to send `L + 1`. -/
def laneW0 (resp : ℕ → Claims.Msg) (n L : ℕ) (b : Array F128) : Array F256 :=
  addScaled (embedF b) (embedF (eqTable (lanePoint resp n L))) (coinOf (resp (L + 1)) 0)

/-- Extra lane `e` over the first `s` positions. -/
def pairLane (extra : ℕ → ℕ → F128) (e s : ℕ) : Array F128 := (Array.range s).map (extra e)

/-- The blinded table after the lane phase: `f + ρ_0 R_{2r} + ρ_1 R_{2r+1}` on the folded table `f`. -/
def blinded (f : Array F256) (rho : F256 × F256) (r0 r1 : Array F128) : Array F256 :=
  (Array.range f.size).map fun c => f[c]! + rho.1.smul r0[c]! + rho.2.smul r1[c]!

/-- **The masked Ligerito's lane phase in closed form**, from `L` sends, at the replies `resp`: with `f_j` and `W_j` the
table `tb` and the weights `laneW0` folded at the first `j` lane coins, the record is the target, `(β, y)` for the sent
out-of-domain value `y = mle(tb)(z) + h_ood`, the lane coins and the pair's; lane message `j < k_0` is
`roundPoly f_j W_j` masked by round `j`'s pads; the pair's sums are `⟨W_{k_0}, R_{2r+i}⟩` masked by theirs; the
blinded claim is `⟨W_{k_0}, f_{k_0} + ρ_0 R_{2r} + ρ_1 R_{2r+1}⟩`. -/
structure LaneForm (resp : ℕ → Claims.Msg) (sch : Schedule) (T : F128) (tb b : Array F128) (lay : Zk.PadLayout)
    (hv : Array F128) (rep : ℕ) (extra : ℕ → ℕ → F128) (L : ℕ) (P : LigeritoProof) (zl : Zk.Level0Msgs)
    (rc : Zk.L0Rec) : Prop where
  record : rc =
    { target := T, ood := #[(coinOf (resp (L + 1)) 0, P.oodValues[0]!)],
      challenges := (laneCh resp sch.k0 L).toArray, rho := laneRho resp sch.k0 L }
  ood : P.oodValues[0]! = mleF tb (lanePoint resp sch.n L) + hv[lay.lgOod 0]!
  msgs_size : sch.k0 ≤ P.messages.size
  msgs : ∀ j < sch.k0, P.messages[j]! = maskMsg lay hv j
    (roundPoly (Lg.foldKs ((laneCh resp sch.k0 L).take j) (embedF tb))
      (Lg.foldKs ((laneCh resp sch.k0 L).take j) (laneW0 resp sch.n L b)))
  e0 : zl.e0 = dotK (Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (embedF (pairLane extra (2 * rep) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size)) +
    ⟨hv[lay.lgE 0 0]!, hv[lay.lgE 0 1]!⟩
  e1 : zl.e1 = dotK (Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
      (embedF (pairLane extra (2 * rep + 1) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size)) +
    ⟨hv[lay.lgE 1 0]!, hv[lay.lgE 1 1]!⟩
  tPrime : zl.tPrime = dotK (Lg.foldKs (laneCh resp sch.k0 L) (laneW0 resp sch.n L b))
    (blinded (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)) (laneRho resp sch.k0 L)
      (pairLane extra (2 * rep) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size)
      (pairLane extra (2 * rep + 1) (Lg.foldKs (laneCh resp sch.k0 L) (embedF tb)).size))

/-! ## The premises -/

/-- **The masked prover's Ligerito returns** wherever its pair coins are independent: at a `--zk` schedule, from any
transcript and `L` sends, on any claim, level-0 commitment, table, weights, pads, extra lanes, paddings, salts and
nonces, where the replies to sends `L + 2 + k_0` and `L + 3 + k_0` are independent over F (`Zk.rhoIndependent`). -/
def ProverRuns : Prop :=
  ∀ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (m : ℕ) (v : Schedule × ℕ) (T : F128) (cm0 : Commit)
    (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (t : Transcript) (l : Array Claims.Msg),
    (∀ x, (h x).size = 64) → 25 ≤ m → m ≤ 35 → Zk.schedule m = .ok v →
    Zk.rhoIndependent (laneRho resp v.1.k0 l.size) = true →
    ∃ res l', ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2
      extra lanePad salt nonce).run t).go l = (.ok res, l')

/-- **The masked prover's Ligerito proof has its shape**: where it returns at a `--zk` schedule on level 0's masked
commitment of its table, its proof has the lengths `Zk.decode` checks at any shape of the schedule and hm96-SHA-512's
digest and salt lengths (`LgShaped`), and `t_pad` padding coefficients. -/
def ProverShape : Prop :=
  ∀ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (m : ℕ) (v : Schedule × ℕ) (sh : Shape) (T : F128)
    (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt0 : ℕ → ByteArray) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (t t' : Transcript) (l l' : Array Claims.Msg) (P : LigeritoProof) (zl : Zk.Level0Msgs) (rc : Zk.L0Rec),
    (∀ x, (h x).size = 64) → 25 ≤ m → m ≤ 35 → Zk.schedule m = .ok v → sh.sched = v.1 →
    sh.hashLen = MerkleScheme.hm96Sha512.digestLen →
    sh.saltLen = (if MerkleScheme.hm96Sha512.saltBytes = 0 then none else some MerkleScheme.hm96Sha512.saltBytes) →
    tb.size = 2 ^ v.1.n → b.size = 2 ^ v.1.n →
    commitPZk (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 (v.1.levels.getD 0 default) v.2 tb extra
      lanePad salt0 = .ok cm0 →
    ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra
      lanePad salt nonce).run t).go l = (.ok ((P, zl, rc), t'), l') →
    LgShaped sh P ∧ zl.ybar.size = v.2

/-- **The masked prover's lane phase in closed form**: where its Ligerito returns at a `--zk` schedule from `L` sends,
on a table and weights of `2^n` entries, its record, masked out-of-domain value, lane messages, pair's sums and blinded
claim are `LaneForm`'s. -/
def Lanes : Prop :=
  ∀ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (m : ℕ) (v : Schedule × ℕ) (T : F128) (cm0 : Commit)
    (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (t t' : Transcript) (l l' : Array Claims.Msg)
    (P : LigeritoProof) (zl : Zk.Level0Msgs) (rc : Zk.L0Rec),
    25 ≤ m → m ≤ 35 → Zk.schedule m = .ok v → tb.size = 2 ^ v.1.n → b.size = 2 ^ v.1.n →
    ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra
      lanePad salt nonce).run t).go l = (.ok ((P, zl, rc), t'), l') →
    LaneForm resp v.1 T tb b lay hv rep extra l.size P zl rc

/-- **The verifier's masked Ligerito returns on the honest masked proof**: where the masked prover's Ligerito returned
at a `--zk` schedule from a transcript that has recorded its rounds and `L + k` sends (`k` its position), as rep
`rep < 2`, on a table `tb` and weights `b` of `2^n` entries with `Σ tb·b = T`, committed at level 0 by `commitPZk`, with
a basis whose value at every original point `q ‖ bits(y)` is the multilinear extension of `b` there, the verifier's
masked Ligerito, on rounds that carry the prover's from its position on, from that position and the prover's pending
bytes, returns the prover's record and ends at the prover's final position and pending bytes, which is its round count
(as plain's `ligeritoWith_of` ends at its model's). -/
def Verifier : Prop :=
  ∀ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (m : ℕ) (v : Schedule × ℕ) (T : F128) (cm0 : Commit)
    (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep : ℕ) (extra lanePad : ℕ → ℕ → F128)
    (salt0 : ℕ → ByteArray) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (basisAt : Array F256 → ℕ → Array F256) (L : ℕ) (t t' : Transcript) (l l' : Array Claims.Msg)
    (P : LigeritoProof) (zl : Zk.Level0Msgs) (rc : Zk.L0Rec) (R : Array Round),
    (∀ x, (h x).size = 64) → 25 ≤ m → m ≤ 35 → Zk.schedule m = .ok v → rep < 2 →
    tb.size = 2 ^ v.1.n → b.size = 2 ^ v.1.n → dot tb b = T →
    commitPZk (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 (v.1.levels.getD 0 default) v.2 tb extra
      lanePad salt0 = .ok cm0 →
    (∀ q e y, q.size + e = v.1.n → y < 2 ^ e →
      FlockSoundness.Refine.gOf (basisAt q e)[y]! =
        FlockSoundness.Model.bHat FlockSoundness.execArith v.1.n (fun w => FlockSoundness.Refine.fOf b[w]!)
          (q.toList.map FlockSoundness.Refine.gOf ++ FlockSoundness.Model.bitsK e y)) →
    (∀ q e, (basisAt q e).size = 2 ^ e) →
    t.k = t.rounds.size → l.size = L + t.k →
    ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) MerkleScheme.hm96Sha512 v.1 T cm0 tb b lay hv rep v.2 extra
      lanePad salt nonce).run t).go l = (.ok ((P, zl, rc), t'), l') →
    Agrees R t'.rounds t.k →
    (Zk.ligeritoWith (Hashes.ofSha512 h : Hashes V) MerkleScheme.hm96Sha512 v.1 T cm0.cap P basisAt
      ⟨v.2, rep, zl⟩).run ⟨R, t.k, t.pending⟩ = .ok (rc, ⟨R, t'.k, t'.pending⟩) ∧ t'.k = t'.rounds.size

end FlockComplete.ZkTables.Level0
