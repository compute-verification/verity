import Proofs.Flock.Claims.Complete.Replay.LigeritoWalk
import Proofs.Flock.Soundness.Refine.Basis

/-!
# Ligerito's final identity from the honest prover's

`LgFinal` (Ligerito's final identity at every rep) from three facts about a rep, none of which mentions the session:

* **The honest Ligerito meets its final identity** (`Lg.Honest`): run on a table `t` committed at level 0, for the
  claim `Σ t·b = T`, the honest prover's Ligerito returns a proof on which the verifier's model (`Lg.out`) ends with
  `lhsE yr ext res = s`, where the verifier's basis at every original point is the multilinear extension of `b`.
* **Ring switching's claim is the packed table's** (`RsSum`): `Σ_y packed(y)·basisVals(y) = T` at the rep's claims and
  coins.
* **The verifier's basis is the batched basis's extension** (`BasisOk`): `basis sws` at every original point is the
  multilinear extension of `basisVals sws`.

And one fact the session walk establishes (`RepWalkP`): the honest rep runs on the trace's packed table, committed at
level 0, and returns the decoded proof.

A returning rep runs the honest Ligerito on its own table, at ring switching's `T` and basis (`repP_lig`), so
`Run.lgFinal_at` is `Lg.Honest` at the rep, and `lgFinal_of` is `LgFinal`.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace Lg

/-- **A level's commitment has the level's codeword rows.** -/
theorem commitP_rows {hs : Hashes V} {ms : MerkleScheme} {lv : Level} {g : Array F128} {salt : ℕ → ByteArray}
    {cm : Commit} (h : commitP hs ms lv g salt = .ok cm) : cm.rows = encodeRows lv g := by
  unfold commitP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  rfl

/-- **The honest Ligerito meets its final identity**: at a fast100 schedule, on a table `tb` and weights `b` of `2^n`
entries with `Σ tb·b = T`, whose level-0 commitment has `tb`'s codeword rows, with a basis whose value at every original
point `q ‖ bits(y)` is the multilinear extension of `b` there, the honest prover's Ligerito, run from a transcript at
which `L + k` bodies have been sent (`k` its position), returns a proof on which the verifier's model, at the replies'
coins, ends with `lhsE yr ext res = s`. -/
def Honest : Prop :=
  ∀ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (m : ℕ) (sch : Schedule) (T : F128)
    (cm0 : Commit) (tb b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (L : ℕ)
    (basisAt : Array F256 → ℕ → Array F256) (t : Transcript) (l : Array Claims.Msg) (P : LigeritoProof)
    (t' : Transcript) (l' : Array Claims.Msg),
    fast100 m = some sch → tb.size = 2 ^ sch.n → b.size = 2 ^ sch.n → dot tb b = T →
    cm0.rows = encodeRows (sch.levels.getD 0 default) tb →
    (∀ q e y, q.size + e = sch.n → y < 2 ^ e →
      FlockSoundness.Refine.gOf (basisAt q e)[y]! =
        FlockSoundness.Model.bHat FlockSoundness.execArith sch.n (fun w => FlockSoundness.Refine.fOf b[w]!)
          (q.toList.map FlockSoundness.Refine.gOf ++ FlockSoundness.Model.bitsK e y)) →
    l.size = L + t.k →
    ((ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b salt nonce).run t).go l = (.ok (P, t'), l') →
    FlockSoundness.Refine.lhsE P.finalProof.yr (out P (cfOf resp L) sch T cm0.cap t).1.2.2.2
      (FlockSoundness.Refine.resE basisAt sch.n (pre P (cfOf resp L) sch T cm0.cap t).1
        (out P (cfOf resp L) sch T cm0.cap t).1.2.2.1 (out P (cfOf resp L) sch T cm0.cap t).1.2.2.2
        (out P (cfOf resp L) sch T cm0.cap t).2.1) = (out P (cfOf resp L) sch T cm0.cap t).2.1.s

end Lg

/-- **A returning rep runs the honest Ligerito** on its table, at ring switching's `T` and the batched basis of its
switched claims, from where its ring switch stopped, with as many bodies sent as the rep's start and that position;
its proof's cap is the commitment's. -/
theorem repP_lig (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (ht : t.k = 0) (l : Array Claims.Msg) {full : Array F128}
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run t).go l =
      (.ok (p, t'), l')) :
    p.cap = cm0.cap ∧ ∃ l₃ t₄,
      l₃.size = l.size + (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
          (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
            (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
                (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size)
          (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
            (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
            (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t))).k ∧
      ((ligeritoP (asksS resp) (Hashes.ofSha512 h) st.merkle sch
        (Rs.T tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
          (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
            (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
                (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size))
        cm0 tbl
        (basisVals (Rs.sws (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
          (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
            (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
                (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size)) (st.m - 7))
        salt nonce).run
        (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
          (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
            (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
                (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size)
          (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
            (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
            (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t)))).go l₃ =
        (.ok (p.ligerito, t₄), l') := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  obtain ⟨l₂, hl₂, hl⟩ := lincheckP_go resp st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
    (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t) l₁ (by rw [hl₁]; exact hfold)
  rw [hl₁] at hl₂ hl
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras) (by simp [abcClaims])
    (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
      (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t)) l₂
  rw [hl₂] at hl₃ hrs
  unfold repP at hrun
  rw [bind_go hz] at hrun
  dsimp only at hrun
  rw [bind_go hl] at hrun
  dsimp only at hrun
  rw [bind_go hrs, StateT.run_bind] at hrun
  obtain ⟨⟨lg, t₄⟩, l₄, h1, h2⟩ := JsonRT.go_bind_ok hrun
  rw [pure_go] at h2
  simp only [Prod.mk.injEq, Except.ok.injEq] at h2
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h2
  exact ⟨rfl, l₃, t₄, by rw [hl₃, Rs.tEnd_k, Lc.tEnd_k, Zc.tEnd_k, ht]; omega, h1⟩

/-! ## The rep's facts -/

/-- **What the session walk establishes at table `i`'s rep `r`, on the packed table**: the table's level-0 commitment
is a `commitP` of the trace's packed table at the schedule's level `0`, and the honest rep on that table and
commitment, run from the rep's first send on and from its own transcript's start, returns the decoded proof. -/
def RepWalkP {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) (i r : ℕ) :
    Prop :=
  ∃ (cm0 : Commit) (s0 : ℕ → ByteArray) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (l : Array Claims.Msg)
    (t' : Transcript) (l' : Array Claims.Msg),
    commitP o.hs o.tabs[i]!.merkle ((o.sch i).levels.getD 0 default) (packed (o.trace i) o.tabs[i]!.m) s0 = .ok cm0 ∧
    l.size = repStart o.spec o.sts (2 * i + r) ∧
    ((repP (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) o.tabs[i]! (o.sch i) (o.trace i)
      (packed (o.trace i) o.tabs[i]!.m) cm0 (o.extras i) salt nonce).run
        (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).go l = (.ok (o.proof i r, t'), l')

/-- **Ring switching's claim is the packed table's** at every rep: `Σ_y packed(y)·basisVals(y) = T` at the rep's
claims and ring-switch coins. -/
def RsSum (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2,
    dot (packed (o.trace i) o.tabs[i]!.m) (basisVals (Rs.sws (o.claims i r) (o.rsCoins i r)) (o.tabs[i]!.m - 7)) =
      Rs.T (o.trace i) (o.claims i r) (o.rsCoins i r)

/-- **The verifier's basis is the batched basis's multilinear extension** at every rep: `basis sws` at `q ‖ bits(y)`
is `Σ_w basisVals(w)·eq(q ‖ bits(y), w)`, at the rep's switched claims. -/
def BasisOk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, ∀ q e y, q.size + e = (o.sch i).n → y < 2 ^ e →
    FlockSoundness.Refine.gOf (basis (Rs.sws (o.claims i r) (o.rsCoins i r)) q e)[y]! =
      FlockSoundness.Model.bHat FlockSoundness.execArith (o.sch i).n
        (fun w => FlockSoundness.Refine.fOf (basisVals (Rs.sws (o.claims i r) (o.rsCoins i r)) (o.tabs[i]!.m - 7))[w]!)
        (q.toList.map FlockSoundness.Refine.gOf ++ FlockSoundness.Model.bitsK e y)

/-- **Ligerito's final identity at a rep** the walk runs on its packed table, from the honest Ligerito's, ring
switching's claim and the verifier's basis. -/
theorem _root_.FlockComplete.Run.lgFinal_at {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2)
    (hw : RepWalkP o i r) (hH : Lg.Honest) (hT : RsSum H rel) (hB : BasisOk H rel) :
    FlockSoundness.Refine.lhsE (o.proof i r).ligerito.finalProof.yr (o.lgOut i r).1.2.2.2
      (FlockSoundness.Refine.resE (basis (Rs.sws (o.claims i r) (o.rsCoins i r))) (o.sch i).n (o.lgPre i r).1
        (o.lgOut i r).1.2.2.1 (o.lgOut i r).1.2.2.2 (o.lgOut i r).2.1) = (o.lgOut i r).2.1.s := by
  obtain ⟨cm0, s0, salt, nonce, l, t', l', h0, hl, hrun⟩ := hw
  have hm : 14 ≤ o.tabs[i]!.m := by have := (o.tab_m hi).2.1; omega
  obtain ⟨full, hfold⟩ := repP_fold o.resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  obtain ⟨hcap, l₃, t₄, hl₃, h1⟩ := repP_lig o.resp (Claims.hOf H) _ _ _ _ cm0 _ salt nonce hm _ rfl l hfold hrun
  rw [hl] at hfold hl₃ h1
  have hfull : o.full i r = full := congrArg val hfold
  rw [← hfull] at hl₃ h1
  obtain ⟨⟨hn7, -⟩, hsm⟩ := FlockSoundness.Refine.fast100_dims _ _ (o.sch_fast100 hi)
  have hn : (o.sch i).n = o.tabs[i]!.m - 7 := by rw [hn7, hsm]
  have key := hH o.resp (Claims.hOf H) o.tabs[i]!.merkle o.tabs[i]!.m (o.sch i) _ cm0 (packed (o.trace i) o.tabs[i]!.m)
    _ salt nonce (repStart o.spec o.sts (2 * i + r)) (basis (Rs.sws (o.claims i r) (o.rsCoins i r))) _ l₃ _ t₄ l'
    (o.sch_fast100 hi) (by simp [packed, hn]) (by simp [basisVals, hn]) (hT o i hi r hr) (Lg.commitP_rows h0)
    (hB o i hi r hr) hl₃ h1
  rw [← hcap] at key
  exact key

/-- **Ligerito's final identity** (`LgFinal`) at every rep the walk runs on its packed table, from the honest
Ligerito's, ring switching's claim and the verifier's basis. -/
theorem lgFinal_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hw : ∀ o : Run H rel, ∀ i < o.tabs.size, ∀ r < 2, RepWalkP o i r) (hH : Lg.Honest) (hT : RsSum H rel)
    (hB : BasisOk H rel) : LgFinal H rel :=
  fun o i hi r hr => o.lgFinal_at hi hr (hw o i hi r hr) hH hT hB

end FlockComplete.Replay
