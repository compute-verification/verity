import Proofs.Flock.Claims.ZK.CoinLayer
import Proofs.Flock.Claims.ZK.Interact
import Flock.Firewall.Coins
import Flock.Firewall.Contract

/-!
# The Lean coin check is the deployed coin layer's (`Flock.Firewall.Coins`)

`flock-firewall coins` (`Flock.Firewall.Coins.cmd`) is the check every coin a firewalled masked prover hands on has passed:
`coin_tree::Checked` asks it at `Hello`, at `Hello`'s answer and at each opening, and its verdict is the transport's. Its
verdict over a session is `Session.start` at the answer, then `Session.openAll` over the openings in order. At SHA-512
(`shaH`, the hash `Coins.hk` takes under the nonce prefix) that verdict is the model's: the keyed coin-tree walk of
`CoinLayer` (`treeChk` at `treeD`, `parseD`), checked by `allOk` up to the first opening that fails.

* `treeChk_sha`: one opening. At the place stream `s`'s round `r` with `n` coins names (`placeOf`), under an answer the
  tree's (`answerOK`), the model's check is `Coins.parses` and `Coins.reach` reaching the answer's root, which are
  `Coins.check`'s two tests (`check_isOk`).
* `openAll_isOk`, `verdict_isOk`: a session. When each request names the model's place (`Follows`: its stream's index in
  the tree, its round as `Session.open` counts it, its count with the session stream's `Hello`'s), `openAll` passes
  exactly where `allOk` does, and so does the whole verdict for a session of at least one opening. A malformed answer
  ends the session before any opening, as it fails every opening's `parseD`.
* `coinLayer_verdict`, `coinLayer_refuses`: the same at `CoinLayer.session nr T`'s check (`refP.chk`), at the nonce whose
  bytes `Hello` carries. `coinLayer_refuses` is the Lean check's side of `LeanRelease.OpeningsRefused` (ZKS10's
  `Claims/ZK/Refine.lean`) at SHA-512: where `allOk` of the openings fails, the Lean check refuses. The transport ends
  the session at that refusal (`coin_tree::Checked`), so the frame takes no coins past it.
-/

namespace Claims.ZKMal.CoinOpen

open Claims FlockSoundness.Merkle FlockSoundness.ZK CoinLayer
open _root_.Flock.Firewall

/-! ## SHA-512 as the oracle -/

/-- SHA-512 on bytes: the hash `Coins.hk` takes under its nonce prefix. -/
def shaH (m : Msg) : Msg := (_root_.Flock.Sha512.hash ⟨m.toArray⟩).data.toList

theorem shaH_length (m : Msg) : (shaH m).length = 64 := by
  unfold shaH
  rw [Array.length_toList, ByteArray.size_data]
  exact _root_.Flock.Firewall.Contract.sha512_size _

theorem hk_length (ν m : List UInt8) : (Coins.hk ν m).length = 64 := shaH_length _

theorem hk_eq (ν m : List UInt8) : Coins.hk ν m = shaH (ν ++ m) := rfl

theorem tag_eq : Coins.tag = tagD := rfl

theorem mix_eq (key : ByteArray) (x : Dg) (y : Salt) : (mixD key x y).val = Coins.mix key x.val y.val := rfl

/-- One call of `H` at SHA-512 answers a digest. -/
theorem hq_sha (q : Msg) (k : Dg → OC Bool) :
    ((hq q k).eval shaH).1 = ((k ⟨shaH q, shaH_length q⟩).eval shaH).1 := by
  have hl := shaH_length q
  simp only [hq, OC.eval, hl, ↓reduceDIte]

theorem chain_length (ν : List UInt8) : ∀ (cs : List (List UInt8)) (tn : List UInt8), tn.length = 64 →
    (Coins.chain ν cs tn).length = 64
  | [], _, h => h
  | _ :: _, _, _ => hk_length _ _

theorem walk_length (ν : List UInt8) : ∀ (idx : ℕ) (a : List UInt8) (sibs : List (List UInt8)), a.length = 64 →
    (Coins.walk ν idx a sibs).length = 64
  | _, _, [], h => h
  | idx, _, _ :: r, _ => walk_length ν (idx / 2) _ r (hk_length _ _)

/-! ## The walk -/

theorem chain_cons (ν c : List UInt8) (l : List (List UInt8)) (tn : List UInt8) :
    Coins.chain ν (c :: l) tn = shaH (ν ++ (tagD "chain" ++ c ++ Coins.chain ν l tn)) := rfl

theorem walk_cons (ν : List UInt8) (idx : ℕ) (a s : List UInt8) (r : List (List UInt8)) :
    Coins.walk ν idx a (s :: r) = Coins.walk ν (idx / 2) (shaH (ν ++ (if idx % 2 = 0 then a ++ s else s ++ a))) r := rfl

/-- The chain at SHA-512 is `Coins.chain`. -/
theorem chainK_sha (ν : List UInt8) : ∀ (cs : List Word) (Tn : Dg) (k : Dg → OC Bool),
    ((chainK ν (tagD "chain") encD cs Tn k).eval shaH).1 =
      ((k ⟨Coins.chain ν (cs.map Subtype.val) Tn.val, chain_length ν _ _ Tn.property⟩).eval shaH).1
  | [], _, _ => rfl
  | c :: r, Tn, k => by
    simp only [chainK, chainK_sha ν r Tn, hq_sha, List.map_cons, chain_cons]
    rfl

/-- The path at SHA-512 is `Coins.walk`. -/
theorem walkK_sha (ν : List UInt8) : ∀ (idx : ℕ) (a : Dg) (sibs : List Dg) (k : Dg → OC Bool),
    ((walkK ν nodeD idx a sibs k).eval shaH).1 =
      ((k ⟨Coins.walk ν idx a.val (sibs.map Subtype.val), walk_length ν _ _ _ a.property⟩).eval shaH).1
  | _, _, [], _ => rfl
  | idx, a, s :: r, k => by
    simp only [walkK, hq_sha, walkK_sha ν (idx / 2) _ r k, List.map_cons, walk_cons]
    rfl

/-- The node an opening's walk reaches at SHA-512, in `Coins`' terms: its leaf (`Coins.leaf`), then its path. -/
def reachVal (ν : List UInt8) (key : ByteArray) (op : GK.CoinOpening Dg Word Salt) : List UInt8 :=
  Coins.walk ν op.idx (Coins.leaf ν key op.pos (Coins.chain ν (op.cs.map Subtype.val) op.Tn.val) op.y.val)
    (op.sibs.map Subtype.val)

/-- **An opening's walk** under `Hello`'s answer `ρ`, at SHA-512, reaches `reachVal`. -/
theorem reachK_sha (ν : List UInt8) (ρ : Msg) (op : GK.CoinOpening Dg Word Salt) (k : Dg → OC Bool) :
    ((reachK ν (treeD ρ) op k).eval shaH).1 =
      ((k ⟨reachVal ν (keyD ρ) op, walk_length ν _ _ _ (hk_length _ _)⟩).eval shaH).1 := by
  simp only [reachK, treeD, chainK_sha, hq_sha, walkK_sha]
  refine congrArg (fun d => ((k d).eval shaH).1) (Subtype.ext ?_)
  simp only [reachVal, Coins.leaf, hk_eq, tag_eq, encD, mix_eq]

/-! ## One opening -/

/-- The place `Coins.check` reads for stream `s`'s round `r` with `n` coins of the tree `ct`, as the model's `Place`. -/
def placeOf (ct : _root_.Flock.CoinTree.Spec) (s r n : ℕ) : Place where
  ok := decide (r < 2 ^ ct.roundsLog ∧ n ≤ ct.block ∧ s < ct.streams.size)
  pos := u32be s ++ u32be r
  idx := s * 2 ^ ct.roundsLog + r
  n := n
  depth := ct.roundsLog + (Nat.nextPowerOfTwo ct.streams.size).log2

/-- Coin `j`'s place at statement `x` is `placeOf` at its slot. -/
theorem place_eq {nr : Msg → ℕ → ℕ} {x : Msg} {j : ℕ} {spec : _root_.Flock.SessionSpec}
    {ct : _root_.Flock.CoinTree.Spec} {s r : ℕ} (h : slot x j = some (spec, ct, s, r)) :
    place nr x j = placeOf ct s r (if j = 0 then spec.points * spec.mPts else nr x j) := by
  unfold place
  rw [h]
  all_goals rfl

theorem join_openOf (ct : _root_.Flock.CoinTree.Spec) (s r n : ℕ) (b : List UInt8) :
    join (openOf (placeOf ct s r n) b) = (Coins.parts n (Coins.depth ct) b).join := by
  simp only [join, openOf, placeOf, Coins.parts, Coins.Parts.join, List.flatMap_def, List.map_map]
  rfl

theorem reach_openOf (ct : _root_.Flock.CoinTree.Spec) (ν : List UInt8) (key : ByteArray) (s r n : ℕ) (m : List UInt8) :
    reachVal ν key (openOf (placeOf ct s r n) (m.drop 5)) = Coins.reach ct ν key s r n m := by
  simp only [reachVal, Coins.reach, openOf, placeOf, Coins.parts, List.map_map]
  rfl

theorem parseD_cond (ct : _root_.Flock.CoinTree.Spec) (s r n : ℕ) (ρ m : Msg) (hρ : answerOK ρ = true) :
    ((placeOf ct s r n).ok && answerOK ρ && decide (m = frame ((placeOf ct s r n).n + 16 + 4 * (placeOf ct s r n).depth)
      (join (openOf (placeOf ct s r n) (m.drop 5))))) = Coins.parses ct s r n m := by
  rw [hρ, Bool.and_true, join_openOf]
  rfl

/-- **One opening**: the model's check at the place stream `s`'s round `r` with `n` coins names, under an answer `ρ` the
tree's, at SHA-512, is `Coins.check`'s two tests. -/
theorem treeChk_sha (ct : _root_.Flock.CoinTree.Spec) (ν : List UInt8) (ρ : Msg) (s r n : ℕ) (m : Msg)
    (hρ : answerOK ρ = true) :
    ((treeChk (treeD ρ) ν (rootD ρ) (placeOf ct s r n).pos (placeOf ct s r n).idx (placeOf ct s r n).n
      (placeOf ct s r n).depth (parseD (placeOf ct s r n) ρ m)).eval shaH).1 =
      (Coins.parses ct s r n m && decide (Coins.reach ct ν (keyD ρ) s r n m = (rootD ρ).val)) := by
  unfold parseD
  rw [parseD_cond ct s r n ρ m hρ]
  cases Coins.parses ct s r n m
  · rfl
  · have hc : (openOf (placeOf ct s r n) (m.drop 5)).pos = (placeOf ct s r n).pos ∧
        (openOf (placeOf ct s r n) (m.drop 5)).idx = (placeOf ct s r n).idx ∧
        (openOf (placeOf ct s r n) (m.drop 5)).cs.length = (placeOf ct s r n).n ∧
        (openOf (placeOf ct s r n) (m.drop 5)).sibs.length = (placeOf ct s r n).depth :=
      ⟨rfl, rfl, by simp [openOf, placeOf], by simp [openOf, placeOf]⟩
    simp only [↓reduceIte, treeChk, hc, and_self, reachK_sha, OC.eval, Bool.true_and, ← reach_openOf]
    exact decide_eq_decide.2 Subtype.ext_iff

/-- `Coins.check` passes exactly where the opening parses and walks to the root. -/
theorem check_isOk (ct : _root_.Flock.CoinTree.Spec) (ν : List UInt8) (key : ByteArray) (root : List UInt8) (s r n : ℕ)
    (m : List UInt8) :
    (Coins.check ct ν key root s r n m).isOk =
      (Coins.parses ct s r n m && decide (Coins.reach ct ν key s r n m = root)) := by
  unfold Coins.check
  cases Coins.parses ct s r n m
  · rfl
  · by_cases hr : Coins.reach ct ν key s r n m = root
    · simp [hr, Except.isOk, Except.toBool]
    · simp [hr, Except.isOk, Except.toBool]

/-! ## The answer -/

theorem answer_isOk (a : List UInt8) : (Coins.answer a).isOk = answerOK a := by
  have e : Coins.frame 20 [] = frame 20 [] := rfl
  unfold Coins.answer answerOK
  rw [e]
  unfold bne
  cases (a.length == 325 && a.take 5 == frame 20 []) <;> cases (a.getD 324 0 &&& 0x80 == 0) <;> rfl

theorem answer_ok {a : List UInt8} (h : answerOK a = true) : Coins.answer a = .ok (keyD a, (rootD a).val) := by
  have e : Coins.frame 20 [] = frame 20 [] := rfl
  revert h
  unfold Coins.answer answerOK
  rw [e]
  unfold bne
  cases (a.length == 325 && a.take 5 == frame 20 []) <;> cases (a.getD 324 0 &&& 0x80 == 0) <;> intro h <;>
    first | rfl | nomatch h

/-! ## A session -/

theorem open_isOk (S : Coins.Session) (st : String) (n : ℕ) (m : List UInt8) :
    (S.open st n m).isOk = (Coins.check S.hello.spec S.hello.nonce S.key S.root (S.hello.spec.streams.toList.idxOf st)
      (S.opened.count (S.hello.spec.streams.toList.idxOf st)) (if st == Coins.SESSION then S.hello.links else n) m).isOk := by
  unfold Coins.Session.open
  dsimp only
  by_cases hs : S.hello.spec.streams.toList.idxOf st < S.hello.spec.streams.size
  · simp only [hs, ↓reduceIte]
    cases Coins.check S.hello.spec S.hello.nonce S.key S.root _ _ _ m <;> rfl
  · simp only [hs, ↓reduceIte, check_isOk, Coins.parses, and_false, decide_false, Bool.false_and]
    rfl

theorem open_ok {S : Coins.Session} {st : String} {n : ℕ} {m : List UInt8} {S' : Coins.Session} {c : List UInt8}
    (h : S.open st n m = .ok (S', c)) :
    S' = { S with opened := S.opened ++ [S.hello.spec.streams.toList.idxOf st] } := by
  unfold Coins.Session.open at h
  dsimp only at h
  by_cases hs : S.hello.spec.streams.toList.idxOf st < S.hello.spec.streams.size
  · simp only [hs, ↓reduceIte] at h
    cases hc : Coins.check S.hello.spec S.hello.nonce S.key S.root (S.hello.spec.streams.toList.idxOf st)
        (S.opened.count (S.hello.spec.streams.toList.idxOf st)) (if st == Coins.SESSION then S.hello.links else n) m with
    | ok c' =>
      rw [hc] at h
      cases h
      rfl
    | error e =>
      rw [hc] at h
      cases h
  · simp only [hs, ↓reduceIte] at h
    cases h

/-- **The requests follow the places `pl`**: from position `j` on, with the streams `opened` already opened, request `i`
of `os` names the place `pl (j + i)` by its stream's index in the tree `ct`, its round (as `Session.open` counts it) and
its count (the session stream's `links`, `Hello`'s). -/
def Follows (pl : ℕ → Place) (ct : _root_.Flock.CoinTree.Spec) (links : ℕ) :
    ℕ → List ℕ → List (String × ℕ × Msg) → Prop
  | _, _, [] => True
  | j, opened, (st, n, _) :: os =>
    pl j = placeOf ct (ct.streams.toList.idxOf st) (opened.count (ct.streams.toList.idxOf st))
      (if st == Coins.SESSION then links else n) ∧
    Follows pl ct links (j + 1) (opened ++ [ct.streams.toList.idxOf st]) os

/-- The model's opening check at the places `pl`, under `Hello`'s answer `ρ` and the nonce bytes `ν`. -/
def chkAt (pl : ℕ → Place) (ν : List UInt8) (ρ : Msg) (j : ℕ) (m : Msg) : OC Bool :=
  treeChk (treeD ρ) ν (rootD ρ) (pl j).pos (pl j).idx (pl j).n (pl j).depth (parseD (pl j) ρ m)

/-- **The openings after the answer**: `openAll` passes exactly where `allOk` of the model's check does. -/
theorem openAll_isOk (pl : ℕ → Place) (ρ : Msg) (hρ : answerOK ρ = true) :
    ∀ (S : Coins.Session) (j : ℕ) (os : List (String × ℕ × Msg)), S.key = keyD ρ → S.root = (rootD ρ).val →
      Follows pl S.hello.spec S.hello.links j S.opened os →
      (S.openAll os).isOk = ((allOk (chkAt pl S.hello.nonce ρ) j (os.map fun o => o.2.2)).eval shaH).1
  | _, _, [], _, _, _ => rfl
  | S, j, (st, n, m) :: os, hk, hr, ⟨hp, hf⟩ => by
    have ho := open_isOk S st n m
    rw [check_isOk, hk, hr] at ho
    simp only [List.map_cons, allOk, ocBind_eval]
    rw [chkAt, hp, treeChk_sha _ _ _ _ _ _ _ hρ, ← ho]
    unfold Coins.Session.openAll
    cases h : S.open st n m with
    | error e => rfl
    | ok p =>
      obtain ⟨S', c⟩ := p
      rw [open_ok h]
      exact openAll_isOk pl ρ hρ _ (j + 1) os hk hr hf

/-- **A session's verdict**: past `Hello` (`h`) and its answer `ρ`, the Lean check passes the openings `os`, at least
one, exactly where the model's check passes every one of them in order, at SHA-512. -/
theorem verdict_isOk (h : Coins.Hello) (ρ : Msg) (os : List (String × ℕ × Msg)) (pl : ℕ → Place)
    (hf : Follows pl h.spec h.links 0 [] os) (hne : os ≠ []) :
    ((Coins.Session.start h ρ).bind fun S => S.openAll os).isOk =
      ((allOk (chkAt pl h.nonce ρ) 0 (os.map fun o => o.2.2)).eval shaH).1 := by
  cases hρ : answerOK ρ
  · obtain ⟨o, os, rfl⟩ := List.exists_cons_of_ne_nil hne
    have hs : (Coins.Session.start h ρ).isOk = false := by
      have := answer_isOk ρ
      rw [hρ] at this
      unfold Coins.Session.start
      cases ha : Coins.answer ρ with
      | ok p => rw [ha] at this; simp [Except.isOk, Except.toBool] at this
      | error e => rfl
    have hl : ((Coins.Session.start h ρ).bind fun S => S.openAll (o :: os)).isOk = false := by
      cases hS : Coins.Session.start h ρ with
      | ok S => rw [hS] at hs; simp [Except.isOk, Except.toBool] at hs
      | error e => rfl
    have hp : parseD (pl 0) ρ o.2.2 = none := by
      unfold parseD
      rw [hρ, Bool.and_false, Bool.false_and]
      rfl
    rw [hl]
    simp only [List.map_cons, allOk, ocBind_eval, chkAt, hp, treeChk, OC.eval]
    rfl
  · unfold Coins.Session.start
    rw [answer_ok hρ]
    exact openAll_isOk pl ρ hρ _ 0 os rfl rfl hf

/-! ## At the deployed coin layer -/

/-- The deployed session's opening check is `chkAt` at its places and the nonce's bytes. -/
theorem refP_chk_eq (nr : Msg → ℕ → ℕ) (T : TableData) (x : Msg) (ν : Nonce) (ρ : Msg) :
    (session nr T).refP.chk x ν ρ = chkAt (place nr x) (nb ν) ρ := rfl

/-- **The Lean check is `CoinLayer.session`'s**: when `Hello` carries the nonce `ν`'s bytes and the requests follow the
committed schedule at statement `x` (`Follows` at `place nr x`), the Lean check's verdict over a session of at least one
opening is `allOk` of the deployed check (`refP.chk`) at SHA-512. -/
theorem coinLayer_verdict (nr : Msg → ℕ → ℕ) (T : TableData) (x : Msg) (ν : Nonce) (h : Coins.Hello) (ρ : Msg)
    (os : List (String × ℕ × Msg)) (hν : h.nonce = nb ν) (hf : Follows (place nr x) h.spec h.links 0 [] os)
    (hne : os ≠ []) :
    ((Coins.Session.start h ρ).bind fun S => S.openAll os).isOk =
      ((allOk ((session nr T).refP.chk x ν ρ) 0 (os.map fun o => o.2.2)).eval shaH).1 := by
  rw [refP_chk_eq, ← hν]
  exact verdict_isOk h ρ os (place nr x) hf hne

/-- **The Lean check refuses the openings `LeanRelease.OpeningsRefused` names**, at SHA-512: where `allOk` of the
deployed check fails on the openings, the Lean check's verdict over them is a refusal. -/
theorem coinLayer_refuses (nr : Msg → ℕ → ℕ) (T : TableData) (x : Msg) (ν : Nonce) (h : Coins.Hello) (ρ : Msg)
    (os : List (String × ℕ × Msg)) (hν : h.nonce = nb ν) (hf : Follows (place nr x) h.spec h.links 0 [] os)
    (hm : ((allOk ((session nr T).refP.chk x ν ρ) 0 (os.map fun o => o.2.2)).eval shaH).1 = false) :
    ((Coins.Session.start h ρ).bind fun S => S.openAll os).isOk = false := by
  cases os with
  | nil => simp [allOk, OC.eval] at hm
  | cons o os => rw [coinLayer_verdict nr T x ν h ρ (o :: os) hν hf (List.cons_ne_nil _ _)]; exact hm

end Claims.ZKMal.CoinOpen
