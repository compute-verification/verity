import Proofs.Flock.Claims.ZK.Refine
import Proofs.Flock.Claims.ZK.Coins

/-!
# A history the release's coin check passes, at every tape

The release reads the transport's coin check at the tape's nonce (`LeanRelease.opens`), and at the deployed coin layer
that check is the session's (`LeanRelease.checkIs_coinLayer`, from `LeanRelease.HelloParses` and
`LeanRelease.ReqsFollow`). Those premises leave the frame rounds a history to be live at: at every tape, past the
server's `Hello` and the link, there is a history whose answer and first opening the check passes
(`LeanRelease.opens_coinPass`), so the coins the frame takes there are the frame's own (`answered = frameAnswer`). A
premise that asked the openings to pass under every nonce would leave none, since the walk keys every hash with it.

* `ansOf ν p b k`: `Hello`'s answer, `Resp::Coins` of 20 words: the root the opening `b` of the place `p` walks to under
  the nonce bytes `ν` and the session key `k`, then `k`. The verifier builds its tree after `Hello`, under its nonce,
  so its answer reads `ν`.
* `openMsg p b`: `Resp::Coins` of the opening's `n + 16 + 4·depth` words `b`, for any `b` that reads back as itself at
  `p` (`Reads`); the all-zero opening does (`reads_zeros`).
* `chkAt_open`: the session's check passes that opening under that answer, at SHA-512.
* `LeanRelease.opens_coinPass`: at coin `0`'s place (the `session` stream's round `0`), where it is in the schedule and
  the protocol has a coin, the release's check passes `[(m₀, ansOf …), (m₁, openMsg …)]` at every tape, every key with
  bit 2047 clear and every opening that reads back; `opens_coinPass_zeros` at the all-zero key and opening.

Whether the frame round there is live reads nothing more of the check: it is the frame's own taking of the coins
(`frameAnswer`) and the round (`Frame.round`). `HelloParses` and `ReqsFollow` are hypotheses here, as
`Flock.zk_server_lean_coinLayer` takes them; at the session loop both are theorems (`Claims/ZK/SessionLean.lean`).
-/

namespace Claims.ZKMal.CoinPass

open Claims FlockSoundness.Merkle FlockSoundness.ZK CoinLayer

/-- The opening of the place `p` whose words are all zero. -/
def zeros (p : Place) : List UInt8 := List.replicate (16 * p.n + 256 + 64 * p.depth) 0

/-- The all-zero session key. -/
def zeroKey : List UInt8 := List.replicate 256 0

/-- **The opening `b` reads back as itself** at the place `p`: its words, read at `p` (`openOf`) and joined again, are
`b`. -/
def Reads (p : Place) (b : List UInt8) : Prop := join (openOf p b) = b

/-- `Resp::Coins` of the opening `b` of the place `p`: its `n + 16 + 4·depth` words. -/
def openMsg (p : Place) (b : List UInt8) : Msg := frame (p.n + 16 + 4 * p.depth) b

/-- **`Hello`'s answer that the opening `b` of the place `p` opens** under the nonce bytes `ν` and the session key `k`:
`Resp::Coins` of 20 words, the root the opening's walk reaches, then `k`. -/
def ansOf (ν : List UInt8) (p : Place) (b k : List UInt8) : Msg :=
  frame 20 [] ++ CoinOpen.reachVal ν ⟨k.toArray⟩ (openOf p b) ++ k

/-! ## The all-zero opening reads back -/

theorem fix_zeros (n a : ℕ) : (fix n (List.replicate a (0 : UInt8))).val = List.replicate n (0 : UInt8) := by
  show (List.replicate a (0 : UInt8) ++ List.replicate n (0 : UInt8)).take n = List.replicate n (0 : UInt8)
  rw [List.replicate_append_replicate, List.take_replicate, Nat.min_eq_left (Nat.le_add_left n a)]

theorem fix_drop_zeros (n a k : ℕ) :
    (fix n ((List.replicate a (0 : UInt8)).drop k)).val = List.replicate n (0 : UInt8) := by
  rw [List.drop_replicate]
  exact fix_zeros n (a - k)

theorem flatMap_fix_zeros (k a : ℕ) (f : ℕ → ℕ) : ∀ l : List ℕ,
    (l.map fun i => fix k ((List.replicate a (0 : UInt8)).drop (f i))).flatMap Subtype.val =
      List.replicate (k * l.length) (0 : UInt8)
  | [] => by simp
  | i :: l => by
    rw [List.map_cons, List.flatMap_cons, fix_drop_zeros, flatMap_fix_zeros k a f l, List.replicate_append_replicate,
      List.length_cons]
    congr 1
    ring

theorem reads_zeros (p : Place) : Reads p (zeros p) := by
  have hcs : (openOf p (zeros p)).cs.flatMap Subtype.val = List.replicate (16 * p.n) (0 : UInt8) := by
    have h := flatMap_fix_zeros 16 (16 * p.n + 256 + 64 * p.depth) (fun i => 16 * i) (List.range p.n)
    rw [List.length_range] at h
    exact h
  have htn : (openOf p (zeros p)).Tn.val = List.replicate 64 (0 : UInt8) := fix_drop_zeros 64 _ _
  have hy : (openOf p (zeros p)).y.val = List.replicate 192 (0 : UInt8) := fix_drop_zeros 192 _ _
  have hsib : (openOf p (zeros p)).sibs.flatMap Subtype.val = List.replicate (64 * p.depth) (0 : UInt8) := by
    have h := flatMap_fix_zeros 64 (16 * p.n + 256 + 64 * p.depth) (fun i => 16 * p.n + 256 + 64 * i)
      (List.range p.depth)
    rw [List.length_range] at h
    exact h
  unfold Reads join
  rw [hcs, htn, hy, hsib, List.replicate_append_replicate, List.replicate_append_replicate,
    List.replicate_append_replicate]
  unfold zeros
  congr 1

/-! ## The answer -/

theorem u32le_length (v : ℕ) : (u32le v).length = 4 := by
  simp [u32le, u32be]

theorem frame_drop (ws : ℕ) (b : List UInt8) : (frame ws b).drop 5 = b := by
  show (u32le ws ++ b).drop 4 = b
  exact List.drop_left' (u32le_length ws)

theorem frame20_length : (frame 20 []).length = 5 := by
  simp [frame, u32le_length]

theorem reach_length (ν : List UInt8) (key : ByteArray) (op : GK.CoinOpening Dg Word Salt) :
    (CoinOpen.reachVal ν key op).length = 64 :=
  CoinOpen.walk_length ν _ _ _ (CoinOpen.hk_length _ _)

theorem getD_append_len (l' : List UInt8) (n : ℕ) (d : UInt8) :
    ∀ l : List UInt8, (l ++ l').getD (l.length + n) d = l'.getD n d
  | [] => by simp
  | a :: l => by
    rw [List.cons_append, List.length_cons, Nat.add_right_comm, List.getD_cons_succ]
    exact getD_append_len l' n d l

theorem ans_head (ν : List UInt8) (p : Place) (b k : List UInt8) :
    (frame 20 [] ++ CoinOpen.reachVal ν ⟨k.toArray⟩ (openOf p b)).length = 69 := by
  rw [List.length_append, frame20_length, reach_length]

theorem ans_length (ν : List UInt8) (p : Place) (b k : List UInt8) (hk : k.length = 256) :
    (ansOf ν p b k).length = 325 := by
  unfold ansOf
  rw [List.length_append, ans_head, hk]

theorem ans_take (ν : List UInt8) (p : Place) (b k : List UInt8) : (ansOf ν p b k).take 5 = frame 20 [] := by
  unfold ansOf
  rw [List.append_assoc]
  exact List.take_left' frame20_length

theorem ans_drop5 (ν : List UInt8) (p : Place) (b k : List UInt8) :
    (ansOf ν p b k).drop 5 = CoinOpen.reachVal ν ⟨k.toArray⟩ (openOf p b) ++ k := by
  unfold ansOf
  rw [List.append_assoc]
  exact List.drop_left' frame20_length

theorem ans_drop69 (ν : List UInt8) (p : Place) (b k : List UInt8) : (ansOf ν p b k).drop 69 = k :=
  List.drop_left' (ans_head ν p b k)

theorem ans_getD (ν : List UInt8) (p : Place) (b k : List UInt8) : (ansOf ν p b k).getD 324 0 = k.getD 255 0 := by
  have h := getD_append_len k 255 0 (frame 20 [] ++ CoinOpen.reachVal ν ⟨k.toArray⟩ (openOf p b))
  rw [ans_head] at h
  exact h

/-- **The answer is the coin tree's** (`answerOK`): 20 words, the key's bit 2047 clear. -/
theorem ansOf_ok (ν : List UInt8) (p : Place) (b k : List UInt8) (hk : k.length = 256)
    (hbit : k.getD 255 0 &&& 0x80 = 0) : answerOK (ansOf ν p b k) = true := by
  unfold answerOK
  rw [ans_length ν p b k hk, ans_take, ans_getD, hbit]
  rfl

theorem keyD_ans (ν : List UInt8) (p : Place) (b k : List UInt8) (hk : k.length = 256) :
    keyD (ansOf ν p b k) = ⟨k.toArray⟩ := by
  unfold keyD
  rw [ans_drop69, List.take_of_length_le (Nat.le_of_eq hk)]

theorem rootD_ans (ν : List UInt8) (p : Place) (b k : List UInt8) :
    (rootD (ansOf ν p b k)).val = CoinOpen.reachVal ν ⟨k.toArray⟩ (openOf p b) := by
  show ((ansOf ν p b k).drop 5 ++ List.replicate 64 (0 : UInt8)).take 64 = _
  rw [ans_drop5, List.append_assoc]
  exact List.take_left' (reach_length _ _ _)

theorem zeroKey_length : zeroKey.length = 256 := by
  unfold zeroKey
  exact List.length_replicate

theorem zeroKey_bit : zeroKey.getD 255 0 &&& 0x80 = 0 := by
  rfl

/-! ## The opening passes -/

/-- An opening that reads back parses at its place, under an answer that is the coin tree's. -/
theorem parseD_open (p : Place) (ρ b : Msg) (hp : p.ok = true) (hρ : answerOK ρ = true) (hb : Reads p b) :
    parseD p ρ (openMsg p b) = some (openOf p b) := by
  have hd : (openMsg p b).drop 5 = b := frame_drop _ _
  have hj : join (openOf p b) = b := hb
  unfold parseD
  rw [hd, hj, hp, hρ]
  have hm : decide (openMsg p b = frame (p.n + 16 + 4 * p.depth) b) = true := decide_eq_true rfl
  rw [hm]
  rfl

/-- **The session's check passes the opening** at the place `pl j` in the schedule, under the answer it opens, at
SHA-512. -/
theorem chkAt_open (pl : ℕ → Place) (j : ℕ) (hok : (pl j).ok = true) (ν b k : List UInt8) (hb : Reads (pl j) b)
    (hk : k.length = 256) (hbit : k.getD 255 0 &&& 0x80 = 0) :
    ((CoinOpen.chkAt pl ν (ansOf ν (pl j) b k) j (openMsg (pl j) b)).eval CoinOpen.shaH).1 = true := by
  unfold CoinOpen.chkAt
  rw [parseD_open (pl j) _ b hok (ansOf_ok ν (pl j) b k hk hbit) hb]
  have hc : (openOf (pl j) b).pos = (pl j).pos ∧ (openOf (pl j) b).idx = (pl j).idx ∧
      (openOf (pl j) b).cs.length = (pl j).n ∧ (openOf (pl j) b).sibs.length = (pl j).depth :=
    ⟨rfl, rfl, by simp [openOf], by simp [openOf]⟩
  have hv : CoinOpen.reachVal ν (keyD (ansOf ν (pl j) b k)) (openOf (pl j) b) = (rootD (ansOf ν (pl j) b k)).val :=
    (congrArg (fun key => CoinOpen.reachVal ν key (openOf (pl j) b)) (keyD_ans ν (pl j) b k hk)).trans
      (rootD_ans ν (pl j) b k).symm
  simp only [treeChk, hc, and_self, ↓reduceIte, CoinOpen.reachK_sha, OC.eval, decide_eq_true_eq]
  exact Subtype.ext hv

end Claims.ZKMal.CoinPass

namespace Claims.ZKServer

open Claims Claims.ZKMal

namespace LeanRelease

/-- **The release's coin check passes a history at every tape** at the deployed coin layer: where the coin check reads
the session's `Hello` with the tape's nonce (`HelloParses`), the transport requests the openings at the
schedule's places (`ReqsFollow`), coin `0`'s place is in the schedule and the protocol has a coin, the check passes
`Hello`'s answer `ansOf` (the root the opening `b` walks to under the tape's nonce and the key `k`) and the opening
`openMsg` of coin `0`, at every tape, key with bit 2047 clear and opening that reads back. There the coins the frame
takes are its own (`frameAnswer`). -/
theorem opens_coinPass (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) (x : Msg)
    (hP : HelloParses nr T L x) (hF : ReqsFollow nr T L x) (hok : (CoinLayer.place nr x 0).ok = true)
    (h0 : 0 < refK x) (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape) (m₀ m₁ : Msg)
    (b k : List UInt8) (hb : CoinPass.Reads (CoinLayer.place nr x 0) b) (hk : k.length = 256)
    (hbit : k.getD 255 0 &&& 0x80 = 0) :
    L.opens x (ν, t) [(m₀, CoinPass.ansOf (CoinLayer.nb ν) (CoinLayer.place nr x 0) b k),
        (m₁, CoinPass.openMsg (CoinLayer.place nr x 0) b)] = true ∧
      L.answered x (ν, t) [(m₀, CoinPass.ansOf (CoinLayer.nb ν) (CoinLayer.place nr x 0) b k),
          (m₁, CoinPass.openMsg (CoinLayer.place nr x 0) b)] =
        L.frameAnswer x (ν, t) [(m₀, CoinPass.ansOf (CoinLayer.nb ν) (CoinLayer.place nr x 0) b k),
          (m₁, CoinPass.openMsg (CoinLayer.place nr x 0) b)] := by
  have hC := checkIs_coinLayer nr T L x hP hF ν t m₀ (CoinPass.ansOf (CoinLayer.nb ν) (CoinLayer.place nr x 0) b k)
    [(m₁, CoinPass.openMsg (CoinLayer.place nr x 0) b)] (by simp only [List.length_cons, List.length_nil]; omega)
  have ho : L.opens x (ν, t) [(m₀, CoinPass.ansOf (CoinLayer.nb ν) (CoinLayer.place nr x 0) b k),
      (m₁, CoinPass.openMsg (CoinLayer.place nr x 0) b)] = true := by
    refine hC.trans ?_
    try rw [CoinOpen.refP_chk_eq]
    refine (allOk_eval CoinOpen.shaH _ _ 0).2 fun i hi => ?_
    have hi0 : i = 0 := by simpa using hi
    subst hi0
    exact CoinPass.chkAt_open (CoinLayer.place nr x) 0 hok (CoinLayer.nb ν) b k hb hk hbit
  exact ⟨ho, L.answered_of_opens x (ν, t) _ ho⟩

/-- **`opens_coinPass` at the all-zero key and opening.** -/
theorem opens_coinPass_zeros (nr : Msg → ℕ → ℕ) (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session nr T).refP.Nonce × (CoinLayer.session nr T).refP.Rand)) (x : Msg)
    (hP : HelloParses nr T L x) (hF : ReqsFollow nr T L x) (hok : (CoinLayer.place nr x 0).ok = true)
    (h0 : 0 < refK x) (ν : (CoinLayer.session nr T).Nonce) (t : (CoinLayer.session nr T).Tape) (m₀ m₁ : Msg) :
    L.opens x (ν, t) [(m₀, CoinPass.ansOf (CoinLayer.nb ν) (CoinLayer.place nr x 0)
        (CoinPass.zeros (CoinLayer.place nr x 0)) CoinPass.zeroKey),
      (m₁, CoinPass.openMsg (CoinLayer.place nr x 0) (CoinPass.zeros (CoinLayer.place nr x 0)))] = true :=
  (opens_coinPass nr T L x hP hF hok h0 ν t m₀ m₁ _ _ (CoinPass.reads_zeros _) CoinPass.zeroKey_length
    CoinPass.zeroKey_bit).1

end LeanRelease

end Claims.ZKServer
