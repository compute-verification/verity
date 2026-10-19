import Proofs.Flock.Claims.ZK.Os
import Proofs.Flock.Claims.Complete.Zk.Tape

/-!
# The frame's draws as the masked prover's tape

The deployed frame draws each rep's pads itself, one read of `drawBytes` bytes at the rep's start (`Frame.drawsWith`,
parsed by `Draws.ofBytes`), and the workers read no pads. Read as the masked prover's tape (`FlockComplete.TapeZk`), the
frames' bytes give the masked prover exactly the frames' pads:

* **The tape of the frames' draws** (`tapeOfDraws`): table `i`'s rep `r`, which is frame rep `rep i r`, reads its `h`,
  `μ` and their padding as the words of that frame's bytes, its `τ` salt and its pads tree's leaf salts as the frame's,
  and every other entry from a given tape (the worker's).
* **Its pads are the frame's** (`padsOf_tapeOfDraws`, `padsOf_drawsAt`): `padsOf` at the rep's layout is the frame's
  `Draws.ofBytes`, `h_ab` and `w` included, wherever the bytes reach `τ`'s salt, as a frame's draws of `drawBytes` bytes
  do (`tau_le_drawBytes`); the leaf salts the masked prover commits under are the frame's (`innerSalt_fit`).
* **At the release** (`cfg_tapeOfDraws`): at a round of frame rep `rep i r`, the configuration's pads and the pads
  tree's rows are those of the masked prover's table `i`, rep `r`. **At the operating system's bytes**
  (`padsOf_os`): with `O.release L`'s frames reading the tape's streams (`OsTape`), every stream is `drawBytes` long,
  so the pads agree at every rep with a stream.
-/

namespace Claims.ZKServer

open _root_.Flock.Firewall.Frame (Draws drawBytes codeLen SALT)

/-- `n` bytes of exactly `n` bytes are those bytes. -/
theorem fitBytes_of_size (n : ℕ) (b : ByteArray) (h : b.size = n) : FlockComplete.fitBytes n b = b := by
  subst h
  unfold FlockComplete.fitBytes
  rw [Nat.min_self, Nat.sub_self, ByteArray.extract_zero_size]
  apply ByteArray.ext
  simp [ByteArray.data_append]

/-- A rep's draws as the masked prover's pads (`zk_veil::Pads`): all but the pads tree's leaf salts. -/
def drawsPads (d : Draws) : FlockComplete.PadsZk := ⟨d.h, d.mu, d.padH, d.padMu, d.tauSalt⟩

/-- Word `k` of `b` from byte `o` (`Frame.words`' `k`-th). -/
def wordAt (b : ByteArray) (o k : ℕ) : _root_.Flock.F128 :=
  ⟨_root_.Flock.u64OfLE b (o + 16 * k), _root_.Flock.u64OfLE b (o + 16 * k + 8)⟩

/-- Where `τ`'s salt starts in the draws of layout `l` (`Draws.ofBytes`). -/
def tauAt (l : _root_.Flock.Zk.PadLayout) : ℕ := 32 * l.len + 32 * _root_.Flock.Zk.K_PAD

/-- **The masked prover's tape of the frames' draws**: table `i`'s rep `r` is frame rep `rep i r`, of layout
`lay (rep i r)` and bytes `B (rep i r)`. Its `h`, `μ` and their padding are those bytes' words, as `Draws.ofBytes` reads
them (`h` before `h_ab` and `w` are set, which `padsOf` sets as the frame does), its `τ` salt and leaf `j`'s salt are the
frame's, and the rest is `t`'s. -/
def tapeOfDraws (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) : FlockComplete.TapeZk :=
  { t with
    h := fun i r => wordAt (B (rep i r)) 0
    mu := fun i r => wordAt (B (rep i r)) (16 * (lay (rep i r)).len)
    padH := fun i r => wordAt (B (rep i r)) (32 * (lay (rep i r)).len)
    padMu := fun i r => wordAt (B (rep i r)) (32 * (lay (rep i r)).len + 16 * _root_.Flock.Zk.K_PAD)
    tauSalt := fun i r => (Draws.ofBytes (lay (rep i r)) (B (rep i r))).tauSalt
    innerSalt := fun i r j => (Draws.ofBytes (lay (rep i r)) (B (rep i r))).salts.extract (SALT * j) (SALT * j + SALT) }

/-- The draws' `τ` salt is `TAU_SALT_BYTES` long wherever the bytes reach its end. -/
theorem tauSalt_size (l : _root_.Flock.Zk.PadLayout) (b : ByteArray)
    (hb : tauAt l + _root_.Flock.Zk.TAU_SALT_BYTES ≤ b.size) :
    (Draws.ofBytes l b).tauSalt.size = _root_.Flock.Zk.TAU_SALT_BYTES := by
  show (b.extract (tauAt l) (tauAt l + _root_.Flock.Zk.TAU_SALT_BYTES)).size = _
  simp only [ByteArray.size_extract]
  omega

/-- A rep's draws of `drawBytes` bytes reach the end of `τ`'s salt. -/
theorem tau_le_drawBytes (l : _root_.Flock.Zk.PadLayout) :
    tauAt l + _root_.Flock.Zk.TAU_SALT_BYTES ≤ drawBytes l := by
  unfold tauAt drawBytes
  omega

/-- **The masked prover's pads are the frame's**: table `i`'s rep `r`'s pads on the tape of the frames' draws, at its
frame rep's layout, are that frame's draws' pads, wherever its bytes reach the end of `τ`'s salt. -/
theorem padsOf_tapeOfDraws (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hb : tauAt (lay (rep i r)) + _root_.Flock.Zk.TAU_SALT_BYTES ≤ (B (rep i r)).size) :
    FlockComplete.padsOf (tapeOfDraws t lay B rep) i r (lay (rep i r)) =
      drawsPads (Draws.ofBytes (lay (rep i r)) (B (rep i r))) := by
  have ht : FlockComplete.fitBytes _root_.Flock.Zk.TAU_SALT_BYTES ((tapeOfDraws t lay B rep).tauSalt i r) =
      (Draws.ofBytes (lay (rep i r)) (B (rep i r))).tauSalt :=
    fitBytes_of_size _ _ (tauSalt_size _ _ hb)
  simp only [FlockComplete.padsOf, ht]
  rfl

/-- **The leaf salts the masked prover commits under are the frame's**: at a leaf of the pads tree (`j < codeLen`), on
draws of `drawBytes` bytes, the masked prover's `fitBytes 192` of its tape's leaf salt (`innerCommitP`) is that salt,
the frame's leaf `j` salt (`padsRoot`). -/
theorem innerSalt_fit (t : FlockComplete.TapeZk) (lay : ℕ → _root_.Flock.Zk.PadLayout) (B : ℕ → ByteArray)
    (rep : ℕ → ℕ → ℕ) (i r j : ℕ) (hs : (B (rep i r)).size = drawBytes (lay (rep i r)))
    (hj : j < codeLen (lay (rep i r))) :
    FlockComplete.fitBytes 192 ((tapeOfDraws t lay B rep).innerSalt i r j) =
      (Draws.ofBytes (lay (rep i r)) (B (rep i r))).salts.extract (SALT * j) (SALT * j + SALT) := by
  apply fitBytes_of_size
  show ((B (rep i r)).extract _ _ |>.extract _ _).size = _
  simp only [ByteArray.size_extract, hs, drawBytes, SALT]
  omega

/-- **`padsOf_tapeOfDraws` at a release's draws** of `drawBytes` bytes: table `i`'s rep `r`'s pads on the tape of the
release's draws at `ρ` are its frame rep's draws' pads (`drawsAt`). -/
theorem padsOf_drawsAt {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (t : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hs : (L.draws ρ (rep i r)).size = drawBytes (L.lay x (rep i r))) :
    FlockComplete.padsOf (tapeOfDraws t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r)) =
      drawsPads (L.drawsAt x ρ (rep i r)) :=
  padsOf_tapeOfDraws t (L.lay x) (L.draws ρ) rep i r (by rw [hs]; exact tau_le_drawBytes _)

/-- **The release's configuration carries the masked prover's pads**: at a round of frame rep `rep i r`, on draws of
`drawBytes` bytes, the configuration's pads and the pads tree's two rows are table `i`'s rep `r`'s on the tape of the
release's draws. -/
theorem cfg_tapeOfDraws {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (hs : List (Msg × Msg))
    (t : FlockComplete.TapeZk) (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hr : L.rep x hs.length = rep i r)
    (hz : (L.draws ρ (rep i r)).size = drawBytes (L.lay x (rep i r))) :
    (L.cfg x ρ hs).h = (FlockComplete.padsOf (tapeOfDraws t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).h ∧
    (L.cfg x ρ hs).rowH =
      (FlockComplete.padsOf (tapeOfDraws t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).h ++
        (FlockComplete.padsOf (tapeOfDraws t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).padH ∧
    (L.cfg x ρ hs).rowMu =
      (FlockComplete.padsOf (tapeOfDraws t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).mu ++
        (FlockComplete.padsOf (tapeOfDraws t (L.lay x) (L.draws ρ) rep) i r (L.lay x (rep i r))).padMu := by
  rw [padsOf_drawsAt L x ρ t rep i r hz]
  unfold LeanRelease.cfg
  rw [hr]
  exact ⟨rfl, rfl, rfl⟩

/-- **At the operating system's bytes**: with `O.release L`'s frames reading the tape's streams, table `i`'s rep `r`'s
pads on the tape of those streams are its frame rep's draws' pads, at every frame rep with a stream. -/
theorem padsOf_os {W R : Type} (L : LeanRelease W R) (x : Msg) (O : OsTape R (L.lay x)) (ρ : R)
    (t : FlockComplete.TapeZk) (rep : ℕ → ℕ → ℕ) (i r : ℕ) (hk : rep i r < O.reps) :
    FlockComplete.padsOf (tapeOfDraws t (L.lay x) (O.frameBytes ρ) rep) i r (L.lay x (rep i r)) =
      drawsPads ((O.release L).drawsAt x ρ (rep i r)) :=
  padsOf_drawsAt (O.release L) x ρ t rep i r (O.frameBytes_size ρ ⟨rep i r, hk⟩)

end Claims.ZKServer
