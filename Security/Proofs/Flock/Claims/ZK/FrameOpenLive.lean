import Proofs.Flock.Claims.ZK.FrameOpenCut
import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# Level 0's cut opening at the pads and at zero pads

`ligeritoPZk_cutL0` gives the bodies the masked Ligerito sends at level 0 at any pads. The rep runs it at the frame's
draws' pads `d.h`, the firewall's live opening (`openLive`) at zero pads. Bodies `0` and `k_0 + 3` carry no pad, and
each body `1 ≤ i ≤ k_0 + 2` is one of level 0's masked rounds with its squeeze, the zero-pad round and the round at the
pads of an `L0Round`, so the frame releases the live opening's body as the rep's:

* **A sent level-0 round** (`L0Sent`): two bodies that are an `L0Round`'s zero-pad round and round at the pads, each
  with one squeeze.
* **The rounds sent** (`l0Sent_ood`, `l0Sent_lane`, `l0Sent_pair`): the out-of-domain round with `β_0`'s squeeze, lane
  round `j`'s masked message with its challenge's (`j < k_0`) and the pair's sums with `ρ_0`'s, at any pads zero at
  the round's positions; at zero pads, every body `1 ≤ i ≤ k_0 + 2` (`l0Sent_zero`). The bodies without pads are the
  same at any two pads (`l0BodyAt_free`).
* **At the release** (`L0Sent.release`, `L0Sent.roundHonest`, `L0Sent.roundFrames`): where the frame takes the coins
  before such a round at its position, it releases the zero-pad body as the body at the pads.
* **The two openings, cut** (`ligeritoPZk_cutL0_live`): cut after level 0's `j`-th send, the opening at zero pads and
  the rep's at `d.h` both stop past the openings, having sent bodies that agree where they carry no pad and are sent
  level-0 rounds where they do.
-/

namespace FlockComplete.LigLock

open Flock
open _root_.Flock.Firewall.Frame (Cfg Draws MaskSlot Masker)
open FlockSoundness.Refine (Frame encs)
open Claims.ZKServer (L0Round LeanRelease absPos zeros_get!)

/-- **Two bodies that are a level-0 masked round** at configuration `c` and absorbed position `p`: the zero-pad round
`G` and the round at the pads `H` of an `L0Round` of `n` values, each with its squeeze of `nc` coins of kind `k`. -/
abbrev L0Sent (c : Cfg) (p n : ℕ) (k : UInt8) (nc : ℕ) (Bz Bh : Claims.Msg) : Prop :=
  ∃ G H, L0Round c p n G H ∧ Bz = encs G ++ (Frame.sq k nc).enc ∧ Bh = encs H ++ (Frame.sq k nc).enc

section Sent

variable {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray} {sch : Schedule}
  {t b : Array F128} {lay : Zk.PadLayout} {rep : ℕ} {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64}
  {d : Draws} {mk : Option MaskSlot}

/-- **The out-of-domain round** with `β_0`'s squeeze is body `1`. -/
theorem l0Sent_ood (hk : 0 < lay.k0) (z : Array F128) (hz : z[lay.lgOod 0]! = 0) :
    L0Sent (Cfg.ofDraws lay d mk) (absPos lay (lay.lgOod 0)) 1 1 1
      (l0BodyAt coin p0 P T cap sch t b lay z rep extra nonce 1)
      (l0BodyAt coin p0 P T cap sch t b lay d.h rep extra nonce 1) :=
  ⟨_, _, Claims.ZKServer.l0Round_ood lay d mk hk _ z hz _ (by
      rw [FlockSoundness.Discharge.ZkView.le64_size]; norm_num), l0BodyAt_one, l0BodyAt_one⟩

/-- **Lane round `j`'s masked message** with its challenge's squeeze is body `j + 2`, for `j < k_0`. -/
theorem l0Sent_lane (hk : 0 < lay.k0) {j : ℕ} (hj : j < sch.k0) (hjl : j < lay.k0) (z : Array F128)
    (hz : ∀ q < 4, z[lay.lgMsg j q]! = 0) :
    L0Sent (Cfg.ofDraws lay d mk) (absPos lay (lay.lgMsg j 0)) 4 2 2
      (l0BodyAt coin p0 P T cap sch t b lay z rep extra nonce (j + 2))
      (l0BodyAt coin p0 P T cap sch t b lay d.h rep extra nonce (j + 2)) :=
  ⟨_, _, Claims.ZKServer.l0Round_msg lay d mk hk hjl (roundPoly (tabS coin p0 t j) (wS coin p0 b sch.n j)) z hz,
    l0BodyAt_msg hj, l0BodyAt_msg hj⟩

/-- **The pair's sums** with `ρ_0`'s squeeze are body `k_0 + 2`. -/
theorem l0Sent_pair (hk : 0 < lay.k0) (z : Array F128) (hz : ∀ i < 2, ∀ c < 2, z[lay.lgE i c]! = 0) :
    L0Sent (Cfg.ofDraws lay d mk) (absPos lay (lay.lgE 0 0)) 4 2 2
      (l0BodyAt coin p0 P T cap sch t b lay z rep extra nonce (sch.k0 + 2))
      (l0BodyAt coin p0 P T cap sch t b lay d.h rep extra nonce (sch.k0 + 2)) :=
  ⟨_, _, Claims.ZKServer.l0Round_e lay d mk hk _ _ z hz, l0BodyAt_e, l0BodyAt_e⟩

/-- **At zero pads every masked body is a sent level-0 round**: body `1 ≤ i ≤ k_0 + 2` at `openLive`'s zero pads and
at the draws' pads, its squeeze one coin of kind `1` or two of kind `2`. -/
theorem l0Sent_zero (hk : 0 < lay.k0) (hkk : sch.k0 ≤ lay.k0) (len : ℕ) {i : ℕ} (hi1 : 1 ≤ i)
    (hi : i ≤ sch.k0 + 2) :
    ∃ p n k nc, (k = 2 ∨ (k = 1 ∧ nc = 1)) ∧ nc ≠ 0 ∧
      L0Sent (Cfg.ofDraws lay d mk) p n k nc
        (l0BodyAt coin p0 P T cap sch t b lay (Array.replicate len 0) rep extra nonce i)
        (l0BodyAt coin p0 P T cap sch t b lay d.h rep extra nonce i) := by
  rcases Nat.lt_or_ge i 2 with h2 | h2
  · obtain rfl : i = 1 := by omega
    exact ⟨_, _, _, _, .inr ⟨rfl, rfl⟩, one_ne_zero, l0Sent_ood hk _ (zeros_get! _ _)⟩
  rcases Nat.lt_or_ge i (sch.k0 + 2) with h3 | h3
  · obtain ⟨j, rfl⟩ : ∃ j, i = j + 2 := ⟨i - 2, by omega⟩
    exact ⟨_, _, _, _, .inl rfl, two_ne_zero,
      l0Sent_lane hk (by omega) (by omega) _ (fun _ _ => zeros_get! _ _)⟩
  · obtain rfl : i = sch.k0 + 2 := by omega
    exact ⟨_, _, _, _, .inl rfl, two_ne_zero, l0Sent_pair hk _ (fun _ _ _ _ => zeros_get! _ _)⟩

/-- **The bodies without pads**, the opening's binding with `z`'s squeeze and `ρ_1`'s squeeze, are the same at any
two pads. -/
theorem l0BodyAt_free (z h : Array F128) {i : ℕ} (hi : i = 0 ∨ i = sch.k0 + 3) :
    l0BodyAt coin p0 P T cap sch t b lay z rep extra nonce i =
      l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce i := by
  rcases hi with rfl | rfl
  · rw [l0BodyAt_zero, l0BodyAt_zero]
  · rw [l0BodyAt_r1, l0BodyAt_r1]

end Sent

/-! ## At the release -/

section Release

variable {W R : Type} {L : LeanRelease W R} {x : Claims.Msg} {ρ : R} {hh : List (Claims.Msg × Claims.Msg)}
  {p n : ℕ} {k : UInt8} {Bz Bh : Claims.Msg} {m : Masker}

/-- **The frame releases a sent level-0 round's zero-pad body as its body at the pads**, tagged `0`, where it takes
the coins before the round at level 0's masked phase and the round's position (`LeanRelease.release_l0`). -/
theorem L0Sent.release (st : Bool) (hS : L0Sent (L.cfg x ρ hh) p n k (L.coins x hh.length) Bz Bh)
    (hA : L.answered x ρ hh = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ hh).order.size → (L.cfg x ρ hh).order[q]! < (L.cfg x ρ hh).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x hh.length = 1)) (h0 : ¬ (L.coins x hh.length = 0 ∧ k = 2))
    (hn : L.coins x hh.length < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ hh).order.size) :
    L.release x ρ hh st Bz = 0 :: Bh := by
  obtain ⟨G, H, hR, rfl, rfl⟩ := hS
  exact Claims.ZKServer.LeanRelease.release_l0 L x ρ hh st hR k hA hc hon hi hpos hk h0 hn hb

/-- **A live sent level-0 round of an honest device** (`RoundHonest`, `LeanRelease.roundHonest_l0`). -/
theorem L0Sent.roundHonest (hS : L0Sent (L.cfg x ρ hh) p n k (L.coins x hh.length) Bz Bh)
    (hA : L.answered x ρ hh = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ hh).order.size → (L.cfg x ρ hh).order[q]! < (L.cfg x ρ hh).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x hh.length = 1)) (h0 : ¬ (L.coins x hh.length = 0 ∧ k = 2))
    (hn : L.coins x hh.length < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ hh).order.size) :
    L.RoundHonest x ρ hh Bz (0 :: Bh) := by
  obtain ⟨G, H, hR, rfl, rfl⟩ := hS
  exact Claims.ZKServer.LeanRelease.roundHonest_l0 L x ρ hh hR k hA hc hon hi hpos hk h0 hn hb

/-- **A sent level-0 round in frames** (`RoundFrames`, `LeanRelease.roundFrames_l0`), for a stopped rep or not. -/
theorem L0Sent.roundFrames (st : Bool) (hS : L0Sent (L.cfg x ρ hh) p n k (L.coins x hh.length) Bz Bh)
    (hA : L.answered x ρ hh = .ok m)
    (hc : ∀ q, q < (L.cfg x ρ hh).order.size → (L.cfg x ρ hh).order[q]! < (L.cfg x ρ hh).h.size)
    (hon : m.on = true) (hi : m.inner = none) (hpos : m.pos = p)
    (hk : k = 2 ∨ (k = 1 ∧ L.coins x hh.length = 1)) (h0 : ¬ (L.coins x hh.length = 0 ∧ k = 2))
    (hn : L.coins x hh.length < 2 ^ 64) (hb : p + n ≤ (L.cfg x ρ hh).order.size) :
    L.RoundFrames x ρ hh st Bz (0 :: Bh) := by
  obtain ⟨G, H, hR, rfl, rfl⟩ := hS
  exact Claims.ZKServer.LeanRelease.roundFrames_l0 L x ρ hh st hR k hA hc hon hi hpos hk h0 hn hb

end Release

/-! ## The two openings, cut -/

/-- **The live opening and the rep's, cut at level 0**: answered by `asksCut` from `l` and cut after level 0's `j`-th
send (`j < k_0 + 4`), the masked Ligerito at `openLive`'s zero pads and at the draws' pads `d.h` both stop past the
openings, each having sent its first `j + 1` level-0 bodies after `l`; the two agree at the bodies without pads, and
each body `1 ≤ i ≤ k_0 + 2` is a sent level-0 round, the live opening's its zero-pad round. -/
theorem ligeritoPZk_cutL0_live (coin : ℕ → ℕ → F128) (hs : Hashes PS) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (d : Draws) (mk : Option MaskSlot)
    (rep tPad : ℕ) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (tr : Transcript) (l : Array Claims.Msg) {j : ℕ} (hj : j < sch.k0 + 4) (hk : 1 ≤ sch.k0) (hn : sch.n ≠ 0)
    (hl0 : 0 < lay.k0) (hkk : sch.k0 ≤ lay.k0) :
    let B := fun (h : Array F128) =>
      l0BodyAt coin l.size tr.pending.data.toList T (flat cm0.cap) sch t b lay h rep extra nonce
    let z := Array.replicate lay.len (0 : F128)
    CutStop (fun L => L.toList = l.toList ++ (List.range (j + 1)).map (B z))
        (((ligeritoPZk (asksCut coin (l.size + j)) hs ms sch T cm0 t b lay z rep tPad extra lanePad salt nonce).run
          tr).go l) ∧
      CutStop (fun L => L.toList = l.toList ++ (List.range (j + 1)).map (B d.h))
        (((ligeritoPZk (asksCut coin (l.size + j)) hs ms sch T cm0 t b lay d.h rep tPad extra lanePad salt
          nonce).run tr).go l) ∧
      (∀ i, (i = 0 ∨ i = sch.k0 + 3) → B z i = B d.h i) ∧
      (∀ i, 1 ≤ i → i ≤ sch.k0 + 2 → ∃ p n k nc, (k = 2 ∨ (k = 1 ∧ nc = 1)) ∧ nc ≠ 0 ∧
        L0Sent (Cfg.ofDraws lay d mk) p n k nc (B z i) (B d.h i)) := by
  intro B z
  exact ⟨ligeritoPZk_cutL0 coin hs ms sch T cm0 t b lay _ rep tPad extra lanePad salt nonce tr l hj hk hn,
    ligeritoPZk_cutL0 coin hs ms sch T cm0 t b lay _ rep tPad extra lanePad salt nonce tr l hj hk hn,
    fun _ hi => l0BodyAt_free _ _ hi,
    fun _ hi1 hi => l0Sent_zero hl0 hkk _ hi1 hi⟩

end FlockComplete.LigLock
