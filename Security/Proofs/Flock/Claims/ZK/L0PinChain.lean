import Proofs.Flock.Claims.ZK.L0PinLive
import Proofs.Flock.Claims.ZK.FrameState

/-!
# A live history's level-0 messages are its bodies at the frame's pads

From the binding on, the frame's state moves through level 0's masked rounds by their values (`answered_openBind`,
`answered_l0`), so where the firewall schedules the opening's binding and then level 0's bodies at zero pads, each at
its count, a live history's messages are the binding and the bodies at the rep's draws' pads:

* **The rounds' places** (`l0Off`, `l0N`, `l0K`, `l0Nc`, `nPiop_ood`, `l0_size`, `l0Sent_at`): body `1 ≤ i ≤ k₀ + 2`
  absorbs `l0N i` values from `l0Off i` past the algebraic phase's position, which is level 0's out-of-domain pad's
  absorbed place, and squeezes `l0Nc i` coins of kind `l0K i`.
* **`LeanRelease.l0_chain`**: the binding's message is the binding, and body `i`'s is the body at the draws' pads,
  tagged `0`.
-/

namespace Claims.ZKServer

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (Cfg Masker Draws MaskSlot absorbed nLig)
open FlockSoundness.Refine (Frame encs)

/-! ## The rounds' places -/

/-- Level 0's body `i`'s first value past the algebraic phase's position: the out-of-domain value, then four a round. -/
def l0Off (i : ℕ) : ℕ := if i ≤ 1 then 0 else 1 + 4 * (i - 2)

/-- Level 0's body `i`'s values. -/
def l0N (i : ℕ) : ℕ := if i ≤ 1 then 1 else 4

/-- Level 0's body `i`'s squeeze's kind. -/
def l0K (i : ℕ) : UInt8 := if i ≤ 1 then 1 else 2

/-- Level 0's body `i`'s squeeze's coins. -/
def l0Nc (i : ℕ) : ℕ := if i ≤ 1 then 1 else 2

theorem l0Off_succ {i : ℕ} (hi : 1 ≤ i) : l0Off i + l0N i = l0Off (i + 1) := by
  unfold l0Off l0N
  split_ifs <;> omega

theorem l0K_ok (i : ℕ) :
    (l0K i = 2 ∨ (l0K i = 1 ∧ l0Nc i = 1)) ∧ ¬ (l0Nc i = 0 ∧ l0K i = 2) ∧ l0Nc i < 2 ^ 64 := by
  unfold l0K l0Nc
  split_ifs <;> decide

theorem absPos_ood_add (l : _root_.Flock.Zk.PadLayout) (t : ℕ) :
    absPos l (l.lgOod 0) + t = absPos l (l.lgOod 0 + t) :=
  (absPos_add l (by have := l0_ge l (le_refl (l.lgOod 0)); omega) t).symm

/-- **The algebraic phase ends at level 0's out-of-domain pad's place.** -/
theorem nPiop_ood (l : _root_.Flock.Zk.PadLayout) (d : Draws) (mk : Option MaskSlot) (hk : 0 < l.k0) :
    (Cfg.ofDraws l d mk).nPiop = absPos l (l.lgOod 0) := by
  show (absorbed l).size - nLig l = _
  have hlig : nLig l = l.nOod0 + 4 * l.k0 + 4 := by
    unfold nLig
    rw [if_pos hk]
  rw [absorbed_size, hlig, absPos_ge l (j := l.lgOod 0) (by have := l0_ge l (le_refl (l.lgOod 0)); omega)]
  simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
    _root_.Flock.Zk.PadLayout.fc]
  omega

/-- **Level 0's bodies fill the layout**: after body `i`, the remaining `k₀ + 2 - i` rounds of four values reach the
absorbed values' end. -/
theorem l0_size (l : _root_.Flock.Zk.PadLayout) (hk : 0 < l.k0) (hood : l.nOod0 = 1) {i : ℕ} (hi1 : 1 ≤ i)
    (hi : i ≤ l.k0 + 2) :
    absPos l (l.lgOod 0) + l0Off i + l0N i + 4 * (l.k0 + 2 - i) = (absorbed l).size := by
  have hlig : nLig l = l.nOod0 + 4 * l.k0 + 4 := by
    unfold nLig
    rw [if_pos hk]
  rw [absorbed_size, hlig, absPos_ge l (j := l.lgOod 0) (by have := l0_ge l (le_refl (l.lgOod 0)); omega)]
  simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.zp, _root_.Flock.Zk.PadLayout.lc,
    _root_.Flock.Zk.PadLayout.fc, hood]
  unfold l0Off l0N
  split_ifs <;> omega

end Claims.ZKServer

namespace FlockComplete.LigLock

open Flock
open _root_.Flock.Firewall.Frame (Cfg Draws MaskSlot Masker)
open Claims.ZKServer (absPos zeros_get! l0Off l0N l0K l0Nc absPos_ood_add)

variable {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray} {sch : Schedule}
  {t b : Array F128} {lay : Zk.PadLayout} {rep : ℕ} {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64}
  {d : Draws} {mk : Option MaskSlot}

/-- **Body `1 ≤ i ≤ k₀ + 2` at zero pads is a sent level-0 round at its place** (`l0Sent_zero`, its position and
counts named). -/
theorem l0Sent_at (hk : 0 < lay.k0) (hood : lay.nOod0 = 1) (hkk : sch.k0 = lay.k0) (len : ℕ) {i : ℕ}
    (hi1 : 1 ≤ i) (hi : i ≤ sch.k0 + 2) :
    L0Sent (Cfg.ofDraws lay d mk) (absPos lay (lay.lgOod 0) + l0Off i) (l0N i) (l0K i) (l0Nc i)
      (l0BodyAt coin p0 P T cap sch t b lay (Array.replicate len 0) rep extra nonce i)
      (l0BodyAt coin p0 P T cap sch t b lay d.h rep extra nonce i) := by
  rcases Nat.lt_or_ge i 2 with h2 | h2
  · obtain rfl : i = 1 := by omega
    have e1 : l0Off 1 = 0 := rfl
    have e2 : l0N 1 = 1 := rfl
    have e3 : l0K 1 = 1 := rfl
    have e4 : l0Nc 1 = 1 := rfl
    rw [e1, e2, e3, e4, Nat.add_zero]
    exact l0Sent_ood hk _ (zeros_get! _ _)
  rcases Nat.lt_or_ge i (sch.k0 + 2) with h3 | h3
  · obtain ⟨j, rfl⟩ : ∃ j, i = j + 2 := ⟨i - 2, by omega⟩
    have hp : absPos lay (lay.lgOod 0) + l0Off (j + 2) = absPos lay (lay.lgMsg j 0) := by
      rw [show l0Off (j + 2) = 1 + 4 * j by unfold l0Off; split_ifs <;> omega, absPos_ood_add]
      congr 1
      simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.lgMsg, hood]
      omega
    rw [hp, show l0N (j + 2) = 4 by unfold l0N; split_ifs <;> first | rfl | omega,
      show l0K (j + 2) = 2 by unfold l0K; split_ifs <;> first | rfl | omega,
      show l0Nc (j + 2) = 2 by unfold l0Nc; split_ifs <;> first | rfl | omega]
    exact l0Sent_lane hk (by omega) (by omega) _ (fun _ _ => zeros_get! _ _)
  · obtain rfl : i = sch.k0 + 2 := by omega
    have hp : absPos lay (lay.lgOod 0) + l0Off (sch.k0 + 2) = absPos lay (lay.lgE 0 0) := by
      rw [show l0Off (sch.k0 + 2) = 1 + 4 * sch.k0 by unfold l0Off; split_ifs <;> omega, absPos_ood_add]
      congr 1
      simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.lgE, hood, hkk]
      omega
    rw [hp, show l0N (sch.k0 + 2) = 4 by unfold l0N; split_ifs <;> first | rfl | omega,
      show l0K (sch.k0 + 2) = 2 by unfold l0K; split_ifs <;> first | rfl | omega,
      show l0Nc (sch.k0 + 2) = 2 by unfold l0Nc; split_ifs <;> first | rfl | omega]
    exact l0Sent_pair hk _ (fun _ _ _ _ => zeros_get! _ _)

end FlockComplete.LigLock

namespace Claims.ZKServer.LeanRelease

open _root_.Flock (F128)
open _root_.Flock.Firewall.Frame (Cfg Masker Draws MaskSlot absorbed)
open FlockSoundness.Refine (Frame encs)

/-- **A live history's level-0 messages**: where the frame takes the coins before round `J` outside a masked phase,
past the algebraic phase's opening and before level 0, at the algebraic phase's last position, rounds `J` to
`J + k₀ + 2` are one rep's frame rounds, the stream answers each with its count, and the firewall schedules round `J`
as the opening's binding with `z`'s squeeze and round `J + i` as level 0's body `i` at zero pads with its counts,
round `J`'s message is the binding and round `J + i`'s the body at the rep's draws' pads, tagged `0`. -/
theorem l0_chain {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hown : L.toSrv.OwnHist L.fillOf wk x w ρ κ h) (hle : h.length ≤ L.last x)
    (hrf : L.refusesCoins x ρ h = false) (hP : L.ProgChecks x w)
    {J : ℕ} {sch : _root_.Flock.Schedule} (hJ : 2 ≤ J) (hJh : J + sch.k0 + 2 < h.length)
    (hkd : ∀ j, J ≤ j → j ≤ J + sch.k0 + 2 → L.kindAt x j = .masked)
    (hrep : ∀ j, J ≤ j → j ≤ J + sch.k0 + 2 → L.rep x j = L.rep x J)
    (hcs : ∀ j, J ≤ j → j ≤ J + sch.k0 + 2 → (L.coinsOf x (h.take (j + 1))).size = L.coins x j)
    (hk : 0 < sch.k0) (hlk : (L.lay x (L.rep x J)).k0 = sch.k0) (hood : (L.lay x (L.rep x J)).nOod0 = 1)
    {m : Masker} (hA : L.answered x ρ (h.take J) = .ok m)
    (hon : m.on = false) (hd : m.done = true) (hl0 : m.doneL0 = false) (hi : m.inner = none)
    (hpos : m.pos = (L.cfg x ρ (h.take J)).nPiop)
    (T₀ : F128) (cap₀ : ByteArray) (hcap : cap₀.size < 2 ^ 64)
    (hs0 : L.scheduled x w ρ (h.take J) (computedOf (L.computed x w ρ (h.take J))) =
      encs (openBind T₀ cap₀) ++ (Frame.sq 2 (L.coins x J)).enc)
    (hn0 : L.coins x J ≠ 0) (hn : L.coins x J < 2 ^ 64)
    {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray} {t b : Array F128} {rep : ℕ}
    {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64} (len : ℕ)
    (hsi : ∀ i, 1 ≤ i → i ≤ sch.k0 + 2 →
      L.scheduled x w ρ (h.take (J + i)) (computedOf (L.computed x w ρ (h.take (J + i)))) =
        FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b (L.lay x (L.rep x J)) (Array.replicate len 0) rep
          extra nonce i)
    (hci : ∀ i, 1 ≤ i → i ≤ sch.k0 + 2 → L.coins x (J + i) = l0Nc i) :
    (h[J]'(by omega)).1 = 0 :: (encs (openBind T₀ cap₀) ++ (Frame.sq 2 (L.coins x J)).enc) ∧
      ∀ i (hi1 : 1 ≤ i) (hi2 : i ≤ sch.k0 + 2), (h[J + i]'(by omega)).1 =
        0 :: FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b (L.lay x (L.rep x J))
          (L.drawsAt x ρ (L.rep x J)).h rep extra nonce i := by
  have hlt : ∀ j, j ≤ h.length → (h.take j).length = j := fun j hj => by
    rw [List.length_take]
    omega
  have hc0 : L.cfg x ρ (h.take J) =
      Cfg.ofDraws (L.lay x (L.rep x J)) (L.drawsAt x ρ (L.rep x J)) (L.slot x (L.rep x J)) := by
    unfold cfg
    rw [hlt J (by omega)]
  have hcfg : ∀ j, J ≤ j → j ≤ J + sch.k0 + 2 → L.cfg x ρ (h.take j) = L.cfg x ρ (h.take J) := by
    intro j h1 h2
    unfold cfg
    rw [hlt j (by omega), hlt J (by omega), hrep j h1 h2]
  have hord : ∀ j, J ≤ j → j ≤ J + sch.k0 + 2 →
      ∀ q, q < (L.cfg x ρ (h.take j)).order.size → (L.cfg x ρ (h.take j)).order[q]! < (L.cfg x ρ (h.take j)).h.size := by
    intro j h1 h2
    rw [hcfg j h1 h2, hc0]
    exact ofDraws_order (L.lay x (L.rep x J)) (L.draws ρ (L.rep x J)) (L.slot x (L.rep x J))
  have hosz : (L.cfg x ρ (h.take J)).order.size = (absorbed (L.lay x (L.rep x J))).size := by
    rw [hc0]
    rfl
  have hnp : (L.cfg x ρ (h.take J)).nPiop = absPos (L.lay x (L.rep x J)) ((L.lay x (L.rep x J)).lgOod 0) := by
    rw [hc0]
    exact nPiop_ood _ _ _ (by omega)
  have hsnoc : ∀ j (hj : j < h.length) (M : Msg), (h[j]'hj).1 = M → h.take j ++ [(M, (h[j]'hj).2)] = h.take (j + 1) := by
    intro j hj M e
    rw [List.take_add_one, List.getElem?_eq_getElem hj, ← e]
    rfl
  have hop : ∀ j, 3 ≤ j → j ≤ h.length → L.opens x ρ (h.take j) = true := fun j h3 hj =>
    (L.coins_taken_of_live wk x w ρ κ h hl hle hrf h3 hj).1
  have hlig : (L.cfg x ρ (h.take J)).lig = true := by
    rw [hc0]
    show decide ((L.lay x (L.rep x J)).k0 > 0) = true
    rw [hlk]
    exact decide_eq_true hk
  have hb0 := L.msg_openBind wk x w ρ κ h hl hown J (by omega) (hkd J le_rfl (by omega)) hA T₀ cap₀ hs0 hon hd hlig
    hl0 hi hcap hn0 hn
  refine ⟨hb0, ?_⟩
  have hsz : ∀ i, 1 ≤ i → i ≤ sch.k0 + 2 →
      absPos (L.lay x (L.rep x J)) ((L.lay x (L.rep x J)).lgOod 0) + l0Off i + l0N i +
        4 * ((L.lay x (L.rep x J)).k0 + 2 - i) = (absorbed (L.lay x (L.rep x J))).size :=
    fun i hi1 hi2 => l0_size _ (by omega) hood hi1 (by omega)
  -- the frame's state before body `i`
  have hst : ∀ i, 1 ≤ i → i ≤ sch.k0 + 2 → ∃ m', L.answered x ρ (h.take (J + i)) = .ok m' ∧ m'.on = true ∧
      m'.l0 = true ∧ m'.inner = none ∧
      m'.pos = absPos (L.lay x (L.rep x J)) ((L.lay x (L.rep x J)).lgOod 0) + l0Off i := by
    intro i hi1
    induction i, hi1 using Nat.le_induction with
    | base =>
      intro _
      obtain ⟨b, hb⟩ := L.answered_openBind x w ρ (h.take J) T₀ cap₀ (h[J]'(by omega)).2 hP hA
        (by rw [hlt J (by omega)]; exact hs0) hon hd hlig hl0 hi hcap (by rw [hlt J (by omega)]; exact hn0)
        (by rw [hlt J (by omega)]; exact hn)
        (by
          rw [hpos, hosz, hnp]
          have := hsz 1 le_rfl (by omega)
          omega)
        (by rw [hlt J (by omega)]; exact hkd J le_rfl (by omega))
        (by rw [hlt J (by omega), hrep (J + 1) (by omega) (by omega)])
        (by rw [hlt J (by omega), hsnoc J (by omega) _ hb0]; exact hop (J + 1) (by omega) (by omega))
        (by rw [hlt J (by omega), hsnoc J (by omega) _ hb0]; exact hcs J le_rfl (by omega))
      rw [hlt J (by omega), hsnoc J (by omega) _ hb0] at hb
      refine ⟨_, hb, rfl, rfl, hi, ?_⟩
      show m.pos = _
      rw [hpos, hnp]
      rfl
    | succ i hi1 ih =>
      intro hi2
      obtain ⟨m', hA', hon', hl0', hi', hpos'⟩ := ih (by omega)
      have hS := FlockComplete.LigLock.l0Sent_at (coin := coin) (p0 := p0) (P := P) (T := T) (cap := cap)
        (t := t) (b := b) (rep := rep) (extra := extra) (nonce := nonce) (d := L.drawsAt x ρ (L.rep x J))
        (mk := L.slot x (L.rep x J)) (by omega) hood hlk.symm len hi1 (by omega)
      obtain ⟨G, H, hR, hBz, hBh⟩ := id hS
      have hJi : J + i < h.length := by omega
      obtain ⟨hk2, h0, hnc⟩ := l0K_ok i
      have hc' := hci i hi1 (by omega)
      have hmsg := L.msg_l0 (n := l0N i) (k := l0K i)
        (Bh := FlockComplete.LigLock.l0BodyAt coin p0 P T cap sch t b (L.lay x (L.rep x J))
          (L.drawsAt x ρ (L.rep x J)).h rep extra nonce i) wk x w ρ κ h hl hown (J + i) hJi
        (hkd (J + i) (by omega) (by omega)) hA'
        (by rw [hcfg (J + i) (by omega) (by omega), hc0, hc']; exact hS)
        (hsi i hi1 (by omega)) (hord (J + i) (by omega) (by omega)) hon' hi' hpos' (by rw [hc']; exact hk2)
        (by rw [hc']; exact h0) (by rw [hc']; exact hnc)
        (by
          rw [hcfg (J + i) (by omega) (by omega), hosz]
          have := hsz i hi1 (by omega)
          omega)
      rw [hBh] at hmsg
      obtain ⟨b', hb'⟩ := L.answered_l0 x w ρ (h.take (J + i))
        (by rw [hcfg (J + i) (by omega) (by omega), hc0]; exact hR) (l0K i) (h[J + i]'hJi).2 hP hA'
        (by rw [hlt (J + i) (by omega), hc', hsi i hi1 (by omega), hBz])
        (hord (J + i) (by omega) (by omega)) hon' hi' hl0' hpos'
        (by rw [hlt (J + i) (by omega), hc']; exact hk2) (by rw [hlt (J + i) (by omega), hc']; exact h0)
        (by rw [hlt (J + i) (by omega), hc']; exact hnc)
        (by
          rw [hcfg (J + i) (by omega) (by omega), hosz]
          have := hsz i hi1 (by omega)
          omega)
        (by rw [hlt (J + i) (by omega)]; exact hkd (J + i) (by omega) (by omega))
        (by rw [hlt (J + i) (by omega), hrep (J + i + 1) (by omega) (by omega), hrep (J + i) (by omega) (by omega)])
        (by
          rw [hlt (J + i) (by omega), hc', hsnoc (J + i) hJi (0 :: (encs H ++ (Frame.sq (l0K i) (l0Nc i)).enc)) hmsg]
          exact hop (J + i + 1) (by omega) (by omega))
        (by
          rw [hlt (J + i) (by omega), hc', hsnoc (J + i) hJi (0 :: (encs H ++ (Frame.sq (l0K i) (l0Nc i)).enc)) hmsg]
          rw [← hc']
          exact hcs (J + i) (by omega) (by omega))
      rw [hlt (J + i) (by omega), hc', hsnoc (J + i) hJi (0 :: (encs H ++ (Frame.sq (l0K i) (l0Nc i)).enc)) hmsg] at hb'
      rw [show J + (i + 1) = J + i + 1 by omega]
      refine ⟨_, hb', hon', hl0', hi', ?_⟩
      show absPos _ _ + l0Off i + l0N i = _
      rw [Nat.add_assoc, l0Off_succ hi1]
  intro i hi1 hi2
  obtain ⟨m', hA', hon', -, hi', hpos'⟩ := hst i hi1 hi2
  have hS := FlockComplete.LigLock.l0Sent_at (coin := coin) (p0 := p0) (P := P) (T := T) (cap := cap)
    (t := t) (b := b) (rep := rep) (extra := extra) (nonce := nonce) (d := L.drawsAt x ρ (L.rep x J))
    (mk := L.slot x (L.rep x J)) (by omega) hood hlk.symm len hi1 hi2
  obtain ⟨hk2, h0, hnc⟩ := l0K_ok i
  have hc' := hci i hi1 hi2
  exact L.msg_l0 (n := l0N i) (k := l0K i) wk x w ρ κ h hl hown (J + i) (by omega)
    (hkd (J + i) (by omega) (by omega)) hA'
    (by rw [hcfg (J + i) (by omega) (by omega), hc0, hc']; exact hS)
    (hsi i hi1 hi2) (hord (J + i) (by omega) (by omega)) hon' hi' hpos' (by rw [hc']; exact hk2)
    (by rw [hc']; exact h0) (by rw [hc']; exact hnc)
    (by
      rw [hcfg (J + i) (by omega) (by omega), hosz]
      have := hsz i hi1 hi2
      omega)

end Claims.ZKServer.LeanRelease
