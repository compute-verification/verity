module

public import Proofs.Flock.SessionLoop.Fill
public import Proofs.Flock.Claims.ZK.FrameProg

/-!
# A scheduled round with values in its free words passes the frame's checks

`Loop.RoundsPass` at the device `Loop.valsDev vs L` asks that each frame round's device round pass the frame's checks.
Outside the inner proof that device round is the scheduled round with a value of `vs` written into each word the frame
frees in it (`Loop.fill`). Where the frame takes the scheduled round, of frames none of which is the inner proof's label
(`Sched.MidAll`), at the state the stream's coins leave, at a rep not stopped:

* `fill_frames`: the filled round is the bytes of frames alike to the scheduled round's, and the scheduled round outside
  the words the frame frees;
* `check_fill`: the frame takes the filled round, freeing the same words, and its check passes it against the scheduled
  round;
* `LeanRelease.roundOk_valsDev`: so the frame round passes the frame's checks at `valsDev` wherever `cross` passes its
  device round. At the inner proof's round 0 or 2 `valsDev` would write `final_c'` or `τ`; outside the inner proof neither
  applies, as the filled round is not the inner proof's round 0 (`Loop.roundFree_own0_inner`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS roundFree passes check standIn firstDiff inFree)
open FlockSoundness.Refine (Frame encs toBytes_length)

namespace HonestPass

theorem setWord_getElem?_out {L : List UInt8} {o i : ℕ} (a : F128) (h : o + 16 ≤ L.length)
    (hi : i < o ∨ o + 16 ≤ i) : (Loop.setWord L o a)[i]? = L[i]? := by
  rcases hi with hi | hi
  · exact Loop.setWord_getElem?_lt a (by omega) hi
  · unfold Loop.setWord
    have h1 : (L.take o).length = o := by rw [List.length_take]; omega
    rw [List.getElem?_append_right (by rw [h1]; omega), h1,
      List.getElem?_append_right (by rw [toBytes_length]; omega), toBytes_length, List.getElem?_drop]
    congr 1
    omega

/-- **`fill` keeps every byte outside the words it sets.** -/
theorem fill_getElem?_out (L : List UInt8) (fr : Array ℕ) (vals : ℕ → F128) (i : ℕ)
    (hb : ∀ k < fr.size, fr[k]! + 16 ≤ L.length) (hi : ∀ k < fr.size, i < fr[k]! ∨ fr[k]! + 16 ≤ i) :
    ∀ m, m ≤ fr.size → ((List.range m).foldl (fun L k => Loop.setWord L fr[k]! (vals k)) L)[i]? = L[i]?
  | 0, _ => rfl
  | m + 1, hm => by
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
      setWord_getElem?_out _ (by rw [Loop.fill_length L fr vals hb m (by omega)]; exact hb m (by omega))
        (hi m (by omega)),
      fill_getElem?_out L fr vals i hb hi m (by omega)]

theorem firstDiff_eq_none {v r : ByteArray} {free : Array ℕ}
    (h : ∀ i < v.size, inFree free i = false → v.get! i = r.get! i) : firstDiff v r free = none := by
  unfold firstDiff
  split
  · rfl
  · refine List.find?_eq_none.mpr fun i hi => ?_
    have hi' := List.mem_range.1 hi
    cases hf : inFree free i
    · simp [h i hi' hf]
    · simp

/-- **The frame's check passes a device round that `passes` and that `round` takes**, at a rep not stopped. -/
theorem check_of_passes {c : Cfg} {s : Masker} {v r : ByteArray} {n : ℕ} {out : ByteArray} {s' : Masker}
    (hs : s.stopped = false) (hp : passes c s v r n = .ok none)
    (h : _root_.Flock.Firewall.Frame.round c s v n = .ok (out, s')) : check c s v r n = .ok (out, s', none) := by
  unfold check
  simp only [hs, Bool.false_eq_true, ↓reduceIte]
  rw [hp]
  show (match _root_.Flock.Firewall.Frame.round c s v n with
    | .ok (out, s') => (Except.ok (out, s', none) : Except String _)
    | .error e => if e.startsWith _root_.Flock.Firewall.Frame.STOP then throw e else standIn c s r n e) = _
  rw [h]

/-- **A scheduled round with values in its free words**: where the frame takes the scheduled round of frames `fs`
(none the inner proof's label) and its squeeze, freeing the words at `fr`, the round `fill` writes `vals` into is as long,
keeps every byte outside those words, and is the bytes of frames alike to `fs` and the same squeeze. -/
theorem fill_frames {c : Cfg} {m : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} (hin : m.inner = none)
    (hM : Sched.MidAll fs) (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64)
    {o : ByteArray} {t : Masker} {fr : Array ℕ}
    (hq : roundFree c m (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok (o, t, fr))
    (vals : ℕ → F128) :
    (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).length = (encs fs ++ (Frame.sq k n).enc).length ∧
      (∀ i, inFree fr i = false →
        (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals)[i]? = (encs fs ++ (Frame.sq k n).enc)[i]?) ∧
      ∃ fs', List.Forall₂ Alike fs fs' ∧ Sched.MidAll fs' ∧
        Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals = encs fs' ++ (Frame.sq k n).enc := by
  have hs : m.inner ≠ some INNER_ROUNDS := by rw [hin]; nofun
  have hmid : ∀ f ∈ fs, midFrame f := fun f hf => (hM f hf).1
  have hsp : Loop.Spaced fr.toList (encs fs).length := Loop.roundFree_spaced c m fs k n hs hmid hk h0 hn hq
  have hbd := hsp.get.2
  have hb : ∀ j < fr.size, fr[j]! + 16 ≤ (encs fs ++ (Frame.sq k n).enc).length :=
    fun j hj => le_trans (hbd j hj) (by rw [List.length_append]; omega)
  have hlen : (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).length = (encs fs ++ (Frame.sq k n).enc).length :=
    Loop.fill_length _ fr vals hb fr.size le_rfl
  have hout : ∀ i, (∀ j < fr.size, i < fr[j]! ∨ fr[j]! + 16 ≤ i) →
      (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals)[i]? = (encs fs ++ (Frame.sq k n).enc)[i]? :=
    fun i hi => fill_getElem?_out _ fr vals i hb hi fr.size le_rfl
  have hfree : ∀ p ∈ fr, ∀ i, p ≤ i → i < p + 16 → valBytes fs i :=
    roundFree_free c m fs k n hs hmid hk h0 hn hq
  refine ⟨hlen, fun i hi => hout i fun j hj => ?_, ?_⟩
  · unfold _root_.Flock.Firewall.Frame.inFree at hi
    have hj' := Array.any_eq_false.1 hi j hj
    rw [getElem!_pos fr j hj]
    rcases Nat.lt_or_ge i fr[j] with h | h
    · exact Or.inl h
    · exact Or.inr (Nat.le_of_not_lt fun h' => hj' (by simp [h, h']))
  · obtain ⟨fs', hA, he⟩ := encs_refill fs (Frame.sq k n).enc _ hlen fun i hi => hout i fun j hj => by
      have hmem : fr[j]! ∈ fr := by rw [getElem!_pos fr j hj]; exact Array.getElem_mem hj
      rcases Nat.lt_or_ge i fr[j]! with h | h
      · exact Or.inl h
      · exact Or.inr (Nat.le_of_not_lt fun h' => hi (hfree _ hmem i h h'))
    exact ⟨fs', hA, Sched.midAll_alike hA hM, he⟩

/-- **The frame takes the filled round and passes it**: where it takes the scheduled round of frames `fs` (none the inner
proof's label) at a rep not stopped outside the inner proof, freeing the words at `fr`, it takes the round with values
in those words, freeing the same words, and its check passes that round against the scheduled one. -/
theorem check_fill {c : Cfg} {m : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} (hst : m.stopped = false)
    (hin : m.inner = none) (hM : Sched.MidAll fs) (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2))
    (hn : n < 2 ^ 64) {o : ByteArray} {t : Masker} {fr : Array ℕ}
    (hq : roundFree c m (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok (o, t, fr))
    (vals : ℕ → F128) :
    ∃ o' t', roundFree c m (ByteArray.mk (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).toArray) n =
        .ok (o', t', fr) ∧
      check c m (ByteArray.mk (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).toArray)
        (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok (o', t', none) := by
  obtain ⟨hlen, hout, fs', hA, hM', hV⟩ := fill_frames hin hM hk h0 hn hq vals
  have hs : m.inner ≠ some INNER_ROUNDS := by rw [hin]; nofun
  have hR := roundFree_alike (c := c) (c' := c) ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ m false fs fs' k n hs
    (fun f hf => (hM f hf).1) (fun f hf => (hM' f hf).1) hk h0 hn hA
    (fun _ hq' => Sched.foldlM_midAll_inner hin hM hq')
  rw [hq, Sched.with_stopped_self m hst, ← hV] at hR
  obtain ⟨⟨o', t', fr'⟩, hq', he⟩ := relM_ok_left hR
  have hfr : fr' = fr := (congrArg Prod.snd he).symm
  rw [hfr] at hq'
  refine ⟨o', t', hq', check_of_passes hst ?_ (round_of_roundFree hq')⟩
  have hsz : (ByteArray.mk (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).toArray).size =
      (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray).size := by
    rw [size_list, size_list, hlen]
  have hfd : firstDiff (ByteArray.mk (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).toArray)
      (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) fr = none :=
    firstDiff_eq_none fun i _ hi => by simp only [get!_list, getElem!_def, hout i hi]
  unfold passes
  rw [hq]
  simp only [bind, Except.bind, show ((ByteArray.mk (Loop.fill (encs fs ++ (Frame.sq k n).enc) fr vals).toArray).size !=
      (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray).size) = false by simp [hsz], Bool.false_eq_true,
    ↓reduceIte, hfd, Option.map_none]
  rfl

end HonestPass

namespace LeanRelease

open HonestPass

theorem roundOk_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (x : Msg) (w : W) (ρ : R)
    (g : List (Msg × Msg)) :
    (L.withDevice d).roundOk x w ρ g =
      (L.cross x g.length w ρ g (d x w ρ g (computedOf (L.computed x w ρ g))) &&
        L.pass x w ρ g (d x w ρ g (computedOf (L.computed x w ρ g))) &&
        L.takes x ρ g (d x w ρ g (computedOf (L.computed x w ρ g)))) :=
  rfl

/-- **A frame round outside the inner proof passes the frame's checks at `valsDev`**: where the frame took the stream's
coins (`answered`) at a state outside the inner proof, and takes there, at a rep not stopped, the scheduled round, of
frames none the inner proof's label and its squeeze, `valsDev`'s round is the scheduled round with `vs`'s values in the
words it frees (`fill`), which the frame's check passes and the frame takes (`check_fill`). The cross-round check is
`L`'s own (`hcr`). -/
theorem roundOk_valsDev (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (L : LeanRelease Flock.Wit Loop.Tp)
    {x : Msg} {w : Flock.Wit} {ρ : Loop.Tp} {g : List (Msg × Msg)} {m : Masker} {fs : List Frame} {k : UInt8}
    {o : ByteArray} {t : Masker} {fr : Array ℕ} (hm : L.answered x ρ g = .ok m) (hin : m.inner = none)
    (hr : L.scheduled x w ρ g (computedOf (L.computed x w ρ g)) = encs fs ++ (Frame.sq k (L.coins x g.length)).enc)
    (hM : Sched.MidAll fs) (hk : k = 2 ∨ (k = 1 ∧ L.coins x g.length = 1))
    (h0 : ¬ (L.coins x g.length = 0 ∧ k = 2)) (hn : L.coins x g.length < 2 ^ 64)
    (hq : roundFree (L.cfg x ρ g) { m with stopped := false }
      (ByteArray.mk (encs fs ++ (Frame.sq k (L.coins x g.length)).enc).toArray) (L.coins x g.length) = .ok (o, t, fr))
    (hcr : L.cross x g.length w ρ g (Loop.valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true) :
    (L.withDevice (Loop.valsDev vs L)).roundOk x w ρ g = true := by
  obtain ⟨o', t', hq', hc⟩ := check_fill (m := { m with stopped := false }) rfl hin hM hk h0 hn hq
    (fun j => Loop.vsAt vs L x w ρ g ((L.cfg x ρ g).order[m.pos + j]!))
  obtain ⟨-, -, fs', -, hM', hV⟩ := fill_frames (m := { m with stopped := false }) hin hM hk h0 hn hq
    (fun j => Loop.vsAt vs L x w ρ g ((L.cfg x ρ g).order[m.pos + j]!))
  have hnot : ¬ ∃ w' : ByteArray, w'.size = 16 ∧
      Loop.fill (encs fs ++ (Frame.sq k (L.coins x g.length)).enc) fr
        (fun j => Loop.vsAt vs L x w ρ g ((L.cfg x ρ g).order[m.pos + j]!)) = (Loop.inner0 w').data.toList := by
    rintro ⟨w', hw', he⟩
    have hq0 : roundFree (L.cfg x ρ g) { m with stopped := false }
        (_root_.Flock.Firewall.Frame.ownRound (L.cfg x ρ g) m w' (ByteArray.mk (Array.replicate 64 0))
          (L.coins x g.length) 0) (L.coins x g.length) = .ok (o', t', fr) := by
      rw [he, Array.toArray_toList] at hq'
      exact hq'
    have h1 : t'.inner = some 0 := Loop.roundFree_own0_inner hw' (by simp [ByteArray.size]) hq0
    have h2 : t'.inner = none := Sched.roundFree_inner (s := { m with stopped := false })
      (r := (ByteArray.mk (Loop.fill (encs fs ++ (Frame.sq k (L.coins x g.length)).enc)
        fr (fun j => Loop.vsAt vs L x w ρ g ((L.cfg x ρ g).order[m.pos + j]!))).toArray, L.coins x g.length))
      ⟨fs', k, by rw [hV], hM', hk, h0, hn⟩ hin hq'
    rw [h1] at h2
    cases h2
  have hd : Loop.valsDev vs L x w ρ g (computedOf (L.computed x w ρ g)) =
      Loop.fill (encs fs ++ (Frame.sq k (L.coins x g.length)).enc) fr
        (fun j => Loop.vsAt vs L x w ρ g ((L.cfg x ρ g).order[m.pos + j]!)) := by
    have hF : Loop.Frames (encs fs ++ (Frame.sq k (L.coins x g.length)).enc) (L.coins x g.length) :=
      ⟨fs, k, rfl, fun f hf => (hM f hf).1, hk, h0, hn⟩
    unfold Loop.valsDev
    rw [hm]
    simp only
    rw [hr, hq]
    simp only
    rw [ite_eq_left hF, Loop.setTau, ite_eq_right, Loop.setFinalC, ite_eq_right hnot]
    rintro ⟨h2, -⟩
    rw [hin] at h2
    cases h2
  have hp : L.pass x w ρ g (Loop.valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true := by
    rw [hd]
    unfold LeanRelease.pass
    rw [hm]
    simp only
    rw [hr, hc]
  have ht : L.takes x ρ g (Loop.valsDev vs L x w ρ g (computedOf (L.computed x w ρ g))) = true := by
    rw [hd]
    unfold LeanRelease.takes LeanRelease.step
    rw [hm]
    simp only [bind, Except.bind]
    rw [round_of_roundFree hq']
  rw [roundOk_withDevice, Bool.and_eq_true, Bool.and_eq_true]
  exact ⟨⟨hcr, hp⟩, ht⟩

end LeanRelease

end Claims.ZKServer
