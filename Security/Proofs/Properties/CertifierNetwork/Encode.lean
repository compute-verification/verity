import Proofs.Properties.CertifierNetwork.Defs
import Proofs.Properties.CertifierDevice.Issuer
import Proofs.Properties.CertifierDevice.Basic
import Proofs.Properties.NetworkCertifier.Grid

/-!
# A device's record has an encoding

`record_head`, `RecordHead.encode` and `certificate_message` refuse what doesn't fit their fields; a record the device
publishes fits them: its grid is `T` packed rows of `r` slots (`Packed`), each frame `B` bytes, and its integers, link and
key identifier are within their widths (`prepare_of`).
-/

namespace Properties.CertifierNetwork.Proofs

open _root_.CertifierDevice
open NetworkCheck (RecordHead Direction uBE text recordHead realFrames rowFrames framesRoot leaves frameLeaf
  leafInput block frameTag depth pairs top empty node statusCode certificateMessage messageInput certificateDigest
  saltBytes)
open Properties.CertifierDevice (Packed ValidKey)
open Properties.NetworkCertifier.Bytes
open Properties.CertifierDevice.Proofs.Issuer (except_ok_bind)

set_option maxRecDepth 20000

theorem uBE_of {w : Nat} {x : Int} (n : String) (h0 : 0 ≤ x) (h1 : x < 2 ^ (8 * w)) :
    ∃ b, uBE w x n = .ok b ∧ b.size = w := by
  refine ⟨⟨(List.range w).toArray.map fun i => (x.toNat >>> (8 * (w - 1 - i))).toUInt8⟩, ?_, ?_⟩
  · unfold uBE
    rw [ite_eq_left ⟨h0, h1⟩]
    rfl
  · simp [ByteArray.size]

theorem cast_lt {n w : Nat} (h : n < 2 ^ (8 * w)) : (n : Int) < 2 ^ (8 * w) := by exact_mod_cast h

theorem text_of {s : String} (n : String) (h : s.toUTF8.size < 2 ^ 16) : ∃ b, text s n = .ok b := by
  obtain ⟨u, hu, -⟩ := uBE_of (w := 2) (x := s.toUTF8.size) s!"len({n})" (by omega) (cast_lt (w := 2) h)
  refine ⟨u ++ s.toUTF8, ?_⟩
  simp only [text, hu]
  rfl

theorem ok_of {ε α : Type} {x : Except ε α} (h : ∀ e, x = .error e → False) : ∃ a, x = .ok a := by
  cases x with
  | error e => exact (h e rfl).elim
  | ok a => exact ⟨a, rfl⟩

theorem forIn_ok {β : Type} (f : Int → β → Except String (ForInStep β)) :
    ∀ (ns : List Int) (a : β), (∀ n ∈ ns, ∀ s, ∃ s', f n s = .ok (.yield s')) → ∀ e,
      forIn ns a f = .error e → False
  | [], a, _, e, h => by simp [pure, Except.pure] at h
  | n :: ns, a, hf, e, h => by
    obtain ⟨s', hs'⟩ := hf n (by simp) a
    rw [List.forIn_cons, hs'] at h
    exact forIn_ok f ns s' (fun m hm => hf m (by simp [hm])) e h

theorem encode_of {h : RecordHead} (hc : (h.counts.size : Int) = h.buckets)
    (hcnt : ∀ n ∈ h.counts.toList, 0 ≤ n ∧ n ≤ h.allocation) (hpar : h.parameters.size = 64) (hroot : h.root.size = 64)
    (hl : h.link.toUTF8.size < 2 ^ 16) (hw : 0 ≤ h.window ∧ h.window < 2 ^ (8 * 8))
    (hT : 0 ≤ h.buckets ∧ h.buckets < 2 ^ (8 * 4)) (hr : 0 ≤ h.allocation ∧ h.allocation < 2 ^ (8 * 4))
    (hB : 0 ≤ h.frameBytes ∧ h.frameBytes < 2 ^ (8 * 4)) (hτ : 0 ≤ h.bucketMs ∧ h.bucketMs < 2 ^ (8 * 4))
    (hst : h.status = none) : ∃ x, h.encode = .ok x := by
  refine ok_of fun e he => ?_
  have hc' : ¬ (h.counts.size ≠ h.buckets ∨ h.counts.any (fun n => ¬(0 ≤ n ∧ n ≤ h.allocation))) := by
    simp only [ne_eq, Array.any_eq_true, decide_eq_true_eq, not_or, not_exists, not_and, not_not]
    refine ⟨hc, fun i hi hh => ?_⟩
    have m : h.counts[i] ∈ h.counts.toList := by simp
    exact hh (hcnt _ m).1 (hcnt _ m).2
  have hp' : ¬ (h.parameters.size ≠ NetworkCheck.digestBytes ∨ h.root.size ≠ NetworkCheck.digestBytes) := by
    simp [NetworkCheck.digestBytes, hpar, hroot]
  obtain ⟨lk, hlk⟩ := text_of "link" hl
  obtain ⟨dir, hdir, -⟩ := uBE_of (w := 1) (x := h.direction.code) "direction" (by omega)
    (by cases h.direction <;> simp [NetworkCheck.Direction.code])
  obtain ⟨win, hwin, -⟩ := uBE_of "window" hw.1 hw.2
  obtain ⟨bk, hbk, -⟩ := uBE_of "T" hT.1 hT.2
  obtain ⟨al, hal, -⟩ := uBE_of "r" hr.1 hr.2
  obtain ⟨fb, hfb, -⟩ := uBE_of "B" hB.1 hB.2
  obtain ⟨bm, hbm, -⟩ := uBE_of "τ_b" hτ.1 hτ.2
  obtain ⟨st, hst', -⟩ := uBE_of (w := 1) (x := ((0 : Nat) : Int)) "status" (by omega) (by norm_num)
  obtain ⟨dl, hdl, -⟩ := uBE_of (w := 1) (x := if h.deadline = true then 1 else 0) "deadline"
    (by split <;> omega) (by split <;> norm_num)
  unfold RecordHead.encode at he
  simp only [bind, Except.bind, pure, Except.pure] at he
  rw [ite_eq_right hc', ite_eq_right hp', hst] at he
  simp only [hlk, hdir, hwin, hbk, hal, hfb, hbm, NetworkCheck.statusCode, pure, Except.pure, hst', hdl] at he
  rw [← Array.forIn_toList] at he
  split at he
  · rename_i e' hy
    refine forIn_ok _ _ _ (fun n hn s => ?_) e' hy
    obtain ⟨v, hv, -⟩ := uBE_of (w := 4) (x := n) "count" (hcnt n hn).1 (lt_of_le_of_lt (hcnt n hn).2 hr.2)
    exact ⟨s ++ v, by rw [hv]⟩
  · cases he
/-! ## The grid -/

theorem packed_size {g : Params} {row : Slots} (h : Packed g row) : row.size = g.allocation := by
  obtain ⟨fs, hle, -, rfl⟩ := h
  simp only [Array.size_append, Array.size_map, Array.size_replicate]
  omega

theorem packed_count {g : Params} {row : Slots} (h : Packed g row) :
    ∃ fs : Array ByteArray, (row.filter Option.isSome).size = fs.size ∧ fs.size ≤ g.allocation ∧
      (∀ f ∈ fs, f.size = g.frameBytes) ∧ row.filterMap id = fs ∧
      (row.extract 0 (row.filter Option.isSome).size).all Option.isSome = true := by
  obtain ⟨fs, hle, hsz, rfl⟩ := h
  have hc := Properties.CertifierDevice.Proofs.rowCount_packed fs (g.allocation - fs.size)
  unfold rowCount at hc
  refine ⟨fs, hc, hle, hsz, Properties.CertifierDevice.Proofs.filterMap_packed _ _, ?_⟩
  rw [hc, Array.extract_append]
  simp only [Array.size_map, Nat.sub_self, Array.extract_zero, Array.append_empty]
  rw [Array.all_eq_true]
  intro i hi
  simp

theorem rowFrames_of (g : Params) :
    ∀ rows : List Slots, (∀ row ∈ rows, Packed g row) →
      ∃ F, rowFrames g.allocation rows = .ok F ∧ F.size ≤ rows.length * g.allocation ∧ ∀ f ∈ F, f.size = g.frameBytes
  | [], _ => ⟨#[], rfl, by simp, by simp⟩
  | row :: rows, h => by
    obtain ⟨F, hF, hle, hs⟩ := rowFrames_of g rows (fun r hr => h r (by simp [hr]))
    have hp := h row (by simp)
    obtain ⟨fs, hc, hfle, hfs, hfm, hall⟩ := packed_count hp
    refine ⟨fs ++ F, ?_, ?_, ?_⟩
    · have hne : ¬ (row.size ≠ g.allocation) := by simp [packed_size hp]
      simp only [rowFrames, hne, ↓reduceIte, hall, hF, hfm, bind, Except.bind, pure, Except.pure]
    · simp only [Array.size_append, List.length_cons]
      rw [Nat.succ_mul]
      omega
    · intro f hf
      rcases Array.mem_append.1 hf with hf | hf
      · exact hfs f hf
      · exact hs f hf

theorem realFrames_of {g : Params} {grid : Array Slots} (hs : grid.size = g.buckets)
    (hp : ∀ row ∈ grid, Packed g row) :
    ∃ F, realFrames grid g = .ok F ∧ F.size ≤ g.buckets * g.allocation ∧ ∀ f ∈ F, f.size = g.frameBytes := by
  obtain ⟨F, hF, hle, hsz⟩ := rowFrames_of g grid.toList (fun r hr => hp r (Array.mem_toList_iff.1 hr))
  refine ⟨F, ?_, by rw [Array.length_toList, hs] at hle; exact hle, hsz⟩
  unfold realFrames
  rw [ite_eq_right (by omega)]
  exact hF

theorem frameTag_size : frameTag.size = 34 := by decide

theorem block_of {x : ByteArray} (h : x.size ≤ 128) : block x = .ok (x ++ ⟨Array.replicate (128 - x.size) 0⟩) := by
  unfold block
  rw [ite_eq_left h]
  rfl

theorem frameLeaf_of {f : ByteArray} {B : Nat} (hB : B < 2 ^ 32) (hf : f.size = B) :
    ∃ x, frameLeaf f B = .ok x ∧ x.size = 64 := by
  obtain ⟨u, hu, hus⟩ := uBE_of (w := 4) (x := B) "B" (by omega) (cast_lt (w := 4) hB)
  have hblk := block_of (x := frameTag ++ u) (by rw [ByteArray.size_append, frameTag_size, hus]; omega)
  have hne : ¬ (f.size ≠ B) := fun h => h hf
  have hl : leafInput f B = .ok (frameTag ++ u ++ ⟨Array.replicate (128 - (frameTag ++ u).size) 0⟩ ++ f) := by
    unfold leafInput
    simp only [hne, ↓reduceIte]
    rw [hu]
    show (do let b ← block (frameTag ++ u); pure (b ++ f) : Except String ByteArray) = _
    rw [hblk]
    rfl
  refine ⟨Flock.Sha512.hash (frameTag ++ u ++ ⟨Array.replicate (128 - (frameTag ++ u).size) 0⟩ ++ f), ?_,
    hash_size _⟩
  unfold frameLeaf
  rw [hl]
  rfl

theorem mapM_of {α β ε : Type} {f : α → Except ε β} {P : β → Prop} :
    ∀ l : List α, (∀ a ∈ l, ∃ b, f a = .ok b ∧ P b) → ∃ r, l.mapM f = .ok r ∧ ∀ b ∈ r, P b
  | [], _ => ⟨[], rfl, by simp⟩
  | a :: l, h => by
    obtain ⟨b, hb, hP⟩ := h a (by simp)
    obtain ⟨r, hr, hPr⟩ := mapM_of l (fun a' ha' => h a' (by simp [ha']))
    refine ⟨b :: r, ?_, ?_⟩
    · rw [List.mapM_cons, hb, hr]
      rfl
    · intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hP
      · exact hPr x hx

theorem leaves_of {g : Params} {F : Array ByteArray} (hle : F.size ≤ g.buckets * g.allocation)
    (hB : g.frameBytes < 2 ^ 32) (hs : ∀ f ∈ F, f.size = g.frameBytes) :
    ∃ ls, leaves F g = .ok ls ∧ ∀ x ∈ ls, x.size = 64 := by
  obtain ⟨r, hr, hP⟩ := mapM_of (f := (frameLeaf · g.frameBytes)) (P := fun x : ByteArray => x.size = 64) F.toList
    (fun f hf => frameLeaf_of hB (hs f (Array.mem_toList_iff.1 hf)))
  refine ⟨r.toArray, ?_, fun x hx => hP x (by simpa using hx)⟩
  unfold leaves
  rw [ite_eq_right (by omega)]
  show Array.mapM (frameLeaf · g.frameBytes) F = _
  rw [Array.mapM_eq_mapM_toList, hr]
  rfl

theorem pairs_size (lv : Nat) : ∀ L : List ByteArray, (∀ x ∈ L, x.size = 64) → ∀ x ∈ pairs lv L, x.size = 64
  | a :: b :: rest, h, x, hx => by
    simp only [pairs, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact hash_size _
    · exact pairs_size lv rest (fun y hy => h y (by simp [hy])) x hx
  | [a], _, x, hx => by
    simp only [pairs, List.mem_singleton] at hx
    subst hx
    exact hash_size _
  | [], _, x, hx => by simp [pairs] at hx

theorem top_size : ∀ (k lv : Nat) (L : List ByteArray), (∀ x ∈ L, x.size = 64) → ∀ x ∈ top lv k L, x.size = 64
  | 0, _, _, h => h
  | k + 1, lv, L, h => top_size k (lv + 1) (pairs lv L) (pairs_size lv L h)

theorem framesRoot_of {g : Params} {F : Array ByteArray} (hle : F.size ≤ g.buckets * g.allocation)
    (hB : g.frameBytes < 2 ^ 32) (hs : ∀ f ∈ F, f.size = g.frameBytes) :
    ∃ rt, framesRoot F g = .ok rt ∧ rt.size = 64 := by
  obtain ⟨ls, hls, hsz⟩ := leaves_of hle hB hs
  refine ⟨(top 0 (depth g) ls.toList).headD (empty (depth g)), by unfold framesRoot; rw [hls]; rfl, ?_⟩
  have := top_size (depth g) 0 ls.toList (fun x hx => hsz x (Array.mem_toList_iff.1 hx))
  cases h : top 0 (depth g) ls.toList with
  | nil => exact empty_size _
  | cons x xs => exact this x (by rw [h]; simp)

/-- A grid of `T` packed rows has a head under `record_head`, its root 64 bytes. -/
theorem recordHead_of {g : Params} {grid : Array Slots} (hs : grid.size = g.buckets)
    (hp : ∀ row ∈ grid, Packed g row) (hB : g.frameBytes < 2 ^ 32) (l : String) (d : Direction) (w : Int)
    (st : Option String) (par : ByteArray) (dl : Bool) :
    ∃ h, recordHead l d w grid st g par dl = .ok h := by
  obtain ⟨F, hF, hle, hsz⟩ := realFrames_of hs hp
  obtain ⟨rt, hrt, -⟩ := framesRoot_of hle hB hsz
  simp only [recordHead, hF, hrt, bind, Except.bind, pure, Except.pure, Except.ok.injEq, exists_eq']

theorem framesRoot_size {F : Array ByteArray} {g : Params} {rt : ByteArray} (h : framesRoot F g = .ok rt) :
    rt.size = 64 := by
  obtain ⟨ls, hl, rfl⟩ := Properties.NetworkCertifier.Tree.framesRoot_ok h
  obtain ⟨-, hm⟩ := Properties.NetworkCertifier.Tree.leaves_ok hl
  have hls : ∀ x ∈ ls.toList, x.size = 64 := fun x hx => by
    obtain ⟨f, -, hf⟩ := Properties.NetworkCertifier.Tree.mapM_mem hm x hx
    obtain ⟨y, -, rfl⟩ := Properties.NetworkCertifier.Tree.frameLeaf_ok hf
    exact hash_size _
  cases ht : top 0 (depth g) ls.toList with
  | nil => exact empty_size _
  | cons x xs => exact top_size _ _ _ hls x (by rw [ht]; simp)

theorem recordHead_fields {l : String} {d : Direction} {w : Int} {grid : Array Slots} {st : Option String}
    {g : Params} {par : ByteArray} {dl : Bool} {h : RecordHead} (e : recordHead l d w grid st g par dl = .ok h) :
    h.link = l ∧ h.direction = d ∧ h.window = w ∧ h.buckets = g.buckets ∧ h.allocation = g.allocation ∧
      h.frameBytes = g.frameBytes ∧ h.bucketMs = g.bucketMs ∧ h.status = st ∧ h.deadline = dl ∧ h.parameters = par ∧
      h.counts = grid.map (fun row => ((row.filter Option.isSome).size : Int)) ∧ h.root.size = 64 := by
  unfold recordHead at e
  cases hF : realFrames grid g with
  | error _ => rw [hF] at e; exact (error_ne_ok e).elim
  | ok F =>
    cases hr : framesRoot F g with
    | error _ =>
      simp only [hF, hr, bind, Except.bind] at e
      exact (error_ne_ok e).elim
    | ok rt =>
      simp only [hF, hr, bind, Except.bind, pure, Except.pure] at e
      cases Except.ok.inj e
      exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, framesRoot_size hr⟩

/-! ## The digest, the message and `prepare` -/

theorem digest_of {h : RecordHead} {x : ByteArray} (he : h.encode = .ok x) :
    ∃ dg, h.digest = .ok dg ∧ dg.size = 64 := by
  obtain ⟨y, -, hy⟩ := Properties.Identity.taggedSha512_of (tag := NetworkCheck.recordTag) (payload := x) (by decide)
    (by decide) (encode_size he)
  refine ⟨Flock.Sha512.hash y, ?_, hash_size _⟩
  unfold RecordHead.digest
  rw [he]
  exact hy

theorem messageInput_of {k s : ByteArray} {l : String} {d : Direction} {w : Int} {g : ByteArray} (hg : g.size = 64)
    (hs : s.size = 32) (hk : k.size = 16) (hw : 0 ≤ w ∧ w < 2 ^ (8 * 8)) : ∃ x, messageInput k s l d w g = .ok x := by
  refine ok_of fun e he => ?_
  obtain ⟨wb, hwb, -⟩ := uBE_of "window" hw.1 hw.2
  unfold messageInput at he
  simp only [bind, Except.bind, pure, Except.pure] at he
  rw [ite_eq_right (by simp [NetworkCheck.digestBytes, hg])] at he
  simp only [hwb] at he
  rw [ite_eq_right (by simp [NetworkCheck.sessionBytes, hs])] at he
  rw [ite_eq_right (by simp [hk])] at he
  cases he

theorem certificateMessage_of {k s : ByteArray} {l : String} {d : Direction} {w : Int} {g : ByteArray}
    (hg : g.size = 64) (hs : s.size = 32) (hk : k.size = 16) (hw : 0 ≤ w ∧ w < 2 ^ (8 * 8)) :
    ∃ m, certificateMessage k s l d w g = .ok m := by
  obtain ⟨x, hx⟩ := messageInput_of (l := l) (d := d) hg hs hk hw
  refine ⟨Flock.Sha512.hash x, ?_⟩
  unfold certificateMessage
  rw [hx]
  rfl

theorem parameters_size {g : Params} {ss : Option _root_.NetworkCertifier.SyncSet} {pd : ByteArray}
    (h : parameters g ss = .ok pd) : pd.size = 64 := by
  unfold parameters NetworkCheck.parametersDigest at h
  cases hc : Flock.canonUtf8 (NetworkCheck.parametersManifest g (ss.map checkSyncSet)) with
  | error e => rw [hc] at h; cases h
  | ok j =>
    rw [hc] at h
    have h' : Identity.taggedSha512 NetworkCheck.parametersTag j.toUTF8 = .ok pd := h
    obtain ⟨x, -, rfl⟩ := Properties.Identity.taggedSha512_ok.mp h'
    exact hash_size _

theorem certificateDigest_size (salt record : ByteArray) : (certificateDigest salt record).size = 64 := by
  unfold certificateDigest
  exact hash_size _

theorem prepare_eq {i : Issuer} {salt : ByteArray} (hsalt : salt.size = saltBytes) {r : Record} {d : Direction}
    {g : Params} {pd : ByteArray} {dl : Bool} {head : RecordHead} {dg m : ByteArray}
    (hh : recordHead r.link d r.window r.grid ((Status.reason? r.status).map reasonValue) g pd dl = .ok head)
    (hdg : head.digest = .ok dg)
    (hm : certificateMessage i.key.I i.session r.link d r.window (certificateDigest salt dg) = .ok m) :
    i.prepare salt r d g pd dl = .ok (head, certificateDigest salt dg, m) := by
  unfold Issuer.prepare
  simp only [hsalt, ne_eq, not_true_eq_false, ↓reduceIte]
  -- `rw`, not `simp`: the kernel overflows checking simp's steps through the signed digest
  generalize certificateDigest salt = D at hm ⊢
  rw [hh, except_ok_bind, hdg, except_ok_bind, hm]
  rfl

/-- A record whose grid is `T` packed rows, complete, of a link and window that fit their fields, under a 64-byte
parameters digest, has a head, digest and message under `prepare` (with a 64-byte salt, a 16-byte key identifier and
a 32-byte session id): the head `record_head` makes of its grid. -/
theorem prepare_of {i : Issuer} (hk : i.key.I.size = 16) (hss : i.session.size = 32) {salt : ByteArray}
    (hsalt : salt.size = saltBytes) {r : Record} {d : Direction} {g : Params} {pd : ByteArray}
    (hs : r.grid.size = g.buckets) (hp : ∀ row ∈ r.grid, Packed g row) (hpd : pd.size = 64)
    (hl : r.link.toUTF8.size < 2 ^ 16) (hw : r.window < 2 ^ 64) (hT : g.buckets < 2 ^ 32) (hr : g.allocation < 2 ^ 32)
    (hB : g.frameBytes < 2 ^ 32) (hτ : g.bucketMs < 2 ^ 32) (hst : r.status = .complete) :
    ∃ head dg m, NetworkCheck.recordHead r.link d r.window r.grid none g pd true = .ok head ∧
      i.prepare salt r d g pd true = .ok (head, dg, m) := by
  obtain ⟨head, hh⟩ := recordHead_of hs hp hB r.link d r.window none pd true
  obtain ⟨hl', -, hw', hT', hr', hB', hτ', hst', -, hpar', hcnt', hroot'⟩ := recordHead_fields hh
  have hcnt : ∀ n ∈ head.counts.toList, 0 ≤ n ∧ n ≤ head.allocation := by
    intro n hn
    rw [hcnt'] at hn
    simp only [Array.toList_map, List.mem_map, Array.mem_toList_iff] at hn
    obtain ⟨row, hrow, rfl⟩ := hn
    have h1 : (row.filter Option.isSome).size ≤ row.size := Array.size_filter_le
    have h2 := packed_size (hp row hrow)
    rw [hr']
    omega
  obtain ⟨x, hx⟩ := encode_of (h := head) (by rw [hcnt', hT', Array.size_map, hs]) hcnt (hpar' ▸ hpd) hroot'
    (hl' ▸ hl) (by rw [hw']; exact ⟨by omega, cast_lt (w := 8) hw⟩) (by rw [hT']; exact ⟨by omega, cast_lt (w := 4) hT⟩)
    (by rw [hr']; exact ⟨by omega, cast_lt (w := 4) hr⟩) (by rw [hB']; exact ⟨by omega, cast_lt (w := 4) hB⟩)
    (by rw [hτ']; exact ⟨by omega, cast_lt (w := 4) hτ⟩) hst'
  obtain ⟨dg, hdg, -⟩ := digest_of hx
  obtain ⟨m, hm⟩ := certificateMessage_of (l := r.link) (d := d) (w := r.window) (certificateDigest_size salt dg) hss
    hk ⟨by omega, cast_lt (w := 8) hw⟩
  refine ⟨head, certificateDigest salt dg, m, hh, ?_⟩
  have hn : (Status.reason? r.status).map reasonValue = none := by rw [hst]; rfl
  exact prepare_eq hsalt (by rw [hn]; exact hh) hdg hm

end Properties.CertifierNetwork.Proofs
