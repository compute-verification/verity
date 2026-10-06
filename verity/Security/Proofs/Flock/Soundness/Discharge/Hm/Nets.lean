/-
The two row nets of a parsed circuit: `HmNets.check` makes its `sha512x3` and `hm96` nets `Net.ofRows` of the verifier's
layouts (`shaNet_ofRows`, `hmNet_ofRows`), so their input ports sit at `cv_k = 1536 k`, `msg_k = 1536 k + 512`
(`shaIn_eq`) and `cv = 0`, `pad = 512`, `y = 1536` (`hmIn_eq`), and their outputs at the layouts' output groups
(`shaOut_eq`, `hmOut_eq`). `port_commit_rows` is `port_commit` with the slots holding the nets' rows (`LaidRows`) in
place of the builder's rows, and the nets' places from `HmNets.check`.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Laid
import Proofs.Flock.Soundness.Discharge.Hm.Port

namespace FlockSoundness.Discharge.Hm

open Flock
open Flock.HmNets hiding and

theorem shaNet_ofRows {c : Circuit} (h : HmNets.check c = .ok ()) :
    Net.ofRows "sha512x3" (shaLaid ()).useful (shaLaid ()).constPos (shaLaid ()).inGroups (shaLaid ()).outGroups
      (shaLaid ()).ra (shaLaid ()).rb = .ok (c.nets.getD c.shaNet default).2 := by
  obtain ⟨⟨g, hg, he⟩, -⟩ := check_ok h
  rw [he, ← hg, sha512x3Net_get]
  rfl

theorem hmNet_ofRows {c : Circuit} (h : HmNets.check c = .ok ()) :
    Net.ofRows "hm96" (hmLaid ()).useful (hmLaid ()).constPos (hmLaid ()).inGroups (hmLaid ()).outGroups
      (hmLaid ()).ra (hmLaid ()).rb = .ok (c.nets.getD c.hmNet default).2 := by
  obtain ⟨-, ⟨g, hg, he⟩⟩ := check_ok h
  rw [he, ← hg, hm96Net_get]
  rfl

theorem shaNet_groups {c : Circuit} (h : HmNets.check c = .ok ()) :
    (c.nets.getD c.shaNet default).2.inGroups = #[⟨0, groupWords (SHA_IN.foldl (· + ·) 0), SHA_IN⟩] ∧
      (c.nets.getD c.shaNet default).2.outGroups =
        #[⟨outStart (compression3.run St.empty).2 SHA_IN SHA_OUT / WORD, groupWords (SHA_OUT.foldl (· + ·) 0), SHA_OUT⟩] := by
  obtain ⟨-, -, -, -, -, hi, ho, -⟩ := Net.ofRows_ok (shaNet_ofRows h)
  rw [hi, ho, shaLaid_eq, layout_inGroups, layout_outGroups]
  exact ⟨rfl, rfl⟩

theorem hmNet_groups {c : Circuit} (h : HmNets.check c = .ok ()) :
    (c.nets.getD c.hmNet default).2.inGroups = #[⟨0, groupWords (HM_IN.foldl (· + ·) 0), HM_IN⟩] ∧
      (c.nets.getD c.hmNet default).2.outGroups =
        #[⟨outStart ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT / WORD, groupWords (HM_OUT.foldl (· + ·) 0), HM_OUT⟩] := by
  obtain ⟨-, -, -, -, -, hi, ho, -⟩ := Net.ofRows_ok (hmNet_ofRows h)
  rw [hi, ho, hmLaid_eq, layout_inGroups, layout_outGroups]
  exact ⟨rfl, rfl⟩

/-- **The `sha512x3` net's inputs**: `cv_s` at column `1536 s`, `msg_s` at `1536 s + 512`. -/
theorem shaIn_eq {c : Circuit} (h : HmNets.check c = .ok ()) :
    ∀ s < 3, shaIn c (2 * s) = 1536 * s ∧ shaIn c (2 * s + 1) = 1536 * s + 512 := by
  intro s hs
  unfold shaIn
  rw [(shaNet_groups h).1]
  obtain _ | _ | _ | s := s
  · decide
  · decide
  · decide
  · omega

/-- **The `hm96` net's inputs**: `cv` at column 0, `pad` at 512, `y` at 1536. -/
theorem hmIn_eq {c : Circuit} (h : HmNets.check c = .ok ()) : hmIn c 0 = 0 ∧ hmIn c 1 = 512 ∧ hmIn c 2 = 1536 := by
  unfold hmIn
  rw [(hmNet_groups h).1]
  decide

theorem portCols_SHA_OUT (S : ℕ) (hS : S / WORD * WORD = S) {s : ℕ} (hs : s < 3) :
    ((portCols #[⟨S / WORD, groupWords (SHA_OUT.foldl (· + ·) 0), SHA_OUT⟩]).getD s default).1 = S + 512 * s := by
  generalize S / WORD = x0 at hS
  subst hS
  obtain _ | _ | _ | s := s
  · show x0 * WORD = _
    omega
  · show x0 * WORD + 512 = _
    omega
  · show x0 * WORD + 512 + 512 = _
    omega
  · omega

theorem portCols_HM_OUT (S : ℕ) (hS : S / WORD * WORD = S) :
    ((portCols #[⟨S / WORD, groupWords (HM_OUT.foldl (· + ·) 0), HM_OUT⟩]).getD 0 default).1 = S := by
  generalize S / WORD = x0 at hS
  subst hS
  rfl

/-- **The `sha512x3` net's outputs**: `out_s` at its output group's start plus `512 s`. -/
theorem shaOut_eq {c : Circuit} (h : HmNets.check c = .ok ()) {s : ℕ} (hs : s < 3) :
    shaOut c s = outStart (compression3.run St.empty).2 SHA_IN SHA_OUT + 512 * s := by
  unfold shaOut
  rw [(shaNet_groups h).2]
  exact portCols_SHA_OUT _ (outStart_word _ _ _) hs

/-- **The `hm96` net's output** `b ‖ c` at its output group's start. -/
theorem hmOut_eq {c : Circuit} (h : HmNets.check c = .ok ()) :
    hmOut c = outStart ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT := by
  unfold hmOut
  rw [(hmNet_groups h).2]
  exact portCols_HM_OUT _ (outStart_word _ _ _)

theorem compression3_lens (z : Var → Bool) (h : stHolds z (compression3.run St.empty).2) :
    ∀ o ∈ (compression3.run St.empty).1.2, o.length = 512 := by
  rw [compression3_run, snd_pair] at h
  rw [compression3_run, fst_pair, snd_pair]
  exact ((sound_shaOuts sha3Ins ⟨4608, []⟩).2 z h sha3Ins_len).2.1

theorem sum_SHA_IN : SHA_IN.foldl (· + ·) 0 = 4608 := by decide
theorem sum_HM_IN : HM_IN.foldl (· + ·) 0 = 3072 := by decide
theorem inRows_SHA_IN : inRowsOf SHA_IN = 8192 := by decide
theorem inRows_HM_IN : inRowsOf HM_IN = 4096 := by decide

/-- **A port's row at its columns, from the slots' rows.** As `port_commit`, with what the slots hold stated on the
circuit's own nets: the circuit passed `HmNets.check` (its `sha512x3` and `hm96` nets are the verifier's), each
compression `k`'s `sha512x3` slot holds that net's rows (`LaidRows`, from row 8192, past its input rows), and the row's
`hm96` slot holds that net's rows (from row 4096). The block holds Δ's copies (`hcv0`, `hchain`, `hcvh`, `hpad`). Then
the 128 bytes at `bcCol` are `commitString (rowPrefix words ‖ row) salt`, the row the `2 · words` bytes at
`Typed.msgCol` and the salt the 192 bytes at `saltCol`. -/
theorem port_commit_rows (c : Circuit) (w : ℕ → Bool) (g p : ℕ) {words n : ℕ} (hs : c.ports[p]!.segs = none)
    (hn : 0 < n)
    (hw : 2 * words = 128 * n) (hslots : n ≤ ((HmRow.rowSlots c.ports).2[p]!).size) (hck : HmNets.check c = .ok ())
    (hsub : ∀ k < n, shaSub c p k < 3)
    (hsha : ∀ k < n, LaidRows (shaLaid ()) 8192 fun j => w (shaBase c g p k + j))
    (hhm : LaidRows (hmLaid ()) 4096 fun j => w (c.slotCol c.hmNet g p + j))
    (hcv0 : ∀ t < 512, w (shaBase c g p 0 + shaIn c (2 * shaSub c p 0) + t) = wordsBit (HmRow.midstate words) t)
    (hchain : ∀ k, k + 1 < n → ∀ t < 512, w (shaBase c g p (k + 1) + shaIn c (2 * shaSub c p (k + 1)) + t) =
      w (shaBase c g p k + shaOut c (shaSub c p k) + t))
    (hcvh : ∀ t < 512, w (c.slotCol c.hmNet g p + hmIn c 0 + t) =
      w (shaBase c g p (n - 1) + shaOut c (shaSub c p (n - 1)) + t))
    (hpad : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmIn c 1 + t) = wordsBit (HmRow.padWords words) t) :
    bytesOf w 128 (bcCol c g p) = Binding.commitString
      ((HmRow.rowPrefix words).data.toList ++ bytesOf w (2 * words) (Typed.msgCol c g p)) (bytesOf w 192 (saltCol c g p)) := by
  have hshaL : ∀ k < n, LaidRows (layout (compression3.run St.empty).2 SHA_IN SHA_OUT (compression3.run St.empty).1.2.flatten.toArray)
      (inRowsOf SHA_IN) fun j => w (shaBase c g p k + j) := by
    intro k hk
    have := hsha k hk
    rw [shaLaid_eq] at this
    rw [inRows_SHA_IN]
    exact this
  have hhmL : LaidRows (layout ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray)
      (inRowsOf HM_IN) fun j => w (c.slotCol c.hmNet g p + j) := by
    have := hhm
    rw [hmLaid_eq] at this
    rw [inRows_HM_IN]
    exact this
  have hgs : Good (SHA_IN.foldl (· + ·) 0) (compression3.run St.empty).2 := by rw [sum_SHA_IN]; exact good_compression3
  have hgh : Good (HM_IN.foldl (· + ·) 0) ((hm96Row Hm96.Default512.key).run St.empty).2 := by rw [sum_HM_IN]; exact good_hm96Row _
  let zs : ℕ → Var → Bool := fun k x => w (shaBase c g p k + col (SHA_IN.foldl (· + ·) 0) (inRowsOf SHA_IN) x)
  let zh : Var → Bool := fun x => w (c.slotCol c.hmNet g p + col (HM_IN.foldl (· + ·) 0) (inRowsOf HM_IN) x)
  have hrows : ∀ k < n, stHolds (zs k) (compression3.run St.empty).2 := fun k hk => laid_rows _ _ _ _ _ hgs (hshaL k hk)
  have hh : stHolds zh ((hm96Row Hm96.Default512.key).run St.empty).2 := laid_rows _ _ _ _ _ hgh hhmL
  have hzs : ∀ k < n, ∀ v : ℕ, v < 4608 → zs k v = w (shaBase c g p k + v) := by
    intro k _ v hv
    show w (shaBase c g p k + col _ _ v) = _
    rw [col_input (by rw [sum_SHA_IN]; exact hv)]
  have hzh : ∀ v : ℕ, v < 3072 → zh v = w (c.slotCol c.hmNet g p + v) := by
    intro v hv
    show w (c.slotCol c.hmNet g p + col _ _ v) = _
    rw [col_input (by rw [sum_HM_IN]; exact hv)]
  have hout : ∀ k < n, ∀ t < 512, w (shaBase c g p k + shaOut c (shaSub c p k) + t) =
      bit (zs k) ((compression3.run St.empty).1.2.getD (shaSub c p k) []) t := by
    intro k hk t ht
    have hs := hsub k hk
    obtain ⟨-, hl3, -⟩ := compression3_spec (zs k) (hrows k hk)
    have hlens := compression3_lens (zs k) (hrows k hk)
    have hflat : (compression3.run St.empty).1.2.flatten.length = 1536 := by
      rw [List.length_flatten, List.map_congr_left (g := fun _ => 512) hlens, List.map_const', List.sum_replicate, hl3]
      rfl
    have hj : 512 * shaSub c p k + t < (compression3.run St.empty).1.2.flatten.toArray.size := by
      rw [List.size_toArray, hflat]; omega
    have := laid_outs (compression3.run St.empty).2 SHA_IN SHA_OUT _ (fun j => w (shaBase c g p k + j))
      (by rw [List.size_toArray, hflat]; decide) (hshaL k hk) _ hj
    rw [shaOut_eq hck hs, Nat.add_assoc, Nat.add_assoc, this, List.getElem_toArray,
      ← bit_of_lt (by rw [hflat]; omega), bit_flatten_of_len (zs k) hlens ht]
  have hbc : ∀ t < 1024, w (c.slotCol c.hmNet g p + hmOut c + t) =
      bit zh ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t := by
    intro t ht
    obtain ⟨-, -, -, hl, -⟩ := hm96Row_spec _ zh hh
    have hj : t < ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.toArray.size := by
      rw [List.size_toArray, hl]; exact ht
    have := laid_outs ((hm96Row Hm96.Default512.key).run St.empty).2 HM_IN HM_OUT _ (fun j => w (c.slotCol c.hmNet g p + j))
      (by rw [List.size_toArray, hl]; decide) hhmL _ hj
    rw [hmOut_eq hck, Nat.add_assoc, this, List.getElem_toArray, ← bit_of_lt (by rw [hl]; exact ht)]
  exact port_commit c w g p hs hn hw hslots hsub (shaIn_eq hck) (hmIn_eq hck) zs hzs hrows hout zh hzh hh hbc hcv0 hchain
    hcvh hpad

end FlockSoundness.Discharge.Hm
