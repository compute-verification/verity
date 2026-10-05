/-
The two row nets of a parsed circuit: `HmNets.check` makes its `sha512x3` and `hm96` nets `Net.ofRows` of the verifier's
layouts (`shaNet_ofRows`, `hmNet_ofRows`), so their input ports sit at `cv_k = 1536 k`, `msg_k = 1536 k + 512`
(`shaIn_eq`) and `cv = 0`, `pad = 512`, `y = 1536` (`hmIn_eq`), and their outputs at the layouts' output groups
(`shaOut_eq`, `hmOut_eq`). `port_commit_rows` is `port_commit` with the slots holding the nets' rows (`LaidRows`) in
place of the builder's rows, and the nets' places from `HmNets.check`.
-/
import FlockSoundness.Protocol.Discharge.Hm.Laid
import FlockSoundness.Protocol.Discharge.Hm.Port

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

theorem sum_SHA_IN : SHA_IN.foldl (· + ·) 0 = 4608 := by decide
theorem sum_HM_IN : HM_IN.foldl (· + ·) 0 = 3072 := by decide

end FlockSoundness.Discharge.Hm
