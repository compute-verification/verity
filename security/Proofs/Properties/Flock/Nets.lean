module

public import Properties.Flock.Nets
public import Proofs.Properties.Flock.Nets.Lines
public import Proofs.Properties.Flock.Nets.Outs
public import Proofs.Flock.SessionLoop.Concrete.Total

/-!
# The verifier's two fixed nets build (`Properties.Flock.Nets`)

The kernel doesn't reduce `HmNets.symm`, the support merge every XOR of the builder runs, which is well-founded
recursion, so neither layout evaluates in the kernel as written. `Nets.Twin` states the builder over a structurally
recursive `symm` and proves it equal to the deployed one. Each SHA-512 compression of the two circuits then runs in the
kernel chunk by chunk against a certificate of where its run puts each committed word (`Nets.Cert`, `Nets.Seg`): four
schedule chunks, ten round chunks and the feed-forward, for `sha512x3`'s three compressions (`Nets.Sha0` to `Nets.Sha2`)
and `hm96`'s three (`Nets.Hm1` to `Nets.Hm3`, the salt's midstate a literal by `Nets.SaltMid`). Composed, they give each
circuit's next bit, the widths of its rows and its outputs (`Nets.Outs`).

The layout's side holds for any builder state whose rows output the bits after its inputs in creation order and read
only earlier bits: `Net.ofRows` succeeds on the layout (`Nets.Build`) and each line of its text fits (`Nets.Lines`).
`Good` and `FlockDeploy.Concrete.Total` give both circuits those facts.
-/

@[expose] public section

open Flock FlockSoundness.Discharge.Hm FlockDeploy.Concrete.Total Properties.Flock.Nets
open Flock.HmNets hiding and

namespace FlockDeploy.Concrete.Nets

/-- **`sha512x3` builds** at its header fields. -/
theorem sha_builds : Builds "sha512x3" (shaLaid ()) 256001 ⟨0, 64, SHA_IN⟩ ⟨1984, 16, SHA_OUT⟩ 64 := by
  have ht : Tot [] compression3 _ := tot_compression 3
  have ht := ht St.empty scoped_empty (belowL_nil _)
  have hlens := compression3_lens _ (stHolds_fwd (fun _ => false) ht.1)
  have hg := good_compression3
  obtain ⟨new, hrun, hnew⟩ := seg_shaOuts []
  have hst : (compression3.run St.empty).2 = ⟨248814, new ++ []⟩ := by
    rw [compression3_run, snd_pair, hrun]
  have houts : (compression3.run St.empty).1.2 =
      [(ffOut sha0Cv sha0).flatten, (ffOut sha1Cv sha1).flatten, (ffOut sha2Cv sha2).flatten] := by
    rw [compression3_run, fst_pair, snd_pair, hrun]
  rw [shaLaid_eq]
  generalize compression3.run St.empty = p at *
  have hnext : p.2.next = 248814 := by rw [hst]
  have hrows : p.2.rows = new ++ [] := by rw [hst]
  have hlen : p.2.rows.length = 244206 := by
    have := hg.1
    omega
  have hflat : p.1.2.flatten.length = 1536 := by
    rw [houts] at hlens
    rw [houts, List.flatten_cons, List.flatten_cons, List.flatten_cons, List.flatten_nil, List.length_append,
      List.length_append, List.length_append, hlens _ List.mem_cons_self,
      hlens _ (List.mem_cons_of_mem _ List.mem_cons_self),
      hlens _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))]
    rfl
  refine builds_layout "sha512x3" p.2 SHA_IN SHA_OUT p.1.2.flatten.toArray 4608 64 16 253952 128 16 sum_SHA_IN
    (by decide) (by decide) ?_ hg.1 (fun k hk => hg.2 k _ (List.getElem?_eq_getElem hk))
    (fun r hr => chain_below ht.1.1 r hr) ?_ ?_ (by decide) (by decide) ?_ ?_ (by decide) (by decide) sha_header
  · rw [hlen]
    decide
  · rw [List.toList_toArray]
    exact (belowL_append.1 ht.2.2).2
  · rw [List.size_toArray, hflat]
    decide
  · rw [hrows, List.append_nil]
    exact hnew
  · rw [List.toList_toArray, houts]
    exact flatten_width shaOuts_width

/-- **`hm96` builds** at its header fields. -/
theorem hm_builds : Builds "hm96" (hmLaid ()) 246785 ⟨0, 32, HM_IN⟩ ⟨1920, 8, HM_OUT⟩ 32 := by
  have ht := tot_hm96Row Hm96.Default512.key St.empty scoped_empty (belowL_nil _)
  have hg := good_hm96Row Hm96.Default512.key
  obtain ⟨new, hrun, hnew⟩ := seg_hmTail Hm96.Default512.key []
  have hr : (hm96Row Hm96.Default512.key).run St.empty =
      (hmTail Hm96.Default512.key (vars 0 512) (vars 512 1024) (vars 1536 1536)).run ⟨3072, []⟩ :=
    hm96Row_run _ _
  have hst : ((hm96Row Hm96.Default512.key).run St.empty).2 = ⟨244562, new ++ []⟩ := by
    rw [hr, hrun]
  have hout : ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 = hmOut Hm96.Default512.key := by
    rw [hr, hrun]
  rw [hmLaid_eq]
  generalize (hm96Row Hm96.Default512.key).run St.empty = p at *
  have hnext : p.2.next = 244562 := by rw [hst]
  have hrows : p.2.rows = new ++ [] := by rw [hst]
  have hlen : p.2.rows.length = 241490 := by
    have := hg.1
    omega
  refine builds_layout "hm96" p.2 HM_IN HM_OUT p.1.2.2.2.toArray 3072 32 8 245760 128 1552 sum_HM_IN
    (by decide) (by decide) ?_ hg.1 (fun k hk => hg.2 k _ (List.getElem?_eq_getElem hk))
    (fun r hr => chain_below ht.1.1 r hr) ?_ ?_ (by decide) (by decide) ?_ ?_ (by decide) (by decide) hm_header
  · rw [hlen]
    decide
  · rw [List.toList_toArray]
    exact (belowL_append.1 ht.2.2).2
  · rw [List.size_toArray, hout]
    simp only [hmOut, List.length_append, List.length_map, List.length_range]
    decide
  · rw [hrows, List.append_nil]
    exact hnew
  · rw [List.toList_toArray, hout]
    exact hmOut_width _

end FlockDeploy.Concrete.Nets

namespace Properties.Flock.Nets

theorem fixedNetsBuild : FixedNetsBuild :=
  ⟨FlockDeploy.Concrete.Nets.sha_builds, FlockDeploy.Concrete.Nets.hm_builds⟩

end Properties.Flock.Nets
