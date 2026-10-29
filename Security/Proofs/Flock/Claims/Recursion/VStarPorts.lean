import Proofs.Flock.Claims.Recursion.VStarDeployed

/-!
# Where the stager places the inner prover's rows (the port mapping)

`VStar.In.partIn` builds each part's port inputs from the stager's inputs (`srcOf`). At the fitted records
(`InnerRec.fit`) each commit-string slot of an inner round (`slotsAt`) reads, at its port and instance, the row the inner
prover committed there under its salt. Here, RecCap's port `top` (`capIn`, through `topsAt`):

* a top's row is `Firewall.topRows` of its node (`topRows_wordsBytes`): the node's 32 words, then 32 zero words, so a
  fitted top row (a 64-word `top` port) is read back whole;
* both reps' level-0 tops are round 0's rows, one tree (`topsAt_same`); RecCap's `N` instances read them in order
  (`capIn_ok`, `portRow_ownRows`).
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-! ## Bytes and words -/

theorem wordsBytes_size (v : Array ℕ) : (wordsBytes v).size = 2 * v.size := by
  simp [wordsBytes, ByteArray.size]

theorem wordsBytes_get_even (v : Array ℕ) {j : ℕ} (hj : j < v.size) :
    ((wordsBytes v).get! (2 * j)).toNat = v[j] % 256 := by
  rw [VStarTrace.get!_eq]
  simp [wordsBytes, Array.getElem?_map, Array.getElem?_range, show 2 * j < 2 * v.size by omega,
    getElem!_pos v j hj]

theorem wordsBytes_get_odd (v : Array ℕ) {j : ℕ} (hj : j < v.size) :
    ((wordsBytes v).get! (2 * j + 1)).toNat = v[j] / 256 % 256 := by
  rw [VStarTrace.get!_eq]
  have h1 : (2 * j + 1) % 2 ≠ 0 := by omega
  have h2 : (2 * j + 1) / 2 = j := by omega
  simp [wordsBytes, Array.getElem?_map, Array.getElem?_range, show 2 * j + 1 < 2 * v.size by omega, h1, h2,
    getElem!_pos v j hj]

/-- **A node's top row**: 32 words of 16 bits as bytes, read back by `Firewall.topRows`, then 32 zero words. -/
theorem topRows_wordsBytes (v : Array ℕ) (hv : v.size = 32) (hb : ∀ (j : ℕ) (h : j < v.size), v[j] < 65536) :
    Firewall.topRows (wordsBytes v) = [v ++ Array.replicate 32 0] := by
  unfold Firewall.topRows
  rw [wordsBytes_size, hv]
  simp only [show 2 * 32 / Firewall.NODE_BYTES = 1 from rfl, List.range_one, List.map_cons, List.map_nil,
    Nat.mul_zero, Nat.zero_add]
  congr 2
  apply Array.ext (by simp [hv])
  intro j h1 h2
  have hj : j < v.size := by simp at h1; omega
  simp only [Array.getElem_map, Array.getElem_range]
  rw [wordsBytes_get_even v hj, wordsBytes_get_odd v hj]
  have := hb j hj
  omega

/-! ## RecCap's port `top` -/

theorem sameTop_self (a : Array (ByteArray × ByteArray)) : VStar.In.sameTop a a = true := by
  unfold VStar.In.sameTop
  rw [Bool.and_eq_true, Array.all_eq_true]
  refine ⟨beq_self_eq_true _, fun i hi => ?_⟩
  rw [Array.getElem_zip]
  simp

/-- **A level both reps share**: `topsAt` makes one run of the level's nodes, each rep's first row `0`. -/
theorem topsAt_same {src : VStar.In.Src} {l : ℕ} {a : Array (ByteArray × ByteArray)} (h2 : src.tops.size = 2)
    (h0 : (src.tops[0]!)[l]? = some a) (h1 : (src.tops[1]!)[l]? = some a) (ha : a.all (·.1.size == 64) = true) :
    VStar.In.topsAt src l = .ok
      { rows := #[a].flatMap (fun t => t.flatMap fun (n, _) => (Firewall.topRows n).toArray),
        salts := #[a].flatMap (fun t => t.map (·.2)), first := #[0, 0] } := by
  have hr : List.range' 0 2 1 = [0, 1] := rfl
  have h0' : (src.tops[0]'(by omega))[l]? = some a := by
    rw [← getElem!_pos src.tops 0 (by omega)]
    exact h0
  have h1' : (src.tops[1]'(by omega))[l]? = some a := by
    rw [← getElem!_pos src.tops 1 (by omega)]
    exact h1
  have ha' : ∀ (i : ℕ) (hi : i < a.size), a[i].1.size = 64 := by simpa using ha
  have hf : Array.findIdx? (VStar.In.sameTop a) #[a] = some 0 :=
    Array.findIdx?_eq_some_iff_getElem.2 ⟨by simp, by simpa using sameTop_self a, fun j hj => absurd hj (by omega)⟩
  simp [VStar.In.topsAt, h2, h0', h1', ha', hf, hr]
  rfl

/-- **RecCap's port inputs** where its level's tops number its `N` instances. -/
theorem capIn_ok {src : VStar.In.Src} {l N : ℕ} {T : VStar.In.Tops} (h : VStar.In.topsAt src l = .ok T)
    (hN : T.rows.size = N) :
    VStar.In.capIn src l N = .ok #[("top", VStar.In.ownRows T.rows T.salts),
      ("out", VStar.In.ownRows (RecCoins.zeroRows N VStar.CAP_OUT_WORDS)
        ((Array.range N).map (RecCoins.zeroSalt RecCoins.CAP_ZERO_VALUE)))] := by
  subst hN
  simp [VStar.In.capIn, h, bind, Except.bind, pure, Except.pure]

/-- **An own-rows port's instance `n` reads row `n` under salt `n`.** -/
theorem portRow_ownRows {nm : String} {rows : Array (Array ℕ)} {salts : Array ByteArray}
    {rest : Array (String × VStar.In.PortIn)} {n : ℕ} (hn : n < rows.size) (hs : n < salts.size) :
    portRow (#[(nm, VStar.In.ownRows rows salts)] ++ rest) nm n = some ⟨rows[n], salts[n]⟩ := by
  have hz : n < (rows.zip salts).size := by rw [Array.size_zip]; exact Nat.lt_min.2 ⟨hn, hs⟩
  simp [portRow, VStar.In.ownRows, Array.find?_append, hn, Array.getElem?_map,
    Array.getElem?_eq_getElem hz, Array.getElem_zip]

end RecDeploy.VStarBuild
