import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.LamDeployed

/-!
# The deployed brings' setup in λ₂'s call bound

`BitsLin2` charges each finder `v`, a bound on the compiled verifier's own evaluations at a run's V\* statements
(`Claims.Recursion.setupCalls`). At `deployedB b` those are `deployedSetup`'s at each of the `nS x` statements, so `v`
is at most `nS x · c` when each statement's `deployedSetup` has at most `c` inputs (`setupCalls_deployedB_le`). At
`nS x ≤ 2^16` and `c ≤ 2^39`, `v ≤ 2^55` (`setupCalls_deployedB_le_pow`) meets `lamLin2_deployed_ge_dep`'s `hV`, so
`λ₂ ≥ 100` at `deployedB b` (`lamLin2_deployedB_ge`). Only the statements a run stages need the count
(`setupCalls_deployedB_le_at`). At one statement, `deployedSetup` is check 5's evaluations and the registered reads'
(`deployedSetup_length`), and the latter are at most `|sts| · |own| · N · D` (`regIns_length_le`): one opening per
instance, port and table, each of `2 + log2ceil leaves` inputs (`opensInputs_length`).
-/

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

theorem flatMap_length_le {α β : Type} (f : α → List β) (c : ℕ) :
    ∀ xs : List α, (∀ a ∈ xs, (f a).length ≤ c) → (xs.flatMap f).length ≤ xs.length * c
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, List.length_cons, Nat.add_mul, Nat.one_mul,
      Nat.add_comm (l.length * c)]
    exact Nat.add_le_add (h a (List.mem_cons.2 (Or.inl rfl)))
      (flatMap_length_le f c l fun b hb => h b (List.mem_cons_of_mem _ hb))

/-- **The compiled verifier's own evaluations at a run's V\* statements**: at most `nS x · c` when each of the run's
V\* statements (`vstar`) has at most `c` setup inputs. -/
theorem setupCalls_deployedB_le_at (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) {c : ℕ} (hc : ∀ xO ∈ vstar (deployedB b).st x ps, (deployedSetup xO).length ≤ c) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ nS x * c := by
  have h := flatMap_length_le (fun xO => (deployedSetup xO).map fun u => (u, H u)) c
    (vstar (deployedB b).st x ps) fun xO hx => by simpa only [List.length_map] using hc xO hx
  rwa [vstar_length] at h

/-- **The compiled verifier's own evaluations at the deployed brings**: at most `nS x · c` at a run's V\* statements,
when each statement's `deployedSetup` has at most `c` inputs. -/
theorem setupCalls_deployedB_le (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) {c : ℕ} (hc : ∀ xO, (deployedSetup xO).length ≤ c) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ nS x * c :=
  setupCalls_deployedB_le_at rel b L IV H x ps fun xO _ => hc xO

/-! ## The registered reads' count at one statement -/

theorem climbInputs_length (dom : ByteArray) :
    ∀ (sibs : List ByteArray) (cur : ByteArray) (idx d : ℕ),
      (_root_.Flock.Registered.climbInputs dom cur idx d sibs).length = sibs.length
  | [], _, _, _ => rfl
  | _ :: rest, _, _, _ => by
    simp only [_root_.Flock.Registered.climbInputs, List.length_cons, climbInputs_length dom rest]

/-- **One opening's SHA-512 inputs**: the tree leaf's, the frame-v3 leaf's and one per level of the registered tree. -/
theorem opensInputs_length (reg : _root_.Flock.Registered.Port) (bc : ByteArray) (pos : ℕ) (path : ByteArray) :
    (_root_.Flock.Registered.opensInputs reg bc pos path).length = 2 + _root_.Flock.log2ceil reg.leaves := by
  simp only [_root_.Flock.Registered.opensInputs, _root_.Flock.Registered.leafInputs, List.length_cons,
    climbInputs_length, _root_.Flock.Registered.sibs, List.length_map, List.length_range]
  omega

/-- A port lists at most one opening per instance. -/
theorem portOpenings_length_le (st : _root_.Flock.Stmt) (reg : _root_.Flock.Registered.Port) :
    (portOpenings st reg).length ≤ st.pub.n := by
  unfold portOpenings
  split
  · simp
  · exact Nat.zero_le _

/-- **The setup's registered openings at `xO`**: at most one per instance, port and statement. -/
theorem regOpenings_length_le {xO : Msg} {s : RecIns} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {own : Array _root_.Flock.Registered.Port} (hs : setupO xO = some (s, spec, sts))
    (hown : s.base.registered = some own) {N : ℕ} (hN : ∀ st ∈ sts.toList, st.pub.n ≤ N) :
    (regOpenings xO).length ≤ sts.size * (own.size * N) := by
  simp only [regOpenings, hs, hown]
  have h := flatMap_length_le (fun st => own.toList.flatMap fun reg => (portOpenings st reg).map fun o => (reg, o))
    (own.size * N) sts.toList fun st hst => by
      have h2 := flatMap_length_le (fun reg => (portOpenings st reg).map fun o => (reg, o)) N own.toList
        fun reg _ => by simpa only [List.length_map] using (portOpenings_length_le st reg).trans (hN st hst)
      simpa only [Array.length_toList] using h2
  simpa only [Array.length_toList] using h

/-- **The registered reads' inputs at `xO`**: at most `|sts| · |own| · N · D`, at statements of at most `N` instances
each and registered trees of at most `D - 2` levels. `N` is capped by the setup's `m ≤ 35` refusal, and `|own|` and
`D` are the verifier's own registration's. -/
theorem regIns_length_le {xO : Msg} {s : RecIns} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} {own : Array _root_.Flock.Registered.Port} (hs : setupO xO = some (s, spec, sts))
    (hown : s.base.registered = some own) {N D : ℕ} (hN : ∀ st ∈ sts.toList, st.pub.n ≤ N)
    (hD : ∀ reg ∈ own.toList, 2 + _root_.Flock.log2ceil reg.leaves ≤ D) :
    (regIns xO).length ≤ sts.size * (own.size * N) * D := by
  have hmem : ∀ o ∈ regOpenings xO, o.1 ∈ own.toList := by
    intro o ho
    simp only [regOpenings, hs, hown, List.mem_flatMap, List.mem_map] at ho
    obtain ⟨_, _, reg, hreg, _, _, rfl⟩ := ho
    exact hreg
  unfold regIns
  refine Nat.le_trans (flatMap_length_le
    (fun o : _root_.Flock.Registered.Port × ByteArray × ℕ × ByteArray =>
      (_root_.Flock.Registered.opensInputs o.1 o.2.1 o.2.2.1 o.2.2.2).map Claims.toMsg) D (regOpenings xO)
    fun o ho => ?_) (Nat.mul_le_mul_right D (regOpenings_length_le hs hown hN))
  rw [List.length_map, opensInputs_length]
  exact hD o.1 (hmem o ho)

/-- No registered reads where the setup refuses or the verifier holds none. -/
theorem regIns_eq_nil {xO : Msg}
    (h : ∀ s spec sts own, setupO xO = some (s, spec, sts) → s.base.registered ≠ some own) : regIns xO = [] := by
  unfold regIns regOpenings
  cases hs : setupO xO with
  | none => rfl
  | some v =>
    obtain ⟨s, spec, sts⟩ := v
    dsimp only
    cases hown : s.base.registered with
    | none => rfl
    | some own => exact absurd hown (h s spec sts own hs)

theorem deployedSetup_length (xO : Msg) :
    (deployedSetup xO).length = (ResLower.resSetup xO).length + (regIns xO).length := by
  simp only [deployedSetup, List.length_append]

/-- **`v ≤ 2^55` at the deployed brings**, at most `2^16` V\* statements of at most `2^39` setup inputs each:
`lamLin2_deployed_ge_dep`'s `hV`. -/
theorem setupCalls_deployedB_le_pow (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (hn : nS x ≤ 2 ^ 16) (hc : ∀ xO, (deployedSetup xO).length ≤ 2 ^ 39) (ps : List (Msg × Out)) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ 2 ^ 55 :=
  (setupCalls_deployedB_le rel b L IV H x ps hc).trans
    ((Nat.mul_le_mul_right _ hn).trans (by norm_num))

/-- **Route 2C's `λ₂ ≥ 100` at the deployed brings**, at the call bound `v = 2^55` that
`setupCalls_deployedB_le_pow` gives `BitsLin2`'s `v` premise. -/
theorem lamLin2_deployedB_ge (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts',
      PartIn (staged (deployedB b).st x (pick (deployedB b).st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) :=
  lamLin2_deployed_ge_dep rel (deployedB b) L nextE colsE wE join rowIn final w decI hin hO hn hS hU le_rfl

end RecDeploy
