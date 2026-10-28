import Proofs.Flock.Claims.Recursion.VStarPorts
import Proofs.Flock.Claims.Recursion.VStarPin

/-!
# Round 0's rows at RecCap's port `top` (`portRow_cap0`)

At the fitted records (`InnerRec.fit x`) round 0's rows are RecCap's `N` slots at port `top` (`slotsAt`), each a fitted
top: a node's 32 words, then 32 zero words (`fitSlot`, port `top` of V\*'s cap being 64 words, `cap_ports`). The stager
reads level 0's tops off round 0's rows (`topsOf`), one tree both reps share (`topsAt_same`), and RecCap's `N`
instances read them back whole (`topRows_top`, `capIn_ok`, `portRow_ownRows`). At the fitted port inputs (`fitPin`)
nothing changes, as the rows are already of the port's words and the salts 192 bytes, so RecCap's instance `n` reads
round 0's `n`-th row under its salt.
-/

namespace RecDeploy.VStarBuild

open _root_.Flock
open Claims (Msg)

/-! ## A fitted top row -/

/-- Every word of a fitted row is 16 bits. -/
theorem fitRow_lt (k : ℕ) (ws : Array ℕ) : ∀ j (h : j < (fitRow k ws).size), (fitRow k ws)[j] < 65536 := by
  intro j h
  have hm : (fitRow k ws)[j] ∈ (fitRow k ws).toList := Array.mem_toList_iff.2 (Array.getElem_mem h)
  rw [show (fitRow k ws).toList = (ws.toList.map (· % 65536) ++ List.replicate k 0).take k from rfl] at hm
  rcases List.mem_append.1 (List.mem_of_mem_take hm) with h1 | h1
  · obtain ⟨b, -, hb⟩ := List.mem_map.1 h1
    rw [← hb]
    exact Nat.mod_lt _ (by decide)
  · rw [List.eq_of_mem_replicate h1]
    decide

theorem top64_size (ws : Array ℕ) : (fitRow 32 ws ++ Array.replicate 32 0).size = 64 := by
  simp [fitRow_size]

theorem top64_lt (ws : Array ℕ) : ∀ j (h : j < (fitRow 32 ws ++ Array.replicate 32 0).size),
    (fitRow 32 ws ++ Array.replicate 32 0)[j] < 65536 := by
  intro j h
  rw [Array.getElem_append]
  split
  · exact fitRow_lt _ _ _ _
  · simp

theorem extract_top (ws : Array ℕ) : (fitRow 32 ws ++ Array.replicate 32 0).extract 0 32 = fitRow 32 ws := by
  rw [Array.extract_append, Array.extract_eq_self_of_le (by rw [fitRow_size])]
  simp [fitRow_size]

/-- **A fitted node is read back as its top row**: `topRows` of its 32 words, then 32 zero words. -/
theorem topRows_fit (ws : Array ℕ) :
    Firewall.topRows (wordsBytes (fitRow 32 ws)) = [fitRow 32 ws ++ Array.replicate 32 0] :=
  topRows_wordsBytes _ (fitRow_size 32 ws) (fitRow_lt 32 ws)

theorem flatMap_single {α β : Type} (l : List α) (f : α → β) : l.flatMap (fun a => [f a]) = l.map f := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [List.flatMap_cons, ih]

theorem getElem_map_range_toArray {α : Type} (f : ℕ → α) {N n : ℕ}
    (h : n < ((List.range N).map f).toArray.size) : ((List.range N).map f).toArray[n] = f n := by
  simp

theorem pair_append {α : Type} (u v : α) : (#[u, v] : Array α) = #[u] ++ #[v] := by
  simp

theorem saltBytes_size (y : Salt) : (saltBytes y).size = HmRow.SALT_BYTES := by
  show (Array.ofFn y).size = _
  rw [Array.size_ofFn]
  rfl

/-- **A fitted top row is fitted unchanged** at a 64-word port. -/
theorem fitPortRow_top (ws : Array ℕ) (y : Salt) :
    fitPortRow 64 ⟨fitRow 32 ws ++ Array.replicate 32 0, saltBytes y⟩ =
      ⟨fitRow 32 ws ++ Array.replicate 32 0, saltBytes y⟩ :=
  fitPortRow_of (top64_size ws) (top64_lt ws) (saltBytes_size y)

/-! ## V\*'s cap -/

/-- **A staged cap's ports**: `checkParsed` holds them to V\*'s (`vstarRows`): `top` and `out`, 64 words each. -/
theorem cap_ports {l N : ℕ} {c : Circuit} (h : RecVStar.checkParsed (.cap l N) c = .ok ()) :
    ∃ p0 p1 : _root_.Flock.Port, c.ports = #[p0, p1] ∧ p0.name = "top" ∧ p0.words = 64 ∧ p1.name = "out" := by
  obtain ⟨u, hu, -, -, -, hr⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok h
  have hu' : u = _root_.VStar.capUnit l N := by
    cases hu
    rfl
  subst hu'
  have hv : RecVStar.vstarRows (_root_.VStar.Stmt.cap l N).sub (_root_.VStar.capUnit l N) =
      [("top", 64, 1024), ("out", 64, 1024)] := rfl
  rw [hv] at hr
  cases hp : c.ports with
  | mk ps =>
    rw [hp] at hr
    dsimp only at hr
    obtain ⟨p0, l1, rfl, h0, hl1⟩ := List.map_eq_cons_iff.1 hr
    obtain ⟨p1, l2, rfl, h1, hl2⟩ := List.map_eq_cons_iff.1 hl1
    obtain rfl := List.map_eq_nil_iff.1 hl2
    simp only [Prod.mk.injEq] at h0 h1
    exact ⟨p0, p1, rfl, h0.1, h0.2.1, h1.1⟩

theorem capPort_cap {c : Circuit} {p0 p1 : _root_.Flock.Port} (hp : c.ports = #[p0, p1]) (h0 : p0.name = "top") :
    capPort c = some 0 := by
  unfold capPort
  rw [hp]
  exact Array.findIdx?_eq_some_iff_getElem.2 ⟨by simp, by simp [h0], fun j hj => absurd hj (by omega)⟩

/-! ## Round 0 -/

/-- **Level 0's RecCap is a part**: at `capAt`'s index, its statement `.cap 0 N` and its circuit. -/
theorem partAt_cap0 {x : Msg} {r : XR} {p N : ℕ} {c : Circuit} (hr : readX x = some r)
    (hcap : capAt r = some (p, N)) (hc : r.w.parts[p]?.bind partCircuit = some c) :
    partAt x p = some (.cap 0 N, c) := by
  unfold capAt at hcap
  cases hss : (RecVStar.statements r.shape).toOption with
  | none => simp [hss] at hcap
  | some ss =>
    rw [hss, Option.bind_some] at hcap
    split at hcap
    · rename_i N' hb
      simp only [Option.some.injEq, Prod.mk.injEq] at hcap
      obtain ⟨rfl, rfl⟩ := hcap
      have hl : ss[ss.size - 1]? = some (.cap 0 N') := hb
      simp [partAt, hr, hss, hl, hc]
    · cases hcap

/-- **Round 0's rows at the fitted records**: RecCap's `N` fitted tops, each under its salt. -/
theorem roundRows_cap0 {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {p N : ℕ} {c : Circuit} {p0 p1 : _root_.Flock.Port} (hr : readX x = some r)
    (hcap : capAt r = some (p, N)) (hc : r.w.parts[p]?.bind partCircuit = some c) (hp : c.ports = #[p0, p1])
    (h0 : p0.name = "top") (hw : p0.words = 64) :
    roundRows (I.fit x) w ρ hist 0 = (List.range N).map fun n =>
      ⟨fitRow 32 ((I.rows w ρ.1 []).getD n #[]) ++ Array.replicate 32 0, saltBytes (ρ.2 0 n)⟩ := by
  have hq := capPort_cap hp h0
  have hs : slotsAt x 0 = (List.range N).map fun n => (p, 0, n) := by
    simp [slotsAt, hr, hcap, hc, hq]
  have hcx : circAt x p = c := by simp [circAt, hr, hc]
  have hf : ∀ n ws, fitSlot x (p, 0, n) ws = fitRow 32 ws ++ Array.replicate 32 0 := by
    intro n ws
    simp [fitSlot, isTop, widthAt, hcx, hp, h0, hw]
  apply List.ext_getElem
  · simp [roundRows, InnerRec.fit, hs]
  · intro n h1 h2
    simp [roundRows, InnerRec.fit, hs, hf]

/-- **RecCap's instance `n` reads round 0's `n`-th row** at the fitted port inputs: at level 0's RecCap (`capAt`), whose
staged circuit V\* parsed (`checkParsed`), the stager's port inputs exist, and port `top`'s row at each instance below
`N` is round 0's at that index. -/
theorem portRow_cap0 {I : InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : (I.fit x).Tape}
    {hist : List (Msg × Msg)} {r : XR} {rs : Array RecVStar.SchedRound} {p N : ℕ} {c : Circuit}
    (hr : readX x = some r) (hrs : repSched r.shape = some rs) (hcap : capAt r = some (p, N))
    (hc : r.w.parts[p]?.bind partCircuit = some c) (hcp : RecVStar.checkParsed (.cap 0 N) c = .ok ()) :
    ∃ src pin, srcOf (I.fit x) x w ρ hist = some src ∧ _root_.VStar.In.partIn src (progOf c) (.cap 0 N) = .ok pin ∧
      (roundRows (I.fit x) w ρ hist 0).length = N ∧
      ∀ n < N, portRow (fitPin c pin) "top" n = (roundRows (I.fit x) w ρ hist 0)[n]? := by
  obtain ⟨p0, p1, hp, h0, hw, -⟩ := cap_ports hcp
  have hrr := roundRows_cap0 (I := I) (w := w) (ρ := ρ) (hist := hist) hr hcap hc hp h0 hw
  have hnode : (roundRows (I.fit x) w ρ hist 0).map
        (fun (rw : _root_.VStar.In.Row) => (wordsBytes (rw.words.extract 0 32), rw.salt)) =
      (List.range N).map fun n => (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n)) := by
    rw [hrr, List.map_map]
    apply List.map_congr_left
    intro n _
    simp only [Function.comp_apply, extract_top]
  have hsrc : srcOf (I.fit x) x w ρ hist = some { (I.fit x).rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) } := by
    simp [srcOf, hr, hrs]
  have ht2 : (topsOf rs (roundRows (I.fit x) w ρ hist)).size = 2 := by simp [topsOf]
  have ht0 : ((topsOf rs (roundRows (I.fit x) w ρ hist))[0]!)[0]? = some ((List.range N).map fun n =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n))).toArray := by
    rw [← hnode]
    rfl
  have ht1 : ((topsOf rs (roundRows (I.fit x) w ρ hist))[1]!)[0]? = some ((List.range N).map fun n =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n))).toArray := by
    rw [← hnode]
    rfl
  have ha : ((List.range N).map fun n =>
      (wordsBytes (fitRow 32 ((I.rows w ρ.1 []).getD n #[])), saltBytes (ρ.2 0 n))).toArray.all
        (·.1.size == 64) = true := by
    simp [wordsBytes_size, fitRow_size]
  obtain ⟨F, hT⟩ : ∃ F, _root_.VStar.In.topsAt { (I.fit x).rest w ρ.1 hist with
      msgs := msgsOf rs (roundRows (I.fit x) w ρ hist), tops := topsOf rs (roundRows (I.fit x) w ρ hist) } 0 =
      .ok ⟨((List.range N).map fun n => fitRow 32 ((I.rows w ρ.1 []).getD n #[]) ++ Array.replicate 32 0).toArray,
        ((List.range N).map fun n => saltBytes (ρ.2 0 n)).toArray, F⟩ := by
    refine ⟨#[0, 0], (topsAt_same ht2 ht0 ht1 ha).trans ?_⟩
    congr 2
    · apply Array.toList_inj.1
      simp [List.flatMap_map, topRows_fit, flatMap_single]
    · apply Array.toList_inj.1
      simp
  have hpin := capIn_ok (N := N) hT (by simp)
  refine ⟨_, _, hsrc, hpin, by simp [hrr], fun n hn => ?_⟩
  have hpw : portWords c "top" = 64 := by
    have hq : c.ports.findIdx? (·.name == "top") = some 0 := capPort_cap hp h0
    unfold portWords
    rw [hq]
    simp [hp, hw]
  rw [portRow_fitPin, pair_append, portRow_ownRows, Option.map_some, hpw]
  · rw [getElem_map_range_toArray, getElem_map_range_toArray]
    rw [fitPortRow_top, hrr, List.getElem?_map, List.getElem?_range hn, Option.map_some]
  · simpa using hn
  · simpa using hn

end RecDeploy.VStarBuild
