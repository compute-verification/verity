import Proofs.Flock.Claims.Recursion.NextRRows

/-!
# The honest files' `finalC` is at the reader's `z`

`sessionRun_decides_fit` takes `FilesFinalC`: each honest file's `finalC` is the zerocheck's at the reader's `z`, the
first value of rep `k`'s round-2 coin in `repCtx`. Here it holds wherever the replies are the coins' bytes.

* **A rep's sends are its schedule's rounds** (`sendsP_eq`): rep `k`'s sends in the plain session (`JsonRT.sendsP`) are
  `repLens`' `k`th, at a set-up rep (`SchedOf`) whose table has `k_log ≥ 6`.
* **The files' `finalC` at the reader's `z`** (`filesFinalC`): where reply `i` is coin `i`'s bytes, for coins that
  number at least the reps' rounds and two, the reply rep `k`'s zerocheck reads `z` from (`zAtP`) is the coin `repCtx`
  reads it from, in `cs.tail.dropLast`.
* **The honest run decides at its own proofs** (`sessionRun_decides_coins`): `sessionRun_decides_fit` at such coins,
  with no `FilesFinalC` premise.
* **The deployed verifier's coins are enough** (`repLens_sum`, `hlen_of_k`): the reps' rounds are the plain verifier's
  recorded rounds (`rounds false`), so coins numbering `(V false).k x` meet the count premise.
-/

namespace FlockDeploy

open Flock FlockComplete

theorem cutLens_getD {α : Type} (d : α) : ∀ (ns : List ℕ) (l : List α) (k j : ℕ), j < ns.getD k 0 →
    ((cutLens ns l).getD k []).getD j d = l.getD ((ns.take k).sum + j) d
  | [], l, k, j, h => by simp at h
  | n :: ns, l, 0, j, h => by
    simp only [List.getD_cons_zero] at h
    simp only [cutLens, List.getD_cons_zero, List.take_zero, List.sum_nil, Nat.zero_add]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_take, h, ite_true]
  | n :: ns, l, k + 1, j, h => by
    simp only [List.getD_cons_succ] at h
    simp only [cutLens, List.getD_cons_succ, List.take_succ_cons, List.sum_cons]
    rw [cutLens_getD d ns (l.drop n) k j h]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_drop]
    rw [Nat.add_assoc]

theorem sum_take_succ_getD : ∀ (ns : List ℕ) (k : ℕ), (ns.take (k + 1)).sum = (ns.take k).sum + ns.getD k 0
  | [], k => by simp
  | n :: ns, 0 => by simp
  | n :: ns, k + 1 => by
    simp only [List.take_succ_cons, List.sum_cons, List.getD_cons_succ]
    rw [sum_take_succ_getD ns k, Nat.add_assoc]

theorem take_sum_le : ∀ (ns : List ℕ) (k : ℕ), (ns.take k).sum ≤ ns.sum
  | [], k => by simp
  | n :: ns, 0 => by simp
  | n :: ns, k + 1 => by
    simp only [List.take_succ_cons, List.sum_cons]
    have := take_sum_le ns k
    omega

theorem take_eq_map_getD (l : List ℕ) (k : ℕ) (h : k ≤ l.length) :
    l.take k = (List.range k).map fun j => l.getD j 0 := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_map, List.length_range]
    omega
  · intro i h1 h2
    simp only [List.getElem_take, List.getElem_map, List.getElem_range, List.getD_eq_getElem?_getD]
    rw [List.getElem?_eq_getElem (by simp at h1; omega), Option.getD_some]

theorem getD_tail_dropLast {α : Type} (d : α) (l : List α) (i : ℕ) (h : i + 2 < l.length) :
    l.tail.dropLast.getD i d = l.getD (i + 1) d := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_dropLast, List.getElem?_tail, List.length_tail,
    show i < l.length - 1 - 1 by omega, ite_true]

theorem repLens_length (spec : SessionSpec) (sts : Array Stmt) : (repLens spec sts).length = 2 * sts.size := by
  simp [repLens]

/-- **Rep `k`'s sends are its schedule's rounds**: at a set-up rep (`SchedOf`) whose table has `k_log ≥ 6`, its sends
in the plain session at replies whose first is `c₁`'s bytes (`JsonRT.sendsP`) are `repLens`' `k`th. -/
theorem sendsP_eq {resp : ℕ → Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {cs : List Coin} {k : ℕ} (hc : resp 0 = c₁.bytes) (hk : k < 2 * sts.size) (h6 : 6 ≤ sts[k / 2]!.c.kLog)
    (hs : SchedOf (repCtx spec sts c₁ r₀ cs k)) :
    JsonRT.sendsP resp spec (tabsOf sts) k = (repLens spec sts).getD k 0 := by
  obtain ⟨-, hkl, rs, hrs, hsc⟩ := hs
  have h6' : 6 ≤ (repCtx spec sts c₁ r₀ cs k).st.fold.kLog := by
    rw [hkl]
    show 6 ≤ (tabsOf sts)[k / 2]!.kLog
    rw [tabsOf_get! (by omega : k / 2 < sts.size)]
    rw [getElem!_pos sts (k / 2) (by omega)] at h6
    exact h6
  obtain ⟨sch, hsch, hsz⟩ := RecDeploy.repRounds_closed hrs h6'
  have hlen : (repLens spec sts).getD k 0 = rs.size := by
    have h' : (repLens spec sts).getD k 0 = (RecDeploy.repSchedAt spec sts[k / 2]!).length := by
      simp [repLens, List.getD_eq_getElem?_getD, hk]
    rw [h']
    change (repCtx spec sts c₁ r₀ cs k).sched.length = rs.size
    rw [hsc, Array.length_toList]
  rw [hlen, hsz]
  simp only [repCtx] at hsch ⊢
  simp only [JsonRT.sendsP, hc, JsonRT.ptsOf_coin, hsch, Option.getD_some]
  cases (tabsOf sts)[k / 2]!.extra (pointsOf spec c₁) <;> dsimp only

/-- **The honest files' `finalC` is the zerocheck's at the reader's `z`** (`FilesFinalC`): at a `sessionS` run whose
tables are `TableOk` with at most 2,048 extra claims at `c₁`'s link points, where reply `i` is coin `i`'s bytes, for
coins that number at least the reps' rounds and two. Rep `k`'s `z` reply (`zAtP`, `sessionS_reps`) is one past the
link and the earlier reps' rounds, the coin `repCtx` reads in `cs.tail.dropLast`. -/
theorem filesFinalC {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {resp : ℕ → Claims.Msg}
    {spec : SessionSpec} {sts : Array Stmt} {w : Claims.Flock.Wit} {ω : Tape} {c₁ : Coin} {cs : List Coin}
    (hc : resp 0 = c₁.bytes) (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs))
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length) :
    FilesFinalC spec sts c₁ cs.tail.dropLast pfs := by
  unfold FilesFinalC
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts htab hS [] cs.tail.dropLast
  have h6 : ∀ j < 2 * sts.size, 6 ≤ sts[j / 2]!.c.kLog := fun j hj => by
    rw [getElem!_pos sts (j / 2) (by omega)]
    exact (htab _ (by omega)).1.k
  intro k hk
  rw [hsz] at hk
  obtain ⟨-, -, -, -, -, hfz, -⟩ := hall k hk
  have hsum : ((List.range k).map (JsonRT.sendsP resp spec (tabsOf sts))).sum = ((repLens spec sts).take k).sum := by
    rw [take_eq_map_getD _ k (by rw [repLens_length]; omega)]
    congr 1
    refine List.map_congr_left fun j hj => ?_
    have hj' : j < 2 * sts.size := by
      have := List.mem_range.1 hj
      omega
    exact sendsP_eq hc hj' (h6 j hj') (hall j hj').2.1
  have hrl : 2 < (repLens spec sts).getD k 0 := by
    rw [← sendsP_eq hc hk (h6 k hk) (hall k hk).2.1]
    simp only [JsonRT.sendsP]
    omega
  have hS1 := sum_take_succ_getD (repLens spec sts) k
  have hS2 := take_sum_le (repLens spec sts) (k + 1)
  have hz : (repCtx spec sts c₁ [] cs.tail.dropLast k).z =
      (cs.getD (((repLens spec sts).take k).sum + 2 + 1) default).at 0 := by
    show (((cutLens (repLens spec sts) cs.tail.dropLast).getD k []).getD 2 default).at 0 = _
    rw [cutLens_getD (default : Coin) _ _ k 2 hrl,
      getD_tail_dropLast (default : Coin) cs (((repLens spec sts).take k).sum + 2) (by omega)]
  rw [hfz, hz, zAtP, hsum, show ∀ a : ℕ, 1 + a + 2 = a + 2 + 1 from fun a => by omega,
    hresp (((repLens spec sts).take k).sum + 2 + 1) (by omega), JsonRT.coinOf_bytes]

/-- **The honest run decides at its own proofs, at coins the replies are**: `sessionRun_decides_fit`, where reply `i`
is coin `i`'s bytes and the coins number at least the reps' rounds and two, so the files' `finalC` are at the reader's
`z` (`filesFinalC`). -/
theorem sessionRun_decides_coins {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H)
    {rd : Claims.Msg → Option FinalP} {x last : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {c₁ : Coin} {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : Tape} {Ts : ℕ → F128}
    {fl : FinalP} (hx : setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) (hrd : rd last = some fl)
    (hps : fl.ps = filePs spec sts c₁ pfs) (hst : fl.stmt = x) :
    ∃ d, decR rd x (honestMsgs (fileCaps spec sts c₁ pfs) ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
        (Ts k)) last) cs = some d := by
  have hc : resp 0 = c₁.bytes := by
    cases cs with
    | nil => simp at hcs
    | cons c cs' =>
      simp only [List.head?_cons, Option.some.injEq] at hcs
      subst hcs
      simpa using hresp 0 (by simp)
  exact sessionRun_decides_fit h64 hx hcs hc htab hS
    (filesFinalC h64 hc (setup_tables hx).2.symm htab hS hresp hlen) hrd hps hst

theorem sum_range_two_mul (g : ℕ → ℕ) : ∀ n,
    ((List.range (2 * n)).map fun k => g (k / 2)).sum = ((List.range n).map fun i => 2 * g i).sum
  | 0 => by simp
  | n + 1 => by
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by omega, List.range_succ, List.range_succ, List.range_succ (n := n),
      List.map_append, List.map_append, List.map_append, List.sum_append, List.sum_append, List.sum_append,
      sum_range_two_mul g n]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    rw [show (2 * n + 1) / 2 = n by omega, show 2 * n / 2 = n by omega]
    omega

theorem foldl_add_eq_sum {α : Type} (f : α → ℕ) : ∀ (l : List α) (a : ℕ),
    l.foldl (fun acc x => acc + f x) a = a + (l.map f).sum
  | [], a => by simp
  | x :: l, a => by
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [foldl_add_eq_sum f l]
    omega

theorem array_foldl_eq {α : Type} [Inhabited α] (f : α → ℕ) (xs : Array α) :
    xs.foldl (fun acc x => acc + f x) 0 = ((List.range xs.size).map fun i => f xs[i]!).sum := by
  rw [← Array.foldl_toList, foldl_add_eq_sum, Nat.zero_add]
  congr 1
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range]
    rw [getElem!_pos xs i (by simpa using h2)]
    simp

/-- **A table's rep schedule has the plain verifier's rounds** (`repRounds false`), at a `fast100` schedule and
`k_log ≥ 6`. -/
theorem repSchedAt_length {spec : SessionSpec} {st : Stmt} (hf : (fast100 st.m).isSome) (h6 : 6 ≤ st.c.kLog) :
    (RecDeploy.repSchedAt spec st).length = repRounds false spec st := by
  obtain ⟨sch, hsch⟩ := Option.isSome_iff_exists.1 hf
  obtain ⟨-, h35, -⟩ := RecDeploy.fast100_sizes hsch
  have key : ∀ m < 36, (fast100 m).isSome → (RecVStar.repRounds m 6 1).isSome := by decide +kernel
  obtain ⟨out, hout⟩ := Option.isSome_iff_exists.1 (key _ (by omega) hf)
  obtain ⟨T, -, hall⟩ := FlockC3.Compose.repRounds_tail hout
  have h2 := hall st.c.kLog (2 + spec.points * st.regions.size)
  rw [RecDeploy.flockRepRounds_eq h2 h6]
  simp only [RecDeploy.repSchedAt, h2, Option.map_some, Option.getD_some, Array.length_toList]

/-- **The reps' rounds are the plain verifier's recorded rounds** (`rounds false`), at `TableOk` tables. -/
theorem repLens_sum {spec : SessionSpec} {sts : Array Stmt}
    (hok : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i]) :
    (repLens spec sts).sum = rounds false spec sts := by
  unfold repLens rounds
  rw [sum_range_two_mul (fun i => (RecDeploy.repSchedAt spec sts[i]!).length),
    array_foldl_eq (fun st => 2 * repRounds false spec st)]
  congr 1
  refine List.map_congr_left fun i hi => ?_
  have hi' := List.mem_range.1 hi
  show 2 * (RecDeploy.repSchedAt spec sts[i]!).length = 2 * repRounds false spec sts[i]!
  rw [getElem!_pos sts i hi', repSchedAt_length (FlockComplete.fast100_isSome (hok i hi').m22 (hok i hi').m35)
    (hok i hi').k]

/-- **`hlen` holds at the deployed verifier's coin count**: at a statement that sets up with `TableOk` tables, coins
numbering `(V false).k x`, the link's, the reps' rounds (`repLens_sum`) and the last message's, are at least the
reps' rounds and two. -/
theorem hlen_of_k {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {cs : List Coin}
    (hx : setupOf false x = some (s, spec, sts)) (hok : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i])
    (hk : cs.length = (V false).k x) : (repLens spec sts).sum + 2 ≤ cs.length := by
  rw [hk, repLens_sum hok]
  simp only [V, hx]
  omega

/-- **Non-vacuity of the coin premises**: replies that are a coin list's bytes meet `hresp`, and its head's bytes are
the first reply; `hlen` holds at the deployed verifier's coin count (`hlen_of_k`). -/
example (c₁ : Coin) (cs : List Coin) :
    (∀ i < (c₁ :: cs).length, (fun i => ((c₁ :: cs).getD i default).bytes) i = ((c₁ :: cs).getD i default).bytes) ∧
      (c₁ :: cs).head? = some c₁ :=
  ⟨fun _ _ => rfl, rfl⟩

end FlockDeploy
