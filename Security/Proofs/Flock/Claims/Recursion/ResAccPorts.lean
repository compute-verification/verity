import Proofs.Flock.Claims.Recursion.ResLowerD
import Proofs.Flock.C3.PubRows.Check

/-!
# P3's halves that read no other part: a part's end reads, and two chains with one step

`AccPorts` (P3) asks that a piece's copy read the accumulators `elemReadC (Top.accRC rd key)` reads, up to one offset
per component. The copy's `acc0` and `acc1` are rows of the outer witness's own `rec-acc` chain; `accRC` reads
`Top.chainC`'s. Two facts need only the part's own checks or the chain's steps:

* **Where the copy reads** (`ends_of_checkAccReads`, `endReads_inner`): at a statement that is no level,
  `checkAccReads` holds its `acc0` and `acc1` to `endReads`' rows (`RecVStar.checkEndReads`); at a part of the table's
  algebra, `acc0` reads rows `0` and `n` and `acc1` rows `n` and `2 · n` (`n` openings a rep): rep `r`'s ends.
* **Two chains with one step** (`chainHolds_off`): two chains that carry row D's running sums (`Top.ChainHolds`) differ
  along each rep's run of table `t`'s chain by one constant per component, since each step (`consTerm`) adds one term
  to both (`off_of_steps`), and the rep's pieces step every position of its run (`pos_split`).
* **P3 from a chain at the ends** (`accPorts_of_chain`): at a dense decision, a copy whose accumulator ports read a
  chain that carries row D's sums at its rep's ends reads `elemReadC (Top.accRC rd key)` plus one offset per component.
  What's left for `AccPorts` is that chain: the outer witness's `rec-acc` rows, which the copy reads at its ends.
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open FlockSoundness.Refine (fOf fOf_add fOf_injective)

/-! ## Where a part's copy reads the chain -/

/-- **`endReads` at a part of the table's algebra**: `acc0` reads rows `0` and `n`, `acc1` rows `n` and `2 · n`. -/
theorem endReads_inner {i : RecVStar.Inner} {sch : Schedule} (hsch : fast100 i.m = some sch) (j : ℕ) (nm : String)
    (pt : VStar.Part) : RecVStar.endReads i (.inner j nm pt) =
      .ok (some (#[some 0, some (RecVStar.opensBelow sch sch.levels.size)],
        #[some (RecVStar.opensBelow sch sch.levels.size), some (2 * RecVStar.opensBelow sch sch.levels.size)])) := by
  simp only [RecVStar.endReads, hsch]
  rfl

/-- A part of the table's algebra is no level: `accReads` has no rows for it. -/
theorem accReads_inner (i : RecVStar.Inner) (j : ℕ) (nm : String) (pt : VStar.Part) :
    RecVStar.accReads i (.inner j nm pt) = .ok none := rfl

/-- **`checkAccReads` at a statement that is no level**: each of its `accPorts` is `own`'s first entry of its name, and
reads `endReads`' rows of the name (`endWant`). -/
theorem ends_of_checkAccReads {i : RecVStar.Inner} {s : VStar.Stmt} {own : Array Registered.Port}
    (h : RecVStar.checkAccReads i s own = .ok ()) (hn : RecVStar.accReads i s = .ok none)
    {w : Array (Option ℕ) × Array (Option ℕ)} (hw : RecVStar.endReads i s = .ok (some w)) :
    ∀ nm ∈ RecVStar.accPorts s, ∃ p, own.find? (·.name == nm) = some p ∧ p.reads = RecVStar.endWant w nm := by
  unfold RecVStar.checkAccReads at h
  rw [hn] at h
  obtain ⟨_, h0, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h0
  dsimp only at h
  unfold RecVStar.checkEndReads at h
  split at h
  · rename_i he
    intro nm hnm
    rw [List.isEmpty_iff.mp he] at hnm
    cases hnm
  · rw [hw] at h
    obtain ⟨_, h1, h⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
    dsimp only at h
    obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
    refine FlockC3.PubRows.rows_forIn_each _ ?_ _ h
    intro nm r hr
    split at hr
    · rename_i p hf
      split at hr
      · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
        cases h2
      · rename_i hne
        refine ⟨⟨p, hf, by simpa using hne⟩, ?_⟩
        cases hr
        rfl
    · obtain ⟨_, h2, -⟩ := FlockSoundness.Refine.ebind_ok hr
      cases h2

/-! ## Two chains with one step -/

/-- **A step adds one term**: `consTerm`'s component `k` is the running sum's plus the opening's terms. -/
theorem fOf_consTerm (LANES : ℕ) (r a c : Array F128) (k : ℕ) :
    fOf (FlockVBridge.consTerm LANES r a c k) =
      fOf a[k]! + ∑ l ∈ Finset.range LANES, fOf (getElem! c (k * LANES + l) * getElem! r l) := by
  simp only [FlockVBridge.consTerm]
  exact FlockVBridge.fOf_foldl_add (fun l => getElem! c (k * LANES + l) * getElem! r l) LANES a[k]!

/-- **Two chains stepped alike from `base` differ by one constant**: wherever both take each of the `N` steps after
`base` by one `consTerm` (one row, one `coef`), each of their first `N + 1` rows' component `k` is the other's plus
their difference at `base`. -/
theorem off_of_steps {A B : ℕ → Array F128} {k base N : ℕ}
    (hstep : ∀ j < N, ∃ (LANES : ℕ) (r c : Array F128),
      (A (base + j + 1))[k]! = FlockVBridge.consTerm LANES r (A (base + j)) c k ∧
      (B (base + j + 1))[k]! = FlockVBridge.consTerm LANES r (B (base + j)) c k) :
    ∀ j ≤ N, (A (base + j))[k]! = (B (base + j))[k]! + ((A base)[k]! + (B base)[k]!)
  | 0, _ => by
    rw [Nat.add_zero]
    apply fOf_injective
    rw [fOf_add, fOf_add, ← add_assoc, add_comm (fOf (B base)[k]!) (fOf (A base)[k]!), add_assoc,
      CharTwo.add_self_eq_zero, add_zero]
  | j + 1, hj => by
    obtain ⟨L, r, c, hA, hB⟩ := hstep j (by omega)
    have ih := off_of_steps hstep j (by omega)
    rw [← Nat.add_assoc, hA, hB]
    apply fOf_injective
    simp only [fOf_add, fOf_consTerm]
    rw [ih]
    simp only [fOf_add]
    ring

/-- **A position of blocks `f 0, f 1, …` laid end to end** is one block's and one offset in it. -/
theorem pos_split (f : ℕ → ℕ) : ∀ (L p : ℕ), p < ((List.range L).map f).sum →
    ∃ ℓ < L, ∃ j < f ℓ, ((List.range ℓ).map f).sum + j = p
  | 0, p, h => by simp at h
  | L + 1, p, h => by
    rw [List.range_succ, List.map_append, List.sum_append] at h
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero] at h
    by_cases hp : p < ((List.range L).map f).sum
    · obtain ⟨ℓ, hℓ, j, hj, he⟩ := pos_split f L p hp
      exact ⟨ℓ, by omega, j, hj, he⟩
    · exact ⟨L, by omega, p - ((List.range L).map f).sum, by omega, by omega⟩

/-- **Two chains that carry row D's running sums differ by one constant along a rep's run**: at a setup, along rep
`r`'s run of table `t`'s chain (rows `r · N` to `r · N + N`, `N` the rep's openings at the table's schedule), each
row's component `k` of `A` is `B`'s plus their difference at the run's start. Each position of the run is a piece's
(`pos_split`, `mem_rowDPiecesOf`), whose `accHolds` steps both chains by one row and one `coef`. -/
theorem chainHolds_off {rd : Msg → Option FinalP} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {x : Msg}
    {ms : List Msg} {cs : List Coin} {A B : ℕ → ℕ → Array F128} (hA : Top.ChainHolds rd key x ms cs A)
    (hB : Top.ChainHolds rd key x ms cs B) {v} (hv : setupOf false x = some v)
    {t : ℕ} (ht : t < v.2.2.size) {r : ℕ} (hr : r < 2) {k : ℕ} (hk : k < 2) :
    ∀ p ≤ Rows.nOpen (Rows.schOf x t),
      (A t (r * Rows.nOpen (Rows.schOf x t) + p))[k]! =
        (B t (r * Rows.nOpen (Rows.schOf x t) + p))[k]! +
          ((A t (r * Rows.nOpen (Rows.schOf x t)))[k]! + (B t (r * Rows.nOpen (Rows.schOf x t)))[k]!) := by
  have hsch : Rows.schOf x t = Rows.schAt v.2.2 t := by simp only [Rows.schOf, Rows.schAt, hv]
  refine off_of_steps (A := A t) (B := B t) fun p hp => ?_
  obtain ⟨ℓ, hℓ, j, hj, he⟩ := pos_split (fun ℓ => ((Rows.schOf x t).levels.getD ℓ default).q)
    (Rows.schOf x t).levels.size p hp
  have hq : (⟨t, r, ℓ, j, key t r ℓ j⟩ : OpenPiece) ∈ Rows.rowDPieces key x := by
    simp only [Rows.rowDPieces, hv]
    rw [Rows.mem_rowDPiecesOf]
    rw [hsch] at hℓ hj
    refine ⟨ht, hr, hℓ, ?_, rfl⟩
    rwa [← Array.getElem!_eq_getD] at hj
  have hpos : Rows.chainPos (Rows.schOf x t) r ℓ j = r * Rows.nOpen (Rows.schOf x t) + p := by
    simp only [Rows.chainPos]
    rw [Nat.add_assoc, he]
  have hA' := hA _ hq k hk
  have hB' := hB _ hq k hk
  have hs : (Rows.sideAt rd x ms ⟨t, r, ℓ, j, key t r ℓ j⟩).sch = Rows.schOf x t := rfl
  dsimp only at hA' hB'
  rw [hs, hpos] at hA' hB'
  exact ⟨_, _, _, hA', hB'⟩

/-! ## The copy's accumulators against `accRC`'s -/

open Flock.RecAlgebra (el rounds) in
theorem el_append_right {a b : Array F128} {j : ℕ} (h : a.size ≤ j) : el (a ++ b) j = el b (j - a.size) := by
  unfold el
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_append_right h]

open Flock.RecAlgebra (el) in
theorem el_four (a b c d : F128) :
    el #[a, b, c, d] 0 = a ∧ el #[a, b, c, d] 1 = b ∧ el #[a, b, c, d] 2 = c ∧ el #[a, b, c, d] 3 = d :=
  ⟨rfl, rfl, rfl, rfl⟩

open Flock.RecAlgebra (el rounds) in
open FlockDeploy (tabsOf) in
/-- **P3 from a chain at the copy's ends**: at a dense decision, a copy whose accumulator ports read a chain `W` that
carries row D's running sums at its rep's ends (`D.accE` at `W`: `acc0` the run's first row, `acc1` its last) reads
`elemReadC (Top.accRC rd key)` there plus one offset per component, `W`'s minus `chainC`'s at the run's start
(`chainHolds_off`). The accumulator ports are `w`'s last four (`hjb`). -/
theorem accPorts_of_chain {rd : Msg → Option FinalP} {key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ} {xi : Msg}
    {ms : List Msg} {cs : List Coin} (D : Rows.DecAt rd xi ms cs) (cls : FlockSoundness.Discharge.Recursion.HClass)
    {xo : Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto) (hf : Claims.Recursion.Form.Dense sto.form)
    (hx : xi = xIOf xo) {W : ℕ → ℕ → Array F128} (hW : Top.ChainHolds rd key xi ms cs W) (kk : ℕ × ℕ × ℕ × ℕ)
    (hn : kk.2.1 + kk.2.2.1 < 2) (copy : ℕ → F128)
    (hjb : ∀ e j, elemPort xi kk.1 e = some j → accBase xi ≤ j → j < accBase xi + 4)
    (hends : ∀ e j, elemPort xi kk.1 e = some j → accBase xi ≤ j →
      copy e = el (D.accE (fun _ _ => W) (kk.2.1 + kk.2.2.1)) (j - accBase xi)) :
    ∃ δ : ℕ → F128, ∀ e j, elemPort xi kk.1 e = some j → accBase xi ≤ j →
      copy e = Rows.elemReadC (Top.accRC rd key) xi ms cs kk e + δ ((j - accBase xi) % 2) := by
  obtain ⟨jj, o, g, u⟩ := kk
  dsimp only at hn hjb hends ⊢
  obtain ⟨hT1, -⟩ := D.tables_of_dense hst hf hx
  have h0 : 0 < D.sts.size := by omega
  have hG := D.rowG cls (fun _ => #[]) hst hf hx h0
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, -, hd⟩ := Rows.decR_last D.dec
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hdt : D.sts = v.2.2 := by rw [hsts, hd]
  have hsh : D.sh cls 0 = Rows.shapeNoCoin v.2.1 v.2.2 := by
    rw [D.sh_eq cls _ hG h0, hsp, hsts, hd]
  have hbase : accBase xi = (rounds (Rows.shapeNoCoin v.2.1 v.2.2)).flatten.size := by
    simp only [accBase, hv]
  have hlen : (Rows.framesRead xi ms (o + g)).length = accBase xi := by
    have h := D.framesRead_length cls hst hf hx h0 ⟨o + g, hn⟩
    have e : 2 * 0 + (⟨o + g, hn⟩ : Fin 2).val = o + g := by simp
    rw [e, hsh] at h
    rw [h, hbase]
  have hex : ∃ ch, Top.ChainHolds rd key xi ms cs ch := ⟨W, hW⟩
  have hC : Top.ChainHolds rd key xi ms cs (Top.chainC rd key xi ms cs) := by
    have hc : Top.chainC rd key xi ms cs = Classical.choose hex := by rw [Top.chainC, dif_pos hex]
    rw [hc]
    exact Classical.choose_spec hex
  have ht : 0 < v.2.2.size := by rw [← hdt]; exact h0
  have off : ∀ c < 2, ∀ p ≤ Rows.nOpen (Rows.schOf xi 0),
      (W 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0) + p))[c]! =
        (Top.chainC rd key xi ms cs 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0) + p))[c]! +
          ((W 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0)))[c]! +
            (Top.chainC rd key xi ms cs 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0)))[c]!) :=
    fun c hc => chainHolds_off hW hC hv ht hn hc
  have hN : Rows.nOpen ((fast100 (tabsOf D.sts)[0]!.m).getD default) = Rows.nOpen (Rows.schOf xi 0) := by
    rw [hdt]
    simp only [Rows.schOf, hv]
  have hE : ∀ chainAt : Msg → List Msg → ℕ → ℕ → Array F128, D.accE chainAt (o + g) =
      #[(chainAt xi ms 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0)))[0]!,
        (chainAt xi ms 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0)))[1]!,
        (chainAt xi ms 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0) + Rows.nOpen (Rows.schOf xi 0)))[0]!,
        (chainAt xi ms 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0) + Rows.nOpen (Rows.schOf xi 0)))[1]!] := by
    intro chainAt
    have h2 : (o + g) / 2 = 0 := by omega
    have h3 : (o + g) % 2 = o + g := by omega
    unfold Rows.DecAt.accE
    rw [h2, h3, hN]
  have hacc : Top.accRC rd key xi ms cs = D.accE (fun x' ms' => Top.chainC rd key x' ms' cs) := accR_eq _ D
  refine ⟨fun c => (W 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0)))[c]! +
    (Top.chainC rd key xi ms cs 0 ((o + g) * Rows.nOpen (Rows.schOf xi 0)))[c]!, ?_⟩
  intro e j he hj
  have hc4 := hjb e j he hj
  have hpa : Rows.portAt (Rows.shapeNoCoin v.2.1 v.2.2) jj e = some j := by
    unfold elemPort at he
    rw [hv] at he
    exact he
  have hr : Rows.elemReadC (Top.accRC rd key) xi ms cs (jj, o, g, u) e =
      el (Top.accRC rd key xi ms cs (o + g)) (j - accBase xi) := by
    unfold Rows.elemReadC
    rw [hv]
    dsimp only
    rw [hpa]
    dsimp only
    rw [el_append_right (by rw [List.size_toArray, hlen]; exact hj), List.size_toArray, hlen]
  rw [hends e j he hj, hr, hacc, hE, hE]
  obtain ⟨c, hc, hcj⟩ : ∃ c, c < 4 ∧ j - accBase xi = c := ⟨_, by omega, rfl⟩
  rw [hcj]
  rcases (by omega : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3) with rfl | rfl | rfl | rfl
  · rw [(el_four _ _ _ _).1, (el_four _ _ _ _).1]
    have h := off 0 (by decide) 0 (Nat.zero_le _)
    rw [Nat.add_zero] at h
    exact h
  · rw [(el_four _ _ _ _).2.1, (el_four _ _ _ _).2.1]
    have h := off 1 (by decide) 0 (Nat.zero_le _)
    rw [Nat.add_zero] at h
    exact h
  · rw [(el_four _ _ _ _).2.2.1, (el_four _ _ _ _).2.2.1]
    exact off 0 (by decide) _ le_rfl
  · rw [(el_four _ _ _ _).2.2.2, (el_four _ _ _ _).2.2.2]
    exact off 1 (by decide) _ le_rfl

end RecDeploy.ResLower
