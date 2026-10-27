import Proofs.Properties.Nci.BodiesOk.Basic

/-!
# What `buildNode` checks of a node

`Flock.Program.buildNode` rebuilds a node from its arguments and its callee, and refuses one whose arguments don't fit
the callee's parameters. `buildNode_spec` reads off what it accepted: per form, the arguments' types against the
callee's parameters, the batch axes and scan `xs` all of length `n`, and the output type it computed.
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk

open Flock.Program
open private buildNode from Flock.Program

/-- What `buildNode` establishes of a node of form `form` calling `c` with arguments `args`, computing `out`. -/
def Shape (c : Defn) (form : Form) (args : Array Arg) (out : Ty) (n nxs : ℕ) : Prop :=
  match form with
  | .prim | .call => out = c.ret ∧ ∀ i < args.size, (args[i]!).ty = (c.params[i]!).2
  | .batch => out = .array n c.ret ∧ ∀ i < args.size,
      ((args[i]!).dims = none ∧ (args[i]!).ty = (c.params[i]!).2) ∨
      ∃ ax mt, (args[i]!).ty.removeAxis ax = some (n, mt) ∧ mt = (c.params[i]!).2 ∧
        (args[i]!).dims = (args[i]!).ty.axisDims ax
  | .scan => ∃ carry y, c.ret = .tuple #[carry, y] ∧ out = .tuple #[carry, .array n y] ∧
      (c.params[0]!).2 = carry ∧ (args[0]!).ty = carry ∧ ∀ i < args.size, i ≠ 0 →
        (i ≤ nxs → ∃ e, (args[i]!).ty = .array n e ∧ e = (c.params[i]!).2 ∧ (args[i]!).dims = some (n, e.leaves)) ∧
        (¬ i ≤ nxs → (args[i]!).ty = (c.params[i]!).2)

theorem len_eq (L : Array ℕ) (n ln : ℕ) (hm : ln ∈ L) (h1 : ¬ L.size > 1)
    (h2 : ¬ (L.size == 1 && L[0]! != n) = true) : ln = n := by
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hm
  have hs : L.size = 1 := by omega
  have hi0 : i = 0 := by omega
  subst hi0
  simp only [hs, beq_self_eq_true, Bool.true_and, bne_iff_ne, ne_eq, Decidable.not_not] at h2
  rw [← h2, getElem!_pos L 0 hi]

theorem error_ne_ok {ε α : Type} {e : ε} {a : α} : (Except.error e : Except ε α) ≠ Except.ok a := by
  intro h; injection h

theorem buildNode_spec (fid : String) (k : ℕ) (form : Form) (callee : Defn) (args : Array Arg)
    (axes : Array (Option ℕ)) (n nxs : ℕ) (out : Ty) (args' : Array Arg)
    (h : buildNode fid k form callee args axes n nxs = .ok (out, args')) (hscan : form = .scan → 0 < args.size)
    (hdims : ∀ i < args.size, (args[i]!).dims = none) :
    args'.size = args.size ∧ (∀ i < args.size, (args'[i]!).ty = (args[i]!).ty ∧ (args'[i]!).seq = (args[i]!).seq) ∧
      args.size = callee.params.size ∧
      (∀ i < args'.size, ∀ e L : ℕ, (args'[i]!).dims = some (e, L) → L = 0 → (callee.params[i]!).2.leaves = 0) ∧
      Shape callee form args' out n nxs := by
  unfold buildNode at h
  rcases form
  all_goals dsimp only at h
  case prim | call =>
    split at h
    · exact absurd h (by intro h; injection h)
    rename_i hsz
    have hsz' : args.size = callee.params.size := by simpa using hsz
    obtain ⟨u, hloop, hret⟩ := Walk.except_bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hret
    obtain ⟨rfl, rfl⟩ := hret
    rw [← Array.forIn_toList] at hloop
    have hl := forIn_except_inv _ (fun _ => False) callee.params.toList
      (fun i (s : Subarray Arg) => s.array = args ∧ s.start = i ∧ s.stop = args.size ∧
        ∀ j < i, (args[j]!).ty = (callee.params[j]!).2) _ u
      (by obtain ⟨h1, h2, h3⟩ := toStream_array args; exact ⟨h1, h2, h3, fun j hj => by omega⟩)
      (by
        intro i hi b t ⟨ha, hst, hsp, hall⟩ hf
        simp only [Array.length_toList] at hi
        obtain ⟨s', hn, ha', hst', hsp'⟩ := next_at b args i ha hst (by omega)
        simp only [hn] at hf
        split at hf
        · exact absurd hf (by intro h; injection h)
        · rename_i hc
          injection hf with hf; subst hf
          refine ⟨ha', hst', by rw [hsp', hsp], fun j hj => ?_⟩
          rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
          · exact hall j hj
          · have := ty_beq (by simpa using hc)
            rw [this, getElem!_pos callee.params j (by omega)]) hloop
    simp only [Array.length_toList] at hl
    rcases hl with ⟨-, -, -, hl⟩ | hl
    · exact ⟨rfl, fun i _ => ⟨rfl, rfl⟩, hsz', fun i hi e L hd _ => (by rw [hdims i hi] at hd; cases hd), rfl,
        fun i hi => hl i (by omega)⟩
    · exact hl.elim
  case batch =>
    split at h
    · exact absurd h (by intro h; injection h)
    split at h
    · exact absurd h (by intro h; injection h)
    rename_i hax hsz
    have hax' : axes.size = args.size := by simpa using hax
    have hsz' : callee.params.size = args.size := by simpa using hsz
    obtain ⟨⟨L, O, sa, sx⟩, hloop, hrest⟩ := Walk.except_bind_ok h
    dsimp only at hrest
    split at hrest
    · exact absurd hrest (by intro h; injection h)
    rename_i hL1
    split at hrest
    · exact absurd hrest (by intro h; injection h)
    rename_i hL2
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hrest
    obtain ⟨rfl, rfl⟩ := hrest
    rw [← Array.forIn_toList] at hloop
    have hl := forIn_except_inv _ (fun _ => False) callee.params.toList
      (fun i (st : Array ℕ × Array Arg × Subarray Arg × Subarray (Option ℕ)) =>
        st.2.2.1.array = args ∧ st.2.2.1.start = i ∧ st.2.2.1.stop = args.size ∧
        st.2.2.2.array = axes ∧ st.2.2.2.start = i ∧ st.2.2.2.stop = axes.size ∧ st.2.1.size = i ∧
        ∀ j < i, (st.2.1[j]!).ty = (args[j]!).ty ∧ (st.2.1[j]!).seq = (args[j]!).seq ∧
          (((st.2.1[j]!).dims = none ∧ (args[j]!).ty = (callee.params[j]!).2) ∨
           ∃ ax mt ln, (args[j]!).ty.removeAxis ax = some (ln, mt) ∧ mt = (callee.params[j]!).2 ∧
             (st.2.1[j]!).dims = (args[j]!).ty.axisDims ax ∧ ln ∈ st.1)) _ _
      (by
        obtain ⟨h1, h2, h3⟩ := toStream_array args
        obtain ⟨h4, h5, h6⟩ := toStream_array axes
        exact ⟨h1, h2, h3, h4, h5, h6, rfl, fun j hj => by omega⟩)
      (by
        rintro i hi ⟨L, O, sa, sx⟩ t hI hf
        dsimp only at hI ⊢
        obtain ⟨ha, hst, hsp, hx, hxst, hxsp, hO, hall⟩ := hI
        simp only [Array.length_toList] at hi
        obtain ⟨sx', hnx, hx', hxst', hxsp'⟩ := next_at sx axes i hx hxst (by omega)
        obtain ⟨sa', hna, ha', hst', hsp'⟩ := next_at sa args i ha hst (by omega)
        dsimp only at hf
        simp only [hnx, hna] at hf
        have hpi : (callee.params.toList[i]'(by simpa using hi)) = callee.params[i]! := by
          rw [getElem!_pos callee.params i (by omega), Array.getElem_toList]
        rw [hpi] at hf
        have hall' : ∀ j < i, ∀ L' : Array ℕ, (∀ x, x ∈ L → x ∈ L') → ∀ (O' : Array Arg), O'.size = i + 1 →
            (∀ j < i, O'[j]! = O[j]!) →
            (O'[j]!).ty = (args[j]!).ty ∧ (O'[j]!).seq = (args[j]!).seq ∧
            (((O'[j]!).dims = none ∧ (args[j]!).ty = (callee.params[j]!).2) ∨
             ∃ ax mt ln, (args[j]!).ty.removeAxis ax = some (ln, mt) ∧ mt = (callee.params[j]!).2 ∧
               (O'[j]!).dims = (args[j]!).ty.axisDims ax ∧ ln ∈ L') := by
          intro j hj L' hL' O' _ hO'
          rw [hO' j hj]
          obtain ⟨h1, h2, h3⟩ := hall j hj
          refine ⟨h1, h2, ?_⟩
          rcases h3 with h3 | ⟨ax, mt, ln, h4, h5, h6, h7⟩
          · exact Or.inl h3
          · exact Or.inr ⟨ax, mt, ln, h4, h5, h6, hL' _ h7⟩
        split at hf
        · rename_i hnone
          split at hf
          · exact absurd hf (by intro h; injection h)
          rename_i hc
          injection hf with hf; subst hf
          refine ⟨ha', hst', by rw [hsp', hsp], hx', hxst', by rw [hxsp', hxsp], by simp [hO], fun j hj => ?_⟩
          rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
          · exact hall' j hj L (fun x hx => hx) _ (by simp [hO])
              (fun j hj => getElem!_push_lt _ _ (by omega)) 
          · rw [← hO, getElem!_push_eq, hO]
            exact ⟨rfl, rfl, Or.inl ⟨hdims j (by omega), ty_beq (by simpa using hc)⟩⟩
        · rename_i ax hsome
          split at hf
          · rename_i ln mt hrm
            split at hf
            · exact absurd hf (by intro h; injection h)
            rename_i hc
            have hmt : mt = (callee.params[i]!).2 := ty_beq (by simpa using hc)
            split at hf
            all_goals
              injection hf with hf; subst hf
              refine ⟨ha', hst', by rw [hsp', hsp], hx', hxst', by rw [hxsp', hxsp], by simp [hO], fun j hj => ?_⟩
              rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
              · exact hall' j hj _ (fun x hx => by simp [hx]) _ (by simp [hO])
                  (fun j hj => getElem!_push_lt _ _ (by omega))
              · rw [← hO, getElem!_push_eq, hO]
                refine ⟨rfl, rfl, Or.inr ⟨ax, mt, ln, hrm, hmt, rfl, ?_⟩⟩
                first
                  | exact Array.mem_push.2 (Or.inr rfl)
                  | (rename_i hcon; simpa using hcon)
          · exact absurd hf (by intro h; injection h)) hloop
    simp only [Array.length_toList] at hl
    rcases hl with ⟨-, -, -, -, -, -, hO, hl⟩ | hl
    · refine ⟨by omega, fun i hi => ⟨(hl i (by omega)).1, (hl i (by omega)).2.1⟩, hsz'.symm,
        fun i hi e L hd hL => ?_, rfl, fun i hi => ?_⟩
      · obtain ⟨-, -, h3⟩ := hl i (by omega)
        rcases h3 with ⟨h3, -⟩ | ⟨ax, mt, ln, h4, h5, h6, -⟩
        · rw [h3] at hd; cases hd
        · obtain ⟨L', hax, -, hdvd, -⟩ := axis_spec ax _ _ _ h4
          rw [h6, hax] at hd
          simp only [Option.some.injEq, Prod.mk.injEq] at hd
          rw [← h5]
          exact Nat.eq_zero_of_zero_dvd (by rw [← hL, ← hd.2]; exact hdvd)
      obtain ⟨h1, -, h3⟩ := hl i (by omega)
      rcases h3 with ⟨h3, h4⟩ | ⟨ax, mt, ln, h4, h5, h6, h7⟩
      · exact Or.inl ⟨h3, by rw [h1, h4]⟩
      · have := len_eq L n ln h7 hL1 hL2
        subst this
        exact Or.inr ⟨ax, mt, by rw [h1]; exact h4, h5, by rw [h1]; exact h6⟩
    · exact hl.elim
  case scan =>
    have hpos := hscan rfl
    rcases hR : callee.ret with w | ⟨nn, e⟩ | ⟨⟨l⟩⟩ | fs
    rotate_left 2
    rcases l with _ | ⟨a, _ | ⟨b, _ | ⟨c, l⟩⟩⟩
    rotate_left 2
    · rw [hR] at h
      conv at h => lhs; whnf
      generalize instDecidableEqBool (callee.params.size != args.size) true = dd at h
      cases dd with
      | isTrue hc => exact absurd h (by intro h; injection h)
      | isFalse hsz =>
      have hsz' : callee.params.size = args.size := by simpa using hsz
      dsimp only at h
      simp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at h
      split at h
      · exact absurd h (by intro h; injection h)
      rename_i hcarry
      split at h
      · exact absurd h (by intro h; injection h)
      rename_i hinit
      obtain ⟨⟨L, O⟩, hloop, hrest⟩ := Walk.except_bind_ok h
      dsimp only at hrest
      split at hrest
      · exact absurd hrest (by intro h; injection h)
      rename_i hL1
      split at hrest
      · exact absurd hrest (by intro h; injection h)
      rename_i hL2
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hrest
      obtain ⟨rfl, rfl⟩ := hrest
      rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
      have hl := forIn_except_inv _ (fun _ => False) _
        (fun i (st : Array ℕ × Array Arg) => st.2.size = i + 1 ∧ st.2[0]! = args[0]! ∧
          ∀ j < i + 1, j ≠ 0 → (st.2[j]!).ty = (args[j]!).ty ∧ (st.2[j]!).seq = (args[j]!).seq ∧
            (j ≤ nxs → ∃ ln e, (args[j]!).ty = .array ln e ∧ e = (callee.params[j]!).2 ∧
              (st.2[j]!).dims = some (ln, e.leaves) ∧ ln ∈ st.1) ∧
            (¬ j ≤ nxs → (args[j]!).ty = (callee.params[j]!).2 ∧ (st.2[j]!).dims = none)) _ _
        (by refine ⟨rfl, rfl, fun j hj hj0 => by omega⟩)
        (by
          rintro i hi ⟨L, O⟩ t hI hf
          dsimp only at hI ⊢
          obtain ⟨hO, hO0, hall⟩ := hI
          simp only [List.length_range', Std.Legacy.Range.size] at hi
          simp only [List.getElem_range', Nat.one_mul] at hf
          rw [show 1 + i = O.size by omega] at hf
          have hall' : ∀ j < i + 1, ∀ L' : Array ℕ, (∀ x, x ∈ L → x ∈ L') → ∀ (O' : Array Arg),
              (∀ j < i + 1, O'[j]! = O[j]!) → j ≠ 0 → (O'[j]!).ty = (args[j]!).ty ∧ (O'[j]!).seq = (args[j]!).seq ∧
              (j ≤ nxs → ∃ ln e, (args[j]!).ty = .array ln e ∧ e = (callee.params[j]!).2 ∧
                (O'[j]!).dims = some (ln, e.leaves) ∧ ln ∈ L') ∧
              (¬ j ≤ nxs → (args[j]!).ty = (callee.params[j]!).2 ∧ (O'[j]!).dims = none) := by
            intro j hj L' hL' O' hO' hj0
            rw [hO' j hj]
            obtain ⟨h1, h2, h3, h4⟩ := hall j hj hj0
            refine ⟨h1, h2, fun hx => ?_, h4⟩
            obtain ⟨ln, e, h5, h6, h7, h8⟩ := h3 hx
            exact ⟨ln, e, h5, h6, h7, hL' _ h8⟩
          have hpush : ∀ x : Arg, ∀ j < i + 1, (O.push x)[j]! = O[j]! :=
            fun x j hj => getElem!_push_lt _ _ (by omega)
          split at hf
          · rename_i hx
            split at hf
            · rename_i ln e hty
              split at hf <;> rename_i hcon
              all_goals
                split at hf
                · exact absurd hf (by intro h; injection h)
                rename_i hc
                injection hf with hf; subst hf
                refine ⟨by simp [hO], by rw [getElem!_push_lt _ _ (by omega), hO0], fun j hj hj0 => ?_⟩
                rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
                · exact hall' j hj _ (fun x hx => by simp [hx]) _ (hpush _) hj0
                · rw [show i + 1 = O.size by omega, getElem!_push_eq]
                  refine ⟨rfl, rfl, fun _ => ⟨ln, e, hty, ty_beq (by simpa using hc), rfl, ?_⟩, fun h => absurd hx h⟩
                  · first
                      | exact Array.mem_push.2 (Or.inr rfl)
                      | simpa using hcon
            · exact absurd hf (by intro h; injection h)
          · rename_i hx
            split at hf
            · exact absurd hf (by intro h; injection h)
            rename_i hc
            injection hf with hf; subst hf
            refine ⟨by simp [hO], by rw [getElem!_push_lt _ _ (by omega), hO0], fun j hj hj0 => ?_⟩
            rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
            · exact hall' j hj _ (fun x hx => hx) _ (hpush _) hj0
            · rw [show i + 1 = O.size by omega, getElem!_push_eq]
              exact ⟨rfl, rfl, fun h => absurd h hx, fun _ => ⟨ty_beq (by simpa using hc), hdims _ (by omega)⟩⟩)
          hloop
      simp only [List.length_range', Std.Legacy.Range.size] at hl
      rcases hl with ⟨hO, hO0, hl⟩ | hl
      · have hOs : O.size = args.size := by omega
        refine ⟨hOs, fun i hi => ?_, hsz'.symm, fun i hi e L hd hL => ?_, a, b, by rw [hR], rfl,
          ty_beq (by simpa using hcarry), ?_, fun i hi hi0 => ?_⟩
        · rcases Nat.eq_zero_or_pos i with rfl | hi0
          · rw [hO0]; exact ⟨rfl, rfl⟩
          · obtain ⟨h1, h2, -⟩ := hl i (by omega) (by omega)
            exact ⟨h1, h2⟩
        · rcases Nat.eq_zero_or_pos i with rfl | hi0
          · rw [hO0, hdims 0 hpos] at hd; cases hd
          · obtain ⟨-, -, h3, h4⟩ := hl i (by omega) (by omega)
            by_cases hx : i ≤ nxs
            · obtain ⟨ln, e', -, h6, h7, -⟩ := h3 hx
              rw [h7] at hd
              simp only [Option.some.injEq, Prod.mk.injEq] at hd
              rw [← h6, hd.2, hL]
            · rw [(h4 hx).2] at hd; cases hd
        · rw [hO0]; exact ty_beq (by simpa using hinit)
        · obtain ⟨h1, -, h3, h4⟩ := hl i (by omega) hi0
          refine ⟨fun hx => ?_, fun hx => by rw [h1]; exact (h4 hx).1⟩
          obtain ⟨ln, e, h5, h6, h7, h8⟩ := h3 hx
          have := len_eq L n ln h8 hL1 hL2
          subst this
          exact ⟨e, by rw [h1, h5], h6, h7⟩
      · exact hl.elim
    all_goals (rw [hR] at h; conv at h => lhs; whnf)
    all_goals exact absurd h (by intro h; injection h)

/-- What the decoder establishes of a Definition a node may call: a primitive is one gate with one output leaf and an empty
body, a composite's body has its parameters and returns as many references as its type has leaves. -/
def CalleeOk (c : Defn) : Prop :=
  (c.prim = true → c.body = default ∧ c.gates = 1 ∧ c.outLeaves = 1 ∧ ∃ w, c.ret = .value w) ∧
  (c.prim = false → c.body.params = c.params ∧ c.body.ret.count = c.ret.leaves ∧ c.outLeaves = c.ret.leaves)

/-- A node's argument of no leaves to a member (a batch's or scan's) goes to a parameter of `c` of none. -/
def DimsOk (c : Defn) (nd : Node) : Prop :=
  ∀ i < nd.args.size, ∀ e L : ℕ, (nd.args[i]!).dims = some (e, L) → L = 0 → (c.params[i]!).2.leaves = 0

/-- What the decoder establishes of a node calling `c`: one argument per parameter, each as many references as its type
has leaves, a primitive node calling a primitive and a call or scan a composite, `buildNode`'s `Shape`, and `DimsOk`. -/
def NodeOk (c : Defn) (nd : Node) : Prop :=
  nd.args.size = c.params.size ∧ (∀ i < nd.args.size, (nd.args[i]!).seq.count = (nd.args[i]!).ty.leaves) ∧
  (nd.form = .prim → c.prim = true) ∧ (nd.form = .call ∨ nd.form = .scan → c.prim = false) ∧
  Shape c nd.form nd.args nd.out nd.n nd.nxs ∧ DimsOk c nd

theorem beq_prim_batch : (Form.prim == Form.batch) = false := rfl
theorem beq_call_batch : (Form.call == Form.batch) = false := rfl
theorem beq_batch_batch : (Form.batch == Form.batch) = true := rfl
theorem beq_scan_batch : (Form.scan == Form.batch) = false := rfl

theorem lt_mul_of (m n E j : ℕ) (hm : m < n) (hj : j < E) : m * E + j < n * E := by
  have : (m + 1) * E ≤ n * E := Nat.mul_le_mul_right _ hm
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

theorem up_ok (c : Defn) (nd : Node) (hc : CalleeOk c) (hn : NodeOk c nd) : Resolve.UpOk nd c := by
  obtain ⟨hsz, hcnt, hpr, hco, hsh, -⟩ := hn
  intro i hi j hj m hm
  cases hp : c.prim
  · obtain ⟨hbp, hbr, hol⟩ := hc.2 hp
    rw [hbp] at hi hj
    have hia : i < nd.args.size := by omega
    obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
    dsimp only at *
    cases form
    · exact absurd (hpr rfl) (by simp [hp])
    · rw [hcnt i hia, hsh.2 i hia]; exact hj
    · obtain ⟨-, hargs⟩ := hsh
      change m < n at hm
      rcases hargs i hia with ⟨hd, ht⟩ | ⟨ax, mt, hrm, hmt, hd⟩
      · simp only [Resolve.memberIdx, hd]; rw [hcnt i hia, ht]; exact hj
      · obtain ⟨L, hax, hlv, -, hb⟩ := axis_spec ax _ _ _ hrm
        simp only [Resolve.memberIdx, hd, hax]; rw [hcnt i hia]
        exact hb j (by rw [hmt]; exact hj) m hm
    · obtain ⟨carry, y, hret, -, hp0, ha0, hrest⟩ := hsh
      change m < n at hm
      dsimp only
      split
      · subst i
        split
        · rw [hcnt 0 hia, ha0, ← hp0]; exact hj
        · rw [hbr, hret, leaves_tuple2, ← hp0]; omega
      · rename_i hi0
        obtain ⟨hx, hs⟩ := hrest i hia hi0
        split
        · rename_i hix
          obtain ⟨e, hte, hep, hd⟩ := hx hix
          rw [hd, hcnt i hia, hte, leaves_array]
          simp only [Option.map_some, Option.getD_some]
          rw [hep]; exact lt_mul_of m n _ j hm hj
        · rename_i hix
          rw [hcnt i hia, hs hix]; exact hj
  · obtain ⟨hbody, -, -, -⟩ := hc.1 hp
    rw [hbody] at hi
    exact (Nat.not_lt_zero _ hi).elim

theorem down_ok (c : Defn) (nd : Node) (hc : CalleeOk c) (hn : NodeOk c nd) : Resolve.DownOk nd c := by
  obtain ⟨hsz, hcnt, hpr, hco, hsh, -⟩ := hn
  intro j hj
  obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
  dsimp only at *
  have hol : c.outLeaves = c.ret.leaves := by
    cases hp : c.prim
    · exact (hc.2 hp).2.2
    · obtain ⟨-, -, h1, w, hw⟩ := hc.1 hp
      rw [h1, hw, leaves_value]
  cases form
  · have hp := hpr rfl
    exact ⟨hp, (hc.1 hp).2.1⟩
  · have hp := hco (Or.inl rfl)
    obtain ⟨-, hbr, -⟩ := hc.2 hp
    rw [hsh.1] at hj
    exact ⟨hp, by rw [hbr]; exact hj⟩
  · obtain ⟨hout, -⟩ := hsh
    rw [hout, leaves_array] at hj
    have hL : 0 < c.ret.leaves := by
      rcases Nat.eq_zero_or_pos c.ret.leaves with h0 | h0
      · rw [h0] at hj; simp at hj
      · exact h0
    refine ⟨?_, fun hp => (hc.1 hp).2.1, fun hp => ?_⟩
    · rw [hol]; exact (Nat.div_lt_iff_lt_mul hL).2 hj
    · rw [hol, (hc.2 hp).2.1]; exact Nat.mod_lt _ hL
  · have hp := hco (Or.inr rfl)
    obtain ⟨-, hbr, -⟩ := hc.2 hp
    obtain ⟨carry, y, hret, hout, -, -, -⟩ := hsh
    refine ⟨hp, fun hn0 => ⟨carry, y, hret, ?_, ?_⟩⟩
    · rw [hout, leaves_tuple2, leaves_array] at hj
      split
      · omega
      · rename_i hjc
        have hy : 0 < y.leaves := by
          rcases Nat.eq_zero_or_pos y.leaves with h0 | h0
          · rw [h0] at hj; simp at hj; omega
          · exact h0
        exact (Nat.div_lt_iff_lt_mul hy).2 (by omega)
    · rw [hbr, hret, leaves_tuple2]
      rw [hout, leaves_tuple2, leaves_array] at hj
      split
      · omega
      · rename_i hjc
        have hy : 0 < y.leaves := by
          rcases Nat.eq_zero_or_pos y.leaves with h0 | h0
          · rw [h0] at hj; simp at hj; omega
          · exact h0
        have := Nat.mod_lt (j - carry.leaves) hy
        omega

theorem op_ok (c : Defn) (nd : Node) (hn : NodeOk c nd) : Resolve.OpOk nd c := by
  obtain ⟨hsz, hcnt, hpr, hco, hsh, -⟩ := hn
  intro i hi m hm t ht
  obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
  dsimp only at *
  cases form
  case prim | call | scan =>
    simp only [beq_prim_batch, beq_call_batch, beq_scan_batch, Bool.false_and, Bool.false_eq_true,
      ↓reduceIte] at ht ⊢
    exact ht
  change m < n at hm
  simp only [beq_batch_batch, Bool.true_and, ↓reduceIte] at ht ⊢
  obtain ⟨-, hargs⟩ := hsh
  rcases hargs i hi with ⟨hd, -⟩ | ⟨ax, mt, hrm, hmt, hd⟩
  · rw [hd] at ht
    simp only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at ht
    simp only [Resolve.memberIdx, hd]; exact ht
  · obtain ⟨L, hax, hlv, -, hb⟩ := axis_spec ax _ _ _ hrm
    rw [hd, hax] at ht
    simp only [Option.isSome_some, ↓reduceIte] at ht
    simp only [Resolve.memberIdx, hd, hax]
    rw [hcnt i hi]
    exact hb t (by rw [hmt]; exact ht) m hm

/-- **Every reference of a node calling a primitive is read by a member**, unless it is a batch of none: a batch over
an axis of length `n` and leaf stride `L` covers its argument by `memberIdx`'s `(j / L * n + m) * L + j % L`. -/
theorem cover_of_nodeOk (c : Defn) (nd : Node) (hn : NodeOk c nd) (hp : c.prim = true)
    (hb : ¬ (nd.form == .batch && nd.n == 0) = true) :
    ∀ i < nd.args.size, ∀ t < (nd.args[i]!).seq.count, ∃ m < nd.members,
      ∃ t' < (if (nd.form == .batch && (nd.args[i]!).dims.isSome) = true then (c.params[i]!).2.leaves
        else (nd.args[i]!).seq.count),
      (if (nd.form == .batch) = true then Resolve.memberIdx (nd.args[i]!) m t' else t') = t := by
  obtain ⟨-, hcnt, -, hco, hsh, -⟩ := hn
  intro i hi t ht
  obtain ⟨form, fn, args, out, axes, n, nxs⟩ := nd
  dsimp only at *
  cases form
  · refine ⟨0, by simp [Node.members, beq_prim_batch, show (Form.prim == Form.scan) = false from rfl], t, ?_, ?_⟩
    · simpa [beq_prim_batch] using ht
    · simp [beq_prim_batch]
  · exact absurd (hco (.inl rfl)) (by simp [hp])
  · have hn0 : 0 < n := by simp only [beq_batch_batch, Bool.true_and, beq_iff_eq] at hb; omega
    have hmem : (Node.mk .batch fn args out axes n nxs).members = n := by simp [Node.members, beq_batch_batch]
    rw [hmem]
    obtain ⟨-, hargs⟩ := hsh
    rcases hargs i hi with ⟨hd, -⟩ | ⟨ax, mt, hrm, hmt, hd⟩
    · refine ⟨0, hn0, t, by simpa [hd] using ht, ?_⟩
      simp [Resolve.memberIdx, hd]
    · obtain ⟨L, hax, hlv, ⟨s, hs⟩, -⟩ := axis_spec ax _ _ _ hrm
      rw [hcnt i hi, hlv, hs] at ht
      have hL : 0 < L := by
        rcases Nat.eq_zero_or_pos L with h0 | h0
        · rw [h0] at ht; simp at ht
        · exact h0
      have hq : t / L < n * s := by
        rw [Nat.div_lt_iff_lt_mul hL]
        calc t < n * (L * s) := ht
          _ = n * s * L := by ring
      have hq' : t / L / n < s := by
        rw [Nat.div_lt_iff_lt_mul hn0]
        calc t / L < n * s := hq
          _ = s * n := by ring
      have hr := Nat.mod_lt t hL
      refine ⟨t / L % n, Nat.mod_lt _ hn0, t / L / n * L + t % L, ?_, ?_⟩
      · simp only [hd, hax, Option.isSome_some, ← hmt, hs]
        calc t / L / n * L + t % L < t / L / n * L + L := by omega
          _ = (t / L / n + 1) * L := by ring
          _ ≤ s * L := Nat.mul_le_mul_right _ hq'
          _ = L * s := by ring
      · simp only [Resolve.memberIdx, hd, hax]
        have h1 : (t / L / n * L + t % L) / L = t / L / n := by
          rw [show t / L / n * L + t % L = t % L + t / L / n * L by ring, Nat.add_mul_div_right _ _ hL,
            Nat.div_eq_of_lt hr, Nat.zero_add]
        have h2 : (t / L / n * L + t % L) % L = t % L := by
          rw [show t / L / n * L + t % L = t % L + t / L / n * L by ring, Nat.add_mul_mod_self_right,
            Nat.mod_eq_of_lt hr]
        rw [h1, h2]
        have e1 := Nat.div_add_mod (t / L) n
        have e2 := Nat.div_add_mod t L
        calc (t / L / n * n + t / L % n) * L + t % L = (n * (t / L / n) + t / L % n) * L + t % L := by ring
          _ = t := by rw [e1, Nat.mul_comm]; exact e2
  · exact absurd (hco (.inr rfl)) (by simp [hp])

end NciCheck.Properties.SecurityProofs.BodiesOk
