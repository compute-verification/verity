import Proofs.Flock.VBridge.Algebra.Lincheck
import Proofs.Flock.Soundness.Refine.Opening

/-!
# V*'s ring-switch residuals are zero only when ring switching's checks pass

`ringSwitch_checks`: when `ringSwitch`'s residuals (one per claim, `Σ_v w[v]·ŝ[v] + value`, rec_algebra.py:620–636)
are zero over F, each claim passes the verifier's value check (`dot w s == c.value`, Opening.lean:78) on the ring
switches `ringSwitch` read, and the batched sum `ringSwitch` returns is the verifier's `T = Σ_k γ_k·β_k`
(Opening.lean:81–86) with `β_k = Σ_u eq(r''_k, u)·t_k[u]`, when the transposes `ringSwitch` read are the verifier's bit
transposes (`Refine.tOf`) of the ring switches.

`wi_rs` lays out w: claim `k`'s ring switch is 128 consecutive elements, so the transposes V*'s S computes from w
(`.wt` terms) are the verifier's of the decoded ring switches (`decoded_rsT`). `ringSwitch_of_residuals` is the stage at
a rep: each claim's value check, and Ligerito's `T` (the one w holds, which V*'s last residual compares with the batched
sum) the verifier's `T` at the rep's coins.
-/

namespace Flock.RecAlgebra

open Flock FlockSoundness.Refine

/-! ## Where the ring switches are in w -/

/-- The rounds before the ring switches (`rounds`, rec_algebra.py:748–772): zerocheck's, then lincheck's, then a holo
shape's `h` (`hRound`). -/
def preRounds (sh : Shape) : Array (Array Key) :=
  #[(Array.range 64).map .ab ++ (Array.range 64).map .c] ++
    (Array.range (sh.m - 6)).map (fun i => #[.g1 i, .ginf i]) ++ #[#[.fa, .fb]] ++
    (Array.range (sh.kLog - 6)).map (fun j => #[.q1 j, .qinf j]) ++ #[(Array.range 64).map .zp] ++ hRound sh

/-- The ring switches' rounds: `c` blocks of 128, block `k` entry `v` being `g k v`. -/
def blocks (g : Nat → Nat → Key) (c : Nat) : Array Key :=
  ((Array.range c).map fun k => (Array.range 128).map (g k)).flatten

theorem blocks_spec (g : Nat → Nat → Key) :
    ∀ c, (blocks g c).size = 128 * c ∧ ∀ n < 128 * c, (blocks g c)[n]? = some (g (n / 128) (n % 128))
  | 0 => ⟨by simp [blocks, Array.eq_empty_of_size_eq_zero (Array.size_range (n := 0))],
      fun n hn => absurd hn (by omega)⟩
  | c + 1 => by
    obtain ⟨hsz, hget⟩ := blocks_spec g c
    have e : blocks g (c + 1) = blocks g c ++ (Array.range 128).map (g c) := by
      simp only [blocks, Array.range_succ, Array.map_append, Array.flatten_append]
      simp
    rw [e]
    refine ⟨by simp [hsz]; omega, fun n hn => ?_⟩
    by_cases hlt : n < 128 * c
    · rw [Array.getElem?_append_left (by omega), hget n hlt]
    · rw [Array.getElem?_append_right (by omega), hsz]
      have hq : n / 128 = c := by omega
      have hr : n % 128 = n - 128 * c := by omega
      rw [hq, hr]
      simp only [Array.getElem?_map, Array.getElem?_range, show n - 128 * c < 128 by omega, ite_true, Option.map_some]

theorem rounds_split (sh : Shape) :
    rounds sh = preRounds sh ++ (Array.range sh.claims).map (fun k => (Array.range 128).map (.rs k)) ++
      ligRounds sh.sched := rfl

theorem rs_not_pre (sh : Shape) (j v : Nat) : Key.rs j v ∉ (preRounds sh).flatten := by
  cases h : sh.holo <;> simp [preRounds, hRound, h, Array.mem_flatten]

theorem idxOf?_of {xs : Array Key} {a : Key} {i : Nat} (hi : xs[i]? = some a) (hlt : ∀ j < i, xs[j]? ≠ some a) :
    xs.idxOf? a = some i := by
  rcases xs with ⟨l⟩
  rw [List.idxOf?_toArray, List.idxOf?_eq_some_iff]
  simp only [List.getElem?_toArray] at hi hlt
  obtain ⟨h, he⟩ := List.getElem?_eq_some_iff.mp hi
  refine ⟨h, he, fun j hj hje => hlt j hj ?_⟩
  rw [List.getElem?_eq_getElem (by omega), hje]

/-- **Where claim `k`'s ring switch is in w** (`_w_names`, rec_algebra.py:812–813): its 128 elements are consecutive,
after the zerocheck's and lincheck's and the claims before it. -/
theorem wi_rs (sh : Shape) {k v : Nat} (hk : k < sh.claims) (hv : v < 128) :
    wi sh (.rs k v) = some ((preRounds sh).flatten.size + 128 * k + v) := by
  unfold wi wNames
  rw [rounds_split, Array.flatten_append, Array.flatten_append]
  obtain ⟨hsz, hget⟩ := blocks_spec Key.rs sh.claims
  have hB : (((Array.range sh.claims).map fun k => (Array.range 128).map (Key.rs k)).flatten) =
      blocks Key.rs sh.claims := rfl
  rw [hB]
  have hin : 128 * k + v < 128 * sh.claims := by omega
  have hl : ∀ i ≤ (preRounds sh).flatten.size + 128 * k + v,
      i < ((preRounds sh).flatten ++ blocks Key.rs sh.claims).size := by
    intro i hi; simp only [Array.size_append, hsz]; omega
  have hl' : ∀ i ≤ (preRounds sh).flatten.size + 128 * k + v,
      i < ((preRounds sh).flatten ++ blocks Key.rs sh.claims ++ (ligRounds sh.sched).flatten).size := by
    intro i hi; simp only [Array.size_append, hsz]; omega
  apply idxOf?_of
  · rw [Array.getElem?_append_left (hl' _ le_rfl), Array.getElem?_append_left (hl _ le_rfl),
      Array.getElem?_append_right (by omega), show (preRounds sh).flatten.size + 128 * k + v -
        (preRounds sh).flatten.size = 128 * k + v by omega, hget _ hin]
    congr 2 <;> omega
  · intro j hj he
    rw [Array.getElem?_append_left (hl' _ (by omega)), Array.getElem?_append_left (hl _ (by omega))] at he
    by_cases hjp : j < (preRounds sh).flatten.size
    · rw [Array.getElem?_append_left hjp] at he
      exact rs_not_pre sh k v (Array.mem_of_getElem? he)
    · rw [Array.getElem?_append_right (by omega), hget _ (by omega)] at he
      simp only [Option.some.injEq, Key.rs.injEq] at he
      omega

/-! ## The decoded ring switches and their transposes -/

theorem foldl_range_congr {β : Type} {f g : β → Nat → β} {n : Nat} (h : ∀ b, ∀ v < n, f b v = g b v) (init : β) :
    (Array.range n).foldl f init = (Array.range n).foldl g init := by
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_range]
  suffices ∀ (l : List Nat), (∀ v ∈ l, v < n) → ∀ b, l.foldl f b = l.foldl g b from
    this _ (fun v hv => List.mem_range.mp hv) init
  intro l
  induction l with
  | nil => intro _ b; rfl
  | cons a l ih =>
    intro hl b
    rw [List.foldl_cons, List.foldl_cons, h b a (hl a List.mem_cons_self)]
    exact ih (fun v hv => hl v (List.mem_cons_of_mem _ hv)) _

theorem decoded_rs (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) {k : Nat} (hk : k < sh.claims) :
    (decoded sh w comb cons).rs[k]! =
      (Array.range 128).map fun v => w[(preRounds sh).flatten.size + 128 * k + v]! := by
  simp only [decoded, Messages.map, symbolic]
  rw [getElem!_pos _ k (by simpa using hk)]
  simp only [Array.getElem_map, Array.getElem_range, Array.map_map]
  apply Array.ext
  · simp
  · intro v h1 h2
    have h1' : v < 128 := by simpa using h1
    simp only [Array.getElem_map, Array.getElem_range, Function.comp, Lin.eval_var, valOf, wi_rs sh hk h1']

theorem decoded_rs_size (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) :
    (decoded sh w comb cons).rs.size = sh.claims ∧ ∀ k < sh.claims, ((decoded sh w comb cons).rs[k]!).size = 128 := by
  refine ⟨by simp [decoded, Messages.map, symbolic], fun k hk => ?_⟩
  rw [decoded_rs sh w comb cons hk]
  simp

/-- **The decoded transposes.** The transposes V*'s S computes from w are the verifier's bit transposes of the decoded
ring switches. -/
theorem decoded_rsT (sh : Shape) (w comb : Array F128) (cons : Nat → Nat → F128) {k : Nat} (hk : k < sh.claims) :
    (decoded sh w comb cons).rsT[k]! = tOf (decoded sh w comb cons).rs[k]! := by
  rw [decoded_rs sh w comb cons hk]
  simp only [decoded, Messages.map, symbolic]
  rw [getElem!_pos _ k (by simpa using hk)]
  simp only [Array.getElem_map, Array.getElem_range, Array.map_map]
  apply Array.ext
  · simp [tOf]
  · intro u h1 h2
    simp only [tOf, Array.getElem_map, Array.getElem_range, Function.comp, Lin.eval_var, valOf,
      wi_rs sh hk (show 0 < 128 by omega), Nat.add_zero, termValue]
    refine foldl_range_congr (fun b v hv => ?_) 0
    rw [getElem!_pos _ v (by simpa using hv)]
    simp only [Array.getElem_map, Array.getElem_range, Nat.add_assoc]

/-! ## Ring switching over values -/

section Values

variable {α : Type} [Val α]

/-- A claim's weights on the 128 packed positions (rec_algebra.py:624–630; `Refine.wOf`). -/
def wvOf (cl : AClaim α) : Array F128 :=
  (Array.range 128).map fun v => cl.skip[v % 64]! * (if v / 64 == 1 then cl.x[0]! else (1 : F128) + cl.x[0]!)

theorem rsClaims_spec (M : Messages α) :
    ∀ (l : List (AClaim α)) (k k₀ k₁ : Nat) (src : Coins) (outs betas : Array α)
      (pts : Array (Array F128 × Array F128)),
      rsClaims M l k k₀ src = .ok ((outs, betas, pts), k₁) →
      outs.size = l.length ∧ betas.size = l.length ∧ pts.size = l.length ∧ k₁ = k₀ + l.length ∧
      ∀ j (hj : j < l.length), ∃ c, src.round (k₀ + j) 7 = .ok c ∧
        outs[j]? = some (Val.add (dotV (wvOf l[j]) M.rs[k + j]!) l[j].value) ∧
        betas[j]? = some (dotV (eqTable c) M.rsT[k + j]!) ∧ pts[j]? = some (c, l[j].x.extract 1 l[j].x.size)
  | [], k, k₀, k₁, src, outs, betas, pts, h => by
    cases cm_pure h
    exact ⟨rfl, rfl, rfl, rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | cl :: rest, k, k₀, k₁, src, outs, betas, pts, h => by
    simp only [rsClaims] at h
    obtain ⟨⟨out, beta, rpp, z⟩, k₂, h1, h⟩ := cm_bind h
    obtain ⟨⟨outs', betas', pts'⟩, k₃, h2, h⟩ := cm_bind h
    cases cm_pure h
    unfold rsClaim at h1
    obtain ⟨rpp', k₄, h3, h1⟩ := cm_bind h1
    cases cm_pure h1
    rcases coinVec_ok h3 with ⟨h7, -⟩ | ⟨-, hr, rfl⟩
    · cases h7
    obtain ⟨ho, hb, hp, hk, hj⟩ := rsClaims_spec M rest (k + 1) (k₀ + 1) k₁ src outs' betas' pts' h2
    simp only [List.length_cons, Array.size_append, ho, hb, hp, hk]
    refine ⟨by simp; omega, by simp; omega, by simp; omega, by omega, fun j hjl => ?_⟩
    rcases j with _ | j
    · refine ⟨rpp, by simpa using hr, ?_, ?_, ?_⟩ <;> rw [Array.getElem?_append_left (by simp)] <;> rfl
    · obtain ⟨c, hc, h1, h2, h3⟩ := hj j (by omega)
      refine ⟨c, by rw [show k₀ + (j + 1) = k₀ + 1 + j by omega]; exact hc, ?_, ?_, ?_⟩
      · rw [Array.getElem?_append_right (by simp), show k + (j + 1) = k + 1 + j by omega]; simpa using h1
      · rw [Array.getElem?_append_right (by simp), show k + (j + 1) = k + 1 + j by omega]; simpa using h2
      · rw [Array.getElem?_append_right (by simp)]; simpa using h3

/-- **Ring switching's coins.** The rounds `ringSwitch` draws from: `r''_k` for each claim, then the batching coins
`γ`; and what it returns. -/
theorem ringSwitch_spec {M : Messages α} {claims : Array (AClaim α)} (hn : claims.size ≠ 0) {k k' : Nat}
    {src : Coins} {rsr : Array α} {rs : RsPts α} (h : ringSwitch M claims k src = .ok ((rsr, rs), k')) :
    ∃ betas : Array α, rsr.size = claims.size ∧ betas.size = claims.size ∧
      src.round (k + claims.size) claims.size = .ok rs.gamma ∧ k' = k + claims.size + 1 ∧
      rs.T = dotV rs.gamma betas ∧ rs.sws.size = claims.size ∧
      ∀ j (hj : j < claims.size), ∃ c, src.round (k + j) 7 = .ok c ∧
        rsr[j]? = some (Val.add (dotV (wvOf claims[j]) M.rs[j]!) claims[j].value) ∧
        betas[j]? = some (dotV (eqTable c) M.rsT[j]!) ∧
        rs.sws[j]? = some { gamma := rs.gamma[j]!, rpp := c, z := claims[j].x.extract 1 claims[j].x.size } := by
  unfold ringSwitch at h
  obtain ⟨⟨outs, betas, pts⟩, k₁, h1, h⟩ := cm_bind h
  obtain ⟨gamma, k₂, h2, h⟩ := cm_bind h
  cases cm_pure h
  obtain ⟨ho, hb, hp, hk, hj⟩ := rsClaims_spec M claims.toList 0 k k₁ src _ _ _ h1
  simp only [Array.length_toList] at ho hb hp hk hj
  rcases coinVec_ok h2 with ⟨h0, -⟩ | ⟨-, hr, rfl⟩
  · exact absurd h0 hn
  subst hk
  refine ⟨betas, ho, hb, hr, rfl, rfl, by simp, fun j hjn => ?_⟩
  obtain ⟨c, hc, h1, h2, h3⟩ := hj j hjn
  simp only [Nat.zero_add, Array.getElem_toList] at h1 h2 h3
  refine ⟨c, hc, h1, h2, ?_⟩
  rw [Array.getElem?_map, Array.getElem?_range]
  simp only [hjn, ite_true, Option.map_some]
  rw [getElem!_pos pts j (by omega)]
  rw [Array.getElem?_eq_getElem (by omega), Option.some.injEq] at h3
  rw [h3]

end Values

/-! ## Ring switching over F -/

/-- A claim over F as the verifier's (`Piop.Claim`). -/
def AClaim.toClaim (c : AClaim F128) : Claim := ⟨c.value, c.skip, c.x⟩

/-- **Ring switching's checks.** When `ringSwitch` over F returns residuals all zero, each claim passes the verifier's
value check (Opening.lean:78) on the ring switches it read, and its batched sum is `Σ_k γ_k·β_k` with
`β_k = Σ_u eq(r''_k, u)·t_k[u]` for the transposes it read (Opening.lean:81–86). -/
theorem ringSwitch_checks {M : Messages F128} {claims : Array (AClaim F128)} (hn : claims.size ≠ 0) {k k' : Nat}
    {src : Coins} {rsr : Array F128} {rs : RsPts F128} (h : ringSwitch M claims k src = .ok ((rsr, rs), k'))
    (hz : ∀ x ∈ rsr, x = 0) :
    (∀ j (hj : j < claims.size), dot (wOf claims[j].toClaim) M.rs[j]! = claims[j].value) ∧
      rs.T = dot rs.gamma ((List.range claims.size).map fun j => dot (eqTable (rs.sws[j]!).rpp) M.rsT[j]!).toArray := by
  obtain ⟨betas, hrsz, hbsz, -, -, hT, hsw, hj⟩ := ringSwitch_spec hn h
  refine ⟨fun j hjn => ?_, ?_⟩
  · obtain ⟨c, -, h1, -, -⟩ := hj j hjn
    have hx : rsr[j]'(by omega) = 0 := hz _ (Array.getElem_mem _)
    rw [Array.getElem?_eq_getElem (by omega), Option.some.injEq, hx] at h1
    rw [← dotV_eq_dot]
    exact f128_eq_of_add_eq_zero h1.symm
  · rw [hT, dotV_eq_dot]
    congr 1
    apply Array.ext
    · simp [hbsz]
    · intro j h1 h2
      obtain ⟨c, -, -, h3, h4⟩ := hj j (by omega)
      rw [Array.getElem?_eq_getElem h1, Option.some.injEq] at h3
      rw [h3, dotV_eq_dot]
      simp only [List.getElem_toArray, List.getElem_map, List.getElem_range]
      rw [getElem!_pos rs.sws j (by omega)]
      rw [Array.getElem?_eq_getElem (by omega), Option.some.injEq] at h4
      rw [h4]

/-! ## Ring switching at a rep -/

/-- The claims V* hands to the opening, over F, as the verifier's. -/
def claimsF (sh : Shape) (val : Key → F128) (info : Info Lin) (extras : Array Claim) : Array Claim :=
  (claimsOf sh.m sh.kLog (info.zc.map (Lin.eval val)) (info.lc.map (Lin.eval val)) extras).map AClaim.toClaim

theorem claimsF_size (sh : Shape) (val : Key → F128) (info : Info Lin) (extras : Array Claim) :
    (claimsF sh val info extras).size = 2 + extras.size := by
  simp [claimsF, claimsOf]

/-- **Ring switching at a rep.** When `GfResiduals`' residuals at w, on the S `structureOf` builds and the v
`verifierRows` gives at the rep's coins, are all zero, each claim V* hands to the opening passes the verifier's value
check on the ring switches w holds (`decoded`), and the `T` w holds, Ligerito's, is the verifier's batched sum
(Opening.lean:81–86) at the coins `verifierRows` returns. -/
theorem ringSwitch_of_residuals {sh : Shape} {w comb : Array F128} {cons : Nat → Nat → F128}
    {rounds : Array (Array F128)} {extras : Array Claim} {v : Array F128} {info : Info Lin}
    {kappa : Array (Array (Array F128))} {S : GfS} {rs : Array F128}
    (hrows : verifierRows sh rounds extras comb = .ok (v, info, kappa)) (hS : structureOf sh = .ok S)
    (hw : w.size = S.nw) (hc : comb.size = 64)
    (hacc : ∀ r < 2, valOf sh w comb cons (.acc1 r) + valOf sh w comb cons (.acc0 r) = kapSum kappa r cons)
    (hres : residuals S w v = .ok rs) (hzero : ∀ x ∈ rs, x = 0) :
    sh.claims = (claimsF sh (valOf sh w comb cons) info extras).size ∧
      (∀ j < sh.claims, dot (wOf (claimsF sh (valOf sh w comb cons) info extras)[j]!)
          (decoded sh w comb cons).rs[j]! = ((claimsF sh (valOf sh w comb cons) info extras)[j]!).value) ∧
      valOf sh w comb cons .T = dot info.rs.gamma
        ((List.range sh.claims).map fun j =>
          dot (eqTable (info.rs.sws[j]!).rpp) (tOf (decoded sh w comb cons).rs[j]!)).toArray := by
  obtain ⟨rs', k, hrun, hres'⟩ := residuals_eq_run w comb cons hrows hS hw hc hacc
  rw [hres] at hres'
  cases hres'
  obtain ⟨hcl, r0, r1, rsr, lgr, k1, k2, k3, -, -, h3, -, hrs⟩ := run_stages hrun
  have hsz : (claimsOf sh.m sh.kLog ((info.map (Lin.eval (valOf sh w comb cons))).zc)
      ((info.map (Lin.eval (valOf sh w comb cons))).lc) extras).size = sh.claims := by
    simp [claimsOf, hcl]
  obtain ⟨hval, hT⟩ := ringSwitch_checks (by omega) h3 (fun x hx => hzero x (by rw [hrs]; simp [hx]))
  have hlast : Val.add ((decoded sh w comb cons).tAtom.getD (info.map (Lin.eval (valOf sh w comb cons))).rs.T)
      (info.map (Lin.eval (valOf sh w comb cons))).rs.T = 0 := hzero _ (by rw [hrs]; simp)
  have hTatom := f128_eq_of_add_eq_zero hlast
  refine ⟨by rw [claimsF_size]; omega, fun j hj => ?_, ?_⟩
  · have e := hval j (by omega)
    have hj' : j < (claimsF sh (valOf sh w comb cons) info extras).size := by rw [claimsF_size]; omega
    rw [getElem!_pos _ j hj']
    simp only [claimsF, Array.getElem_map]
    exact e
  · rw [show valOf sh w comb cons .T = (decoded sh w comb cons).tAtom.getD
        (info.map (Lin.eval (valOf sh w comb cons))).rs.T by
          simp [decoded, Messages.map, symbolic, Lin.eval_var]]
    rw [hTatom, hT, hsz]
    have e : ((List.range sh.claims).map fun j =>
          dot (eqTable ((info.map (Lin.eval (valOf sh w comb cons))).rs.sws[j]!).rpp)
            (decoded sh w comb cons).rsT[j]!) =
        (List.range sh.claims).map fun j =>
          dot (eqTable (info.rs.sws[j]!).rpp) (tOf (decoded sh w comb cons).rs[j]!) :=
      List.map_congr_left fun j hj => by rw [decoded_rsT sh w comb cons (List.mem_range.mp hj)]; rfl
    rw [e]
    rfl

end Flock.RecAlgebra
