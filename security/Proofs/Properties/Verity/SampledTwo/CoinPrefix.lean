module

public import Proofs.Properties.Verity.SampledTwo.CoinMapAgree

/-!
# The two-stage leg's coin map: the play before a release reads only the coins written before it

The coin map (`place`, `Proofs.Properties.Verity.SampledTwo.CoinMap`) writes a run's coins into a tape one release at a
time: a release the locator puts on round `j` is written from the next elements of `r j` (`slot`), and nothing else of
the run's coins is read before it. So two runs' coins give one play until the first release that is written from
elements they differ at, and that release has the same frame and the same bytes in both.

* **A written chunk** (`coinBit_chunk`, `freshVal_chunk`, `openBits_chunk`, `ChunkAgree`, `LogAgree`): the coins agree
  where a release is written, and where every release of a list is.
* **One until they differ** (`SameUntil`, `SameAt`, `sameUntil_prefix`, `sameUntil_next`, `sameUntil_whole`): two logs
  that are release by release the same frame and bytes, and the same release while the coins agree at what it is
  written from.
* **Along the releases** (`same_placeK`): the outputs at one tape with `r` and with `r'` written in are `SameAt`, by
  the recursion `placeK_congr` and `law_placeKA` take.
* **At the period's game** (`same_place`, `place_prefix`, `place_whole`, `place_prefix_used`).

What this does not say: that the releases before the one `place` puts round `n` on are on rounds below `n`. A strategy
may ask for a later round's release first, and then that round's coin is in the log before round `n`'s release. The
statements here are in the log's order, at the rounds the earlier releases are located on (`usedOf`), whatever those
are; that they are the rounds a history shows is a hypothesis on the locator and the accepted form, not proved here.
-/

@[expose] public section

namespace Verity.SampledTwo.CoinMap

open Verity.Game
open Claims

variable {β : Type}

/-! ## A written chunk -/

/-- **A coin's bit reads one element.** Coins that agree at the elements `u` to `u + m` have the same bits `128 u` to
`128 (u + m)`. -/
theorem coinBit_chunk {c c' : FlockDeploy.Coin} {u m : ℕ}
    (h : ∀ a : Fin 65536, u ≤ a.val → a.val < u + m → c a = c' a) {t : ℕ} (h₁ : 128 * u ≤ t)
    (h₂ : t < 128 * (u + m)) : coinBit c t = coinBit c' t := by
  unfold coinBit FlockDeploy.Coin.at
  by_cases ha : t / 128 < 65536
  · rw [dif_pos ha, dif_pos ha, h ⟨t / 128, ha⟩ (by show u ≤ t / 128; omega) (by show t / 128 < u + m; omega)]
  · rw [dif_neg ha, dif_neg ha]

theorem freshVal_chunk {c c' : FlockDeploy.Coin} {u : ℕ}
    (h : ∀ a : Fin 65536, u ≤ a.val → a.val < u + 4 → c a = c' a) : freshVal c u = freshVal c' u :=
  funext fun j => coinBit_chunk h (Nat.le_add_right _ _) (by have := j.isLt; omega)

theorem openBits_chunk {c c' : FlockDeploy.Coin} {u n : ℕ}
    (h : ∀ a : Fin 65536, u ≤ a.val → a.val < u + n → c a = c' a) : openBits c u n = openBits c' u n := by
  unfold openBits
  congr 1
  funext t
  exact coinBit_chunk h (Nat.le_add_right _ _) (by have := t.isLt; omega)

section Along

variable (L : Locate) {K : ℕ}

/-- **The coins agree where a release is written**, after the releases `pre`: at the elements of its round's coin the
coin map writes it from, where the locator puts it and they fit (`slot`). It reads the release's frame and the bytes
it answers, not its coins. -/
def ChunkAgree (r r' : Fin K → FlockDeploy.Coin) (pre : List Verity.Release) (rl : Verity.Release) : Prop :=
  ∀ (j : Fin K) (u : ℕ), slot L K pre rl.frame rl.answers (widthOf rl.frame) = some (j, u) →
    ∀ a : Fin 65536, u ≤ a.val → a.val < u + widthOf rl.frame → r j a = r' j a

/-- **The coins agree where every release of `q` is written**, after the releases `pre`. -/
def LogAgree (r r' : Fin K → FlockDeploy.Coin) : List Verity.Release → List Verity.Release → Prop
  | _, [] => True
  | pre, rl :: q => ChunkAgree L r r' pre rl ∧ LogAgree r r' (pre ++ [rl]) q

/-- **Two logs after `pre` are one until a release written from elements the coins differ at**: where the first ends
so does the second, and `E` holds; a release of the first is answered in the second by a release of the same frame
and bytes, the same release where the coins agree at what it is written from, and then the logs after it are one in
the same way. -/
def SameUntil (E : Prop) (r r' : Fin K → FlockDeploy.Coin) :
    List Verity.Release → List Verity.Release → List Verity.Release → Prop
  | _, [], log' => log' = [] ∧ E
  | pre, rl :: log, log' => ∃ rl' rest, log' = rl' :: rest ∧ rl'.frame = rl.frame ∧ rl'.answers = rl.answers ∧
      (ChunkAgree L r r' pre rl → rl' = rl ∧ SameUntil E r r' (pre ++ [rl]) log rest)

/-- **Two outputs' logs after `pre` are one until the coins differ**, and the outputs are one where they never do. -/
def SameAt (lg : β → List Verity.Release) (r r' : Fin K → FlockDeploy.Coin) (pre : List Verity.Release) (o o' : β) :
    Prop :=
  SameUntil L (o' = o) r r' pre ((lg o).drop pre.length) ((lg o').drop pre.length)

/-- **Coins that agree at every round the releases `q` continue agree where each is written.** -/
theorem logAgree_of_used {r r' : Fin K → FlockDeploy.Coin} :
    ∀ (q pre : List Verity.Release), (∀ j : Fin K, usedOf L j.val pre q ≠ 0 → r j = r' j) → LogAgree L r r' pre q
  | [], _, _ => trivial
  | rl :: q, pre, h => by
    refine ⟨fun j u hs a ha₁ ha₂ => ?_, logAgree_of_used q (pre ++ [rl]) fun j hj => h j ?_⟩
    · obtain ⟨hl, -, -⟩ := slot_spec L hs
      rw [h j (by simp only [usedOf, if_pos hl]; omega)]
    · simp only [usedOf]
      omega

/-- **The logs are one through `q`**, where the coins agree at what each release of `q` is written from. -/
theorem sameUntil_prefix {E : Prop} {r r' : Fin K → FlockDeploy.Coin} :
    ∀ (q pre : List Verity.Release) {rest log' : List Verity.Release},
      SameUntil L E r r' pre (q ++ rest) log' → LogAgree L r r' pre q →
        ∃ rest', log' = q ++ rest' ∧ SameUntil L E r r' (pre ++ q) rest rest'
  | [], pre, rest, log', h, _ => ⟨log', rfl, by rw [List.append_nil]; exact h⟩
  | rl :: q, pre, rest, log', h, ⟨hc, hq⟩ => by
    obtain ⟨rl', rest₁, rfl, -, -, hstep⟩ := h
    obtain ⟨rfl, hs⟩ := hstep hc
    obtain ⟨rest', rfl, hs'⟩ := sameUntil_prefix q (pre ++ [rl']) hs hq
    refine ⟨rest', rfl, ?_⟩
    rw [List.append_assoc] at hs'
    exact hs'

/-- **The release after `q` has the same frame and bytes**, where the coins agree at what each release of `q` is
written from. -/
theorem sameUntil_next {E : Prop} {r r' : Fin K → FlockDeploy.Coin} {q pre rest log' : List Verity.Release}
    {rl : Verity.Release} (h : SameUntil L E r r' pre (q ++ rl :: rest) log') (hq : LogAgree L r r' pre q) :
    ∃ cs rest', log' = q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' := by
  obtain ⟨rest₁, rfl, hs⟩ := sameUntil_prefix L q pre h hq
  obtain ⟨rl', rest', rfl, hf, ha, -⟩ := hs
  refine ⟨rl'.coins, rest', ?_⟩
  rw [← hf, ← ha]

/-- **The logs are one, and `E` holds**, where the coins agree at what each release of the first is written from. -/
theorem sameUntil_whole {E : Prop} {r r' : Fin K → FlockDeploy.Coin} {q pre log' : List Verity.Release}
    (h : SameUntil L E r r' pre q log') (hq : LogAgree L r r' pre q) : log' = q ∧ E := by
  obtain ⟨rest', rfl, hs⟩ := sameUntil_prefix L q pre (rest := []) (by rw [List.append_nil]; exact h) hq
  obtain ⟨rfl, hE⟩ := hs
  exact ⟨List.append_nil q, hE⟩

/-! ## Along the releases -/

theorem sameAt_leaf {lg : β → List Verity.Release} {f : List Verity.Release → Game β} {pre : List Verity.Release}
    (hf : Records lg f pre) (s : Strategy (f [])) (t : Tape (f []) s) (r r' : Fin K → FlockDeploy.Coin) :
    SameAt L lg r r' pre (Tape.out _ s t) (Tape.out _ s t) := by
  unfold SameAt
  rw [hf [] s _ (playReaches_tapeOut _ s t), List.append_nil, List.drop_length]
  exact ⟨rfl, rfl⟩

/-- One release: two outputs whose logs continue `pre` by releases of one frame and bytes, the same release and one
after it where the coins agree at what it is written from. -/
theorem sameAt_step {lg : β → List Verity.Release} {r r' : Fin K → FlockDeploy.Coin} {pre log log' : List Verity.Release}
    {rl rl' : Verity.Release} {o o' : β} (h : lg o = pre ++ rl :: log) (h' : lg o' = pre ++ rl' :: log')
    (hf : rl'.frame = rl.frame) (ha : rl'.answers = rl.answers)
    (hc : ChunkAgree L r r' pre rl → rl' = rl ∧ SameAt L lg r r' (pre ++ [rl]) o o') : SameAt L lg r r' pre o o' := by
  have e₁ : (lg o).drop pre.length = rl :: log := by
    rw [h]
    exact List.drop_left
  have e₁' : (lg o').drop pre.length = rl' :: log' := by
    rw [h']
    exact List.drop_left
  unfold SameAt
  rw [e₁, e₁']
  refine ⟨rl', log', rfl, hf, ha, fun hch => ?_⟩
  obtain ⟨e, hs⟩ := hc hch
  subst e
  refine ⟨rfl, ?_⟩
  have e₂ : (lg o).drop (pre ++ [rl']).length = log := by
    rw [h, show pre ++ rl' :: log = (pre ++ [rl']) ++ log by simp]
    exact List.drop_left
  have e₂' : (lg o').drop (pre ++ [rl']).length = log' := by
    rw [h', show pre ++ rl' :: log' = (pre ++ [rl']) ++ log' by simp]
    exact List.drop_left
  unfold SameAt at hs
  rw [e₂, e₂'] at hs
  exact hs

theorem sameAt_opt {γ : Type} (n : Game β) (y : γ → Game β) {lg : β → List Verity.Release}
    {r r' : Fin K → FlockDeploy.Coin} {pre : List Verity.Release}
    {fn fn' : (s : Strategy n) → Tape n s → Tape n s}
    {fy fy' : (a : γ) → (s : Strategy (y a)) → Tape (y a) s → Tape (y a) s}
    (hn : ∀ s t, SameAt L lg r r' pre (Tape.out n s (fn s t)) (Tape.out n s (fn' s t)))
    (hy : ∀ a s t, SameAt L lg r r' pre (Tape.out (y a) s (fy a s t)) (Tape.out (y a) s (fy' a s t))) :
    ∀ (o : Option γ) (s : Strategy (optGame o n y)) (t : Tape (optGame o n y) s),
      SameAt L lg r r' pre (Tape.out _ s (placeOpt n y fn fy o s t)) (Tape.out _ s (placeOpt n y fn' fy' o s t))
  | none, s, t => hn s t
  | some a, s, t => hy a s t

theorem sameAt_ite {c : Prop} (dc : Decidable c) (G₁ G₂ : Game β) {lg : β → List Verity.Release}
    {r r' : Fin K → FlockDeploy.Coin} {pre : List Verity.Release}
    {f₁ f₁' : (s : Strategy G₁) → Tape G₁ s → Tape G₁ s} {f₂ f₂' : (s : Strategy G₂) → Tape G₂ s → Tape G₂ s}
    (h₁ : ∀ s t, SameAt L lg r r' pre (Tape.out G₁ s (f₁ s t)) (Tape.out G₁ s (f₁' s t)))
    (h₂ : ∀ s t, SameAt L lg r r' pre (Tape.out G₂ s (f₂ s t)) (Tape.out G₂ s (f₂' s t))) :
    ∀ (s : Strategy (@ite _ c dc G₁ G₂)) (t : Tape (@ite _ c dc G₁ G₂) s),
      SameAt L lg r r' pre (Tape.out _ s (placeIte dc G₁ G₂ s f₁ f₂ t)) (Tape.out _ s (placeIte dc G₁ G₂ s f₁' f₂' t)) := by
  cases dc with
  | isFalse _ => exact h₂
  | isTrue _ => exact h₁

/-- **A coin node**: the node's release has one frame and bytes at every value (`rel`), the two maps write one value
where the coins agree at what the release is written from (`hw`), and the plays after a value are one until the coins
differ (`hrec`). -/
theorem sameAt_coin {lg : β → List Verity.Release} {C : Type} {iF : Fintype C} {iN : Nonempty C} {kf : C → Game β}
    (s : Strategy (@Game.coin β C iF iN kf)) (t : Tape (@Game.coin β C iF iN kf) s) {w w' : C → C}
    {F F' : (c : C) → (s : Strategy (kf c)) → Tape (kf c) s → Tape (kf c) s} {r r' : Fin K → FlockDeploy.Coin}
    {pre : List Verity.Release} (rel : C → Verity.Release)
    (hfr : ∀ x y, (rel x).frame = (rel y).frame ∧ (rel x).answers = (rel y).answers)
    (hlog : ∀ x (τ : Tape (kf x) (s x)), ∃ log, lg (Tape.out (kf x) (s x) τ) = pre ++ rel x :: log)
    (hw : ∀ c y, ChunkAgree L r r' pre (rel y) → w c = w' c)
    (hrec : ∀ x (τ : Tape (kf x) (s x)),
      SameAt L lg r r' (pre ++ [rel x]) (Tape.out _ (s x) (F x (s x) τ)) (Tape.out _ (s x) (F' x (s x) τ))) :
    SameAt L lg r r' pre (Tape.out _ s (placeCoin kf w F s t)) (Tape.out _ s (placeCoin kf w' F' s t)) := by
  let t' : C × ((c : C) → Tape (kf c) (s c)) := t
  have key : ∀ x x' : C, (∀ y, ChunkAgree L r r' pre (rel y) → x = x') →
      SameAt L lg r r' pre (Tape.out (kf x) (s x) (F x (s x) (t'.2 x)))
        (Tape.out (kf x') (s x') (F' x' (s x') (t'.2 x'))) := by
    intro x x' hx
    obtain ⟨log, hl⟩ := hlog x (F x (s x) (t'.2 x))
    obtain ⟨log', hl'⟩ := hlog x' (F' x' (s x') (t'.2 x'))
    refine sameAt_step L hl hl' (hfr x' x).1 (hfr x' x).2 fun hch => ?_
    have e := hx x hch
    subst e
    exact ⟨rfl, hrec x (t'.2 x)⟩
  exact key (w t'.1) (w' t'.1) fun y hy => hw t'.1 y hy

/-- The prefix law along `relK k`, at every layout, trees, releases before and continuation that records them. -/
def RecSame (lg : β → List Verity.Release) {k : ℕ} (rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k) : Prop :=
  ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release) (f : List Verity.Release → Game β),
    Records lg f pre → ∀ (s : Strategy (relK k fs ts f)) (τ : Tape (relK k fs ts f) s) (r r' : Fin K → FlockDeploy.Coin),
      SameAt L lg r r' pre (Tape.out _ s (rc r fs ts pre f s τ)) (Tape.out _ s (rc r' fs ts pre f s τ))

theorem same_fresh {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    (hrec : RecSame L lg rc) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (hf : Records lg f pre) (l : String) (b : ByteArray)
    (s : Strategy (freshG k fs ts f l b)) (t : Tape (freshG k fs ts f l b) s) (r r' : Fin K → FlockDeploy.Coin) :
    SameAt L lg r r' pre (Tape.out _ s (placeFresh L r k (rc r) fs ts pre f l b s t))
      (Tape.out _ s (placeFresh L r' k (rc r') fs ts pre f l b s t)) := by
  unfold placeFresh
  refine sameAt_opt L (f []) (freshY k ts f l b) (fun s t => sameAt_leaf L hf s t r r') (fun fs' s' t' => ?_) _ s t
  refine sameAt_coin L s' t' (fun x => ⟨.fresh l, b, List.ofFn x⟩) (fun _ _ => ⟨rfl, rfl⟩)
    (fun x τ => hf.out ⟨.fresh l, b, List.ofFn x⟩ k fs' ts (s' x) τ) (fun c y hch => ?_)
    (fun x τ => hrec fs' ts _ _ (hf.cons _) (s' x) τ r r')
  rcases hs : slot L K pre (.fresh l) b 4 with _ | ⟨i, u⟩
  · rfl
  · exact freshVal_chunk (hch i u hs)

theorem same_hello {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    (hrec : RecSame L lg rc) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (hf : Records lg f pre) (spec : Flock.CoinTree.Spec) (ν : ByteArray)
    (s : Strategy (helloG k fs ts f spec ν)) (t : Tape (helloG k fs ts f spec ν) s) (r r' : Fin K → FlockDeploy.Coin) :
    SameAt L lg r r' pre (Tape.out _ s (placeHello k (rc r) fs ts pre f spec ν s t))
      (Tape.out _ s (placeHello k (rc r') fs ts pre f spec ν s t)) := by
  unfold placeHello
  refine sameAt_ite L _ (helloX k fs ts f spec ν) (f []) (fun s' t' => ?_) (fun s t => sameAt_leaf L hf s t r r') s t
  exact sameAt_coin L s' t' (fun tr => ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩)
    (fun _ _ => ⟨rfl, rfl⟩)
    (fun tr τ => hf.out ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec tr ++ treeKey tr.1)⟩ k fs _ (s' tr) τ)
    (fun _ _ _ => rfl) (fun tr τ => hrec fs _ _ _ (hf.cons _) (s' tr) τ r r')

theorem same_round {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    (hrec : RecSame L lg rc) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (hf : Records lg f pre) (i sn n : ℕ) (b : ByteArray) (T : OpenTree)
    (p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)) (s : Strategy (openP k fs ts f i sn n b T p))
    (t : Tape (openP k fs ts f i sn n b T p) s) (r r' : Fin K → FlockDeploy.Coin) :
    SameAt L lg r r' pre (Tape.out _ s (placeRound L r k (rc r) fs ts pre f i sn n b T p s t))
      (Tape.out _ s (placeRound L r' k (rc r') fs ts pre f i sn n b T p s t)) := by
  unfold placeRound
  refine sameAt_coin L s t (fun dr => ⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩) (fun _ _ => ⟨rfl, rfl⟩)
    (fun dr τ => hf.out ⟨.opening i sn n, b, TreeDraw.opened dr.1 n⟩ k fs _ (s dr) τ) (fun c y hch => ?_)
    (fun dr τ => hrec fs _ _ _ (hf.cons _) (s dr) τ r r')
  rcases hs : slot L K pre (.opening i sn n) b n with _ | ⟨j, u⟩
  · rfl
  · have e : openBits (r j) u n = openBits (r' j) u n := openBits_chunk (hch j u hs)
    show openW n (openBits (r j) u n) c = openW n (openBits (r' j) u n) c
    rw [e]

theorem same_open {lg : β → List Verity.Release} {k : ℕ} {rc : (Fin K → FlockDeploy.Coin) → PlaceRec β k}
    (hrec : RecSame L lg rc) (fs : List (String × ℕ)) (ts : List OpenTree) (pre : List Verity.Release)
    (f : List Verity.Release → Game β) (hf : Records lg f pre) (i sn n : ℕ) (b : ByteArray)
    (s : Strategy (openG k fs ts f i sn n b)) (t : Tape (openG k fs ts f i sn n b) s)
    (r r' : Fin K → FlockDeploy.Coin) :
    SameAt L lg r r' pre (Tape.out _ s (placeOpen L r k (rc r) fs ts pre f i sn n b s t))
      (Tape.out _ s (placeOpen L r' k (rc r') fs ts pre f i sn n b s t)) := by
  unfold placeOpen
  refine sameAt_ite L _ _ (f []) (fun s' t' => ?_) (fun s t => sameAt_leaf L hf s t r r') s t
  refine sameAt_opt L (f []) (openT k fs ts f i sn n b) (fun s t => sameAt_leaf L hf s t r r') (fun T s₃ t₃ => ?_) _ s' t'
  refine sameAt_opt L (f []) (openP k fs ts f i sn n b T) (fun s t => sameAt_leaf L hf s t r r') (fun p s₄ t₄ => ?_) _
    s₃ t₃
  exact same_round L hrec fs ts pre f hf i sn n b T p s₄ t₄ r r'

/-- **The play at two runs' coins is one until a release written from elements the coins differ at**, along the
device's releases: at every layout, trees, releases before and continuation that records them, the outputs at the
tape `τ` with `r` and with `r'` written in have logs that are one until then (`SameAt`), and are one output where the
coins agree at what every release is written from. -/
theorem same_placeK (lg : β → List Verity.Release) : ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (pre : List Verity.Release) (f : List Verity.Release → Game β), Records lg f pre →
    ∀ (s : Strategy (relK k fs ts f)) (τ : Tape (relK k fs ts f) s) (r r' : Fin K → FlockDeploy.Coin),
      SameAt L lg r r' pre (Tape.out _ s (placeK L r k fs ts pre f s τ)) (Tape.out _ s (placeK L r' k fs ts pre f s τ))
  | 0, _, _, _, _, hf, s, τ, r, r' => sameAt_leaf L hf s τ r r'
  | _ + 1, _, _, _, _, hf, ⟨none, s'⟩, τ, r, r' => sameAt_leaf L hf s' τ r r'
  | k + 1, fs, ts, pre, f, hf, ⟨some (.fresh l, b), s'⟩, τ, r, r' =>
    same_fresh L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f hf s τ r r' => same_placeK lg k fs ts pre f hf s τ r r') fs ts pre f hf l b s' τ r r'
  | k + 1, fs, ts, pre, f, hf, ⟨some (.hello spec, ν), s'⟩, τ, r, r' =>
    same_hello L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f hf s τ r r' => same_placeK lg k fs ts pre f hf s τ r r') fs ts pre f hf spec ν s' τ r r'
  | k + 1, fs, ts, pre, f, hf, ⟨some (.opening i sn n, b), s'⟩, τ, r, r' =>
    same_open L (rc := fun r fs ts pre f s t => placeK L r k fs ts pre f s t)
      (fun fs ts pre f hf s τ r r' => same_placeK lg k fs ts pre f hf s τ r r') fs ts pre f hf i sn n b s' τ r r'

theorem sameAt_along {g g' : Game β} (h : g = g') {Pl Pl' : (s : Strategy g') → Tape g' s → Tape g' s}
    {lg : β → List Verity.Release} {r r' : Fin K → FlockDeploy.Coin} {pre : List Verity.Release}
    (hl : ∀ s t, SameAt L lg r r' pre (Tape.out g' s (Pl s t)) (Tape.out g' s (Pl' s t))) (s : Strategy g)
    (t : Tape g s) : SameAt L lg r r' pre (Tape.out g s (along h Pl s t)) (Tape.out g s (along h Pl' s t)) := by
  subst h
  exact hl s t

end Along

section Period

variable (L : Locate) {K : ℕ}
variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The period's play at two runs' coins is one until a release written from elements the coins differ at**: the
outputs at the tape `ω` with `r` and with `r'` written in (`place`) have logs that are release by release the same
frame and bytes, the same release while the coins agree at what it is written from (`SameUntil`), and are one output
where they agree at what every release is written from. -/
theorem same_place (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (r r' : Fin K → FlockDeploy.Coin) (ω : Tape (V.game θ F c) σ) :
    SameAt L Verity.Exchange.log r r' [] (Tape.out _ σ (place V θ F c σ L r ω))
      (Tape.out _ σ (place V θ F c σ L r' ω)) := by
  refine sameAt_along L (game_eq V θ F c) (fun s t => ?_) σ ω
  rcases s with ⟨m, s'⟩
  exact same_placeK L Verity.Exchange.log _ _ _ _ _
    (fun log _ _ h => by rw [List.nil_append]; exact Verity.Sampled.log_of_finish h) s' t r r'

/-- **The log before a release, and the release's frame and bytes, read the run's coins only where the releases before
it are written.** Where the log at `r` is `q`, then `rl`, and `r'` agrees with `r` at the elements each release of `q`
is written from (`LogAgree`), the log at `r'` is `q`, then a release of `rl`'s frame and bytes. -/
theorem place_prefix (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (r r' : Fin K → FlockDeploy.Coin) (ω : Tape (V.game θ F c) σ)
    {q rest : List Verity.Release} {rl : Verity.Release}
    (h : (Tape.out _ σ (place V θ F c σ L r ω)).log = q ++ rl :: rest) (hq : LogAgree L r r' [] q) :
    ∃ cs rest', (Tape.out _ σ (place V θ F c σ L r' ω)).log = q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' := by
  have hs := same_place L V θ F c σ r r' ω
  unfold SameAt at hs
  rw [List.length_nil, List.drop_zero, List.drop_zero, h] at hs
  exact sameUntil_next L hs hq

/-- **The same, by rounds**: where `r'` is `r` at every round a release of `q` continues. Nothing is asked of the
round `rl` is put on, or of any round only later releases are put on. -/
theorem place_prefix_used (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (r r' : Fin K → FlockDeploy.Coin) (ω : Tape (V.game θ F c) σ)
    {q rest : List Verity.Release} {rl : Verity.Release}
    (h : (Tape.out _ σ (place V θ F c σ L r ω)).log = q ++ rl :: rest)
    (hq : ∀ j : Fin K, usedOf L j.val [] q ≠ 0 → r j = r' j) :
    ∃ cs rest', (Tape.out _ σ (place V θ F c σ L r' ω)).log = q ++ ⟨rl.frame, rl.answers, cs⟩ :: rest' :=
  place_prefix L V θ F c σ r r' ω h (logAgree_of_used L q [] hq)

/-- **The play reads the run's coins only where its releases are written**: where `r'` agrees with `r` at the elements
each release of the log at `r` is written from, the output at `r'` is the output at `r`. -/
theorem place_whole (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Strategy (V.game θ F c)) (r r' : Fin K → FlockDeploy.Coin) (ω : Tape (V.game θ F c) σ)
    (hq : LogAgree L r r' [] (Tape.out _ σ (place V θ F c σ L r ω)).log) :
    Tape.out _ σ (place V θ F c σ L r' ω) = Tape.out _ σ (place V θ F c σ L r ω) := by
  have hs := same_place L V θ F c σ r r' ω
  unfold SameAt at hs
  rw [List.length_nil, List.drop_zero, List.drop_zero] at hs
  exact (sameUntil_whole L hs hq).2

end Period

end Verity.SampledTwo.CoinMap
