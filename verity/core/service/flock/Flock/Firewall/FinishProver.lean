/-
# The firewall's own prover: what `flock-firewall finish` computes a rep's proof file with

The masked prover's phases of the analysis (`FlockComplete.squeezeP` to `FlockComplete.innerP`, live PROTOCOL.md §1–§7),
here as code the firewall runs: definition by definition the analysis's, so that each is that one by unfolding. Two
differ, each by a parameter or a line:

* `ligeritoPZk` takes the commitment of levels ≥ 1 as an argument (`commit`): the analysis's is `commitP`, which
  evaluates every codeword entry on its own, and `flock-firewall finish` passes `Finish.commitFw`, the same commitment by
  the additive FFT (`Link.fft`).
* `innerP` never checks the batched constraint on the pads: the inner proof speaks on schedule (`flock-circuit`'s
  `zk_finish`, `SelfCheck::Skip`), and a constraint that fails on the pads fails the session at `Finish`, as
  `FlockComplete.innerCut` reads it.

The proof file's bytes are the analysis's `encZk` (`FlockSoundness.Discharge.ZkView.encZk`), `flock-circuit --zk`'s
`bincode` of `(Commitment, R1csProofLigerito, InnerProof)`.
-/
import Flock.With.Verify
import Flock.With.Zk
import Flock.Zk
import Flock.ZkProof

namespace Flock.Firewall.Finish

open Flock Transcript

/-! ## The transcript (`FlockComplete.Asks`, `squeezeP`) -/

/-- **What the prover sends**: the link message and a round's framed bytes, each answered by its live coin. -/
structure Asks (M : Type → Type) where
  link : Array ByteArray → Array ByteArray → M (Nat → F128)
  round : ByteArray → M (Nat → F128)

/-- `n` bytes: `b`'s first `n`, zero-padded. -/
def fitBytes (n : Nat) (b : ByteArray) : ByteArray :=
  b.extract 0 n ++ ByteArray.mk (Array.replicate (n - min n b.size) 0)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The prover's squeeze**: a squeeze of `n = 0` frames its header; otherwise the framed bytes are sent as round `k`,
the round's first `n` coins returned, `pending` cleared, and the round recorded. -/
def squeezeP (io : Asks M) (kind : UInt8) (n : Nat) : TMW M (Array F128) := fun t =>
  let c := framed t kind n
  if n = 0 then pure (#[], { t with pending := c })
  else do
    let coin ← io.round c
    let v := (Array.range n).map coin
    pure (v, { t with pending := ByteArray.empty, k := t.k + 1, rounds := t.rounds.push ⟨n, ByteArray.empty, v, some c⟩ })

def sampleP (io : Asks M) : TMW M F128 := do
  let c ← squeezeP io 1 1
  return c.getD 0 0

def sampleVecP (io : Asks M) (n : Nat) : TMW M (Array F128) := squeezeP io 2 n

def sampleKP (io : Asks M) : TMW M F256 := do
  let c ← squeezeP io 2 2
  return ⟨c.getD 0 0, c.getD 1 0⟩

/-! ## Ligerito (`FlockComplete.Commit` to `FlockComplete.queryP`) -/

/-- **A level's commitment**: its codeword rows, the leaves' salts, the Merkle levels from the leaves to the cap, and the
cap. -/
structure Commit where
  rows : Array (Array F128)
  salts : Array ByteArray
  layers : Array (Array ByteArray)
  cap : Array ByteArray
  deriving Inhabited

/-- `Σ_{i < n} f(i)`. -/
def sumF (n : Nat) (f : Nat → F128) : F128 := (List.range n).foldl (fun acc i => acc + f i) 0

/-! ### What compiled code runs (`@[csimp]`: proofs see the definitions above and in `Finish`) -/

namespace Fast

/-- `f` folded from `acc` over `i, i + 1, …, i + k − 1`, with no list. -/
@[specialize] def foldRange {α : Type} (f : α → Nat → α) : Nat → Nat → α → α
  | _, 0, acc => acc
  | i, k + 1, acc => foldRange f (i + 1) k (f acc i)

theorem foldRange_eq {α : Type} (f : α → Nat → α) :
    ∀ (i k : Nat) (acc : α), foldRange f i k acc = (List.range' i k).foldl f acc
  | _, 0, _ => rfl
  | i, k + 1, acc => by rw [foldRange, foldRange_eq f (i + 1) k, List.range'_succ, List.foldl_cons]

theorem foldRange_succ {α : Type} (f : α → Nat → α) (n : Nat) (acc : α) :
    foldRange f 0 (n + 1) acc = f (foldRange f 0 n acc) n := by
  rw [foldRange_eq, foldRange_eq, ← List.range_eq_range', ← List.range_eq_range', List.range_succ, List.foldl_append]
  rfl

/-- `sumF` as compiled code computes it: the same additions in the same order, with no list. Each term comes first
(`F128.add_comm`) so that the sum reuses the accumulator's cell: with the term second, compiled code takes a table entry
`f i = t[i]!` as its own and counts a reference on it, which parallel tasks reading one table contend on. -/
@[inline] def sumFFast (n : Nat) (f : Nat → F128) : F128 := foldRange (fun acc i => f i + acc) 0 n 0

end Fast

open Fast in
@[csimp] theorem sumF_eq_sumFFast : @sumF = @sumFFast :=
  funext fun n => funext fun f => by
    rw [sumF, sumFFast, show (fun acc i => f i + acc) = fun acc i => acc + f i from
      funext fun _ => funext fun _ => F128.add_comm _ _, foldRange_eq, List.range_eq_range']

namespace Fast

theorem sumF_zero (f : Nat → F128) : sumF 0 f = 0 := rfl

theorem sumF_succ (n : Nat) (f : Nat → F128) : sumF (n + 1) f = sumF n f + f n := by
  rw [sumF, sumF, List.range_succ, List.foldl_append]
  rfl

theorem sumF_congr {n : Nat} {f f' : Nat → F128} (h : ∀ i, i < n → f i = f' i) : sumF n f = sumF n f' := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumF_succ, sumF_succ, ih fun i hi => h i (by omega), h n (by omega)]

theorem sumF_eq_zero {n : Nat} {f : Nat → F128} (h : ∀ i, i < n → f i = 0) : sumF n f = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumF_succ, ih fun i hi => h i (by omega), h n (by omega), F128.add_zero]

/-- `Σ_{i < m + n} f(i) = Σ_{i < m} f(i) + Σ_{i < n} f(m + i)`. -/
theorem sumF_add (m n : Nat) (f : Nat → F128) : sumF (m + n) f = sumF m f + sumF n fun i => f (m + i) := by
  induction n with
  | zero => rw [Nat.add_zero, sumF_zero, F128.add_zero]
  | succ n ih => rw [← Nat.add_assoc, sumF_succ, sumF_succ, ih, F128.add_assoc]

/-- `Σ_{i < b·a} f(i) = Σ_{j < a} Σ_{i < b} f(b·j + i)`. -/
theorem sumF_blocks (a b : Nat) (f : Nat → F128) :
    sumF (b * a) f = sumF a fun j => sumF b fun i => f (b * j + i) := by
  induction a with
  | zero => rfl
  | succ a ih => rw [Nat.mul_succ, sumF_add, ih, sumF_succ]

/-- Changing one term `f(k)`, `k < n`, by `d` changes the sum by `d` (`F128.add_comm`: the term's place doesn't matter). -/
theorem sumF_update {n k : Nat} {f f' : Nat → F128} (d : F128) (hk : k < n) (hd : f' k = f k + d)
    (h : ∀ i, i ≠ k → f' i = f i) : sumF n f' = sumF n f + d := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [sumF_succ, sumF_succ]
    by_cases hkn : k = n
    · subst hkn
      rw [sumF_congr fun i hi => h i (by omega), hd, F128.add_assoc]
    · rw [ih (by omega), h n (Ne.symm hkn), F128.add_assoc, F128.add_comm d, ← F128.add_assoc]

theorem getElem!_eq {α : Type} [Inhabited α] (xs : Array α) {i : Nat} (h : i < xs.size) : xs[i]! = xs[i] := by
  simp [h]

theorem getElem!_range_map {α : Type} [Inhabited α] (f : Nat → α) {n i : Nat} (h : i < n) :
    ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_eq _ (by simpa using h)]
  simp

theorem mul_sumF (g : F128) (n : Nat) (f : Nat → F128) : g * sumF n f = sumF n fun i => g * f i := by
  induction n with
  | zero => exact F128.mul_zero g
  | succ n ih => rw [sumF_succ, sumF_succ, F128.mul_add, ih]

theorem getElem!_map_mul (g : F128) (xs : Array F128) (i : Nat) : (xs.map (g * ·))[i]! = g * xs[i]! := by
  by_cases h : i < xs.size
  · rw [getElem!_eq _ (by simpa using h), getElem!_eq xs h, Array.getElem_map]
  · rw [getElem!_neg _ i (by simpa using h), getElem!_neg xs i h]
    exact (F128.mul_zero g).symm

/-- `sumF` in tasks of `b` terms, their sums added in order (`sumF_blocks`). -/
@[specialize] def sumFPar (b n : Nat) (f : Nat → F128) : F128 :=
  let a := n / b
  ((Array.range a).map fun i => Task.spawn fun _ => sumF b fun j => f (b * i + j)).foldl (fun acc t => t.get + acc) 0 +
    sumF (n - b * a) fun j => f (b * a + j)

theorem sumFPar_eq (b n : Nat) (f : Nat → F128) : sumFPar b n f = sumF n f := by
  have hp : ∀ a : Nat, ((Array.range a).map fun i => Task.spawn fun _ => sumF b fun j => f (b * i + j)).foldl
      (fun acc t => t.get + acc) 0 = sumF a fun i => sumF b fun j => f (b * i + j) := by
    intro a
    rw [Array.foldl_map, ← Array.foldl_toList, Array.toList_range, sumF]
    refine congrArg (fun g => List.foldl g (0 : F128) (List.range a)) (funext fun acc => funext fun i => ?_)
    exact F128.add_comm _ _
  unfold sumFPar
  dsimp only
  rw [hp, ← sumF_blocks, ← sumF_add, Nat.add_sub_cancel' (Nat.mul_div_le n b)]

theorem foldRange_succ_at {α : Type} (f : α → Nat → α) (lo n : Nat) (acc : α) :
    foldRange f lo (n + 1) acc = f (foldRange f lo n acc) (lo + n) := by
  rw [foldRange_eq, foldRange_eq, List.range'_1_concat, List.foldl_append]
  rfl

/-- `e + a`, kept out of line: `B.modify k (addTo e)` then hands it the entry with no other reference, so the sum
reuses the entry's cell; inlined, the entry is still in `B` when the sum is built, and every update allocates. -/
@[noinline] def addTo (e a : F128) : F128 := e + a

/-- **Byte `j`'s buckets** of the words `lo, …, lo + n − 1`: entry `b < 256` is
`Σ_{y, byte_j(g[lo + y]) = b} e[lo + y]` (`byteBuckets_go`). -/
def byteBuckets (g e : Array F128) (j lo n : Nat) : Array F128 :=
  foldRange (fun B y => B.modify ((g[y]!).byte j) (addTo e[y]!)) lo n (Array.replicate 256 0)

/-- After `n` words from `lo`, the buckets' sum over the bytes with bit `i` set is
`Σ_{y < n} bit_{8j+i}(g[lo + y])·e[lo + y]`. -/
theorem byteBuckets_go (g e : Array F128) (j lo : Nat) {i : Nat} (hj : j < 16) (hi : i < 8) : ∀ n : Nat,
    (byteBuckets g e j lo n).size = 256 ∧
    (sumF 256 fun b => if b.testBit i then (byteBuckets g e j lo n)[b]! else 0) =
      sumF n fun y => if (g[lo + y]!).bit (8 * j + i) then e[lo + y]! else 0
  | 0 => by
    refine ⟨by simp [byteBuckets, foldRange], ?_⟩
    rw [sumF_zero]
    apply sumF_eq_zero
    intro b hb
    simp only [byteBuckets, foldRange]
    split
    · rw [getElem!_eq _ (by simpa using hb)]; simp
    · rfl
  | n + 1 => by
    obtain ⟨hs, ih⟩ := byteBuckets_go g e j lo hj hi n
    unfold byteBuckets at hs ih ⊢
    rw [foldRange_succ_at]
    generalize foldRange (fun B y => B.modify ((g[y]!).byte j) (addTo e[y]!)) lo n (Array.replicate 256 0) = B
      at hs ih ⊢
    refine ⟨by rw [Array.size_modify, hs], ?_⟩
    have hk := (g[lo + n]!).byte_lt j
    rw [sumF_succ n, ← ih, ← F128.testBit_byte _ hj hi]
    apply sumF_update _ hk
    · rw [getElem!_eq (B.modify _ _) (by rw [Array.size_modify, hs]; exact hk), getElem!_eq B (by rw [hs]; exact hk),
        Array.getElem_modify_self]
      split
      · exact F128.add_comm _ _
      · rw [F128.add_zero]
    · intro b hb
      by_cases hb256 : b < 256
      · rw [getElem!_eq (B.modify _ _) (by rw [Array.size_modify, hs]; exact hb256),
          getElem!_eq B (by rw [hs]; exact hb256), Array.getElem_modify_of_ne (Ne.symm hb)]
      · simp [hs, hb256]

/-- **A claim's byte tables** of `er`: entry `256·j + b` (`j < 16`, `b < 256`) is `Σ_{i < 8, bit_i(b)} er[8j + i]`. -/
def byteTable (er : Array F128) : Array F128 :=
  Array.ofFn (n := 4096) fun k => sumF 8 fun i => if (k.val % 256).testBit i then er[8 * (k.val / 256) + i]! else 0

theorem byteTable_sum (er : Array F128) (v : F128) :
    (sumF 128 fun u => if v.bit u then er[u]! else 0) = sumF 16 fun j => (byteTable er)[256 * j + v.byte j]! := by
  rw [show 128 = 8 * 16 from rfl, sumF_blocks]
  apply sumF_congr
  intro j hj
  have hb := v.byte_lt j
  rw [byteTable, getElem!_eq _ (by rw [Array.size_ofFn]; omega), Array.getElem_ofFn]
  simp only [show (256 * j + v.byte j) % 256 = v.byte j by omega, show (256 * j + v.byte j) / 256 = j by omega]
  apply sumF_congr
  intro i hi
  rw [F128.testBit_byte v hj hi]

end Fast

/-- `X̂_c(ω_q)` for `c < 2^cols`: the products of `Ŵ_i(ω_q)` over the bits of `c`. -/
def xhats (sv : Array F128) (q cols : Nat) : Array F128 :=
  let wh := whats sv q cols
  (List.range cols).foldl (fun t i => t ++ t.map (· * wh[i]!)) #[1]

/-- **The codeword** of an F-table at a level: `C[q][ℓ] = Σ_c g[c·2^lanes + ℓ]·X̂_c(ω_q)`. -/
def encodeRows (lv : Level) (g : Array F128) : Array (Array F128) :=
  let lanes := 2 ^ lv.lanes
  let sv := svTable (lv.cols + 1)
  (Array.range (2 ^ lv.d)).map fun q =>
    let xh := xhats sv q lv.cols
    (Array.range lanes).map fun l => sumF (2 ^ lv.cols) fun c => g[c * lanes + l]! * xh[c]!

/-- The cap of a level's flattened entries. -/
def flat (cap : Array ByteArray) : ByteArray := cap.foldl (· ++ ·) ByteArray.empty

/-- **A level's commitment** of the F-table `g` with the leaves' salts `salt q`, each fitted to the scheme's salt size. -/
def commitP (hs : Hashes M) (ms : MerkleScheme) (lv : Level) (g : Array F128) (salt : Nat → ByteArray) : M Commit := do
  let ops := ms.opsWith hs
  let rows := encodeRows lv g
  let salts := (Array.range (2 ^ lv.d)).map fun q => fitBytes ms.saltBytes (salt q)
  let leaves ← (Array.range (2 ^ lv.d)).mapM fun q => ops.leaf rows[q]! salts[q]!
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:lv.d - lv.cap] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => ops.node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨rows, salts, layers, cur⟩

/-- **A level's opening** at `pos`: the rows, each query's `d − cap` siblings leaf side first, and the salts. -/
def openP (ms : MerkleScheme) (cm : Commit) (lv : Level) (pos : Array Nat) : LevelProof :=
  { rows := pos.map fun p => cm.rows[p]!
    siblings := pos.foldl (fun acc p => acc ++ (Array.range (lv.d - lv.cap)).map fun h => (cm.layers[h]!)[(p >>> h) ^^^ 1]!) #[]
    salts := if ms.saltBytes = 0 then #[] else pos.map fun p => cm.salts[p]! }

/-- A K-table's fold at `r` on its low variable. -/
def foldK (r : F256) (a : Array F256) : Array F256 :=
  (Array.range (a.size / 2)).map fun k => a[2 * k]! + r * (a[2 * k + 1]! + a[2 * k]!)

/-- **A sumcheck message**: `(q(0), q`'s leading coefficient`)` for `q(X) = Σ_{x'} t(X, x')·W(X, x')`. -/
def roundPoly (t w : Array F256) : F256 × F256 :=
  let h := t.size / 2
  ((List.range h).foldl (fun acc k => acc + t[2 * k]! * w[2 * k]!) 0,
   (List.range h).foldl (fun acc k => acc + (t[2 * k]! + t[2 * k + 1]!) * (w[2 * k]! + w[2 * k + 1]!)) 0)

/-- **The split** of a K-table: `g[2x + b] = f[x].c_b`. -/
def splitK (f : Array F256) : Array F128 := f.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]

/-- The weight on a split table: `W'[2x + b] = u^b·W[x]`. -/
def liftU (w : Array F256) : Array F256 := w.foldl (fun acc v => (acc.push v).push (F256.u * v)) #[]

/-- An F-table as a K-table. -/
def embedF (g : Array F128) : Array F256 := g.map F256.ofF

/-- `W + β·V` entrywise. -/
def addScaled (w v : Array F256) (b : F128) : Array F256 :=
  (Array.range w.size).map fun i => w[i]! + (v[i]!).smul b

/-- **The multilinear extension** of an F-table at an F point: `Σ_x g[x]·eq(z, x)`. -/
def mleF (g : Array F128) (z : Array F128) : F128 :=
  let e := eqTable z
  sumF g.size fun x => g[x]! * e[x]!

/-- **The consistency weight** of a level's queries: `V(c) = Σ_j eq(α)[j]·X̂_c(ω_{pos_j})` for `c < 2^cols`. -/
def consW (cols : Nat) (pos : Array Nat) (alpha : Array F128) : Array F128 :=
  let ea := eqTable alpha
  let sv := svTable (cols + 1)
  let xs := pos.map fun p => xhats sv p cols
  (Array.range (2 ^ cols)).map fun c => sumF pos.size fun j => ea[j]! * (xs[j]!)[c]!

/-! ### The Ligerito rounds in tasks (`@[csimp]`: proofs see the definitions above) -/

namespace Fast

/-- `(Array.range n).map f` in tasks of `b` entries (`parTab_eq`), for any entry type; `b = 1` is a task to an entry. -/
@[specialize] def parTab {α : Type} (b : Nat) (f : Nat → α) (n : Nat) : Array α :=
  ((Array.range (n / b)).map fun i => Task.spawn fun _ => (Array.range b).map fun j => f (i * b + j)).foldl
    (fun acc t => acc ++ t.get) #[] ++ (Array.range (n - n / b * b)).map fun j => f (n / b * b + j)

theorem parTab_eq {α : Type} (b : Nat) (f : Nat → α) (n : Nat) : parTab b f n = (Array.range n).map f := by
  have hp : ∀ a, ((Array.range a).map fun i => Task.spawn fun _ => (Array.range b).map fun j => f (i * b + j)).foldl
      (fun acc t => acc ++ t.get) #[] = (Array.range (a * b)).map f := by
    intro a
    induction a with
    | zero => simp only [Nat.zero_mul, Array.range_eq_empty_iff.mpr rfl, Array.map_empty, Array.foldl_empty]
    | succ a ih =>
      rw [Array.range_succ, Array.map_append, Array.foldl_append, ih, Nat.succ_mul, Array.range_add]
      simp only [Task.spawn, List.map_toArray, List.map_cons, List.map_nil, List.size_toArray, List.length_cons,
        List.length_nil, Nat.zero_add, List.foldl_toArray', List.foldl_cons, List.foldl_nil, Array.map_append,
        Array.map_map, Function.comp_def]
  conv => rhs; rw [← Nat.add_sub_cancel' (Nat.div_mul_le_self n b)]
  rw [Array.range_add, Array.map_append, Array.map_map, parTab, hp]
  rfl

theorem range_map_get! {α β : Type} [Inhabited α] (f : α → β) (t : Array α) :
    (Array.range t.size).map (fun i => f t[i]!) = t.map f := by
  apply Array.ext
  · simp
  · intro i h1 _
    simp only [Array.getElem_map, Array.getElem_range]
    rw [getElem!_eq t (by simpa using h1)]

/-- `Σ_{k < n} f(k)` in K, added as `roundPoly` adds. -/
def sumK (n : Nat) (f : Nat → F256) : F256 := (List.range n).foldl (fun acc k => acc + f k) 0

/-- `sumK` as compiled code computes it: the same additions in the same order, with no list. -/
@[inline] def sumKFast (n : Nat) (f : Nat → F256) : F256 := foldRange (fun acc k => acc + f k) 0 n 0

@[csimp] theorem sumK_eq_sumKFast : @sumK = @sumKFast :=
  funext fun n => funext fun f => by rw [sumK, sumKFast, foldRange_eq, List.range_eq_range']

theorem addK_assoc (a b c : F256) : a + b + c = a + (b + c) := by
  show F256.add (F256.add a b) c = F256.add a (F256.add b c)
  simp only [F256.add, F128.add_assoc]

theorem addK_zero (a : F256) : a + 0 = a := by
  cases a
  show F256.add _ F256.zero = _
  simp only [F256.add, F256.zero, F128.add_zero]

theorem addK_congr {a a' : F256} (b : F256) (h : a = a') : a + b = a' + b := h ▸ rfl

theorem sumK_zero (f : Nat → F256) : sumK 0 f = 0 := rfl

theorem sumK_succ (n : Nat) (f : Nat → F256) : sumK (n + 1) f = sumK n f + f n := by
  rw [sumK, sumK, List.range_succ, List.foldl_append]
  rfl

theorem sumK_congr {n : Nat} {f f' : Nat → F256} (h : ∀ i, i < n → f i = f' i) : sumK n f = sumK n f' := by
  induction n with
  | zero => rfl
  | succ n ih => rw [sumK_succ, sumK_succ, ih fun i hi => h i (by omega), h n (by omega)]

theorem sumK_add (m n : Nat) (f : Nat → F256) : sumK (m + n) f = sumK m f + sumK n fun i => f (m + i) := by
  induction n with
  | zero => rw [Nat.add_zero, sumK_zero, addK_zero]
  | succ n ih => rw [← Nat.add_assoc, sumK_succ, sumK_succ, ih, addK_assoc]

theorem sumK_blocks (a b : Nat) (f : Nat → F256) :
    sumK (a * b) f = sumK a fun j => sumK b fun i => f (j * b + i) := by
  induction a with
  | zero => rw [Nat.zero_mul]; rfl
  | succ a ih => rw [Nat.succ_mul, sumK_add, ih, sumK_succ]

/-- `Σ_{k < n}` as `2^12`-term blocks and the rest. -/
theorem sumK_split (n : Nat) (f : Nat → F256) : sumK n f =
    sumK (n / 4096) (fun i => sumK 4096 fun j => f (i * 4096 + j)) +
      sumK (n - n / 4096 * 4096) fun j => f (n / 4096 * 4096 + j) := by
  conv => lhs; rw [← Nat.add_sub_cancel' (Nat.div_mul_le_self n 4096)]
  rw [sumK_add, sumK_blocks]

/-- `roundPoly`'s terms. -/
@[inline] def rpA (t w : Array F256) (k : Nat) : F256 := t[2 * k]! * w[2 * k]!
@[inline] def rpB (t w : Array F256) (k : Nat) : F256 := (t[2 * k]! + t[2 * k + 1]!) * (w[2 * k]! + w[2 * k + 1]!)

/-- `roundPoly` with its sums in `2^12`-term blocks, a task to a block (`roundPoly_eq_fast`). -/
def roundPolyFast (t w : Array F256) : F256 × F256 :=
  let h := t.size / 2
  let ps := parTab 1 (fun i => (sumK 4096 fun j => rpA t w (i * 4096 + j), sumK 4096 fun j => rpB t w (i * 4096 + j)))
    (h / 4096)
  (sumK (h / 4096) (fun i => (ps[i]!).1) + sumK (h - h / 4096 * 4096) (fun j => rpA t w (h / 4096 * 4096 + j)),
   sumK (h / 4096) (fun i => (ps[i]!).2) + sumK (h - h / 4096 * 4096) (fun j => rpB t w (h / 4096 * 4096 + j)))

theorem roundPoly_eq_fast (t w : Array F256) : roundPoly t w = roundPolyFast t w := by
  show (sumK (t.size / 2) (rpA t w), sumK (t.size / 2) (rpB t w)) = _
  simp only [roundPolyFast, parTab_eq]
  rw [sumK_split (t.size / 2) (rpA t w), sumK_split (t.size / 2) (rpB t w)]
  refine Prod.ext (addK_congr _ (sumK_congr fun i hi => ?_)) (addK_congr _ (sumK_congr fun i hi => ?_)) <;>
    simp only [getElem!_range_map _ hi]

/-- `foldK` in tasks of `2^12` entries. -/
def foldKFast (r : F256) (a : Array F256) : Array F256 :=
  parTab 4096 (fun k => a[2 * k]! + r * (a[2 * k + 1]! + a[2 * k]!)) (a.size / 2)

/-- `addScaled` in tasks of `2^12` entries. -/
def addScaledFast (w v : Array F256) (b : F128) : Array F256 :=
  parTab 4096 (fun i => w[i]! + (v[i]!).smul b) w.size

/-- `embedF` in tasks of `2^12` entries. -/
def embedFFast (g : Array F128) : Array F256 := parTab 4096 (fun i => F256.ofF g[i]!) g.size

/-- `consW` with a task to each query's `X̂` and its entries in tasks of `2^8`. -/
def consWFast (cols : Nat) (pos : Array Nat) (alpha : Array F128) : Array F128 :=
  let ea := eqTable alpha
  let sv := svTable (cols + 1)
  let xs := parTab 1 (fun j => xhats sv pos[j]! cols) pos.size
  parTab 256 (fun c => sumF pos.size fun j => ea[j]! * (xs[j]!)[c]!) (2 ^ cols)

end Fast

open Fast in
@[csimp] theorem roundPoly_eq_roundPolyFast : @roundPoly = @roundPolyFast :=
  funext fun t => funext fun w => roundPoly_eq_fast t w

open Fast in
@[csimp] theorem foldK_eq_foldKFast : @foldK = @foldKFast :=
  funext fun r => funext fun a => by rw [foldK, foldKFast, parTab_eq]

open Fast in
@[csimp] theorem addScaled_eq_addScaledFast : @addScaled = @addScaledFast :=
  funext fun w => funext fun v => funext fun b => by rw [addScaled, addScaledFast, parTab_eq]

open Fast in
@[csimp] theorem embedF_eq_embedFFast : @embedF = @embedFFast :=
  funext fun g => by rw [embedF, embedFFast, parTab_eq, range_map_get!]

open Fast in
@[csimp] theorem consW_eq_consWFast : @consW = @consWFast :=
  funext fun cols => funext fun pos => funext fun alpha => by
    simp only [consW, consWFast, parTab_eq]
    rw [range_map_get! (fun p => xhats (svTable (cols + 1)) p cols) pos]

/-- `mleF` with its sum in tasks of `2^12` terms. -/
def Fast.mleFFast (g : Array F128) (z : Array F128) : F128 :=
  let e := eqTable z
  Fast.sumFPar 4096 g.size fun x => g[x]! * e[x]!

open Fast in
@[csimp] theorem mleF_eq_mleFFast : @mleF = @mleFFast :=
  funext fun g => funext fun z => by rw [mleF, mleFFast, sumFPar_eq]

/-- What the prover keeps through the opening: the K-table being folded, the weight, and the proof's lists so far. -/
structure LigState where
  tab : Array F256
  w : Array F256
  msgs : Array (F256 × F256) := #[]
  oods : Array F128 := #[]
  qn : Array UInt64 := #[]
  cn : Array UInt64 := #[]
  cbn : Array UInt64 := #[]
  deriving Inhabited

/-- The prover's Ligerito actions: its transcript and its state. -/
abbrev LP (M : Type → Type) := StateT LigState (TMW M)

def liftT {α : Type} (x : TMW M α) : LP M α := StateT.lift x

/-- Send the next sumcheck message. -/
def sendMsg (m : F256 × F256) : LP M Unit := do
  modify fun st => { st with msgs := st.msgs.push m }
  liftT (liftTM (absorbK m.1))
  liftT (liftTM (absorbK m.2))

/-- A grinding nonce onto list `kind` (`0` query, `1` consistency, `2` claim), absorbed as a message: `0`, the only one the
verifier accepts (R4), so the tape's nonces are unread. -/
def powP (_nonce : Nat → Nat → UInt64) (kind : Nat) : LP M Unit := do
  let st ← get
  set (match kind with
    | 0 => { st with qn := st.qn.push 0 }
    | 1 => { st with cn := st.cn.push 0 }
    | _ => { st with cbn := st.cbn.push 0 })
  liftT (liftTM (pow 0))

/-- A challenge: fold the table and the weight at it; then the next message, after splitting when `split`. -/
def challengeP (io : Asks M) (split : Bool) : LP M F256 := do
  let r ← liftT (sampleKP io)
  modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
  if split then modify fun st => { st with tab := embedF (splitK st.tab), w := liftU st.w }
  let st ← get
  sendMsg (roundPoly st.tab st.w)
  return r

/-- An out-of-domain sample on the current F-table `g` with an intro message (levels ≥ 1). -/
def oodIntroP (io : Asks M) (nonce : Nat → Nat → UInt64) (g : Array F128) (dim : Nat) : LP M Unit := do
  let z ← liftT (sampleVecP io dim)
  let y := mleF g z
  modify fun st => { st with oods := st.oods.push y }
  liftT (liftTM (absorbF y))
  let e := embedF (eqTable z)
  sendMsg (roundPoly (embedF g) e)
  powP nonce 2
  let b ← liftT (sampleP io)
  modify fun st => { st with w := addScaled st.w e b }

/-- A level's queries: the query and consistency coins, the opening, and the consistency weight `V`. -/
def queryP (io : Asks M) (nonce : Nat → Nat → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    LP M (LevelProof × Array F128) := do
  powP nonce 0
  let coins ← liftT (sampleVecP io lv.q)
  let pos := positions lv coins
  powP nonce 1
  let alpha ← liftT (sampleVecP io (Nat.log2 (lv.q - 1) + 1))
  return (openP ms cm lv pos, consW lv.cols pos alpha)

/-! ## The ring switch (`FlockComplete.ringSwitchP`, `basisVals`) -/

/-- **The ring switch** sending claim `k`'s switched vector `sv k`: each claim's `s_k`, and what the verifier derives:
`T` and the switched claims (`FlockComplete.ringSwitchCut`). -/
def ringSwitchP (io : Asks M) (sv : Nat → Array F128) (claims : Array Claim) :
    TMW M (Array (Array F128) × F128 × Array Switched) := do
  let n := claims.size
  liftTM (label "flock-pcs-open-batch-v0")
  let mut ss : Array (Array F128) := #[]
  let mut betas : Array F128 := #[]
  let mut perClaim : Array (Array F128 × Array F128) := #[]
  for k in [0:n] do
    let c := claims[k]!
    let s := sv k
    liftTM (label "flock-ring-switch-v0")
    liftTM (absorbSlice s)
    let rpp ← sampleVecP io 7
    let t := (Array.range 128).map fun u =>
      (Array.range 128).foldl (fun acc v => if (s[v]!).bit u then acc + F128.monomial v else acc) (0 : F128)
    betas := betas.push (dot (eqTable rpp) t)
    perClaim := perClaim.push (rpp, c.x.extract 1 c.x.size)
    ss := ss.push s
  let gamma ← sampleVecP io n
  let T := dot gamma betas
  let sws := (Array.range n).map fun k => { gamma := gamma[k]!, rpp := (perClaim[k]!).1, z := (perClaim[k]!).2 : Switched }
  return (ss, T, sws)

/-- `ringSwitchP` with every claim's switched vector computed first, a task to a claim (`ringSwitchP_eq_fast`). -/
def ringSwitchPFast (io : Asks M) (sv : Nat → Array F128) (claims : Array Claim) :
    TMW M (Array (Array F128) × F128 × Array Switched) :=
  let pre := Fast.parTab 1 sv claims.size
  ringSwitchP io (fun k => if k < claims.size then pre[k]! else sv k) claims

omit [MonadExceptOf String M] in
theorem ringSwitchP_eq_fast (io : Asks M) (sv : Nat → Array F128) (claims : Array Claim) :
    ringSwitchP io sv claims = ringSwitchPFast io sv claims := by
  have h : (fun k => if k < claims.size then (Fast.parTab 1 sv claims.size)[k]! else sv k) = sv := by
    funext k
    split
    · rw [Fast.parTab_eq, Fast.getElem!_range_map _ ‹_›]
    · rfl
  exact congrArg (fun s => ringSwitchP io s claims) h.symm

@[csimp] theorem ringSwitchP_eq_ringSwitchPFast : @ringSwitchP = @ringSwitchPFast := by
  funext M _ _ io sv claims
  exact ringSwitchP_eq_fast (M := M) io sv claims

/-- **The batched basis on the Boolean cube** (§12.4): `b(y) = Σ_k γ_k·Φ_k(eq(z_k, y))` for `y < 2^ℓ`, with
`Φ_k(x^u) = eq(r''_k, u)`. -/
def basisVals (sws : Array Switched) (l : Nat) : Array F128 :=
  let tabs := sws.map fun sw => (sw.gamma, eqTable sw.z, eqTable sw.rpp)
  (Array.range (2 ^ l)).map fun y =>
    tabs.foldl (fun acc (g, ez, er) =>
      let v := ez[y]!
      acc + g * sumF 128 fun u => if v.bit u then er[u]! else 0) 0

/-! ## The masked Ligerito (`FlockComplete.padF` to `FlockComplete.ligeritoPZk`) -/

/-- The sum `Σ_c a[c]·b[c]` of two K-tables. -/
def dotK (a b : Array F256) : F256 := (List.range a.size).foldl (fun acc c => acc + a[c]! * b[c]!) 0

/-- `dotK` with its sum in `2^12`-term blocks, a task to a block (`dotK_eq_dotKFast`). -/
def Fast.dotKFast (a b : Array F256) : F256 :=
  let n := a.size
  let ps := Fast.parTab 1 (fun i => Fast.sumK 4096 fun j => a[i * 4096 + j]! * b[i * 4096 + j]!) (n / 4096)
  Fast.sumK (n / 4096) (fun i => ps[i]!) +
    Fast.sumK (n - n / 4096 * 4096) fun j => a[n / 4096 * 4096 + j]! * b[n / 4096 * 4096 + j]!

open Fast in
@[csimp] theorem dotK_eq_dotKFast : @dotK = @dotKFast :=
  funext fun a => funext fun b => by
    show sumK a.size (fun c => a[c]! * b[c]!) = _
    simp only [dotKFast, parTab_eq]
    rw [sumK_split a.size (fun c => a[c]! * b[c]!)]
    exact addK_congr _ (sumK_congr fun i hi => by simp only [getElem!_range_map _ hi])

/-- A sumcheck message masked by the lane round `j`'s pads (`lgMsg j 0` to `lgMsg j 3`, coordinate by coordinate). -/
def maskMsg (lay : Zk.PadLayout) (h : Array F128) (j : Nat) (m : F256 × F256) : F256 × F256 :=
  (m.1 + ⟨h[lay.lgMsg j 0]!, h[lay.lgMsg j 1]!⟩, m.2 + ⟨h[lay.lgMsg j 2]!, h[lay.lgMsg j 3]!⟩)

/-- **The masked Ligerito prover** on the table `t` committed at level 0 as `cm0`, for the claim `Σ t·b = T`, as rep
`rep` with the pads `h`; levels ≥ 1 committed by `commit` (the analysis's `commitP hs ms`). -/
def ligeritoPZk (io : Asks M) (commit : Level → Array F128 → (Nat → ByteArray) → M Commit) (ms : MerkleScheme)
    (sch : Schedule) (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : Nat)
    (extra lanePad : Nat → Nat → F128) (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64) :
    TMW M (LigeritoProof × Zk.Level0Msgs × Zk.L0Rec) := do
  let n := sch.n
  let R := sch.levels.size - 1
  let lv := fun (i : Nat) => sch.levels.getD i default
  let lanes := 2 ^ (lv 0).lanes
  liftTM (label "flock-ligerito-basis-f256-split-v0")
  liftTM (absorbF T)
  liftTM (absorbBytes (flat cm0.cap))
  liftTM (label Zk.LABEL_OPEN)
  let prog : LP M (Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof × Zk.Level0Msgs ×
      Array F256 × (F256 × F256) × F128 × F128) := do
    -- level 0's out-of-domain sample (no message), its value masked
    let z ← liftT (sampleVecP io n)
    let y := mleF t z + h[lay.lgOod 0]!
    modify fun st => { st with oods := st.oods.push y }
    liftT (liftTM (absorbF y))
    powP nonce 2
    let b0 ← liftT (sampleP io)
    modify fun st => { st with w := addScaled st.w (embedF (eqTable z)) b0 }
    let st ← get
    sendMsg (maskMsg lay h 0 (roundPoly st.tab st.w))
    let mut initCh : Array F256 := #[]
    let mut rho : F256 × F256 := (0, 0)
    let mut zl : Zk.Level0Msgs := default
    for j in [0:sch.k0] do
      let r ← liftT (sampleKP io)
      initCh := initCh.push r
      modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
      if j + 1 == sch.k0 then
        -- the pair's sums, its coins, the blinded claim and the padding's combination
        let st ← get
        let r0 := (Array.range st.tab.size).map (extra (2 * rep))
        let r1 := (Array.range st.tab.size).map (extra (2 * rep + 1))
        let e0 := dotK st.w (embedF r0) + ⟨h[lay.lgE 0 0]!, h[lay.lgE 0 1]!⟩
        let e1 := dotK st.w (embedF r1) + ⟨h[lay.lgE 1 0]!, h[lay.lgE 1 1]!⟩
        liftT (liftTM (absorbK e0))
        liftT (liftTM (absorbK e1))
        let rh0 ← liftT (sampleKP io)
        let rh1 ← liftT (sampleKP io)
        rho := (rh0, rh1)
        if !Zk.rhoIndependent rho then throw "PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128"
        liftT (liftTM (label Zk.LABEL_CLOSE))
        let y1 := (Array.range st.tab.size).map fun c => st.tab[c]! + rh0.smul r0[c]! + rh1.smul r1[c]!
        let tP := dotK st.w y1
        liftT (liftTM (absorbK tP))
        let ef := eqTableK initCh
        let ybar := (Array.range tPad).map fun i =>
          (List.range lanes).foldl (fun acc l => acc + (ef[l]!).smul (lanePad l i))
            (rh0.smul (lanePad (lanes + 2 * rep) i) + rh1.smul (lanePad (lanes + 2 * rep + 1) i))
        liftT (liftTM (absorbSlice (ybar.foldl (fun acc v => (acc.push v.c0).push v.c1) #[])))
        zl := ⟨e0, e1, tP, ybar⟩
        modify fun st => { st with tab := embedF (splitK y1), w := liftU st.w }
        let st ← get
        sendMsg (roundPoly st.tab st.w)
      else
        let st ← get
        sendMsg (maskMsg lay h (j + 1) (roundPoly st.tab st.w))
    let mut splitDim := n - sch.k0 + 1
    let g1 := (← get).tab.map F256.c0
    let cm1 ← liftT (StateT.lift (commit (lv 1) g1 (salt 1)))
    liftT (liftTM (absorbBytes (flat cm1.cap)))
    oodIntroP io nonce g1 splitDim
    oodIntroP io nonce g1 splitDim
    let (op0, v0) ← queryP io nonce ms (lv 0) cm0
    sendMsg (roundPoly (embedF g1) (liftU (embedF v0)))
    powP nonce 2
    let bq ← liftT (sampleP io)
    modify fun st => { st with w := addScaled st.w (liftU (embedF v0)) bq }
    let mut prevCm := cm1
    let mut caps : Array (Array ByteArray) := #[cm1.cap]
    let mut rproofs : Array LevelProof := #[]
    let mut fin : FinalProof := default
    for i in [0:R] do
      let last := i + 1 == R
      let mut f : Array F256 := #[]
      for j in [0:4] do
        if j + 1 == 4 then
          let r ← liftT (sampleKP io)
          modify fun st => { st with tab := foldK r st.tab, w := foldK r st.w }
          let d ← get
          f := d.tab
          modify fun st => { st with tab := embedF (splitK st.tab), w := liftU st.w }
          let st ← get
          -- nothing reads the last level's split: it sends upstream's plain fold message, of the unsplit table
          sendMsg (if last then roundPoly d.tab d.w else roundPoly st.tab st.w)
        else
          let _ ← challengeP io false
      let ext := splitDim - 4
      if !last then
        splitDim := ext + 1
        let g := splitK f
        let cmN ← liftT (StateT.lift (commit (lv (i + 2)) g (salt (i + 2))))
        liftT (liftTM (absorbBytes (flat cmN.cap)))
        caps := caps.push cmN.cap
        oodIntroP io nonce g splitDim
        oodIntroP io nonce g splitDim
        let (op, v) ← queryP io nonce ms (lv (i + 1)) prevCm
        rproofs := rproofs.push op
        sendMsg (roundPoly (embedF g) (liftU (embedF v)))
        powP nonce 2
        let bb ← liftT (sampleP io)
        modify fun st => { st with w := addScaled st.w (liftU (embedF v)) bb }
        prevCm := cmN
      else
        let yr := f.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]
        for x in yr do liftT (liftTM (absorbF x))
        let (op, _) ← queryP io nonce ms (lv R) prevCm
        fin := { yr, rows := op.rows, siblings := op.siblings, salts := op.salts }
        powP nonce 2
        let _ ← liftT (sampleP io)
    return (caps, rproofs, fin, op0, zl, initCh, rho, b0, y)
  let ((caps, rproofs, fin, op0, zl, initCh, rho, b0, y), st) ← prog.run { tab := embedF t, w := embedF b }
  return ({ initialCap := cm0.cap, initialProof := op0, recursiveCaps := caps, recursiveProofs := rproofs,
            finalProof := fin, messages := st.msgs, queryNonces := st.qn, oodValues := st.oods, claimNonces := st.cbn,
            consistencyNonces := st.cn }, zl, { target := T, ood := #[(b0, y)], challenges := initCh, rho })

/-! ## The inner proof (`FlockComplete.PadsZk` to `FlockComplete.innerP`) -/

/-- **One rep's pads** (upstream's `Pads`): `h`, `μ`, their RS padding, and `τ`'s salt. -/
structure PadsZk where
  h : Array F128
  mu : Array F128
  padH : Array F128
  padMu : Array F128
  tauSalt : ByteArray
  deriving Inhabited

/-- **The pads' commitment**: the two rows' codewords, the leaves' salts, the tree's levels and its root. -/
structure InnerCommit where
  ch : Array F128
  cm : Array F128
  salts : Array ByteArray
  layers : Array (Array ByteArray)
  root : ByteArray
  deriving Inhabited

/-- **The pads' commitment** of the rep's pads, leaf `j` salted from `salt j` (fitted to 192 bytes). -/
def innerCommitP (hs : Hashes M) (p : PadsZk) (salt : Nat → ByteArray) : M InnerCommit := do
  let hv := p.h ++ p.padH
  let mv := p.mu ++ p.padMu
  let n := hv.size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE
  let ch := (Array.range n).map (Zk.codeAt hv)
  let cm := (Array.range n).map (Zk.codeAt mv)
  let salts := (Array.range n).map fun j => fitBytes 192 (salt j)
  let leaves ← (Array.range n).mapM fun j => do
    Hm96.Default512.leafWith hs (← hs.sha512 (ch[j]!.toBytes ++ cm[j]!.toBytes)) salts[j]!
  let mut layers := #[leaves]
  let mut cur := leaves
  for _ in [0:Nat.log2 n] do
    cur ← (Array.range (cur.size / 2)).mapM fun i => (MerkleOps.hm96With hs).node cur[2 * i]! cur[2 * i + 1]!
    layers := layers.push cur
  return ⟨ch, cm, salts, layers, cur[0]!⟩

/-- **The inner proof** that the pads satisfy `cons` and the triple's constraints, `finalC` the masked `final_c`; the
batched constraint is not checked on the pads. -/
def innerP (io : Asks M) (hs : Hashes M) (lay : Zk.PadLayout) (p : PadsZk) (ic : InnerCommit) (finalC : F128)
    (cons : Array (String × Zk.Lin)) : TMW M Zk.InnerProof := do
  let n := ic.ch.size
  liftTM (label Zk.LABEL_INNER)
  liftTM (absorbF finalC)
  let eps ← sampleP io
  let rho := eps * p.h[lay.fa]! + p.h[lay.tx]!
  let sigma := p.h[lay.fb]! + p.h[lay.ty]!
  liftTM (absorbSlice #[rho, sigma])
  let cons := cons ++ Zk.triple lay eps rho sigma
  let gammas ← sampleVecP io cons.size
  let l := Zk.Lin.combo gammas (cons.map (·.2)) lay.len
  let tau := dot l.a p.mu
  let tauCommit ← StateT.lift (Zk.tauCommitmentWith hs tau p.tauSalt)
  liftTM (absorbBytes tauCommit)
  let beta ← sampleP io
  if beta.isZero then throw "PROVER-STOPPED: degenerate coin β = 0"
  let hv := p.h ++ p.padH
  let mv := p.mu ++ p.padMu
  let y := (Array.range hv.size).map fun i => hv[i]! + beta * mv[i]!
  liftTM (absorbSlice y)
  let qs := (← sampleVecP io Zk.Q).map fun w => w.lo.toNat % n
  return { root := ic.root, rho, sigma, tauCommit, tauSalt := p.tauSalt, tau, y,
           opened := qs.map fun j => (ic.ch[j]!, ic.cm[j]!), salts := qs.map fun j => ic.salts[j]!,
           paths := qs.map fun j => (Array.range (Nat.log2 n)).map fun k => (ic.layers[k]!)[(j >>> k) ^^^ 1]! }

/-- The shape `Zk.verifyRepWith` decodes a rep's proof file at, with `extras` extra claims. -/
def shapeZk (st : Setup) (sch : Schedule) (extras : Nat) : Shape :=
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 then none else some st.merkle.saltBytes
  { m := st.m, kLog := st.kLog, claims := 2 + extras, sched := sch, hashLen := st.merkle.digestLen, saltLen }

/-! ## The proof file's bytes (`FlockSoundness.Discharge.ZkView.encZk`) -/

/-- A length or a `u64` field, little-endian. -/
def wLen (n : Nat) : ByteArray := le64 n.toUInt64

/-- A `Vec<T>`: its length, then its elements. -/
def wVec {α : Type} (f : α → ByteArray) (xs : Array α) : ByteArray := xs.foldl (fun b x => b ++ f x) (wLen xs.size)

/-- A `Vec<u8>`. -/
def wBytes (b : ByteArray) : ByteArray := wLen b.size ++ b

/-- An `(F128, F128)`. -/
def wPair (p : F128 × F128) : ByteArray := p.1.toBytes ++ p.2.toBytes

/-- An `F256`: `c0`, then `c1`. -/
def wF256 (x : F256) : ByteArray := x.c0.toBytes ++ x.c1.toBytes

/-- Each level's `opened_salts`, present when the scheme salts its leaves. -/
def wSalts (sh : Shape) (xs : Array ByteArray) : ByteArray :=
  match sh.saltLen with
  | none => ByteArray.empty
  | some _ => wVec id xs

/-- A level's openings: its rows, its Merkle siblings, its salts. -/
def wLevel (sh : Shape) (p : LevelProof) : ByteArray :=
  wVec (wVec F128.toBytes) p.rows ++ wVec id p.siblings ++ wSalts sh p.salts

/-- The commitment's parameters. -/
def wParams (p : PcsParams) : ByteArray :=
  le64 p.m ++ le64 p.logInvRate ++ le64 p.logBatchSize ++ le32 p.profile ++
    (match p.numLanes with
     | none => ByteArray.mk #[0]
     | some x => ByteArray.mk #[1] ++ le64 x) ++
    le32 p.merkleHash

/-- `R1csProofLigerito`'s Ligerito proof, level 0's hiding messages excluded. -/
def wLigerito (sh : Shape) (l : LigeritoProof) : ByteArray :=
  wVec id l.initialCap ++ wLevel sh l.initialProof ++ wVec (wVec id) l.recursiveCaps ++
    wVec (wLevel sh) l.recursiveProofs ++
    (wVec F128.toBytes l.finalProof.yr ++ wVec (wVec F128.toBytes) l.finalProof.rows ++ wVec id l.finalProof.siblings ++
      wSalts sh l.finalProof.salts) ++
    wLen 0 ++ wVec (fun m : F256 × F256 => wF256 m.1 ++ wF256 m.2) l.messages ++ wVec le64 l.queryNonces ++
    wVec F128.toBytes l.oodValues ++ wLen 0 ++ wVec le64 l.claimNonces ++ wVec le64 l.consistencyNonces

/-- The commitment and `R1csProofLigerito`, with the `--zk` differences. -/
def wProof (sh : Shape) (p : Proof) : ByteArray :=
  wVec id p.cap ++ wParams p.params ++
    (wVec F128.toBytes p.zerocheck.round1ab ++ wVec F128.toBytes p.zerocheck.round1c ++ wVec wPair p.zerocheck.rounds ++
      p.zerocheck.finalA.toBytes ++ p.zerocheck.finalB.toBytes ++ p.zerocheck.finalC.toBytes ++ wLen 0) ++
    (wVec wPair p.lincheck.rounds ++ wVec F128.toBytes p.lincheck.zPartial ++ wLen 0 ++ wLen 0) ++
    wVec (fun v => wVec F128.toBytes v ++ wLen 0) p.ringSwitches ++ wLen 0 ++
    wLigerito sh p.ligerito

/-- Level 0's hiding messages, as `Some`. -/
def wLevel0 (m : Zk.Level0Msgs) : ByteArray :=
  ByteArray.mk #[1] ++ wF256 m.e0 ++ wF256 m.e1 ++ wF256 m.tPrime ++ wVec wF256 m.ybar

/-- The inner proof (`zk_veil::InnerProof`). -/
def wInner (p : Zk.InnerProof) : ByteArray :=
  wBytes p.root ++ p.rho.toBytes ++ p.sigma.toBytes ++ wBytes p.tauCommit ++ wBytes p.tauSalt ++ p.tau.toBytes ++
    wVec F128.toBytes p.y ++ wVec wPair p.opened ++ wVec id p.salts ++ wVec (wVec id) p.paths

/-- **A `--zk` proof file**: the bytes `flock-circuit --zk` writes for the decoded proof `p` at shape `sh`. -/
def encZk (sh : Shape) (p : Zk.ZkProof) : ByteArray :=
  wProof sh p.proof ++ wLevel0 p.level0 ++ wInner p.inner

end Flock.Firewall.Finish
