/-
# The firewall's level-0 recheck (live PROTOCOL.md §10): a table's level-0 root from its witness and level 0's hiding

Before anything of a firewalled table leaves, the firewall commits the table's level 0 again from the host's witness, the
table's level-0 hiding and its level-0 salts, and the root that leaves in the session's `Commit` is that recheck's
(`flock-circuit`'s `Firewall::new` and `Firewall::root`). This is the recheck of record: the masked prover's level-0
commitment (`FlockComplete.commitPZk` in the analysis), here as the code the firewall runs.

* `rows`: level 0's padded codeword. Row `q` holds the table's `2^lanes` witness lanes, lane `l` the words
  `g[c·2^lanes + l]`, then the four extra lanes `extra e`; each entry is the lane's encoding at `ω_q`,
  `Σ_c v(c)·X̂_c(ω_q)`, plus its padding `(c + Ŵ_cols(ω_q))·Σ_i μ_l[i]·X̂_i(ω_q)` (`padF`, lane `l`'s coefficients
  `lanePad l`, `t_pad` of them). `rowsDirect` evaluates each entry so; `rowsFft` takes each lane's entry as one polynomial
  in the novel basis over `cols + 1` bits (`coeffs`) and evaluates it on all `2^d` points by the additive FFT (`fft`),
  `d·2^d` products a lane where `rowsDirect` takes `2^(d + cols)`. `rows` is `rowsFft` where the padding fits below
  `2^cols` and the domain, wider than the columns, in a word, as at `m` = 25, 26 and 27.
* `leaves`: row `q`'s `hm96-sha512/v1` leaf under the salt `salt q` fitted to 192 bytes, computed in tasks of `2^5` rows
  (`parRange`).
* `cap`: the Merkle nodes `SHA-512(l ‖ r)` `d − cap` levels up from the leaves (`layer`), and `root` their concatenation,
  at level 0 of the table's `--zk` schedule (`Zk.schedule m`, which also fixes `t_pad`).

The firewall's code reaches it through the session loop's `Flock.Firewall.Level0`, which commits the level 0 it draws,
and `flock-firewall finish`'s `table` request (`Flock.Firewall.Finish.commit0`, on `parse`'s bytes). The root that
leaves is one of these, and the finish keeps the table's tree for its reps' openings (`Finish.openLive`): Rust commits
nothing of level 0 under the firewall.
-/
import Flock.Firewall.Coins
import Flock.Ligerito
import Flock.Merkle
import Flock.Zk

namespace Flock.Firewall.Link

/-- `Σ_{i < n} f(i)`. -/
def sumF (n : Nat) (f : Nat → F128) : F128 := (List.range n).foldl (fun acc i => acc + f i) 0

/-- `X̂_c(ω_q)` for `c < 2^cols`: the products of `Ŵ_i(ω_q)` over the bits of `c`. -/
def xhats (sv : Array F128) (q cols : Nat) : Array F128 :=
  let wh := whats sv q cols
  (List.range cols).foldl (fun t i => t ++ t.map (· * wh[i]!)) #[1]

/-- Lane `l`'s padding at a position: `(c + Ŵ_logMsg)·Σ_i μ[i]·X_i` at the position's `Ŵ`s `ws` (`Zk.padAt` on F). -/
def padF (ws : Array F128) (logMsg : Nat) (mu : Array F128) : F128 := (Zk.padAt ws logMsg (mu.map F256.ofF)).c0

/-- Every lane's padding coefficients: the witness lanes', then the extra lanes'. -/
def musOf (lv : Level) (tPad : Nat) (lanePad : Nat → Nat → F128) : Array (Array F128) :=
  (Array.range (2 ^ lv.lanes + Zk.EXTRA_LANES)).map fun l => (Array.range tPad).map (lanePad l)

/-- **Row `q` of level 0's padded codeword**, at the subspace normalisers `sv` and the paddings `mus`. -/
def row (lv : Level) (g : Array F128) (extra : Nat → Nat → F128) (sv : Array F128) (mus : Array (Array F128))
    (q : Nat) : Array F128 :=
  let lanes := 2 ^ lv.lanes
  let xh := xhats sv q lv.cols
  let ws := whats sv q (lv.cols + 1)
  (Array.range (lanes + Zk.EXTRA_LANES)).map fun l =>
    let v : Nat → F128 := if l < lanes then fun c => g[c * lanes + l]! else extra (l - lanes)
    sumF (2 ^ lv.cols) (fun c => v c * xh[c]!) + padF ws lv.cols mus[l]!

/-- **Level 0's padded codeword**, row by row: each entry evaluated on its own (`row`). -/
def rowsDirect (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) : Array (Array F128) :=
  (Array.range (2 ^ lv.d)).map (row lv g extra (svTable (lv.cols + 1)) (musOf lv tPad lanePad))

/-- `whatsGo` with the inverses `svi[i] = sv[i]⁻¹` given, so that a twiddle costs no inversion. -/
def whatsInvGo (sv svi : Array F128) : Nat → Nat → F128 → Array F128 → Array F128
  | 0, _, _, out => out
  | fuel + 1, i, s, out => whatsInvGo sv svi fuel (i + 1) (s * s + sv[i]! * s) (out.push (s * svi[i]!))

/-- Level `s`'s twiddles `Ŵ_s(ω_{r·2^s})`, `r < 2^(d − s)`, for each `s < n`, at the inverses `svi` of `sv`. -/
def twiddlesWith (sv svi : Array F128) (d n : Nat) : Array (Array F128) :=
  (Array.range n).map fun s => (Array.range (2 ^ (d - s))).map fun r =>
    (whatsInvGo sv svi (s + 1) 0 ⟨(r * 2 ^ s).toUInt64, 0⟩ #[])[s]!

/-- Level `s`'s twiddles `Ŵ_s(ω_{r·2^s})`, `r < 2^(d − s)`, for each `s < n`: `sv`'s `n` inverses once, not one for
each twiddle's every `Ŵ_i`. -/
def twiddles (sv : Array F128) (d n : Nat) : Array (Array F128) := twiddlesWith sv (sv.map F128.inv) d n

/-- **The additive FFT** in the novel basis: entry `r < 2^(d − s)` of `fft tw d s n u` is
`Σ_{c < 2^n} u(c)·Π_{i ∈ bits(c)} Ŵ_{s+i}(ω_{r·2^s})`, at the twiddles `tw = twiddles sv d n'` (`s + n ≤ n'`). Its halves
on `u`'s even and odd coefficients read `Ŵ_j` for `j > s` only, which is the same at `ω_{r·2^s}` and `ω_{⌊r/2⌋·2^(s+1)}`. -/
def fft (tw : Array (Array F128)) (d : Nat) : (s n : Nat) → (Nat → F128) → Array F128
  | s, 0, u => Array.replicate (2 ^ (d - s)) (u 0)
  | s, n + 1, u =>
    let e := fft tw d (s + 1) n (fun c => u (2 * c))
    let o := fft tw d (s + 1) n (fun c => u (2 * c + 1))
    let t := tw[s]!
    (Array.range (2 ^ (d - s))).map fun r => e[r / 2]! + t[r]! * o[r / 2]!

theorem spawn_or_pure_get {α : Type} (c : Prop) [Decidable c] (x : α) :
    (if c then Task.spawn fun _ => x else Task.pure x).get = x := by
  split <;> rfl

/-- `fft` with the even half in its own task on the top four levels, so that a lane's FFT takes up to 16 cores
(`fftPar_eq`). -/
def fftPar (tw : Array (Array F128)) (d : Nat) : (s n : Nat) → (Nat → F128) → Array F128
  | s, 0, u => Array.replicate (2 ^ (d - s)) (u 0)
  | s, n + 1, u =>
    let eT := if s < 4 ∧ 4 ≤ n then Task.spawn fun _ => fftPar tw d (s + 1) n (fun c => u (2 * c))
      else Task.pure (fftPar tw d (s + 1) n (fun c => u (2 * c)))
    let o := fftPar tw d (s + 1) n (fun c => u (2 * c + 1))
    let e := eT.get
    let t := tw[s]!
    (Array.range (2 ^ (d - s))).map fun r => e[r / 2]! + t[r]! * o[r / 2]!

theorem fftPar_eq (tw : Array (Array F128)) (d : Nat) :
    ∀ n s u, fftPar tw d s n u = fft tw d s n u
  | 0, s, u => by simp only [fftPar, fft]
  | n + 1, s, u => by simp only [fftPar, fft, fftPar_eq tw d n, spawn_or_pure_get]

@[csimp] theorem fft_eq_fftPar : @fft = @fftPar := by
  funext tw d s n u
  exact (fftPar_eq tw d n s u).symm

/-- Entry `i` of `mu`, 0 past its end. -/
def muAt (mu : Array F128) (i : Nat) : F128 := if i < mu.size then mu[i]! else 0

/-- A lane's entry as one polynomial in the novel basis over `cols + 1` bits: its words `v c` plus its padding
`(c + Ŵ_cols)·Σ_i μ_i·X̂_i`, which is `c·μ_i` at `X̂_i` and `μ_i` at `X̂_{2^cols + i} = X̂_i·Ŵ_cols` (`μ` below `2^cols`). -/
def coeffs (cols : Nat) (v : Nat → F128) (mu : Array F128) (c : Nat) : F128 :=
  if c < 2 ^ cols then v c + Zk.PAD_C * muAt mu c else muAt mu (c - 2 ^ cols)

/-- Lane `l`'s coefficients (`coeffs`): its words in `g` or its extra lane, and its padding. -/
def laneCoeffs (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) (l : Nat) : Nat → F128 :=
  let lanes := 2 ^ lv.lanes
  coeffs lv.cols (if l < lanes then fun c => g[c * lanes + l]! else extra (l - lanes)) ((Array.range tPad).map (lanePad l))

/-- **Level 0's padded codeword** by the FFT, one task to a lane, on each lane's `laneCoeffs`. A lane's paddings `mu` are
built here, once: compiled code eta-expands a definition that returns a function, so `laneCoeffs … l` would rebuild them
at every coefficient the FFT reads. -/
def rowsFft (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) : Array (Array F128) :=
  let nL := 2 ^ lv.lanes + Zk.EXTRA_LANES
  let tw := twiddles (svTable (lv.cols + 1)) lv.d (lv.cols + 1)
  let es := ((Array.range nL).map fun l =>
    let mu := (Array.range tPad).map (lanePad l)
    Task.spawn fun _ => fft tw lv.d 0 (lv.cols + 1)
      (coeffs lv.cols (if l < 2 ^ lv.lanes then fun c => g[c * 2 ^ lv.lanes + l]! else extra (l - 2 ^ lv.lanes)) mu)).map
    Task.get
  (Array.range (2 ^ lv.d)).map fun q => (Array.range nL).map fun l => es[l]![q]!

/-- **Level 0's padded codeword** of the packed table `g`, with the extra lanes `extra` and the paddings `lanePad`: by the
FFT where the padding fits below `2^cols` and the domain, wider than the columns, in a word; else row by row. -/
def rows (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) : Array (Array F128) :=
  if tPad ≤ 2 ^ lv.cols ∧ lv.cols < lv.d ∧ lv.d ≤ 64 then rowsFft lv tPad g extra lanePad
  else rowsDirect lv tPad g extra lanePad

/-- The first `n` bytes of `b`, zero-padded to `n`. -/
def fitBytes (n : Nat) (b : ByteArray) : ByteArray :=
  b.extract 0 n ++ ByteArray.mk (Array.replicate (n - min n b.size) 0)

/-- A row's `hm96-sha512/v1` leaf under its salt, fitted to the scheme's 192 bytes. -/
def leaf (r : Array F128) (salt : ByteArray) : ByteArray := MerkleScheme.hm96Sha512.leaf r (fitBytes 192 salt)

/-- `f 0, …, f (a·b − 1)`, in `a` tasks of `b` each. -/
def parRange {α : Type} (f : Nat → α) (a b : Nat) : Array α :=
  ((Array.range a).map fun i => Task.spawn fun _ => (Array.range b).map fun j => f (i * b + j)).foldl
    (fun acc t => acc ++ t.get) #[]

/-- **`a` tasks of `b` each are one map** over `a·b`. -/
theorem parRange_eq {α : Type} (f : Nat → α) (a b : Nat) : parRange f a b = (Array.range (a * b)).map f := by
  unfold parRange
  induction a with
  | zero => simp only [Nat.zero_mul, Array.range_eq_empty_iff.mpr rfl, Array.map_empty, Array.foldl_empty]
  | succ a ih =>
    rw [Array.range_succ, Array.map_append, Array.foldl_append, ih, Nat.succ_mul, Array.range_add]
    simp only [Task.spawn, List.map_toArray, List.map_cons, List.map_nil, List.size_toArray, List.length_cons,
      List.length_nil, Nat.zero_add, List.foldl_toArray', List.foldl_cons, List.foldl_nil, Array.map_append,
      Array.map_map, Function.comp_def]

/-- **Layer `h` of a Merkle tree** over the leaves `lf q` (`q < 2^d`): layer `h + 1` holds the nodes `SHA-512(l ‖ r)`
of layer `h`'s pairs. -/
def layer (d : Nat) (lf : Nat → ByteArray) : Nat → Array ByteArray
  | 0 => (Array.range (2 ^ d)).map lf
  | h + 1 =>
    let below := layer d lf h
    (Array.range (below.size / 2)).map fun i => Sha512.hash (below[2 * i]! ++ below[2 * i + 1]!)

/-- A cap's entries, concatenated. -/
def flat (cap : Array ByteArray) : ByteArray := cap.foldl (· ++ ·) ByteArray.empty

/-- **What the recheck reads of one table**: its `m`, its packed table `g` (`2^(lanes + cols)` words), its four extra
lanes `extra e c`, every lane's padding coefficients `lanePad l i` and level 0's salt `salt q` at each tree position. -/
structure In where
  m : Nat
  g : Array F128
  extra : Nat → Nat → F128
  lanePad : Nat → Nat → F128
  salt : Nat → ByteArray

/-- Rows per task. -/
def CHUNK : Nat := 5

/-- **The leaves** of level `lv`'s padded codeword under the table's salts, `2^CHUNK` rows to a task. -/
def leaves (lv : Level) (tPad : Nat) (i : In) : Array ByteArray :=
  let rs := rows lv tPad i.g i.extra i.lanePad
  let k := min lv.d CHUNK
  parRange (fun q => leaf rs[q]! (i.salt q)) (2 ^ (lv.d - k)) (2 ^ k)

/-- **The cap** at level `lv`: the nodes `d − cap` levels up from the leaves. -/
def cap (lv : Level) (tPad : Nat) (i : In) : Array ByteArray :=
  let lvs := leaves lv tPad i
  layer lv.d (fun q => lvs[q]!) (lv.d - lv.cap)

/-- **A table's level-0 root**: its cap, concatenated, at level 0 of its `--zk` schedule; empty where `m` has none. -/
def root (i : In) : ByteArray :=
  match Zk.schedule i.m with
  | .ok (s, tPad) => flat (cap (s.levels.getD 0 default) tPad i)
  | .error _ => ByteArray.empty

/-- A request's sizes at `m`: words of the packed table, columns, lanes with the extra ones, `t_pad`, and tree
positions. -/
def sizes (m : Nat) : V (Nat × Nat × Nat × Nat × Nat) := do
  let (s, tPad) ← Zk.schedule m
  let lv := s.levels.getD 0 default
  return (2 ^ (lv.lanes + lv.cols), 2 ^ lv.cols, 2 ^ lv.lanes + Zk.EXTRA_LANES, tPad, 2 ^ lv.d)

/-- A request's bytes at `m`. -/
def wantBytes (m : Nat) : V Nat := do
  let (nG, nC, nL, tPad, nQ) ← sizes m
  return 16 * nG + 16 * Zk.EXTRA_LANES * nC + 16 * nL * tPad + 192 * nQ

/-- The word at byte `o` of `b`: `lo ‖ hi`, little-endian. -/
def wordAt (b : ByteArray) (o : Nat) : F128 := ⟨u64OfLE b o, u64OfLE b (o + 8)⟩

/-- `n` words of `b` from byte `o`. -/
def wordsAt (b : ByteArray) (o n : Nat) : Array F128 := (Array.range n).map fun i => wordAt b (o + 16 * i)

/-- **A request's table** at `m`, from its bytes: the packed table's words, the extra lanes (each `2^cols` words), the
lanes' paddings (each `t_pad` words), then the salts, `192` bytes per tree position; every word `lo ‖ hi`, little-endian. -/
def parse (m : Nat) (b : ByteArray) : V In := do
  let (nG, nC, nL, tPad, nQ) ← sizes m
  let want ← wantBytes m
  if b.size != want then throw s!"link: a {b.size}-byte table where m = {m} takes {want}"
  let g := wordsAt b 0 nG
  let oE := 16 * nG
  let ex := (Array.range Zk.EXTRA_LANES).map fun e => wordsAt b (oE + 16 * nC * e) nC
  let oP := oE + 16 * Zk.EXTRA_LANES * nC
  let pads := (Array.range nL).map fun l => wordsAt b (oP + 16 * tPad * l) tPad
  let oS := oP + 16 * nL * tPad
  return { m, g, extra := fun e c => (ex[e]!)[c]!, lanePad := fun l k => (pads[l]!)[k]!
           salt := fun q => if q < nQ then b.extract (oS + 192 * q) (oS + 192 * (q + 1)) else ByteArray.empty }

end Flock.Firewall.Link
