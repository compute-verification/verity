/-
# The firewall's level-0 recheck (live PROTOCOL.md §10): a table's level-0 root from its witness and level 0's hiding

Before anything of a firewalled table leaves, the firewall commits the table's level 0 again from the host's witness, the
table's level-0 hiding and its level-0 salts, and the root that leaves in the session's `Commit` is that recheck's
(`flock-circuit`'s `Firewall::new` and `Firewall::root`). This is the recheck of record: the masked prover's level-0
commitment (`FlockComplete.commitPZk` in the analysis), here as the code the firewall runs.

* `rows`: level 0's padded codeword. Row `q` holds the table's `2^lanes` witness lanes, lane `l` the words
  `g[c·2^lanes + l]`, then the four extra lanes `extra e`; each entry is the lane's encoding at `ω_q`,
  `Σ_c v(c)·X̂_c(ω_q)`, plus its padding `(c + Ŵ_cols(ω_q))·Σ_i μ_l[i]·X̂_i(ω_q)` (`padF`, lane `l`'s coefficients
  `lanePad l`, `t_pad` of them).
* `leaves`: row `q`'s `hm96-sha512/v1` leaf under the salt `salt q` fitted to 192 bytes, computed in tasks of `2^5` rows
  (`parRange`).
* `cap`: the Merkle nodes `SHA-512(l ‖ r)` `d − cap` levels up from the leaves (`layer`), and `root` their concatenation,
  at level 0 of the table's `--zk` schedule (`Zk.schedule m`, which also fixes `t_pad`).

`flock-firewall link` serves it to the firewall (`cmd`). Rust's `commit_zk` (its additive FFT) is the speed reference: the
firewall holds its own root to this one and sends this one.
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

/-- **Level 0's padded codeword** of the packed table `g`, with the extra lanes `extra` and the paddings `lanePad`. -/
def rows (lv : Level) (tPad : Nat) (g : Array F128) (extra lanePad : Nat → Nat → F128) : Array (Array F128) :=
  (Array.range (2 ^ lv.d)).map (row lv g extra (svTable (lv.cols + 1)) (musOf lv tPad lanePad))

/-- The first `n` bytes of `b`, zero-padded to `n`. -/
def fitBytes (n : Nat) (b : ByteArray) : ByteArray :=
  b.extract 0 n ++ ByteArray.mk (Array.replicate (n - min n b.size) 0)

/-- A row's `hm96-sha512/v1` leaf under its salt, fitted to the scheme's 192 bytes. -/
def leaf (r : Array F128) (salt : ByteArray) : ByteArray := MerkleScheme.hm96Sha512.leaf r (fitBytes 192 salt)

/-- `f 0, …, f (a·b − 1)`, in `a` tasks of `b` each. -/
def parRange {α : Type} (f : Nat → α) (a b : Nat) : Array α :=
  ((Array.range a).map fun i => Task.spawn fun _ => (Array.range b).map fun j => f (i * b + j)).foldl
    (fun acc t => acc ++ t.get) #[]

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
  let sv := svTable (lv.cols + 1)
  let mus := musOf lv tPad i.lanePad
  let k := min lv.d CHUNK
  parRange (fun q => leaf (row lv i.g i.extra sv mus q) (i.salt q)) (2 ^ (lv.d - k)) (2 ^ k)

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

/-- `flock-firewall link`: the level-0 recheck, served on stdin and stdout. `table M LEN` followed by the `LEN` bytes of
a table (`parse`) is answered `ok N` followed by the `N` bytes of its level-0 root (`root`). A malformed request is
answered `err WHY` and ends the session. -/
def cmd (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let refuse := fun (why : String) => do
    stdout.putStrLn s!"err {why.replace "\n" " "}"
    stdout.flush
    return (1 : UInt32)
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | ["table", m, len] =>
      let (some m, some len) := (m.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let b ← Coins.readExact "link" stdin len
      match parse m b with
      | .ok i =>
        let r := root i
        stdout.putStrLn s!"ok {r.size}"
        stdout.write r
        stdout.flush
      | .error e => return (← refuse e)
    | _ => return (← refuse s!"a request `{line}`")
  return 0

end Flock.Firewall.Link
