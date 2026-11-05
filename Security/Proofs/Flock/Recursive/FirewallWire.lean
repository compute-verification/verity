module

public import Flock.Firewall
import all Init.Data.String.Defs

/-!
# The honest worker's bytes read back as it meant them

`decode_putReq`: `decode` reads a request the worker encodes (`putReq`) back to itself wherever each length fits its
word (`ReqFits`). `frames_putFrames`: `frames` reads the frames the worker writes (`putFrames`) back to themselves
wherever each frame's op, kind, count and payload fit (`FrameFits`). `split_putRound`: `split` reads a round the
worker's challenger absorbs (`putRound`) to its caps, its free elements and its coin count, in absorb order, wherever
each item fits and the statement is bound once in a rep's first round and nowhere else (`RoundFits`).

`At b p x` says `x` sits in `b` at `p`; each reader (`rdU32`, `rdU64`, `rdBytes`, `rdStr`, `rdList`) has a lemma
reading what its writer put there. `split`'s loop is read item by item through `Steps`, its body's action on each frame
an honest item makes.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion.FirewallWire

open Flock Flock.Firewall

/-! ## Bytes and their readers -/

/-- `x` sits in `b` at offset `p`. -/
def At (b : ByteArray) (p : Nat) (x : ByteArray) : Prop :=
  p + x.size ≤ b.size ∧ ∀ i (h : i < x.size), b.get! (p + i) = x[i]

theorem get!_eq {b : ByteArray} {i : Nat} (h : i < b.size) : b.get! i = b[i] := by
  cases b with
  | mk d => exact getElem!_pos d i h

theorem at_append_right (a x : ByteArray) : At (a ++ x) a.size x := by
  refine ⟨by simp [ByteArray.size_append], fun i h => ?_⟩
  rw [get!_eq (by simp [ByteArray.size_append]; omega), ByteArray.getElem_append_right (by omega)]
  simp

theorem At.left {b : ByteArray} {p : Nat} {x y : ByteArray} (h : At b p (x ++ y)) : At b p x := by
  refine ⟨by have := h.1; simp [ByteArray.size_append] at this; omega, fun i hi => ?_⟩
  rw [h.2 i (by simp [ByteArray.size_append]; omega), ByteArray.getElem_append_left hi]

theorem At.right {b : ByteArray} {p : Nat} {x y : ByteArray} (h : At b p (x ++ y)) : At b (p + x.size) y := by
  refine ⟨by have := h.1; simp [ByteArray.size_append] at this; omega, fun i hi => ?_⟩
  rw [Nat.add_assoc, h.2 (x.size + i) (by simp [ByteArray.size_append]; omega),
    ByteArray.getElem_append_right (by omega)]
  simp

theorem size_putLE (k n : Nat) : (putLE k n).size = k := by
  simp [putLE, ByteArray.size]

theorem putLE_get (k n i : Nat) (h : i < (putLE k n).size) : ((putLE k n)[i]).toNat = n / 256 ^ i % 256 := by
  rw [ByteArray.getElem_eq_getElem_data]
  simp [putLE]

theorem fold_le {b : ByteArray} {p n : Nat} : ∀ k, (∀ i < k, (b.get! (p + i)).toNat = n / 256 ^ i % 256) →
    (List.range k).foldl (fun x i => x + (b.get! (p + i)).toNat * 256 ^ i) 0 = n % 256 ^ k
  | 0, _ => by simp [Nat.mod_one]
  | k + 1, h => by
    rw [List.range_succ, List.foldl_append, fold_le k (fun i hi => h i (by omega))]
    simp only [List.foldl_cons, List.foldl_nil, h k (by omega), Nat.mod_pow_succ, Nat.mul_comm]

theorem le_at {b : ByteArray} {p k n : Nat} (h : At b p (putLE k n)) :
    (List.range k).foldl (fun x i => x + (b.get! (p + i)).toNat * 256 ^ i) 0 = n % 256 ^ k :=
  fold_le k fun i hi => by
    have hi' : i < (putLE k n).size := by rw [size_putLE]; exact hi
    rw [h.2 i hi', putLE_get]

theorem rdU32_at {b : ByteArray} {p n : Nat} (h : At b p (putLE 4 n)) (hn : n < 2 ^ 32) :
    (rdU32 b).run p = .ok (n, p + (putLE 4 n).size) := by
  rw [size_putLE]
  have hs := h.1
  rw [size_putLE] at hs
  have hv := le_at h
  rw [Nat.mod_eq_of_lt (by simpa using hn)] at hv
  simp only [rdU32, need, StateT.run]
  simp [bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, set, StateT.set, pure, StateT.pure,
    Except.bind, Except.pure, hv, show ¬ p + 4 > b.size by omega]

theorem rdU64_at {b : ByteArray} {p n : Nat} (h : At b p (putLE 8 n)) (hn : n < 2 ^ 64) :
    (rdU64 b).run p = .ok (n, p + (putLE 8 n).size) := by
  rw [size_putLE]
  have hs := h.1
  rw [size_putLE] at hs
  have hv := le_at h
  rw [Nat.mod_eq_of_lt (by simpa using hn)] at hv
  simp only [rdU64, need, StateT.run]
  simp [bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, set, StateT.set, pure, StateT.pure,
    Except.bind, Except.pure, hv, show ¬ p + 8 > b.size by omega]

theorem At.extract {b : ByteArray} {p : Nat} {x : ByteArray} (h : At b p x) : b.extract p (p + x.size) = x := by
  have hs := h.1
  apply ByteArray.ext
  apply Array.ext
  · simp
    omega
  · intro i h1 h2
    have hi : i < x.size := by simpa using h2
    have e := h.2 i hi
    rw [get!_eq (by omega)] at e
    have := @ByteArray.getElem_extract i b p (p + x.size) (by simpa using h1)
    simp only [ByteArray.getElem_eq_getElem_data] at this e
    rw [this, e]

theorem size_putBytes (x : ByteArray) : (putBytes x).size = 4 + x.size := by
  simp [putBytes, ByteArray.size_append, size_putLE]

theorem rdBytes_at {b : ByteArray} {p : Nat} {x : ByteArray} (h : At b p (putBytes x)) (hx : x.size < 2 ^ 32) :
    (rdBytes b).run p = .ok (x, p + (putBytes x).size) := by
  have h1 := rdU32_at h.left hx
  rw [size_putLE] at h1
  have h2 := h.right
  rw [size_putLE] at h2
  have he : b.extract (p + 4) (p + 4 + x.size) = x := h2.extract
  have hs := h2.1
  rw [size_putBytes, ← Nat.add_assoc]
  simp only [rdBytes, StateT.run_bind, h1]
  simp [bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, set, StateT.set, pure, StateT.pure,
    Except.bind, Except.pure, need, StateT.run, show ¬ b.size < p + 4 + x.size by omega]
  exact he


theorem run_sbind {α β : Type} (x : Rd α) (f : α → Rd β) (p : Nat) : (StateT.bind x f).run p = (x >>= f).run p := rfl

theorem ok_bind {α β : Type} (a : α) (f : α → V β) : (Except.ok a : V α) >>= f = f a := rfl

theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  have h : s.toUTF8.IsValidUTF8 := by simpa using s.isValidUTF8
  simp only [String.fromUTF8?, h, dite_true]
  rfl

theorem rdStr_at {b : ByteArray} {p : Nat} {s : String} (h : At b p (putStr s)) (hs : s.toUTF8.size < 2 ^ 32) :
    (rdStr b).run p = .ok (s, p + (putStr s).size) := by
  have h1 := rdBytes_at h hs
  simp only [rdStr, StateT.run_bind, h1]
  simp only [bind, Except.bind, fromUTF8?_toUTF8]
  rfl

/-- Each pair's string and bytes fit their words. -/
def PairsFit (l : List (String × ByteArray)) : Prop := ∀ e ∈ l, e.1.toUTF8.size < 2 ^ 32 ∧ e.2.size < 2 ^ 32

theorem forIn_pairs {b : ByteArray} {f : Nat → Array (String × ByteArray) → Rd (ForInStep (Array (String × ByteArray)))}
    (hf : ∀ i acc p k v, At b p (putStr k ++ putBytes v) → k.toUTF8.size < 2 ^ 32 → v.size < 2 ^ 32 →
      (f i acc).run p = .ok (.yield (acc.push (k, v)), p + (putStr k ++ putBytes v).size)) :
    ∀ (l : List (String × ByteArray)) (s : Nat) acc p, At b p (putPairs l) → PairsFit l →
      (forIn (List.range' s l.length) acc f).run p = .ok (acc ++ l.toArray, p + (putPairs l).size)
  | [], s, acc, p, _, _ => by
    simp [putPairs, StateT.run, pure, StateT.pure, Except.pure]
  | (k, v) :: l, s, acc, p, h, hl => by
    have hkv := hl (k, v) List.mem_cons_self
    have hl' : PairsFit l := fun e he => hl e (List.mem_cons_of_mem _ he)
    simp only [putPairs] at h
    rw [List.length_cons, List.range'_succ, List.forIn_cons, StateT.run_bind, hf s acc p k v h.left hkv.1 hkv.2]
    simp only [bind, Except.bind]
    rw [forIn_pairs hf l _ _ _ h.right hl']
    simp [putPairs, ByteArray.size_append, Nat.add_assoc]

theorem size_putStr (s : String) : (putStr s).size = 4 + s.toUTF8.size := size_putBytes _

theorem rdList_at {b : ByteArray} {p : Nat} {l : List (String × ByteArray)} (h : At b p (putList l))
    (hl : l.length < 2 ^ 32) (he : PairsFit l) : (rdList b).run p = .ok (l, p + (putList l).size) := by
  have h1 := rdU32_at h.left hl
  rw [size_putLE] at h1
  have h2 := h.right
  rw [size_putLE] at h2
  simp only [rdList, StateT.run_bind, h1]
  simp only [bind, Except.bind]
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [show (Std.Legacy.Range.size [:l.length]) = l.length by simp [Std.Legacy.Range.size]]
  rw [forIn_pairs (fun i acc p k v hkv hk hv => ?_) l _ _ _ h2 he]
  · simp [pure, StateT.pure, Except.pure, StateT.run, putList, ByteArray.size_append, size_putLE, Nat.add_assoc]
  · rw [run_sbind, StateT.run_bind, rdStr_at hkv.left hk, ok_bind, run_sbind, StateT.run_bind, rdBytes_at hkv.right hv,
      ok_bind]
    simp only [ByteArray.size_append, Nat.add_assoc]
    rfl

/-! ## Requests -/

theorem at_tag (t : UInt8) (x : ByteArray) : At (⟨#[t]⟩ ++ x) 1 x := at_append_right ⟨#[t]⟩ x

theorem size_tag (t : UInt8) : (⟨#[t]⟩ : ByteArray).size = 1 := rfl

theorem tag_get (t : UInt8) (x : ByteArray) : (⟨#[t]⟩ ++ x : ByteArray).get! 0 = t := by
  rw [get!_eq (by rw [ByteArray.size_append, size_tag]; omega),
    ByteArray.getElem_append_left (by simp [size_tag])]
  rfl

theorem tag_size (t : UInt8) (x : ByteArray) : ((⟨#[t]⟩ ++ x : ByteArray).size == 0) = false := by
  simp [ByteArray.size_append, size_tag]

/-- Each length of a request fits its word. -/
def ReqFits : Req → Prop
  | .hello t => t.toUTF8.size < 2 ^ 32
  | .link y => y.size < 2 ^ 32
  | .openStream st pa l pos => st.toUTF8.size < 2 ^ 32 ∧ pa.toUTF8.size < 2 ^ 32 ∧ l.size < 2 ^ 32 ∧ pos < 2 ^ 64
  | .round st c n => st.toUTF8.size < 2 ^ 32 ∧ c.size < 2 ^ 32 ∧ n < 2 ^ 32
  | .commit f r p => f.size < 2 ^ 32 ∧ r.length < 2 ^ 32 ∧ p.length < 2 ^ 32 ∧ PairsFit r ∧ PairsFit p
  | _ => True

/-- `decode` reads back each request the worker encodes, wherever its lengths fit their words. -/
theorem decode_putReq {r : Req} (h : ReqFits r) : decode (putReq r) = .ok r := by
  cases r with
  | hello t =>
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdStr_at (at_tag 1 _) h, ok_bind]
    rfl
  | link y =>
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdBytes_at (at_tag 2 _) h, ok_bind]
    rfl
  | ping =>
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdBytes_at (at_tag 9 _) (by decide), ok_bind]
    rfl
  | openStream st pa l pos =>
    obtain ⟨h1, h2, h3, h4⟩ := h
    have a := at_tag 3 (putStr st ++ (putStr pa ++ (putBytes l ++ putLE 8 pos)))
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdStr_at a.left h1, ok_bind, StateT.run_bind, rdStr_at a.right.left h2, ok_bind,
      StateT.run_bind, rdBytes_at a.right.right.left h3, ok_bind, StateT.run_bind, rdU64_at a.right.right.right h4,
      ok_bind]
    rfl
  | round st c n =>
    obtain ⟨h1, h2, h3⟩ := h
    have a := at_tag 4 (putStr st ++ (putBytes c ++ putLE 4 n))
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdStr_at a.left h1, ok_bind, StateT.run_bind, rdBytes_at a.right.left h2, ok_bind,
      StateT.run_bind, rdU32_at a.right.right h3, ok_bind]
    rfl
  | proof =>
    have a := at_tag 5 (putStr "" ++ (putLE 4 0 ++ putBytes .empty))
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdStr_at a.left (by decide), ok_bind, StateT.run_bind, rdU32_at a.right.left (by decide),
      ok_bind, StateT.run_bind, rdBytes_at a.right.right (by decide), ok_bind]
    rfl
  | finish => rfl
  | register => rfl
  | commit f rs ps =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := h
    have a := at_tag 7 (putBytes f ++ (putList rs ++ putList ps))
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdBytes_at a.left h1, ok_bind, StateT.run_bind, rdList_at a.right.left h2 h4, ok_bind,
      StateT.run_bind, rdList_at a.right.right h3 h5, ok_bind]
    rfl
  | prime =>
    have a := at_tag 8 (putBytes .empty ++ putLE 4 0)
    simp only [putReq, decode, tag_size, tag_get, Bool.false_eq_true, ite_false]
    rw [StateT.run_bind, rdBytes_at a.left (by decide), ok_bind, StateT.run_bind, rdU32_at a.right (by decide),
      ok_bind]
    rfl

/-! ## A round's frames -/

/-- A frame's payload size, as `frames` reads it from its op and `n`. -/
def frameSize (op n : Nat) : Nat := if op == OP_OBSERVE then 16 * n else if op == OP_SQUEEZE then 0 else n

/-- A frame `frames` reads back: a known op, a kind byte, `n` in its word, and the payload its op and `n` say. -/
def FrameFits : Nat × Nat × Nat × ByteArray → Prop
  | (op, kind, n, p) => op ∈ [OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES] ∧ kind < 256 ∧ n < 2 ^ 64 ∧
      p.size = frameSize op n

theorem size_mk (a : Array UInt8) : (⟨a⟩ : ByteArray).size = a.size := rfl

theorem at_zero (b : ByteArray) : At b 0 b := ⟨by simp, fun i h => by rw [Nat.zero_add, get!_eq h]⟩

theorem size_putFrame (fr : Nat × Nat × Nat × ByteArray) :
    (putFrame fr).size = 16 + (fr.2.2.2.size + 15) / 16 * 16 := by
  obtain ⟨op, kind, n, p⟩ := fr
  simp only [putFrame, ByteArray.size_append, size_putLE, size_mk, Array.size_replicate]
  simp only [List.size_toArray, List.length_cons, List.length_nil]
  omega

theorem size_putFrames : ∀ fs : List (Nat × Nat × Nat × ByteArray), 16 * fs.length ≤ (putFrames fs).size
  | [] => by simp [putFrames]
  | fr :: fs => by
    have := size_putFrames fs
    simp only [putFrames, ByteArray.size_append, size_putFrame, List.length_cons]
    omega

theorem forIn_stop {c : ByteArray} {α : Type} {f : Nat → α × Nat → V (ForInStep (α × Nat))}
    (hstop : ∀ i out, f i (out, c.size) = pure (.yield (out, c.size))) :
    ∀ (s k : Nat) out, forIn (List.range' s k) (out, c.size) f = .ok (out, c.size)
  | s, 0, out => rfl
  | s, k + 1, out => by
    rw [List.range'_succ, List.forIn_cons, hstop]
    exact forIn_stop hstop _ k out

theorem forIn_frames {c : ByteArray}
    {f : Nat → Array (Nat × Nat × Nat × ByteArray) × Nat → V (ForInStep (Array (Nat × Nat × Nat × ByteArray) × Nat))}
    (hstop : ∀ i out, f i (out, c.size) = pure (.yield (out, c.size)))
    (hstep : ∀ i out p fr, At c p (putFrame fr) → FrameFits fr →
      f i (out, p) = pure (.yield (out.push fr, p + (putFrame fr).size))) :
    ∀ (fs : List (Nat × Nat × Nat × ByteArray)) (s k : Nat) out p, At c p (putFrames fs) →
      p + (putFrames fs).size = c.size → fs.length ≤ k → (∀ fr ∈ fs, FrameFits fr) →
      forIn (List.range' s k) (out, p) f = .ok (out ++ fs.toArray, c.size)
  | [], s, k, out, p, _, he, _, _ => by
    have : p = c.size := by simpa [putFrames] using he
    subst this
    rw [forIn_stop hstop]
    simp
  | fr :: fs, s, k + 1, out, p, h, he, hk, hf => by
    simp only [putFrames] at h he
    rw [List.range'_succ, List.forIn_cons, hstep s out p fr h.left (hf fr List.mem_cons_self), pure_bind]
    simp only []
    rw [forIn_frames hstop hstep fs _ k _ _ h.right (by rw [ByteArray.size_append] at he; omega)
      (by simpa using hk) (fun fr' h' => hf fr' (List.mem_cons_of_mem _ h'))]
    simp

/-- `frames` reads back the frames the worker writes, wherever each fits. -/
theorem frames_putFrames {fs : List (Nat × Nat × Nat × ByteArray)} (hf : ∀ fr ∈ fs, FrameFits fr) :
    frames (putFrames fs) = .ok fs := by
  have hsz := size_putFrames fs
  simp only [frames]
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  rw [forIn_frames (c := putFrames fs) (fun i out => ?_) (fun i out q fr hq hfr => ?_) fs _ _ #[] 0 (at_zero _)
    (by simp) (by simp [Std.Legacy.Range.size]; omega) hf]
  · rw [ok_bind]
    simp
    rfl
  · simp
  · obtain ⟨op, kind, n, pl⟩ := fr
    obtain ⟨hop, hk, hn, hs⟩ := hfr
    have hq' := hq
    simp only [putFrame] at hq'
    have hH := hq'.left.left.left
    have hL : At (putFrames fs) (q + 8) (putLE 8 n) := hq'.left.left.right
    have hP : At (putFrames fs) (q + 16) pl := hq'.left.right
    have hop5 : op < 256 := by
      simp only [OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, List.mem_cons, List.mem_nil_iff,
        or_false] at hop
      omega
    have g0 : ((putFrames fs).get! q).toNat = op := by
      have := hH.2 0 (by simp [size_mk])
      rw [Nat.add_zero] at this
      rw [this]
      show (op.toUInt8).toNat = op
      simp [Nat.mod_eq_of_lt hop5]
    have g1 : ((putFrames fs).get! (q + 1)).toNat = kind := by
      have := hH.2 1 (by simp [size_mk])
      rw [this]
      show (kind.toUInt8).toNat = kind
      simp [Nat.mod_eq_of_lt hk]
    have gn : le (putFrames fs) (q + 8) 8 = n := by
      have := le_at hL
      rw [Nat.mod_eq_of_lt (by simpa using hn)] at this
      exact this
    have ge : (putFrames fs).extract (q + 16) (q + 16 + pl.size) = pl := hP.extract
    have hsize := size_putFrame (op, kind, n, pl)
    have hlt := hP.1
    have hs' : (if (op == OP_OBSERVE) = true then 16 * n else if (op == OP_SQUEEZE) = true then 0 else n) = pl.size :=
      hs.symm
    dsimp only
    simp only [g0, g1, gn, hs', ge, show q < (putFrames fs).size by omega, ite_true,
      show ¬ q + 16 > (putFrames fs).size by omega, ite_false, show ¬ q + 16 + pl.size > (putFrames fs).size by omega,
      hop, decide_true, Bool.not_true, Bool.false_eq_true]
    rw [hsize]
    simp only [Nat.add_assoc]

/-! ## A round's split -/

theorem beq_iff {a b : ByteArray} : (a == b) = true ↔ a = b := by
  show (a.data == b.data) = true ↔ a = b
  rw [beq_iff_eq, ByteArray.ext_iff]

/-- The caps an item puts among `split`'s: the binding's level-0 cap, or a cap. -/
def itemCaps : Absorb → List ByteArray
  | .bind _ c => [c]
  | .cap c => [c]
  | _ => []

/-- The elements an item puts in `split`'s free message. -/
def itemFree : Absorb → List Nat
  | .observe _ es => es
  | _ => []

def isBind : Absorb → Bool
  | .bind _ _ => true
  | _ => false

/-- An item `split` reads as the worker meant it: each length in its word, a cap a whole number of nodes, a nonce of
`NONCE_BYTES`, an observe of a kind `split` takes, and no label that would start a binding. -/
def ItemFits : Absorb → Prop
  | .domain d => d.size < 2 ^ 64
  | .label l => l.size < 2 ^ 64 ∧ l ≠ LABEL_R1CS
  | .bind d c => d.size < 2 ^ 64 ∧ c.size < 2 ^ 64 ∧ isCap c = true
  | .cap c => c.size < 2 ^ 64 ∧ isCap c = true
  | .nonce x => x.size = NONCE_BYTES
  | .observe kind es => (kind = 1 ∧ es.length = 1 ∨ kind = 2) ∧ es.length < 2 ^ 64 ∧ ∀ e ∈ es, e < 2 ^ 128
  | .squeeze kind => kind < 256

/-- `split`'s caps and free message, with an item's added. -/
def addItem (s : Split) (a : Absorb) : Split :=
  { s with caps := s.caps ++ (itemCaps a).toArray, free := s.free ++ (itemFree a).toArray }

def putElems (es : List Nat) : ByteArray := es.foldr (fun e p => putLE 16 e ++ p) .empty

theorem size_putElems : ∀ es : List Nat, (putElems es).size = 16 * es.length
  | [] => rfl
  | e :: es => by
    show (putLE 16 e ++ putElems es).size = _
    rw [ByteArray.size_append, size_putLE, size_putElems es, List.length_cons]
    omega

theorem at_elem {b : ByteArray} : ∀ (es : List Nat) (p : Nat), At b p (putElems es) →
    ∀ i (h : i < es.length), At b (p + 16 * i) (putLE 16 es[i])
  | [], _, _, i, h => absurd h (by simp)
  | e :: es, p, hb, 0, _ => by
    simp only [putElems, List.foldr_cons] at hb
    simpa using hb.left
  | e :: es, p, hb, i + 1, h => by
    simp only [putElems, List.foldr_cons] at hb
    have := at_elem es _ (by have := hb.right; rwa [size_putLE] at this) i (by simpa using h)
    rw [List.getElem_cons_succ]
    rwa [show p + 16 * (i + 1) = p + 16 + 16 * i by omega]

theorem elems_putElems {es : List Nat} (he : ∀ e ∈ es, e < 2 ^ 128) :
    (Array.range es.length).map (fun i => le (putElems es) (16 * i) 16) = es.toArray := by
  apply Array.ext
  · simp
  · intro i h1 h2
    have hi : i < es.length := by simpa using h1
    have a := at_elem es 0 (at_zero _) i hi
    rw [Nat.zero_add] at a
    have := le_at a
    rw [Nat.mod_eq_of_lt (by simpa using he _ (List.getElem_mem hi))] at this
    simp only [Array.getElem_map, Array.getElem_range, List.getElem_toArray]
    exact this

section loop

variable {first : Bool} {f : Nat × Nat × Nat × ByteArray → Split × Option Nat → V (ForInStep (Split × Option Nat))}

/-- `split`'s loop on each frame an honest item makes, at each binding state it meets. -/
structure Steps (first : Bool)
    (f : Nat × Nat × Nat × ByteArray → Split × Option Nat → V (ForInStep (Split × Option Nat))) : Prop where
  domain : ∀ d st, f (OP_DOMAIN, 0, d.size, d) st = pure (.yield st)
  label : ∀ l st, l ≠ LABEL_R1CS → f (OP_LABEL, 0, l.size, l) st = pure (.yield st)
  r1cs : ∀ s, f (OP_LABEL, 0, LABEL_R1CS.size, LABEL_R1CS) (s, none) = pure (.yield (s, if first then some 0 else none))
  digest : ∀ d s, f (OP_BYTES, 0, d.size, d) (s, some 0) = pure (.yield (s, some 1))
  level0 : ∀ c s, isCap c = true →
    f (OP_BYTES, 0, c.size, c) (s, some 1) = pure (.yield ({ s with caps := s.caps.push c }, some 2))
  nonce : ∀ x s b, b ≠ some 0 → b ≠ some 1 → x.size = NONCE_BYTES →
    f (OP_BYTES, 0, x.size, x) (s, b) = pure (.yield (s, b))
  cap : ∀ c s b, b ≠ some 0 → b ≠ some 1 → isCap c = true →
    f (OP_BYTES, 0, c.size, c) (s, b) = pure (.yield ({ s with caps := s.caps.push c }, b))
  observe : ∀ k es s b, (k = 1 ∧ es.length = 1 ∨ k = 2) → (∀ e ∈ es, e < 2 ^ 128) →
    f (OP_OBSERVE, k, es.length, putElems es) (s, b) = pure (.yield ({ s with free := s.free ++ es.toArray }, b))
  squeeze : ∀ k m st, f (OP_SQUEEZE, k, m, .empty) st = pure (.yield st)

theorem forIn_item (H : Steps first f) {a : Absorb} (ha : ItemFits a) (hb : isBind a = false) {s : Split}
    {b : Option Nat} (h0 : b ≠ some 0) (h1 : b ≠ some 1) (rest : List (Nat × Nat × Nat × ByteArray)) :
    forIn (a.frames ++ rest) (s, b) f = forIn rest (addItem s a, b) f := by
  cases a with
  | domain d =>
    rw [Absorb.frames, List.cons_append, List.nil_append, List.forIn_cons, H.domain, pure_bind]
    simp [addItem, itemCaps, itemFree]
  | label l =>
    rw [Absorb.frames, List.cons_append, List.nil_append, List.forIn_cons, H.label _ _ ha.2, pure_bind]
    simp [addItem, itemCaps, itemFree]
  | bind d c => simp [isBind] at hb
  | cap c =>
    rw [Absorb.frames, List.cons_append, List.nil_append, List.forIn_cons, H.cap _ _ _ h0 h1 ha.2, pure_bind]
    simp [addItem, itemCaps, itemFree]
  | nonce x =>
    rw [Absorb.frames, List.cons_append, List.nil_append, List.forIn_cons, H.nonce _ _ _ h0 h1 ha, pure_bind]
    simp [addItem, itemCaps, itemFree]
  | observe k es =>
    have e : es.foldr (fun e p => putLE 16 e ++ p) .empty = putElems es := rfl
    rw [Absorb.frames, e, List.cons_append, List.nil_append, List.forIn_cons, H.observe _ _ _ _ ha.1 ha.2.2,
      pure_bind]
    simp [addItem, itemCaps, itemFree]
  | squeeze k =>
    rw [Absorb.frames, List.cons_append, List.nil_append, List.forIn_cons, H.squeeze, pure_bind]
    simp [addItem, itemCaps, itemFree]

theorem forIn_items (H : Steps first f) {b : Option Nat} (h0 : b ≠ some 0) (h1 : b ≠ some 1)
    (rest : List (Nat × Nat × Nat × ByteArray)) :
    ∀ (items : List Absorb) (s : Split), (∀ a ∈ items, ItemFits a ∧ isBind a = false) →
      forIn (items.flatMap Absorb.frames ++ rest) (s, b) f = forIn rest (items.foldl addItem s, b) f
  | [], s, _ => rfl
  | a :: items, s, h => by
    rw [List.flatMap_cons, List.append_assoc, forIn_item H (h a List.mem_cons_self).1 (h a List.mem_cons_self).2 h0 h1,
      forIn_items H h0 h1 rest items _ (fun a' h' => h a' (List.mem_cons_of_mem _ h'))]
    rfl

theorem forIn_bind (H : Steps true f) {d c : ByteArray} (hc : isCap c = true) (s : Split)
    (rest : List (Nat × Nat × Nat × ByteArray)) :
    forIn ((Absorb.bind d c).frames ++ rest) (s, none) f = forIn rest (addItem s (.bind d c), some 2) f := by
  rw [Absorb.frames]
  simp only [List.cons_append, List.nil_append, List.forIn_cons, H.r1cs, pure_bind, ite_true, H.digest, H.level0 c s hc]
  simp [addItem, itemCaps, itemFree]

theorem foldl_addItem : ∀ (items : List Absorb) (s : Split), items.foldl addItem s =
    { s with caps := s.caps ++ (items.flatMap itemCaps).toArray, free := s.free ++ (items.flatMap itemFree).toArray }
  | [], s => by simp
  | a :: items, s => by
    rw [List.foldl_cons, foldl_addItem items]
    simp [addItem, Array.append_assoc]

theorem frames_fit {a : Absorb} (ha : ItemFits a) : ∀ fr ∈ a.frames, FrameFits fr := by
  cases a with
  | domain d =>
    simp only [ItemFits] at ha
    simp [Absorb.frames, FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, ha]
  | label l =>
    simp only [ItemFits] at ha
    simp [Absorb.frames, FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, ha.1]
  | bind d c =>
    obtain ⟨hd, hc, -⟩ := ha
    simp [Absorb.frames, FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, hd, hc]
    decide
  | cap c =>
    simp only [ItemFits] at ha
    simp [Absorb.frames, FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, ha.1]
  | nonce x =>
    simp only [ItemFits] at ha
    simp [Absorb.frames, FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, ha, NONCE_BYTES]
  | observe k es =>
    obtain ⟨hk, hl, -⟩ := ha
    have e : es.foldr (fun e p => putLE 16 e ++ p) .empty = putElems es := rfl
    simp only [Absorb.frames, e, List.mem_cons, List.mem_nil_iff, or_false, forall_eq, FrameFits]
    refine ⟨by decide, by omega, hl, ?_⟩
    rw [size_putElems]
    rfl
  | squeeze k =>
    simp only [ItemFits] at ha
    simp [Absorb.frames, FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, ha]

theorem forIn_round (H : Steps first f) {items : List Absorb} {kind n : Nat} (hi : ∀ a ∈ items, ItemFits a)
    (hb : if first then ∃ pre d c post, items = pre ++ .bind d c :: post ∧ ∀ a ∈ pre ++ post, isBind a = false
      else ∀ a ∈ items, isBind a = false) :
    forIn (items.flatMap Absorb.frames ++ [(OP_SQUEEZE, kind, n, .empty)]) (({} : Split), (none : Option Nat)) f =
      .ok ({ caps := (items.flatMap itemCaps).toArray, free := (items.flatMap itemFree).toArray },
        if first then some 2 else none) := by
  cases first with
  | true =>
    simp only [ite_true] at hb
    obtain ⟨pre, d, c, post, rfl, hnb⟩ := hb
    have hc : isCap c = true := by
      have := hi (.bind d c) (by simp)
      exact this.2.2
    rw [List.flatMap_append, List.flatMap_cons, List.append_assoc, List.append_assoc,
      forIn_items H (by simp) (by simp) _ pre _ (fun a h => ⟨hi a (by simp [h]), hnb a (by simp [h])⟩),
      forIn_bind H hc, forIn_items H (by simp) (by simp) _ post _
        (fun a h => ⟨hi a (by simp [h]), hnb a (by simp [h])⟩), List.forIn_cons, H.squeeze, pure_bind]
    simp [foldl_addItem, addItem, itemCaps, itemFree]
    rfl
  | false =>
    simp only [Bool.false_eq_true, ite_false] at hb
    rw [forIn_items H (by simp) (by simp) _ items _ (fun a h => ⟨hi a h, hb a h⟩), List.forIn_cons, H.squeeze,
      pure_bind]
    simp [foldl_addItem]
    rfl

/-- A round `split` reads as the worker meant it: each item fits, the squeeze's kind and count fit their words, and one
`bind` item binds the statement where the round is a rep's first, and none elsewhere. -/
def RoundFits (first : Bool) (items : List Absorb) (kind n : Nat) : Prop :=
  (∀ a ∈ items, ItemFits a) ∧ kind < 256 ∧ n < 2 ^ 64 ∧
    if first then ∃ pre d c post, items = pre ++ .bind d c :: post ∧ ∀ a ∈ pre ++ post, isBind a = false
    else ∀ a ∈ items, isBind a = false

/-- `split` reads a round the worker's challenger absorbs to its caps and free elements in absorb order, and its coin
count. -/
theorem split_putRound {first : Bool} {items : List Absorb} {kind n : Nat} (h : RoundFits first items kind n) :
    split (putRound items kind n) first =
      .ok { caps := (items.flatMap itemCaps).toArray, free := (items.flatMap itemFree).toArray, n := n } := by
  obtain ⟨hi, hk, hn, hb⟩ := h
  have hfit : ∀ fr ∈ items.flatMap Absorb.frames ++ [(OP_SQUEEZE, kind, n, ByteArray.empty)], FrameFits fr := by
    intro fr hfr
    rcases List.mem_append.mp hfr with h' | h'
    · obtain ⟨a, ha, hfa⟩ := List.mem_flatMap.mp h'
      exact frames_fit (hi a ha) fr hfa
    · simp only [List.mem_cons, List.mem_nil_iff, or_false] at h'
      subst h'
      simp [FrameFits, frameSize, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, OP_BYTES, hk, hn]
  simp only [split, putRound]
  rw [frames_putFrames hfit, ok_bind, List.getLast?_concat]
  simp only [show (OP_SQUEEZE != OP_SQUEEZE) = false from rfl, Bool.false_eq_true, ite_false]
  rw [forIn_round ?H hi hb]
  · cases first <;> rfl
  · constructor
    · intro d st
      simp [OP_DOMAIN, OP_LABEL, OP_BYTES, OP_OBSERVE]
    · intro l st hl
      have : (l == LABEL_R1CS) = false := Bool.eq_false_iff.mpr fun h => hl (beq_iff.mp h)
      simp [OP_LABEL, this]
    · intro s
      cases first <;> simp [OP_LABEL, beq_iff.mpr rfl]
    · intro d s
      simp [OP_LABEL, OP_BYTES]
    · intro c s hc
      simp [OP_LABEL, OP_BYTES, hc]
    · intro x s b h0 h1 hx
      simp [OP_LABEL, OP_BYTES, h0, h1, hx]
    · intro c s b h0 h1 hc
      have h8 : c.size ≠ NONCE_BYTES := by
        simp only [isCap, NODE_BYTES, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
        simp only [NONCE_BYTES]
        omega
      simp [OP_LABEL, OP_BYTES, h0, h1, hc, h8]
    · intro k es s b hk he
      simp [OP_LABEL, OP_BYTES, OP_OBSERVE, elems_putElems he]
      intro h
      rcases hk with ⟨h1, h2⟩ | h1 <;> omega
    · intro k m st
      simp [OP_LABEL, OP_BYTES, OP_OBSERVE, OP_SQUEEZE]

end loop

end FlockSoundness.Discharge.Recursion.FirewallWire
