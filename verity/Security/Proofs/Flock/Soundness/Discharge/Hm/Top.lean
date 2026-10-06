/-
The two slot circuits as the verifier builds them (`Flock.HmNets.hmLaid`, `shaLaid`): run from the empty state, so their
inputs are the committed bits `0, 1, …` in port order, and what their rows force on those bits.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Hm96

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- The `n` committed input bits from `k`, as `input` allocates them. -/
def vars (k n : ℕ) : List Lin := (List.range n).map fun i => Lin.var (k + i)

theorem length_vars (k n : ℕ) : (vars k n).length = n := by simp [vars]

theorem bit_vars (z : Var → Bool) (k n : ℕ) {j : ℕ} (hj : j < n) : bit z (vars k n) j = z (k + j) := by
  simp [bit, vars, hj]

theorem fst_pair {α β : Type} (a : α) (b : β) : (a, b).1 = a := rfl
theorem snd_pair {α β : Type} (a : α) (b : β) : (a, b).2 = b := rfl

theorem run_bind {α β : Type} (m : M α) (f : α → M β) (s : St) :
    (m >>= f).run s = (f (m.run s).1).run (m.run s).2 := rfl

theorem run_bind_pair {α β : Type} (a : α) (m : M β) (s : St) :
    (m >>= fun b => pure (a, b)).run s = ((a, (m.run s).1), (m.run s).2) := rfl

/-! ## The hm96 slot -/

theorem run_inputs3 {β : Type} (a b c : ℕ) (g : List Lin → List Lin → List Lin → M β) (s : St) :
    (input a >>= fun x => input b >>= fun y => input c >>= fun w => g x y w).run s =
      (g (vars s.next a) (vars (s.next + a) b) (vars (s.next + a + b) c)).run ⟨s.next + a + b + c, s.rows⟩ := rfl

theorem hm96Row_run (key : ByteArray) (s : St) : (hm96Row key).run s =
    (hmTail key (vars s.next 512) (vars (s.next + 512) 1024) (vars (s.next + 512 + 1024) 1536)).run
      ⟨s.next + 512 + 1024 + 1536, s.rows⟩ := by
  rw [hm96Row_eq]
  exact run_inputs3 512 1024 1536 (hmTail key) s

/-- **The verifier's hm96 net computes `b ‖ c`**: built from the empty state, its inputs are the committed bits
`0 … 511` (`cv`), `512 … 1535` (`pad`) and `1536 … 3071` (the salt `y`, byte-string order), and whenever its rows hold
under `z`, its 1024 outputs (byte-string order) are: bit `t < 512` is bit `t` of `F(cv, pad)` XOR bit `t` of `M(key)·y`,
bit `512 + t` is bit `t` of `F(F(mid, y₀), y₁ ‖ tail)`. -/
theorem hm96Row_spec (key : ByteArray) (z : Var → Bool) (h : stHolds z ((hm96Row key).run St.empty).2) :
    ((hm96Row key).run St.empty).1.1 = vars 0 512 ∧ ((hm96Row key).run St.empty).1.2.1 = vars 512 1024 ∧
      ((hm96Row key).run St.empty).1.2.2.1 = vars 1536 1536 ∧ ((hm96Row key).run St.empty).1.2.2.2.length = 1024 ∧
      ∀ t < 512,
        bit z ((hm96Row key).run St.empty).1.2.2.2 t =
            (wbit (compressV ((words (vars 0 512) 8).map (bv z)) ((words (vars 512 1024) 16).map (bv z))) (wordBit t) ^^
              maskV key (bit z (vars 1536 1536)) t) ∧
          bit z ((hm96Row key).run St.empty).1.2.2.2 (512 + t) =
            wbit (compressV (compressV saltMid (((saltWords (vars 1536 1536)).map (bv z)).take 16))
              (((saltWords (vars 1536 1536)).map (bv z)).drop 16 ++ saltTail)) (wordBit t) := by
  rw [hm96Row_run] at h ⊢
  obtain ⟨h1, h2, h3, h4⟩ := (sound_hmTail key _ _ _ _).2 z h
  exact ⟨h1, h2, h3, h4 (length_vars _ _) (length_vars _ _)⟩

/-! ## The sha512x3 slot -/

/-- `Flock.HmNets.compression`'s inputs: `cv` (512 bits) and `msg` (1024) for each of `n` compressions. -/
def shaIns (n : ℕ) : M (List (List Lin × List Lin)) :=
  (List.range n).mapM fun _ => do
    let cv ← input 512
    let msg ← input 1024
    pure (cv, msg)

/-- `Flock.HmNets.compression`'s compressions, each output's words concatenated. -/
def shaOuts (ins : List (List Lin × List Lin)) : M (List (List Lin)) :=
  ins.mapM fun x => do
    let out ← compress (words x.1 8) (words x.2 16)
    pure out.flatten

theorem compression_eq (n : ℕ) :
    compression n = (shaIns n >>= fun ins => shaOuts ins >>= fun outs => pure (ins, outs)) := rfl

/-- The sha512x3 slot's inputs from the empty state: `cv_k`, `msg_k` for `k < 3`, in that order. -/
def sha3Ins : List (List Lin × List Lin) :=
  [(vars 0 512, vars 512 1024), (vars 1536 512, vars 2048 1024), (vars 3072 512, vars 3584 1024)]

theorem shaIns_run : (shaIns 3).run St.empty = (sha3Ins, ⟨4608, []⟩) := rfl

theorem sha3Ins_getD {k : ℕ} (hk : k < 3) :
    sha3Ins.getD k ([], []) = (vars (1536 * k) 512, vars (1536 * k + 512) 1024) := by
  obtain _ | _ | _ | k := k
  · rfl
  · rfl
  · rfl
  · omega

/-- A bit of a concatenation of `n`-bit pieces is its piece's bit. -/
theorem bit_flatten_of_len (z : Var → Bool) {n : ℕ} : ∀ {ws : List Word}, (∀ w ∈ ws, w.length = n) →
    ∀ {k q : ℕ}, q < n → bit z ws.flatten (n * k + q) = bit z (ws.getD k []) q
  | [], _, k, q, _ => by simp [bit]
  | w :: ws, h, k, q, hq => by
      have hw : w.length = n := h w List.mem_cons_self
      rw [List.flatten_cons]
      cases k with
      | zero =>
          simp only [Nat.mul_zero, Nat.zero_add, List.getD_cons_zero, bit]
          rw [List.getElem?_append_left (by omega)]
      | succ k =>
          have ih := bit_flatten_of_len z (fun x hx => h x (List.mem_cons_of_mem _ hx)) (k := k) hq
          simp only [List.getD_cons_succ, bit] at ih ⊢
          rw [List.getElem?_append_right (by rw [hw]; nlinarith), hw,
            show n * (k + 1) + q - n = n * k + q by rw [Nat.mul_succ]; omega, ih]

theorem sound_compressFlat (cv msg : List Lin) :
    Sound (compress (words cv 8) (words msg 16) >>= fun out => pure out.flatten) fun z out =>
      cv.length = 512 → msg.length = 1024 → out.length = 512 ∧ ∀ q < 512,
        bit z out q = wbit (compressV ((words cv 8).map (bv z)) ((words msg 16).map (bv z))) q := by
  refine Sound.bind' (sound_compress _ _) fun x => Sound.pure' fun z hx hcv hmsg => ?_
  obtain ⟨x1, x2⟩ := hx (length_words _ _) (words64_words (by omega)) (length_words _ _) (words64_words (by omega))
  have hxl : x.length = 8 := by rw [length_of_map_bv x2, length_compressV (by simp [length_words])]
  refine ⟨?_, fun q _ => ?_⟩
  · rw [List.length_flatten, List.map_congr_left (g := fun _ => 64) x1, List.map_const', List.sum_replicate, hxl]
    rfl
  · rw [bit_flatten z x1, ← x2, wbit, getD_map_bv]

theorem sound_shaOuts : ∀ ins : List (List Lin × List Lin), Sound (shaOuts ins) fun z outs =>
    (∀ p ∈ ins, p.1.length = 512 ∧ p.2.length = 1024) → outs.length = ins.length ∧ (∀ o ∈ outs, o.length = 512) ∧
      ∀ k < ins.length, ∀ q < 512, bit z (outs.getD k []) q =
        wbit (compressV ((words (ins.getD k ([], [])).1 8).map (bv z)) ((words (ins.getD k ([], [])).2 16).map (bv z))) q
  | [] => Sound.pure' fun z _ => ⟨rfl, fun _ h => absurd h (by simp), fun k hk => absurd hk (by simp)⟩
  | p :: l => by
      unfold shaOuts
      rw [List.mapM_cons]
      obtain ⟨cv, msg⟩ := p
      refine Sound.bind' (sound_compressFlat cv msg) fun o => Sound.bind' (sound_shaOuts l) fun os =>
        Sound.pure' fun z hos ho hl => ?_
      obtain ⟨hp1, hp2⟩ := hl (cv, msg) List.mem_cons_self
      obtain ⟨o1, o2⟩ := ho hp1 hp2
      obtain ⟨s1, s2, s3⟩ := hos fun q hq => hl q (List.mem_cons_of_mem _ hq)
      refine ⟨by simp [s1], fun x hx => ?_, fun k hk q hq => ?_⟩
      · rcases List.mem_cons.1 hx with rfl | hx
        · exact o1
        · exact s2 x hx
      · cases k with
        | zero => simpa using o2 q hq
        | succ k => simpa using s3 k (by simpa using hk) q hq

theorem length_sha3Ins : sha3Ins.length = 3 := rfl

theorem sha3Ins_len : ∀ p ∈ sha3Ins, p.1.length = 512 ∧ p.2.length = 1024 := by
  intro p hp
  simp only [sha3Ins, List.mem_cons, List.not_mem_nil, or_false] at hp
  rcases hp with rfl | rfl | rfl <;> simp only [length_vars, and_self]

theorem compression3_run : compression3.run St.empty =
    ((sha3Ins, ((shaOuts sha3Ins).run ⟨4608, []⟩).1), ((shaOuts sha3Ins).run ⟨4608, []⟩).2) := by
  rw [show compression3 = compression 3 from rfl, compression_eq, run_bind, shaIns_run]
  exact run_bind_pair _ _ _

/-- **The verifier's sha512x3 net computes three SHA-512 compressions**: built from the empty state, its inputs are
`cv_k` = bits `1536 k … 1536 k + 511` and `msg_k` = the next 1024, and whenever its rows hold under `z`, bit `q < 512`
of its output `k < 3` (word order) is bit `q` of `F(cv_k, msg_k)`. -/
theorem compression3_spec (z : Var → Bool) (h : stHolds z (compression3.run St.empty).2) :
    (compression3.run St.empty).1.1 = sha3Ins ∧ (compression3.run St.empty).1.2.length = 3 ∧
      ∀ k < 3, ∀ q < 512, bit z ((compression3.run St.empty).1.2.getD k []) q =
        wbit (compressV ((words (vars (1536 * k) 512) 8).map (bv z))
          ((words (vars (1536 * k + 512) 1024) 16).map (bv z))) q := by
  rw [compression3_run, snd_pair] at h
  rw [compression3_run, fst_pair, fst_pair, snd_pair]
  obtain ⟨h1, -, h3⟩ := (sound_shaOuts sha3Ins ⟨4608, []⟩).2 z h sha3Ins_len
  refine ⟨rfl, h1.trans length_sha3Ins, fun k hk q hq => ?_⟩
  rw [h3 k hk q hq, sha3Ins_getD hk, fst_pair, snd_pair]

end FlockSoundness.Discharge.Hm
