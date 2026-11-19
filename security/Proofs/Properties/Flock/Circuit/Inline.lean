module

public import Properties.Flock.Circuit.Inline
public import Proofs.Properties.Flock.EvalBool
public import Proofs.Properties.Flock.DecodeBoolean
import Batteries.Tactic.OpenPrivate
meta import Properties.Flock.Circuit.Inline
meta import Proofs.Properties.Flock.EvalBool
meta import Flock.Test.Lower

/-!
# `inline` against the evaluator

On `Flock.Test.Lower`'s Boolean program and on the format's descriptor vector (`WORD_DESC`), on four input seeds each,
every bit of every gate of C, read from the inlined gates, is the evaluator's (`Prog.evalBool`), and the inlining's AND
gates are at most the bodies' ANDs (`bodyAnds`). On `WORD_DESC` the block of its `TWiden16_v1` gate and that gate's 16-bit
operand gives the same bits gate by gate, each gate's emitted span within its `gateAnds`; and a Definition handed fewer
wires than its parameters have reads past them, where the inlining's gate 0 gives the evaluator's `false`.

`decode_gates`: in a decoded Program every primitive is spelt as its family's id and every gate has §9's signature.
-/

@[expose] public section

namespace FlockSoundness.Inline

open Flock.Program Flock.OfProgram

/-- The raw gates' values on inputs `a`, by index. -/
def rawEval (ops : Array RawOp) (a : ℕ → Bool) : Array Bool :=
  (List.range ops.size).foldl (fun v g => v.push (match ops[g]! with
    | .input => a g | .and x y => v[x]! && v[y]! | .xor x y => xor v[x]! v[y]! | .not x => !v[x]! | .zero => false)) #[]

/-- C's gates' bits by the evaluator, input gate `g`'s bit `i` given by `inp g i`. -/
def cEval (P : Prog) (inp : ℕ → ℕ → Bool) : Array (List Bool) :=
  (List.range (size P)).foldl (fun vals g =>
    let w := widthAt P g
    if inputAt P g then vals.push ((List.range w).map (inp g)) else
    match gateAt P g with
    | none => vals.push []
    | some (p, _) =>
      let xs := (opsBelowN P g).flatMap fun a => vals[a]!
      vals.push ((List.range w).map fun i => (P.evalBool p xs).getD i false)) #[]

def seedBit (s x : ℕ) : Bool := ((x * 2654435761 + s * 40503 + 12345) / 65536) % 3 == 0

def andsIn (ops : Array RawOp) (lo hi : ℕ) : ℕ :=
  ((List.range (hi - lo)).filter fun k => match ops[lo + k]! with | .and _ _ => true | _ => false).length

/-- On seed `s`, each gate `g` of `gs`: its bits, inlined and evaluated, are the evaluator's, and its emitted span has at
most its `gateAnds` AND gates. The walk is run once. -/
def agrees (P : Prog) (gs : List ℕ) (s : ℕ) : Bool :=
  let (done, ops) := walk P
  let bit (g i : ℕ) : ℕ := (done.getD g default).bits.getD i (ops.size - 1)
  let v := rawEval ops (seedBit s)
  let c := cEval P fun g i => seedBit s (bit g i)
  gs.all fun g =>
    let x := done.getD g default
    ((List.range (widthAt P g)).all fun i => v.getD (bit g i) false == (c.getD g []).getD i false) &&
      andsIn ops x.lo x.hi ≤ (if inputAt P g then 0 else match gateAt P g with | some (p, _) => bodyAnds P p | none => 0)

def wholeOk (P : Prog) : Bool := (List.range 4).all (agrees P (List.range (size P)))

def wordP : Prog := match decode FlockSoundness.EvalBool.WORD_DESC.toUTF8 with | .ok P => P | .error _ => default
def toyP : Prog := match Flock.Test.Lower.prog with | .ok P => P | .error _ => default

#guard size toyP > 0 && wholeOk toyP
#guard size wordP > 0 && wholeOk wordP
#guard let ops := (walk toyP).2; andsIn ops 0 ops.size > 0

/-- The block of `TWiden16_v1`'s gate and its operands. -/
def widenBlock (P : Prog) : List ℕ :=
  ((List.range (size P)).filter fun g => keyAt P g == "TWiden16_v1").flatMap fun g => g :: opsBelowN P g

#guard (widenBlock wordP).any (fun g => widthAt wordP g == 16) &&
  (widenBlock wordP).any (fun g => keyAt wordP g == "TWiden16_v1") &&
  (List.range 4).all (agrees wordP (widenBlock wordP))

/-- Definition `d` on `k` input wires, inlined and evaluated against `evalDef?` on the same bits, on seed `s`. -/
def shortAgrees (P : Prog) (d k s : ℕ) : Bool :=
  let fuel := 2 * P.defs.size + 2
  let ins := (List.range k).map (· + 1)
  match (inlDef? P fuel d ins).run (#[RawOp.zero] ++ Array.replicate k RawOp.input) with
  | (some ws, ops) =>
    let v := rawEval ops (seedBit s)
    match evalDef? P fuel d (ins.map (seedBit s)) with
    | some bs => ws.length == bs.length && (ws.zip bs).all fun (w, b) => v.getD w false == b
    | none => false
  | (none, _) => false

#guard match wordP.bodyOf? "TAdd32_v1" with
  | some b => (wordP.defs[b]!).params.foldl (fun n p => n + p.2.leaves) 0 > 10 &&
      (List.range 4).all fun s => shortAgrees wordP b 10 s
  | none => false

/-! ## The gates' signatures in a decoded Program -/

section
open InferenceOnlyVerifier.Properties.SecurityProofs
open BodiesOk (DM Inv Pre go_inv decode_steps)
open DecodeKeys (Kept Keys of_triple go_kept)
open FlockSoundness.DecodeBoolean
open private getDef composite staticFns decodeAll decodeBodies extendReach primEntries DecState DecState.defs
  DecState.index DecState.busy Desc Desc.ddefs liftV from Circuits.Program

theorem params_beq {a b : Array (String × Ty)} (h : (a == b) = true) : a = b := by
  obtain ⟨hs, hr⟩ := Array.isEqv_iff_rel.1 h
  refine Array.ext hs (fun i h1 h2 => ?_)
  have hi := hr i h1
  generalize a[i] = p at hi ⊢
  generalize b[i] = q at hi ⊢
  obtain ⟨p1, p2⟩ := p
  obtain ⟨q1, q2⟩ := q
  change (p1 == q1 && p2 == q2) = true at hi
  rw [Bool.and_eq_true, beq_iff_eq] at hi
  rw [hi.1, BodiesOk.ty_beq hi.2]

theorem forIn_ok_all {α ε : Type} {f : α → PUnit → Except ε (ForInStep PUnit)} {Q : α → Prop}
    (hf : ∀ x r, f x PUnit.unit = .ok r → Q x ∧ r = .yield PUnit.unit) :
    ∀ (l : List α) {u : PUnit}, forIn l PUnit.unit f = .ok u → ∀ x ∈ l, Q x
  | [], _, _, x, hx => absurd hx List.not_mem_nil
  | y :: l, u, h, x, hx => by
    rw [List.forIn_cons] at h
    cases hy : f y PUnit.unit with
    | error e => rw [hy] at h; cases h
    | ok r =>
      obtain ⟨hq, rfl⟩ := hf y r hy
      rw [hy] at h
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hq
      · exact forIn_ok_all hf l h x hx

/-- `booleanKeys` accepts only Definitions whose primitives are spelt as their family's ids and whose gates have
§9's signature. -/
theorem booleanKeys_sig {defs : Array Defn} {index : Std.HashMap String Nat} {prims : Array String} {bm : Lean.Json}
    {pairs : Array (String × String)} (h : booleanKeys defs index prims bm = .ok pairs) :
    ∀ d ∈ defs, d.prim = true → misspelt d.key = false ∧
      ∀ g, gateKind? d.key = some g → d.params = g.signature.1 ∧ d.ret = g.signature.2 := by
  unfold booleanKeys at h
  obtain ⟨u, h1, -⟩ := Walk.except_bind_ok h
  rw [← Array.forIn_toList] at h1
  intro d hd
  refine forIn_ok_all (Q := fun d => d.prim = true → misspelt d.key = false ∧
      ∀ g, gateKind? d.key = some g → d.params = g.signature.1 ∧ d.ret = g.signature.2) ?_ _ h1 d
    (Array.mem_toList_iff.2 hd)
  intro x r hx
  dsimp only at hx
  split at hx
  · cases hx
  rename_i hms
  split at hx
  · rename_i g hg
    generalize hsig : g.signature = sg at hx
    obtain ⟨ps, rt⟩ := sg
    dsimp only at hx
    split at hx
    · cases hx
    rename_i hbeq
    cases hx
    refine ⟨fun hp => ⟨by simpa [hp] using hms, fun g' hg' => ?_⟩, rfl⟩
    simp only [hp, ↓reduceIte, hg'] at hg
    cases hg
    simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true] at hbeq
    rw [hsig]
    exact ⟨params_beq hbeq.1, BodiesOk.ty_beq hbeq.2⟩
  · rename_i hg
    cases hx
    refine ⟨fun hp => ⟨by simpa [hp] using hms, fun g' hg' => ?_⟩, rfl⟩
    simp [hp, hg'] at hg

/-- **Every primitive of a decoded Program is spelt as its family's id, and each gate has §9's signature**:
`booleanKeys` checks every Definition `decodeAll` leaves, and every primitive is keyed by a primitive entry, which
`decodeAll` decodes before `booleanKeys`. -/
theorem decode_gates (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ w < P.defs.size, (P.defs[w]!).prim = true → misspelt (P.defs[w]!).key = false ∧
      ∀ g, gateKind? (P.defs[w]!).key = some g →
        (P.defs[w]!).params = g.signature.1 ∧ (P.defs[w]!).ret = g.signature.2 := by
  intro w hw' hprim
  obtain ⟨kv, desc, rootId, programFn, bm, root, s0, st, st1, st2, pairs, hdd, hgo, hd0, hi0', -, -, -, hall, hbk, hbod,
    hd, -, -, -⟩ := decode_steps bytes P h
  have hk0 := of_triple (go_kept _ _ _) hgo (DecodeKeys.kept_empty desc hd0 hi0')
  obtain ⟨hi0, -⟩ := go_inv _ _ _ _ _ _ hgo
    ⟨fun i hi => absurd hi (by rw [hd0]; exact Nat.not_lt_zero i), fun k v hk => absurd hk (by rw [hi0']; simp)⟩
  obtain ⟨hb1, -, hall'⟩ := decodeAll_run desc _ hall ⟨hk0, hi0⟩
  obtain ⟨-, hextra⟩ := booleanKeys_ok hbk
  have hw : ∀ p ∈ pairs, ∃ wi, (DecState.index st1)[p.1]? = some wi := fun p hp => by
    have hg := (hextra p hp).2
    unfold IsWord at hg
    cases hix : (DecState.index st1)[p.1]? with
    | none => rw [hix] at hg; simp at hg
    | some wi => exact ⟨wi, rfl⟩
  obtain ⟨hb2, hp2, -⟩ := decodeBodies_run desc _ _ hbod hb1 hw
  have hK1 := hb1.1.1
  have hK2 := hb2.1.1
  rw [hd] at hw' hprim ⊢
  obtain ⟨d, hdd', hform⟩ := hb2.1.2.2.2 w hw' hprim
  rw [hdd, ← Array.foldl_toList] at hdd'
  have hmem : ((DecState.defs st2)[w]!.key, d) ∈ kv := by
    rcases hm_foldl_some (fun m x => m.insert x.1 x.2) id (fun _ _ => rfl) _ _ kv.toList ∅ hdd' with h0 | ⟨x, hx, he⟩
    · simp at h0
    · simp only [id] at he
      subst he
      simpa using hx
  have hkp := mem_primEntries hmem hform
  have hcont : (Desc.ddefs desc).contains ((DecState.defs st2)[w]!.key) = true := by
    rw [hdd, ← Array.foldl_toList, Std.HashMap.contains_eq_isSome_getElem?, hdd']
    rfl
  obtain ⟨j, hj⟩ := hall' _ hkp hcont
  have hj2 := idx_keep hK1 hK2 hp2 hj
  have hw2 := hK2.2.1 w hw'
  have hjw : j = w := by rw [hj2] at hw2; exact Option.some.inj hw2
  subst hjw
  have hjlt := (hK1.1 _ _ hj).1
  rw [hp2.2 _ hjlt] at hprim ⊢
  have hm : (DecState.defs st1)[j]! ∈ DecState.defs st1 := by
    rw [getElem!_pos _ j hjlt]
    exact Array.getElem_mem hjlt
  exact booleanKeys_sig hbk _ hm hprim

end

end FlockSoundness.Inline
