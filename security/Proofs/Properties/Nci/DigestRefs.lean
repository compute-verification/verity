module

public import Circuits.Program
public import NciCheck.DefinitionDigest
import Batteries.Tactic.OpenPrivate
import all Circuits.Program
import all Lean.Data.Json.Basic

/-!
# The decoder's references and the canonical entry's flat parts

The decoder parses each reference of a body (`parsePartsIn`): an alias stands for an earlier argument's parts, a
concatenation splices its members' parts, and any other reference is one part. The canonical entry writes each reference
flat (`flatParts`): an alias as the parts the earlier argument's flat reference holds, a concatenation spliced. Whatever
the decoder parses for a reference, `flatParts` writes flat parts that, one by one and in order, parse to the same parts
(`flat_of_parse`), given that every earlier argument an alias may name is a row's flat parts that parse to its parts
(`RowsArgs`). A part with neither an alias nor a concatenation parses alike at any fuel and with any earlier arguments
(`parse_single`); `parse_alias` and `parse_concat` take one step of the decoder, and `flat_single`, `flat_alias` and
`flat_concat` one step of `flatParts`.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json JsonNumber)
open Flock Flock.Program
open NciCheck.DefinitionDigest (flatParts written)
open private natJ? from Circuits.Program

/-- One part, parsed: what `parsePartsIn` makes of a reference with no alias and no concatenation, at any fuel and
with any earlier arguments. -/
def parsePart1 (j : Json) (leaves : Bool → Nat → Option Nat) : V (Array Part) :=
  parsePartsIn 1 j leaves fun _ _ => none

theorem parse_single {n : Nat} {j : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    (ha : (j.getObjVal? "alias").toOption = none) (hc : (j.getObjVal? "concat").toOption = none) :
    parsePartsIn (n + 1) j leaves args = parsePart1 j leaves := by
  cases h1 : j.getObjVal? "alias" with
  | ok a => rw [h1] at ha; cases ha
  | error e =>
    cases h2 : j.getObjVal? "concat" with
    | ok c => rw [h2] at hc; cases hc
    | error e' =>
      have key : ∀ (m : Nat) (args' : Nat → Nat → Option Seq) b b',
          parsePartsIn._f leaves args (n + 1) b j = parsePartsIn._f leaves args' (m + 1) b' j := by
        intro m args' b b'
        delta parsePartsIn._f
        cases j with
        | obj kvs =>
          dsimp only
          rw [h1, h2]
        | _ => rfl
      unfold parsePart1
      delta parsePartsIn
      exact key 0 _ _ _

theorem parse_alias {n : Nat} {j a : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} (h : parsePartsIn (n + 1) j leaves args = .ok qs) (h1 : j.getObjVal? "alias" = .ok a) :
    ∃ k x k' x' s, a = .arr #[k, x] ∧ natJ? k = some k' ∧ natJ? x = some x' ∧ args k' x' = some s ∧
      qs = s.parts := by
  delta parsePartsIn at h
  delta parsePartsIn._f at h
  cases j with
  | obj kvs =>
    dsimp only at h
    rw [h1] at h
    dsimp only at h
    rcases a with _ | _ | _ | _ | ⟨⟨l⟩⟩ | _
    all_goals try (injection h; done)
    rcases l with _ | ⟨k, _ | ⟨x, _ | ⟨y, l⟩⟩⟩
    all_goals try (injection h; done)
    conv at h => lhs; whnf
    split at h
    all_goals try (injection h; done)
    rename_i k' x' hk hx
    split at h
    all_goals try (injection h; done)
    rename_i s hs
    injection h with h
    exact ⟨k, x, k', x', s, rfl, hk, hx, hs, h.symm⟩
  | _ => cases h

theorem parse_concat {n : Nat} {j c : Json} {leaves : Bool → Nat → Option Nat} {args : Nat → Nat → Option Seq}
    {qs : Array Part} (h : parsePartsIn (n + 1) j leaves args = .ok qs) (h1 : (j.getObjVal? "alias").toOption = none)
    (h2 : j.getObjVal? "concat" = .ok c) :
    ∃ cs r, c = .arr cs ∧ cs.mapM (fun x => parsePartsIn n x leaves args) = .ok r ∧ qs = r.foldl (· ++ ·) #[] := by
  delta parsePartsIn at h
  dsimp only at h
  generalize hb : (Nat.rec ⟨parsePartsIn._f leaves args Nat.zero PUnit.unit, PUnit.unit⟩
    (fun n n_ih => ⟨parsePartsIn._f leaves args n.succ n_ih, n_ih⟩) n :
      Nat.below (motive := fun _ => Json → V (Array Part)) (n + 1)) = b at h
  have hb1 : ∀ x, b.1 x = parsePartsIn n x leaves args := by
    subst hb
    intro x
    cases n with
    | zero => rfl
    | succ m => delta parsePartsIn; rfl
  delta parsePartsIn._f at h
  cases h1' : j.getObjVal? "alias" with
  | ok a => rw [h1'] at h1; cases h1
  | error e =>
  cases j with
  | obj kvs =>
    dsimp only at h
    rw [h1', h2] at h
    dsimp only at h
    rcases c with _ | _ | _ | _ | cs | _
    all_goals try (injection h; done)
    dsimp only at h
    simp only [hb1] at h
    rcases hm : cs.mapM (fun x => parsePartsIn n x leaves args) with e | r <;> simp only [hm, bind, Except.bind] at h
    · exact absurd h (by intro h; injection h)
    injection h with h
    exact ⟨cs, r, rfl, hm, h.symm⟩
  | _ => cases h


theorem natJ_getNat {x : Json} {w : Nat} (h : natJ? x = some w) : x.getNat? = .ok w := by
  unfold natJ? at h
  split at h
  · rename_i n
    split at h
    · rename_i hc
      cases h
      simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
      obtain ⟨m, e⟩ := n
      simp only at hc
      obtain ⟨rfl, hm⟩ := hc
      obtain ⟨m', rfl⟩ := Int.eq_ofNat_of_zero_le hm
      rfl
    · cases h
  · cases h

theorem flat_single {rows : Array (Array (Array Json))} {n : Nat} {j : Json}
    (h1 : (j.getObjVal? "alias").toOption = none) (h2 : (j.getObjVal? "concat").toOption = none) :
    flatParts rows (n + 1) j = .ok #[j] := by
  cases h1' : j.getObjVal? "alias" with
  | ok a => rw [h1'] at h1; cases h1
  | error e =>
  cases h2' : j.getObjVal? "concat" with
  | ok c => rw [h2'] at h2; cases h2
  | error e' =>
  delta flatParts flatParts._f
  dsimp only
  rw [h1', h2']
  rfl

theorem flat_alias {rows : Array (Array (Array Json))} {n : Nat} {j k x : Json} {k' x' : Nat} {ps : Array Json}
    (h1 : j.getObjVal? "alias" = .ok (.arr #[k, x])) (hk : natJ? k = some k') (hx : natJ? x = some x')
    (hr : rows[k']?.bind (·[x']?) = some ps) : flatParts rows (n + 1) j = .ok ps := by
  delta flatParts flatParts._f
  dsimp only
  rw [h1]
  dsimp only
  split
  · rename_i ps' heq
    conv at heq => lhs; whnf
    dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ] at heq
    rw [natJ_getNat hk, natJ_getNat hx] at heq
    dsimp only at heq
    rw [hr] at heq
    cases heq
    rfl
  · rename_i hne
    exfalso
    apply hne ps
    conv => lhs; whnf
    dsimp only [Array.getLit, List.getElem_toArray, List.getElem_cons_zero, List.getElem_cons_succ]
    rw [natJ_getNat hk, natJ_getNat hx]
    exact hr

theorem flat_concat {rows : Array (Array (Array Json))} {n : Nat} {j : Json} {cs : Array Json}
    {r : Array (Array Json)} (h1 : (j.getObjVal? "alias").toOption = none) (h2 : j.getObjVal? "concat" = .ok (.arr cs))
    (hm : cs.mapM (flatParts rows n) = .ok r) : flatParts rows (n + 1) j = .ok (r.foldl (· ++ ·) #[]) := by
  cases h1' : j.getObjVal? "alias" with
  | ok a => rw [h1'] at h1; cases h1
  | error e =>
  delta flatParts
  dsimp only
  generalize hb : (Nat.rec ⟨flatParts._f rows Nat.zero PUnit.unit, PUnit.unit⟩
    (fun n n_ih => ⟨flatParts._f rows n.succ n_ih, n_ih⟩) n :
      Nat.below (motive := fun _ => Json → V (Array Json)) (n + 1)) = b
  have hb1 : ∀ x, b.1 x = flatParts rows n x := by
    subst hb
    intro x
    cases n with
    | zero => rfl
    | succ m => delta flatParts; rfl
  delta flatParts._f
  dsimp only
  rw [h1', h2]
  dsimp only
  have hm' : ∀ f : Json → V (Array Json), (∀ x, f x = flatParts rows n x) → Array.mapM f cs = .ok r := by
    intro f hf
    rw [show f = flatParts rows n from funext hf]
    exact hm
  rw [hm' _ hb1]
  rfl

/-- Flat parts and what the decoder parses from them, in order: each flat part has no alias and no concatenation and
parses alone to its parts. -/
inductive PartsRel (leaves : Bool → Nat → Option Nat) : List Json → List Part → Prop
  | nil : PartsRel leaves [] []
  | cons {p : Json} {ps : List Json} {q : Array Part} {qs : List Part} :
      (p.getObjVal? "alias").toOption = none → (p.getObjVal? "concat").toOption = none →
      parsePart1 p leaves = .ok q → PartsRel leaves ps qs → PartsRel leaves (p :: ps) (q.toList ++ qs)

theorem PartsRel.append {leaves : Bool → Nat → Option Nat} : ∀ {a c : List Json} {b d : List Part},
    PartsRel leaves a b → PartsRel leaves c d → PartsRel leaves (a ++ c) (b ++ d)
  | _, _, _, _, .nil, h => h
  | _, _, _, _, .cons h1 h2 h3 h, h' => by
    rw [List.cons_append, List.append_assoc]
    exact .cons h1 h2 h3 (h.append h')

/-- The rows the canonical entry has written so far stand for the decoder's earlier arguments: every argument an
alias may name is a row's flat parts, which parse to its parts. -/
def RowsArgs (leaves : Bool → Nat → Option Nat) (rows : Array (Array (Array Json))) (args : Nat → Nat → Option Seq) :
    Prop :=
  ∀ k x s, args k x = some s → ∃ ps, rows[k]?.bind (·[x]?) = some ps ∧ PartsRel leaves ps.toList s.parts.toList

theorem toList_foldl_append {α : Type} : ∀ (l : List (Array α)) (acc : Array α),
    (l.foldl (· ++ ·) acc).toList = acc.toList ++ (l.map Array.toList).flatten
  | [], acc => by simp
  | y :: l, acc => by
    rw [List.foldl_cons, toList_foldl_append l, Array.toList_append, List.map_cons, List.flatten_cons,
      List.append_assoc]

theorem flat_of_parse {leaves : Bool → Nat → Option Nat} {rows : Array (Array (Array Json))}
    {args : Nat → Nat → Option Seq} (H : RowsArgs leaves rows args) :
    ∀ (fuel : Nat) (j : Json) (qs : Array Part), parsePartsIn fuel j leaves args = .ok qs →
      ∃ fps, flatParts rows fuel j = .ok fps ∧ PartsRel leaves fps.toList qs.toList := by
  intro fuel
  induction fuel with
  | zero => intro j qs h; delta parsePartsIn at h; exact nomatch h
  | succ n ih =>
    intro j qs h
    cases h1 : j.getObjVal? "alias" with
    | ok a =>
      obtain ⟨k, x, k', x', s, rfl, hk, hx, hs, rfl⟩ := parse_alias h h1
      obtain ⟨ps, hr, hrel⟩ := H k' x' s hs
      exact ⟨ps, flat_alias h1 hk hx hr, hrel⟩
    | error e =>
      have h1o : (j.getObjVal? "alias").toOption = none := by rw [h1]; rfl
      cases h2 : j.getObjVal? "concat" with
      | ok c =>
        obtain ⟨cs, r, rfl, hm, rfl⟩ := parse_concat h h1o h2
        have key : ∀ (l : List Json) (r : List (Array Part)), l.mapM (fun x => parsePartsIn n x leaves args) = .ok r →
            ∃ r' : List (Array Json), l.mapM (flatParts rows n) = .ok r' ∧
              PartsRel leaves (r'.map Array.toList).flatten (r.map Array.toList).flatten := by
          intro l
          induction l with
          | nil =>
            intro r hr
            simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at hr
            subst hr
            exact ⟨[], rfl, .nil⟩
          | cons y l ihl =>
            intro r hr
            rw [List.mapM_cons] at hr
            simp only [bind, Except.bind] at hr
            split at hr
            · cases hr
            rename_i q hq
            split at hr
            · cases hr
            rename_i qs' hqs
            simp only [pure, Except.pure, Except.ok.injEq] at hr
            subst hr
            obtain ⟨fy, hfy, hy⟩ := ih y q hq
            obtain ⟨r', hr', hl⟩ := ihl qs' hqs
            refine ⟨fy :: r', ?_, ?_⟩
            · rw [List.mapM_cons]
              simp only [bind, Except.bind, hfy, hr', pure, Except.pure]
            · simp only [List.map_cons, List.flatten_cons]
              exact hy.append hl
        rw [Array.mapM_eq_mapM_toList] at hm
        simp only [Functor.map, Except.map] at hm
        split at hm
        · cases hm
        rename_i l hl
        simp only [Except.ok.injEq] at hm
        subst hm
        obtain ⟨r', hr', hrel⟩ := key cs.toList l hl
        have hm' : cs.mapM (flatParts rows n) = .ok r'.toArray := by
          rw [Array.mapM_eq_mapM_toList, hr']
          rfl
        refine ⟨_, flat_concat h1o h2 hm', ?_⟩
        rw [← Array.foldl_toList, ← Array.foldl_toList, toList_foldl_append, toList_foldl_append]
        simpa using hrel
      | error e' =>
        have h2o : (j.getObjVal? "concat").toOption = none := by rw [h2]; rfl
        rw [parse_single h1o h2o] at h
        refine ⟨#[j], flat_single h1o h2o, ?_⟩
        simpa using PartsRel.cons h1o h2o h .nil

end NciCheck.Properties.Record.SecurityProofs
