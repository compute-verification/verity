import Definitions.Claims.Sampling
import Definitions.Flock.Circuit

/-!
# C-Flock's session as a sampling statement

The C-Flock session's units, read as the sampling claims' statement, partition, commitment and witness
(`Claims.Sampling`), so a game-form bound over the live audit can conclude in the claims' relation `RelS`:

* `progOf C κ`: the circuit's gates lowered to the Boolean profile's (`BGate`), in order; a row
  `(⊕ a) ∧ (⊕ b)` is three gates (the two XORs, then their AND), every other gate one. An input gate the constants
  `κ` fix to `b` is the constant `.const b`, public circuit structure, which a proof unit reads with no opening. Wire
  `g` sits at `posOf C g`, its last gate.
* `partOf P`: one class, one stratum, a replay unit of one proof unit per C-Flock unit, slot `u` unit `u`'s lowered
  gates (`unitGates`). The input gates are in no unit: C-Flock's input unit is not a proof unit, and its values are
  committed values that the units read.
* `designOf kd`: that class drawn `subset kd`, checking every proof unit of a drawn replay unit.
* `stmtOf P κ kd desc q roots`: the statement of the program at constants `κ`, the query `q`, that design and the
  roots.
* `commitOf pos bit val com cs`: an opening at wire `g`'s position passes when some opening of the registered commit
  string at `g`'s position (`cs (pos g)`) carries its value on `g` (`bit (val op) g`), as `RightRegZ` reads the
  registration's commitments. It reads neither the roots nor `H`.
* `witOf C X`: an opening of every wire at its position, of its value in `C`'s evaluation on `X`'s inputs.
-/

namespace Flock.Bridge

open FlockSoundness.Audit

/-- The Boolean-profile gates one C-Flock gate lowers to: three for a row, one for any other. -/
def gateSize {N : ℕ} : Op N → ℕ
  | .row _ _ => 3
  | _ => 1

/-- Where wire `g`'s lowered gates start: the lowered sizes of the gates before it. -/
def off (C : Circuit) (g : ℕ) : ℕ :=
  (((List.finRange C.N).filter fun h => decide (h.val < g)).map fun h => gateSize (C.op h)).sum

/-- **Wire `g`'s position** in the lowered program: its last lowered gate. -/
def posOf (C : Circuit) (g : Fin C.N) : ℕ := off C g.val + gateSize (C.op g) - 1

/-- Gate `g` lowered: a row to the XOR of its first list, the XOR of its second, and their AND; an input the
constants `κ` fix to `b` to `.const b`. -/
def lower (C : Circuit) (κ : Fin C.N → Option Bool) (g : Fin C.N) : List Claims.Sampling.BGate :=
  match C.op g with
  | .input => match κ g with
    | some b => [.const b]
    | none => [.input]
  | .and a b => [.and [posOf C a, posOf C b]]
  | .xor a b => [.xor [posOf C a, posOf C b]]
  | .not a => [.not (posOf C a)]
  | .row a b => [.xor (a.map (posOf C)), .xor (b.map (posOf C)), .and [off C g.val, off C g.val + 1]]

/-- **The circuit as a Boolean-profile program** at constants `κ`: its gates lowered in order, returning its outputs'
positions. -/
def progOf (C : Circuit) (κ : Fin C.N → Option Bool) : Claims.Sampling.Prog where
  gates := (List.finRange C.N).flatMap (lower C κ)
  outs := ((List.finRange C.N).filter fun g => decide (g ∈ C.outputs)).map (posOf C)

/-- Unit `u`'s lowered gates. -/
def unitGates {C : Circuit} {n : ℕ} (P : Partition C n) (u : Fin n) : List ℕ :=
  (List.finRange C.N).flatMap fun g =>
    if P.unit g = some u then (List.range (gateSize (C.op g))).map (off C g.val + ·) else []

/-- **The partition**: one class of one stratum, slot `u` the replay unit of the one proof unit `unitGates P u`. -/
def partOf {C : Circuit} {n : ℕ} (P : Partition C n) : Claims.Sampling.Partition :=
  ⟨[[(List.finRange n).map fun u => [unitGates P u]]]⟩

/-- **The design**: the one class drawn `subset kd`, every proof unit of a drawn replay unit checked. -/
def designOf (kd : ℕ) : Claims.Sampling.Design := ⟨[⟨.subset kd, none⟩]⟩

/-- **The statement**: `progOf`'s program at constants `κ` under the query `q` and the descriptor bytes `desc`, the
design `designOf kd`, and the roots. -/
def stmtOf {C : Circuit} {n : ℕ} (_P : Partition C n) (κ : Fin C.N → Option Bool) (kd : ℕ) (desc : Claims.Msg)
    (q : Claims.Sampling.Query) (roots : List Claims.Msg) : Claims.Sampling.Stmt :=
  ⟨desc, progOf C κ, q, designOf kd, roots⟩

open Classical in
/-- **The registration's commitment**: an opening at wire `g`'s position passes when some opening `op` of the
registered commit string at `g`'s position carries its value on `g`. -/
noncomputable def commitOf {C : Circuit} {Pos Val Opn Com : Type} (pos : Fin C.N → Pos) (bit : Val → Fin C.N → Bool)
    (val : Opn → Val) (com : Opn → Com) (cs : Pos → Com) : Claims.Sampling.Commitment Bool where
  opens _ o := .ret (decide (∃ g : Fin C.N, posOf C g = o.pos ∧ ∃ op : Opn, com op = cs (pos g) ∧ bit (val op) g = o.val))
  layer _ _ _ := []

/-- **The witness**: an opening of every wire at its position, of its value in `C`'s evaluation on `X`'s inputs. -/
def witOf (C : Circuit) (X : Fin C.N → Bool) : Claims.Sampling.Wit :=
  ⟨(List.finRange C.N).map fun g => ⟨posOf C g, C.eval X g, []⟩⟩

end Flock.Bridge
