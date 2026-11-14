module

public import Properties.Nci.Opening
public import Properties.Verity.Sha512
public import Flock.Hm96
public import Proofs.Accounting.ReadUnit
public import Proofs.Accounting.ReadUnit.BitsPorts
public import Proofs.Properties.Nci.RecordForm

/-!
# A pinned reader's unit returns its table's row

`PinnedReadBinds` is a lemma, proved by `pinnedReadBinds`, with its statement beside its proof: no lock lists it and
nothing reads it. Let a check hold a record's row as an opening under a pin, and let its reader's circuit file parse
to a templated statement whose unit template the read's check accepts (`Flock.ReadNet.check`) and whose unit returns
the circuit's outputs. Let the file's unit be the record's (`laidOutAs`), with the read unit's gates `q` at the
reader's roles (`portsOf`). Where the unit's constant is registered as 1, the unit is correct in a transcript, its
verdict bit is 1 there, and a table is committed under the root and the domain id its root and domain gates hold: the
row the unit returns in its local evaluation is that table's row at the key its key gates hold, or the fill row where
the key is at or past the table's count, or SHA-512 collides on what the unit's check and the table's opening hashed.

The record's names reach the circuit's gates by `Properties.Nci.Opening`'s map alone, and the map's gates are the
ones the read's check proves the read at:

* `portsAtGates_spec`: which gates `portsAtGates`'s are, bit by bit; `portsAtGates_eq`: so they are the gates at any
  columns each of those bits' gates sits at.
* `fieldBit_at`: `Spot.Bits.fieldBit` at a bit of a field's first instance, where no two fields share a name.
* `rowGate_copyGate`, `outRowGate_copyGate`: the map's gate on a parameter's bit, and on an output row's, is the copy's
  gate at the bit's column (`Accounting.ReadUnit.copyGate_colAt_inGate`, `copyGate_colAt_outGate`).
* `portsOf_portsAt`: so the map's ports are the read template's (`Accounting.ReadUnit.portsAt`), at copy 0.

There the read's gates compute its reference (`Accounting.ReadUnit.gatesRefine_check`). Inside the table the row is
the table's by `Accounting.SecurityProofs.readUnitBinds`; at or past the count it is the fill row by the gates'
refinement alone.

**What it does not say.** It is on the row returned, not on the opened ports: at a key at or past the count the
reference accepts any witness, and the opened ports hold what the prover likes. The output row returned is the one
the opening declares (`Form.openedRow`); of an opening that declares none it says nothing. Of `laidOutAs` the
proof reads the unit's number alone: that `Spot.Bits.check` gives no reason is what makes a row's name mean its port,
and no step here needs it. It gives no file, record and transcript that meet its hypotheses (`PinnedRead`):
`PinnedReadHonest`, a statement not proved, says the check's own form has some, the honest instance a listing asks.
-/

@[expose] public section

namespace NciCheck.Properties.Record

open Lean (Json)
open Flock.Program (Prog Defn)
open NciCheck.RecordText (Row Held)
open Accounting

/-- **A pinned read, as a check holds it and its circuit file lays it out**: the hypotheses of `PinnedReadBinds`, in
its order. The check reads the record and holds its row `r` as an opening under a pin, the opening declaring the output
row `out`; the reader's Definition `dfn` has the roles `ro`; the file `bytes` parses to the templated statement `ct`
with the unit template `tu`, which the read's check accepts; the file's unit `u` is `r`'s, with the read unit's gates
`q`; the unit's constant is registered as 1, `u` is correct in `Yc` with its registered constants there, its verdict
bit is 1, and `table` is committed under the root and domain id its gates hold. -/
def PinnedRead (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn)
    (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray))
    (ct : Flock.Circuit) (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening) : Prop :=
  F.read j tr = .ok c ∧
    Flock.Program.decode (F.inst c).program.toUTF8 = .ok P ∧
    F.held c r = some (.opened name p pin) ∧ F.openedRow c r = some out ∧
    P.index[r.tid]? = some k ∧ P.defs[k]? = some dfn ∧
    F.layout dfn = some ro ∧
    Flock.Desc.parseStmt tags bytes tables = .ok (.tmpl ct) ∧
    ct.tmpl = some tu ∧ Flock.ReadNet.check q.count q.words tu.t = true ∧ ct.outNet = ct.unitNet ∧
    laidOutAs ct P r.tid u.val r.unit = true ∧
    portsOf ct hB n dfn r.unit ro out q.count q.words = some q ∧
    consts (Flock.Bridge.constGate (.tmpl ct) hB n u) = some true ∧
    (Flock.Bridge.partsOf (.tmpl ct) hB n).Correct Yc u ∧
    (∀ g ∈ (Flock.Bridge.partsOf (.tmpl ct) hB n).io u, ∀ b, consts g = some b → ReadUnit.bit Yc g = b) ∧
    q.okOf Yc = true ∧
    ∀ i < q.count, Hash.RowLeaf.accepts Verity.sha512 Verity.sha512
      (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc)) (q.rootOf Yc) i (table i)

/-- **A pinned reader's unit returns its table's row.** Let a check read a record and hold its row `r` as an opening
under a pin, the reader's Definition having the roles `ro`. Let the circuit file `bytes` parse to the templated
statement `ct`, as the verifier reads it, with the unit template `tu`; let the read's check accept `tu` for a table of
`q.count` rows of `q.words` words, and the unit return the circuit's outputs. Let the file's unit `u` be `r`'s unit,
and `q` be the read unit's gates there at the roles and the output row `out` the opening declares as its row. Let
the unit's constant be registered as 1. Let `u` be correct in the transcript `Yc` and the registered constants hold
there; let the verdict bit be 1; and let `table` be committed under the root and the domain id the root and domain
gates hold (`PinnedRead`). Then the row `u` returns in its local evaluation is `table`'s row at the key its key gates
hold there, or the fill row where the key is at or past the table's count (`Hash.EmbeddingRead.h`), or SHA-512
collides on what the unit's check and `table`'s opening at the key hashed. -/
def PinnedReadBinds : Prop :=
  ∀ (F : Form) (c : F.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ) (dfn : Defn) (ro : Roles)
    (out : String) (tags : Flock.Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) (ct : Flock.Circuit)
    (tu : Flock.TmplUnit) (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedRead F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table →
    q.hOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) =
        Hash.EmbeddingRead.h q.count q.words (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
          (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))) ∨
      Hash.CollisionIn Verity.sha512
        (Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
            (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
          Hash.RowLeaf.outerHashed Verity.sha512 Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))
            (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)))) ∨
      Hash.CollisionIn Verity.sha512
        (Hash.RowLeaf.innerHashed Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (q.opening ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u)) ++
          Hash.RowLeaf.innerHashed Verity.sha512
            (q.verifier Flock.Hm96.Default512.key.data.toList (q.domainOf Yc))
            (table (q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u))))

/-- **An honest pinned read**: of the check's own form (`NciCheck.Record.SecurityProofs.form`), some instance, record,
circuit file and transcript meet `PinnedRead`, so `PinnedReadBinds` is not vacuous of it. So that it does not hold
emptily, by a table of one constant row: the check accepts the record (`Accepts`); the table has two different rows
within its count; and the transcript chooses the index, which no index gate's registered constant fixes and which
lies within the table's count. Not proved: that the record reads with its opening held rests on `Provenance` for the
check's form, and that the read's check accepts the template and the file parses are runs of code on a file, which
the kernel does not evaluate. -/
def PinnedReadHonest : Prop :=
  ∃ (c : NciCheck.Record.SecurityProofs.form.Case) (j tr : Json) (P : Prog) (r : Row) (name p pin : String) (k : ℕ)
    (dfn : Defn) (ro : Roles) (out : String) (tags : Flock.Tags) (bytes : ByteArray)
    (tables : List (String × ByteArray)) (ct : Flock.Circuit) (tu : Flock.TmplUnit)
    (hB : FlockSoundness.Built (Flock.Desc.tmpl ct).unroll.unit) (n : ℕ)
    (consts : Fin (Flock.Bridge.circuitOf (.tmpl ct) hB n).N → Option Bool) (u : Fin n)
    (q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n))
    (Yc : (Flock.Bridge.circuitOf (.tmpl ct) hB n).Trace) (table : ℕ → Hash.RowLeaf.Opening),
    PinnedRead NciCheck.Record.SecurityProofs.form c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n
      consts u q Yc table ∧
    Accepts NciCheck.Record.SecurityProofs.form j tr c ∧
    (∃ i i', i < q.count ∧ i' < q.count ∧ (table i).row ≠ (table i').row) ∧
    (∀ b, consts (q.tok b) = none) ∧
    q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count

end NciCheck.Properties.Record

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Defn)
open Accounting

/-! ## The gates `portsAtGates` gives -/

/-- **The gates `gatesOf` gives are `gate`'s**, each at its index. -/
theorem gatesOf_some {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f)
    (i : Fin k) : gate i.val = some (f i).val := by
  unfold gatesOf at h
  by_cases hg : ∀ i : Fin k, ∃ g : Fin N, gate i.val = some g.val
  · rw [dite_eq_left hg] at h
    obtain rfl := Option.some.inj h
    exact Classical.choose_spec (hg i)
  · rw [dite_eq_right hg] at h
    cases h

/-- `gatesOf_some` at an index and its bound. -/
theorem gatesOf_at {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f) (i : ℕ)
    (hi : i < k) : gate i = some (f ⟨i, hi⟩).val :=
  gatesOf_some h ⟨i, hi⟩

/-- **The gates `gatesOf` gives are the gates at the columns `col`**, where each bit's gate is the gate at its
column. -/
theorem gatesOf_eq {N k : ℕ} {gate : ℕ → Option ℕ} {f : Fin k → Fin N} (h : gatesOf N k gate = some f)
    (at_ : ℕ → Fin N) (col : ℕ → ℕ) (hat : ∀ b < k, ∀ g, gate b = some g → (at_ (col b)).val = g) :
    f = fun i => at_ (col i.val) :=
  funext fun i => Fin.ext (hat i.val i.isLt _ (gatesOf_some h i)).symm

/-- **The gates `portsAtGates` gives are the roles' gates**, bit by bit: the index's are the key parameter's, the
root's and the domain id's the root and domain parameters', the row returned the output row `out`'s and the verdict
the first bit of the verdict's row; the opened parameters are three, whose bits are the witness row's, its salt's and
its path's, a digest a level. -/
theorem portsAtGates_spec {C : Verity.Circuit} {inG outG : String → ℕ → Option ℕ} {ro : Roles} {out : String}
    {count words : ℕ} {q : ReadUnit.Ports C} (h : portsAtGates C inG outG ro out count words = some q) :
    (∀ i : Fin 32, inG ro.key i.val = some (q.tok i).val) ∧
      (∀ i : Fin 512, inG ro.root i.val = some (q.root i).val) ∧
      (∀ i : Fin 512, inG ro.domain i.val = some (q.domain i).val) ∧
      (∀ i : Fin (16 * q.words), outG out i.val = some (q.h i).val) ∧
      outG ro.verdict 0 = some q.ok.val ∧
      ro.opened.length = 3 ∧
      (∀ i : Fin (16 * q.words), inG (ro.opened.getD 0 "") i.val = some (q.row i).val) ∧
      (∀ i : Fin (8 * Hash.Hm96.SALT_BYTES), inG (ro.opened.getD 1 "") i.val = some (q.salt i).val) ∧
      ∀ (l : Fin (Hash.Tree.depth q.count)) (i : Fin 512),
        inG (ro.opened.getD 2 "") (512 * l.val + i.val) = some (q.path l i).val := by
  unfold portsAtGates at h
  by_cases hlen : ro.opened.length = 3
  · rw [ite_eq_left hlen] at h
    obtain ⟨tok, htok, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨root, hroot, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨domain, hdomain, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨rowG, hrow, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨saltG, hsalt, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨pathG, hpath, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨hG, hh, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨okG, hok, h⟩ := Option.map_eq_some_iff.1 h
    subst h
    exact ⟨gatesOf_some htok, gatesOf_some hroot, gatesOf_some hdomain, gatesOf_some hh,
      gatesOf_at hok 0 Nat.one_pos, hlen, gatesOf_some hrow, gatesOf_some hsalt,
      fun l i => gatesOf_at hpath (512 * l.val + i.val) _⟩
  · rw [ite_eq_right hlen] at h
    cases h

/-- Two read units' gates of one table are equal where each of their ports' gates are. -/
theorem ports_congr {C : Verity.Circuit} {count words : ℕ} {position : ℕ → ℕ} {tok tok' : Fin 32 → Fin C.N}
    {root root' domain domain' : Fin 512 → Fin C.N} {row row' : Fin (16 * words) → Fin C.N}
    {salt salt' : Fin (8 * Hash.Hm96.SALT_BYTES) → Fin C.N}
    {path path' : Fin (Hash.Tree.depth count) → Fin 512 → Fin C.N} {h h' : Fin (16 * words) → Fin C.N}
    {ok ok' : Fin C.N} (h1 : tok = tok') (h2 : root = root') (h3 : domain = domain') (h4 : row = row')
    (h5 : salt = salt') (h6 : path = path') (h7 : h = h') (h8 : ok = ok') :
    (ReadUnit.Ports.mk count words position tok root domain row salt path h ok : ReadUnit.Ports C) =
      ReadUnit.Ports.mk count words position tok' root' domain' row' salt' path' h' ok' := by
  subst h1 h2 h3 h4 h5 h6 h7 h8
  rfl

/-- **The gates `portsAtGates` gives are the gates at the read's columns**, where each bit's gate is: with `at_ x`
the gate at column `x`, and each bit of the key, the root, the domain id, the witness row, its salt, its path, the
row returned and the verdict having its gate, if any, at its column (`ckey`, `croot`, `cdomain`, `crow`, `csalt`,
`cpath`, `ch`, `cok`), the read unit's gates are the gates at those columns. -/
theorem portsAtGates_eq {C : Verity.Circuit} {inG outG : String → ℕ → Option ℕ} {ro : Roles} {out : String}
    {count words : ℕ} {q : ReadUnit.Ports C} (h : portsAtGates C inG outG ro out count words = some q)
    (at_ : ℕ → Fin C.N) (ckey croot cdomain crow csalt ch : ℕ → ℕ) (cpath : ℕ → ℕ → ℕ) (cok : ℕ)
    (hkey : ∀ b < 32, ∀ g, inG ro.key b = some g → (at_ (ckey b)).val = g)
    (hroot : ∀ b < 512, ∀ g, inG ro.root b = some g → (at_ (croot b)).val = g)
    (hdomain : ∀ b < 512, ∀ g, inG ro.domain b = some g → (at_ (cdomain b)).val = g)
    (hrow : ∀ b < 16 * words, ∀ g, inG (ro.opened.getD 0 "") b = some g → (at_ (crow b)).val = g)
    (hsalt : ∀ b < 8 * Hash.Hm96.SALT_BYTES, ∀ g, inG (ro.opened.getD 1 "") b = some g → (at_ (csalt b)).val = g)
    (hpath : ∀ l < Hash.Tree.depth count, ∀ i < 512, ∀ g,
      inG (ro.opened.getD 2 "") (512 * l + i) = some g → (at_ (cpath l i)).val = g)
    (hh : ∀ b < 16 * words, ∀ g, outG out b = some g → (at_ (ch b)).val = g)
    (hok : ∀ g, outG ro.verdict 0 = some g → (at_ cok).val = g) :
    q = { count := count, words := words, position := fun r => r, tok := fun i => at_ (ckey i),
          root := fun i => at_ (croot i), domain := fun i => at_ (cdomain i), row := fun i => at_ (crow i),
          salt := fun i => at_ (csalt i), path := fun l i => at_ (cpath l i), h := fun i => at_ (ch i),
          ok := at_ cok } := by
  unfold portsAtGates at h
  by_cases hlen : ro.opened.length = 3
  · rw [ite_eq_left hlen] at h
    obtain ⟨tok, htok, h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨root, hroot', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨domain, hdomain', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨rowG, hrow', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨saltG, hsalt', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨pathG, hpath', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨hG, hh', h⟩ := Option.bind_eq_some_iff.1 h
    obtain ⟨okG, hok', h⟩ := Option.map_eq_some_iff.1 h
    subst h
    exact ports_congr (gatesOf_eq htok at_ ckey hkey) (gatesOf_eq hroot' at_ croot hroot)
      (gatesOf_eq hdomain' at_ cdomain hdomain) (gatesOf_eq hrow' at_ crow hrow)
      (gatesOf_eq hsalt' at_ csalt hsalt)
      (funext fun l => funext fun i => Fin.ext (hpath l.val l.isLt i.val i.isLt _ (gatesOf_some hpath'
        ⟨512 * l.val + i.val, by have := l.isLt; have := i.isLt; omega⟩)).symm)
      (gatesOf_eq hh' at_ ch hh) (Fin.ext (hok _ (gatesOf_some hok' ⟨0, Nat.one_pos⟩)).symm)
  · rw [ite_eq_right hlen] at h
    cases h

/-! ## A field's bit among a unit's ports -/

/-- **The first field named `f` is field `i`**, where field `i` is named `f` and no two fields share a name. -/
theorem findIdx?_fst : ∀ (ws : List (String × ℕ)) (i : ℕ) (f : String), (ws.map (·.1))[i]? = some f →
    (ws.map (·.1)).Nodup → ws.findIdx? (·.1 == f) = some i
  | [], _, _, h, _ => by cases h
  | x :: ws, 0, f, h, _ => by
    have hx : x.1 = f := Option.some.inj h
    rw [List.findIdx?_cons, ite_eq_left (by rw [hx]; exact beq_self_eq_true f)]
  | x :: ws, i + 1, f, h, hnd => by
    have hnd' : x.1 ∉ ws.map (·.1) ∧ (ws.map (·.1)).Nodup := List.nodup_cons.1 hnd
    have hf : (ws.map (·.1))[i]? = some f := h
    have hne : ¬ (x.1 == f) = true := fun he =>
      hnd'.1 (eq_of_beq he ▸ List.mem_of_getElem? hf)
    rw [List.findIdx?_cons, ite_eq_right hne, findIdx?_fst ws i f hf hnd'.2]
    rfl

/-- **`Spot.Bits.fieldBit` at a bit of field `i`'s first instance**: where field `i` of `ws` is named `f`, no two
fields share a name and the file has an instance, bit `β` of `f`'s first instance's words is instance 0's port, or
output leaf, `P + β / 16`, `P` the words of the fields before it, at bit `β % 16`. -/
theorem fieldBit_at {c : Flock.Circuit} {ws : List (String × ℕ)} {f : String} {i β : ℕ}
    (hname : (ws.map (·.1))[i]? = some f) (hnd : (ws.map (·.1)).Nodup) (hupv : 0 < c.upv)
    (hβ : β < 16 * (ws.getD i default).2) :
    Spot.Bits.fieldBit c ws f β = some (0, ((ws.take i).map (·.2)).sum + β / 16, β % 16) := by
  have hw : β / 16 < (ws.getD i default).2 := by omega
  have hle : (ws.getD i default).2 ≤ c.upv * (ws.getD i default).2 := Nat.le_mul_of_pos_left _ hupv
  unfold Spot.Bits.fieldBit
  rw [findIdx?_fst ws i f hname hnd]
  show (if ((ws.getD i default).2 == 0 || decide (c.upv * (ws.getD i default).2 ≤ β / 16)) = true then none
    else some (β / 16 / (ws.getD i default).2, ((ws.take i).map (·.2)).sum + β / 16 % (ws.getD i default).2,
      β % 16)) = _
  have hc : ¬ ((ws.getD i default).2 == 0 || decide (c.upv * (ws.getD i default).2 ≤ β / 16)) = true := by
    simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq]
    omega
  rw [ite_eq_right hc, Nat.div_eq_of_lt hw, Nat.mod_eq_of_lt hw]

/-! ## The map's gates are the read template's -/

/-- **The map's gate on a parameter's bit is the copy's gate at the bit's column.** In a templated circuit whose unit
template the read's check accepts, let the Definition `d`'s parameters be `ws`, no two sharing a name, with parameter
`i` named `port`, on `W` words after `P` words of parameters, all among the read's ports. Where `rowGate` gives the
gate `g` on bit `b` of `port` in unit `u`, the gate of copy 0 of `u` at column `16 P + b` is `g`. -/
theorem rowGate_copyGate {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ) (u : Fin n)
    (hj : 0 < (Flock.HmOut.unrolled ct).upv) {d : Defn} {ws : List (String × ℕ)} (hws : paramWords d = ws) {i : ℕ}
    {port : String} {W P : ℕ} (hname : (ws.map (·.1))[i]? = some port) (hnd : (ws.map (·.1)).Nodup)
    (hW : (ws.getD i default).2 = W) (hP : ((ws.take i).map (·.2)).sum = P)
    (hfit : P + W ≤ 162 + H + 32 * Hash.Tree.depth V) {b col g : ℕ} (hb : b < 16 * W) (hcol : col = 16 * P + b)
    (h : rowGate (Flock.HmOut.unrolled ct) d n u.val port b = some g) :
    (Flock.Bridge.copyGate (.tmpl ct) hB n u hj (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB) col)).val = g := by
  subst hws
  have hf : Spot.Bits.fieldBit (Flock.HmOut.unrolled ct) (paramWords d) port b = some (0, P + b / 16, b % 16) := by
    rw [fieldBit_at hname hnd hj (by rw [hW]; exact hb), hP]
  unfold rowGate at h
  rw [hf] at h
  have hc : col = 16 * (P + b / 16) + b % 16 := by omega
  exact (congrArg (fun x => (Flock.Bridge.copyGate (.tmpl ct) hB n u hj
      (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB) x)).val) hc).trans
    (ReadUnit.copyGate_colAt_inGate ht hsz hcheck hB n u hj (p := P + b / 16) (t := b % 16) (by omega)
      (Nat.mod_lt _ (by decide)) h)

/-- **The map's gate on an output row's bit is the copy's gate at the bit's column.** Let the Definition `d`'s output
rows be `ws`, no two sharing a name, with output row `i` named `out`, on `W` words after `P` words of output rows.
Where `outRowGate` gives the gate `g` on bit `b` of `out` in unit `u` of the circuit `D` describes, the gate of copy 0
of `u` at the column of bit `b % 16` of returned output `P + b / 16` (`Layout.outCol`) is `g`. -/
theorem outRowGate_copyGate (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (u : Fin n)
    (hj : 0 < D.unroll.upv) {d : Defn} {ws : List (String × ℕ)} (hws : outWords d = ws) {i : ℕ} {out : String}
    {W P : ℕ} (hname : (ws.map (·.1))[i]? = some out) (hnd : (ws.map (·.1)).Nodup)
    (hW : (ws.getD i default).2 = W) (hP : ((ws.take i).map (·.2)).sum = P)
    {b g : ℕ} (hb : b < 16 * W) (h : outRowGate D.unroll d n u.val out b = some g) :
    (Flock.Bridge.copyGate D hU n u hj (ReadUnit.colAt (FlockSoundness.Rows.ofNet hU)
      (FlockSoundness.Discharge.Layout.outCol D.unroll (P + b / 16) (b % 16)))).val = g := by
  subst hws
  have hf : Spot.Bits.fieldBit D.unroll (outWords d) out b = some (0, P + b / 16, b % 16) := by
    rw [fieldBit_at hname hnd hj (by rw [hW]; exact hb), hP]
  unfold outRowGate at h
  rw [hf] at h
  exact ReadUnit.copyGate_colAt_outGate D hU n u hj h

/-- **What `readRows` decides**: the Definition's parameters and output rows are the read's, none sharing a name. -/
theorem readRows_spec {d : Defn} {ro : Roles} {out : String} {count words : ℕ}
    (h : readRows d ro out count words = true) :
    paramWords d = [(ro.key, 2), (ro.root, 32), (ro.domain, 32), (ro.opened.getD 0 "", words),
        (ro.opened.getD 1 "", Hash.Hm96.SALT_BYTES / 2), (ro.opened.getD 2 "", 32 * Hash.Tree.depth count)] ∧
      outWords d = [(out, words), (ro.verdict, 1)] ∧
      [ro.key, ro.root, ro.domain, ro.opened.getD 0 "", ro.opened.getD 1 "", ro.opened.getD 2 ""].Nodup ∧
      [out, ro.verdict].Nodup := by
  unfold readRows at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2, h.2⟩

/-- A templated statement `Desc.parseStmt` reads is the one `HmOut.parseTmpl` reads. -/
theorem parseTmpl_of_parseStmt {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {ct : Flock.Circuit} (h : Flock.Desc.parseStmt tags bytes tables = .ok (.tmpl ct)) :
    Flock.HmOut.parseTmpl tags bytes tables = .ok ct := by
  unfold Flock.Desc.parseStmt at h
  split at h
  · split at h
    · next c hc => cases h; exact hc
    · cases h
  · unfold Flock.Desc.parse at h
    split at h <;> cases h

/-- **The map's ports are the read template's.** Let the templated statement parse into `ct`, with its unit template
passing the read's check for a table of `V` rows of `H` words and its unit returning the circuit's outputs. Where
`portsOf` gives the read unit's gates `q` of unit `u` at the roles `ro` of a Definition and the output row `out`, they
are the gates of copy 0 of `u` at the read template's columns (`Accounting.ReadUnit.portsAt`). -/
theorem portsOf_portsAt {V H : ℕ} {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {ct : Flock.Circuit} {tu : Flock.TmplUnit} (hparse : Flock.HmOut.parseTmpl tags bytes tables = .ok ct)
    (ht : ct.tmpl = some tu) (hcheck : Flock.ReadNet.check V H tu.t = true) (ho : ct.outNet = ct.unitNet)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ) (u : Fin n) {d : Defn} {ro : Roles}
    {out : String} {q : ReadUnit.Ports (Flock.Bridge.circuitOf (.tmpl ct) hB n)}
    (h : portsOf ct hB n d u.val ro out V H = some q) :
    ∃ hj : 0 < (Flock.HmOut.unrolled ct).upv, q = ReadUnit.portsAt (.tmpl ct) hB n u hj V H := by
  have hsz : ct.unitNet < ct.nets.size := by
    obtain ⟨_, _, P, -, rfl, -, -⟩ := FlockSoundness.TmplValue.parseTmpl_facts hparse
    exact P.unitNet
  have hj : 0 < (Flock.HmOut.unrolled ct).upv := (ReadUnit.Gates.unrolled_leaves_of_parse hparse ho).1
  unfold portsOf at h
  obtain ⟨hrows, h⟩ := Option.ite_none_right_eq_some.1 h
  obtain ⟨hp, hw, hnp, hno⟩ := readRows_spec hrows
  have hS : Hash.Hm96.SALT_BYTES = 192 := rfl
  have key := fun (i : ℕ) (port : String) (W P : ℕ) =>
    @rowGate_copyGate V H ct tu ht hsz hcheck hB n u hj d _ hp i port W P
  have outk := fun (i : ℕ) (o : String) (W P : ℕ) =>
    @outRowGate_copyGate (.tmpl ct) hB n u hj d _ hw i o W P
  exact ⟨hj, portsAtGates_eq h
    (fun x => Flock.Bridge.copyGate (.tmpl ct) hB n u hj (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB) x))
    (fun b => b) (fun b => Flock.ReadNet.ROOT_AT + b) (fun b => Flock.ReadNet.DOMAIN_AT + b)
    (fun b => Flock.ReadNet.ROW_AT + b) (fun b => Flock.ReadNet.saltAt H + b)
    (fun b => FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (b / 16) (b % 16))
    (fun l i => Flock.ReadNet.pathAt H + 512 * l + i)
    (FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) H 0)
    (fun b hb g hg => key 0 ro.key 2 0 rfl hnp rfl rfl (by omega) (by omega) (by omega) hg)
    (fun b hb g hg => key 1 ro.root 32 2 rfl hnp rfl rfl (by omega) (by omega)
      (by unfold Flock.ReadNet.ROOT_AT; omega) hg)
    (fun b hb g hg => key 2 ro.domain 32 34 rfl hnp rfl rfl (by omega) (by omega)
      (by unfold Flock.ReadNet.DOMAIN_AT; omega) hg)
    (fun b hb g hg => key 3 (ro.opened.getD 0 "") H 66 rfl hnp rfl rfl (by omega) hb
      (by unfold Flock.ReadNet.ROW_AT; omega) hg)
    (fun b hb g hg => key 4 (ro.opened.getD 1 "") (Hash.Hm96.SALT_BYTES / 2) (66 + H) rfl hnp rfl
      (show 2 + (32 + (32 + (H + 0))) = 66 + H by omega) (by omega) (by omega)
      (by unfold Flock.ReadNet.saltAt; omega) hg)
    (fun l hl i hi g hg => key 5 (ro.opened.getD 2 "") (32 * Hash.Tree.depth V) (162 + H) rfl hnp rfl
      (show 2 + (32 + (32 + (H + (Hash.Hm96.SALT_BYTES / 2 + 0)))) = 162 + H by omega) (by omega) (by omega)
      (by unfold Flock.ReadNet.pathAt; omega) hg)
    (fun b hb g hg => (congrArg (fun x => (Flock.Bridge.copyGate (.tmpl ct) hB n u hj
        (ReadUnit.colAt (FlockSoundness.Rows.ofNet hB)
          (FlockSoundness.Discharge.Layout.outCol (Flock.HmOut.unrolled ct) x (b % 16)))).val)
        (Nat.zero_add (b / 16)).symm).trans (outk 0 out H 0 rfl hno rfl rfl hb hg))
    (fun g hg => outk 1 ro.verdict 1 H rfl hno rfl rfl (show 0 < 16 * 1 by omega) hg)⟩

/-! ## The statement -/

/-- **A pinned reader's unit returns its table's row** (`PinnedReadBinds`). -/
theorem pinnedReadBinds : PinnedReadBinds := by
  intro F c j tr P r name p pin k dfn ro out tags bytes tables ct tu hB n consts u q Yc table h
  unfold PinnedRead at h
  obtain ⟨-, -, -, -, -, -, -, hstmt, ht, hcheck, ho, hlaid, hports, hone, hcorrect, hconsts, hok, htable⟩ := h
  have hparse := parseTmpl_of_parseStmt hstmt
  have hu : u.val = r.unit := by
    unfold laidOutAs at hlaid
    simp only [Bool.and_eq_true, beq_iff_eq] at hlaid
    exact hlaid.2
  rw [← hu] at hports
  obtain ⟨hj, hq⟩ := portsOf_portsAt hparse ht hcheck ho hB n u hports
  have hgates : ReadUnit.GatesRefine Verity.sha512 Verity.sha512 Flock.Hm96.Default512.key.data.toList consts
      (Flock.Bridge.partsOf (.tmpl ct) hB n) u q := by
    rw [hq]
    exact ReadUnit.gatesRefine_check hparse ht hcheck ho hB n consts u hone hj
  have hrows : ReadUnit.RegisteredRows consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q (q.rootOf Yc)
      (q.domainOf Yc) Yc :=
    ⟨rfl, rfl, hok, hconsts⟩
  have hb := Accounting.SecurityProofs.readUnitBinds Verity.sha512 Verity.sha512
    Flock.Hm96.Default512.key.data.toList consts (Flock.Bridge.partsOf (.tmpl ct) hB n) u q _ _ Yc hgates hcorrect
    hrows
  obtain ⟨hh, -⟩ := hgates.computes Yc hcorrect hconsts
  by_cases hlt : q.tokOf ((Flock.Bridge.partsOf (.tmpl ct) hB n).localEval Yc u) < q.count
  · rcases hb (table _) (htable _ hlt) with h | h | h
    · left
      rw [h]
      unfold Hash.EmbeddingRead.h
      exact (ite_eq_left hlt).symm
    · exact .inr (.inl h)
    · exact .inr (.inr h)
  · left
    rw [hh]
    unfold Hash.EmbeddingRead.h
    exact (ite_eq_right hlt).trans (ite_eq_right hlt).symm

end NciCheck.Properties.Record.SecurityProofs
