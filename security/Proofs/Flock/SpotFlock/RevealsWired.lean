module

public import Proofs.Flock.SpotFlock.AddressesWired
public import Proofs.Flock.SpotFlock.WiredAt

/-!
# The sessions' openings at the wired program

`AddressesWired h ω keep` (`AddressesWired.lean`) says where each gate's value sits at the members' programs wired by
`ω`. This file proves, from it, what `SessionOpensTwo` asks of a session at the wired program: the witness reveals
each value a checked unit's proof block needs there, by an opening that passes the deployed check, and values that
agree with the openings that pass make the block correct.

* **What a session reveals** (`revealWired`). A proof block, without the gates `ω` moves, needs gate `g` at the wired
  program only where the block needs, side by side, a gate `i` with `ω i = g` (`Prog.needs_wired`). So the reveal
  holds one opening for each gate `i` the block needs side by side (`needsSide`), in that order: the frame's opening
  of `i`, at position `ω i`, under the roots the wired program opens `ω i` under (`rootsAtWired`). The reveal count
  is the one side by side (`length_revealWired_le`), and the reveal doesn't read the statement's program
  (`revealWired_prog`), so a code whose statement holds the wired program reveals the same list.
* **The frame** (`revealCodeWired`, `BuiltWired`): the base frame over the addresses, revealing that list.
* **The openings pass** (`CheckedRun.opens_slot`, `CheckedRun.opens_at`, `CheckedRun.reveals_wired`). The opening of
  `i` holds the row the unit's session holds `i` in. Its place and its slot are read at position `ω i`, and are those
  of `i`'s address, since a gate the wiring moves has the place and the slot of the gate it is sent to. The row opens
  the registered read of its port (`rows_open`) or its population's published tree (`rows_rooted`), and the frame
  checks it at the entry the slot names (`port_frameAt`), which has that read's or that tree's root (`EntriesAt`).
* **Two reads of one committed bit** (`CheckedRun.correct_wired`). Two gates a unit holds or reads that `ω` sends to
  one gate are each needed side by side (`Prog.Rewires.needs_of_moved`, `Prog.Rewires.needs_of_target`), so each has
  an opening in the reveal, at one position under one set of roots. Values that agree with the openings that pass
  therefore hold one value there, and the unit's two decoded values are equal. No condition on the map says so: it
  is the commitment's binding, which `SessionOpensTwo`'s own hypothesis on the values carries.
* `stmtOpensWired`: every checked unit of an accepted run, at the wired program (`StmtOpensWired`, `StmtOpens`'s
  shape there).
-/

@[expose] public section

namespace Claims.Sampling

/-- **A gate a proof block's correctness reads is one it holds or reads.** -/
theorem Prog.touches_of_needs {C : Prog} {u : List ℕ} {i : ℕ} (hn : C.Needs u i) : C.Touches u i := by
  unfold Prog.Needs at hn
  unfold Prog.Touches
  rcases hn with hn | hn
  · exact Or.inl hn.1
  · exact Or.inr hn.1

/-- **A gate the map moves, that a proof block holds or reads, is one its correctness reads**: it is an input, so
the block commits it where it holds it and reads it opened where it doesn't. -/
theorem Prog.Rewires.needs_of_moved {C : Prog} {P : Partition} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hW : C.Rewires P ω keep) {u : List ℕ} {i : ℕ} (ht : C.Touches u i) (hm : ω i ≠ i) : C.Needs u i := by
  have hin := hW.input i hm
  unfold Prog.Touches at ht
  unfold Prog.Needs
  by_cases hu : i ∈ u
  · exact Or.inl ⟨hu, Prog.boundary_of_input u hin⟩
  · rcases ht with ht | ht
    · exact absurd ht hu
    · exact Or.inr ⟨ht, hu, fun b hb => by rw [hin] at hb; cases hb⟩

/-- **A gate the map sends a moved gate to, that a proof block holds or reads, is one its correctness reads**: the
program returns it and it is no constant. -/
theorem Prog.Rewires.needs_of_target {C : Prog} {P : Partition} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hW : C.Rewires P ω keep) {u : List ℕ} {i j : ℕ} (ht : C.Touches u i) (hj : ω j ≠ j) (hji : ω j = i) :
    C.Needs u i := by
  have hout := hW.out j hj
  have hnc := hW.noConst j hj
  rw [hji] at hout hnc
  unfold Prog.Touches at ht
  unfold Prog.Needs
  by_cases hu : i ∈ u
  · exact Or.inl ⟨hu, Prog.boundary_of_out u hout⟩
  · rcases ht with ht | ht
    · exact absurd ht hu
    · exact Or.inr ⟨ht, hu, hnc⟩

end Claims.Sampling

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)

/-! ## An opening read at another position -/

/-- **A list zipped with its own image**, mapped: the map of each element with its image. -/
theorem map_zip_map_self {α β γ : Type} (G : α → β) (F : α × β → γ) :
    ∀ l : List α, (l.zip (l.map G)).map F = l.map fun a => F (a, G a)
  | [] => rfl
  | a :: l => by
    show F (a, G a) :: (l.zip (l.map G)).map F = F (a, G a) :: l.map fun a => F (a, G a)
    rw [map_zip_map_self G F l]

/-- **Two openings with one value and one data, at positions of one place, have one row.** -/
theorem row_at {place place' : ℕ → (w : ℕ) × Fin w} {bits : Msg → ℕ → Bool} {o o' : Opening Bool}
    (hp : place o.pos = place' o'.pos) (hv : o.val = o'.val) (hd : o.data = o'.data) :
    row place bits o = row place' bits o' := by
  unfold row rowBits
  rw [hp, hv, hd]

theorem saltOf_pos (o : Opening Bool) (p : ℕ) : saltOf ({ o with pos := p } : Opening Bool) = saltOf o := rfl

theorem pathOf_pos (o : Opening Bool) (p : ℕ) : pathOf ({ o with pos := p } : Opening Bool) = pathOf o := rfl

/-! ## What a session reveals at the wired program -/

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

/-- **The proof-block partition at the wired program**: each proof block without the gates `ω` moves
(`Partition.cut`). -/
def partitionWired (s : AuditStatement) : Option Partition := (partition h s).map fun P => P.cut ω

/-- **The two-stage design at the wired program**: its replay partition, each replay block without the gates `ω`
moves. -/
def twoWired (s : AuditStatement) : Option Design.Two :=
  (two h s).map fun d => ({ d with part := d.part.cut ω } : Design.Two)

/-- **The gates unit `u`'s proof block needs opened, side by side**: `Reveal.needs` at the members' programs. It
reads the statement's roots only, so it is the same list at the statement the verifier parses and at that statement
with the wired program. -/
def needsSide (s : AuditStatement) (u : ℕ) : List ℕ :=
  Reveal.needs (partition h) { s with prog := bitsProgs h.ms } u

/-- **The roots the wired program opens gate `g` of unit `u` under** (`Reveal.rootsAt` at the members' programs
wired by `ω`, the wired partition and the wired design): the interior roots of `u`'s replay block where `g` is one of
that block's gates that `ω` leaves alone and the wired program doesn't commit as the block sees it, and the
statement's otherwise. -/
def rootsAtWired (s : AuditStatement) (u : ℕ) (v : Seen) (g : ℕ) : List Msg :=
  Reveal.rootsAt (partitionWired h ω) (twoWired h ω) (iroots []) { s with prog := (bitsProgs h.ms).wired ω keep } u v g

/-- **What a session's witness reveals for unit `u` at the wired program**, at a frame `f`: the frame's openings, in
the order of the gates the unit's proof block needs side by side (`needsSide`), the opening of gate `i` at position
`ω i` and under the roots the wired program opens `ω i` under. The frame is asked at the statement with the members'
programs side by side, the one its session runs at, so the reveal reads the statement without its program
(`revealWired_prog`). -/
def revealWired (f : Code) (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    List (List Msg × Opening Bool) :=
  ((needsSide h s u).zip (f.reveal { s with prog := bitsProgs h.ms } u v wF)).map fun p =>
    (rootsAtWired h ω keep s u v (ω p.1), ({ p.2.2 with pos := ω p.1 } : Opening Bool))

/-- **The reveal at the wired program doesn't read the statement's program**: it is one list at the statement the
verifier parses, with the members' programs side by side, and at that statement with the wired program. -/
theorem revealWired_prog (f : Code) (s : AuditStatement) (p : Prog) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    revealWired h ω keep f { s with prog := p } u v wF = revealWired h ω keep f s u v wF :=
  rfl

/-- **The reveal at the wired program is no longer than the list of gates the proof block needs side by side**,
whatever the frame: the reveal count is the one side by side. -/
theorem length_revealWired_le (f : Code) (s : AuditStatement) (u : ℕ) (v : Seen) (wF : Claims.Flock.Wit) :
    (revealWired h ω keep f s u v wF).length ≤ (needsSide h s u).length := by
  unfold revealWired
  rw [List.length_map, List.length_zip]
  exact Nat.min_le_left _ _

/-- **The reveal at the wired program holds the frame's opening of each gate the proof block needs side by side**,
at a frame that reveals one opening a gate, each at the gate `ω` sends it to. -/
theorem mem_revealWired {f : Code} {s : AuditStatement} {u : ℕ} {v : Seen} {wF : Claims.Flock.Wit}
    {R : ℕ → List Msg} {op : ℕ → Opening Bool}
    (hG : f.reveal { s with prog := bitsProgs h.ms } u v wF =
      (needsSide h s u).map fun j => (R j, ({ op j with pos := ω j } : Opening Bool)))
    {i : ℕ} (hi : i ∈ needsSide h s u) :
    (rootsAtWired h ω keep s u v (ω i), ({ op i with pos := ω i } : Opening Bool)) ∈
      revealWired h ω keep f s u v wF := by
  unfold revealWired
  rw [hG, map_zip_map_self]
  exact List.mem_map.2 ⟨i, hi, rfl⟩

/-- **The base frame at the wired program**: `revealCode` at the addresses, revealing, for each gate `i` a proof
block needs side by side, the address's opening of `i` (`addressOp`) at position `ω i`, under the roots the wired
program opens `ω i` under. -/
noncomputable def revealCodeWired (addr : ℕ → Option Address) (c₀ : Code) : Code :=
  { revealCode h addr c₀ with
    reveal := fun s u v wF => (needsSide h s u).map fun i =>
      (rootsAtWired h ω keep s u v (ω i), ({ addressOp h addr s u v wF i [] with pos := ω i } : Opening Bool)) }

/-- **Its own ports are `revealOwn`**, of depth 64. Stated at `revealOwn` and never through `revealCode`: the two codes
differ in what they reveal, and the unifier would compare them field by field before it reads one field of each. -/
theorem depth_revealCodeWired (addr : ℕ → Option Address) (c₀ : Code) :
    depth (revealCodeWired h ω keep addr c₀).own = 64 :=
  depth_revealOwn

/-! Its place, bits, salt and path are the base frame's, stated as functions: a proof rewrites with these before it
names an opening, so the unifier never unfolds an opening's data to compare a field with its value. -/

theorem place_revealCodeWired (addr : ℕ → Option Address) (c₀ : Code) :
    (revealCodeWired h ω keep addr c₀).place = addressPlace addr :=
  rfl

theorem bits_revealCodeWired (addr : ℕ → Option Address) (c₀ : Code) :
    (revealCodeWired h ω keep addr c₀).bits = bitsOf :=
  rfl

theorem salt_revealCodeWired (addr : ℕ → Option Address) (c₀ : Code) :
    (revealCodeWired h ω keep addr c₀).salt = saltOf :=
  rfl

theorem path_revealCodeWired (addr : ℕ → Option Address) (c₀ : Code) :
    (revealCodeWired h ω keep addr c₀).π = pathOf :=
  rfl

open Classical in
/-- **The address map at the wired program**: the addresses `AddressesWired` names, where it holds. -/
noncomputable def addressMapWired (g : ℕ) : Option Address :=
  if hx : AddressesWired h ω keep then some (Exists.choose hx g) else none

theorem addressMapWired_eq (hA : AddressesWired h ω keep) :
    addressMapWired h ω keep = fun g => some (Exists.choose hA g) :=
  funext fun _ => dite_eq_left hA

/-- **The held code's frame is SPOT's own at the wired program**: `frameAt` at the addresses' slots and
`revealCodeWired` at the addresses, from some code `c₀` for the fields the base frame doesn't set. -/
def BuiltWired : Prop :=
  ∃ c₀ : Code, h.frame =
    frameAt (slotsOf (addressMapWired h ω keep)) (revealCodeWired h ω keep (addressMapWired h ω keep) c₀)

/-! ## The openings pass -/

namespace CheckedRun

variable {h ω keep} {r : CheckedRun h.stage}

/-- **The gates an accepted run's checked unit needs side by side are `needsSide`'s.** -/
theorem mem_needsSide (hr : r.Ok h) {i : ℕ} : i ∈ needsSide h r.s r.u ↔ r.s.prog.Needs r.b i := by
  obtain ⟨Pt, _, _, hx, hP, -, -, -, -, -, hb, -, -, -⟩ := hr
  have e := Reveal.mem_needs (partition h) (s := { r.s with prog := bitsProgs h.ms }) (g := i) hP hb
  rw [parse_prog h hx]
  exact e

/-- **The roots the wired program opens a gate under, at an accepted run's checked unit**, are
`CheckedRun.rootsWired`'s: the unit's proof block, without the gates `ω` moves, lies in its replay block without
them (`replayOf_cut`). -/
theorem rootsAtWired_eq (hr : r.Ok h) (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) (g : ℕ) :
    rootsAtWired h ω keep r.s r.u r.seen g = r.rootsWired ω keep g := by
  obtain ⟨Pt, d, _, hx, hP, h2, hin, -, -, -, -, hρ, hrb, -⟩ := hr
  have hW' := rewires_stmt h hx hP hW
  have hP' : partitionWired h ω { r.s with prog := (bitsProgs h.ms).wired ω keep } = some (Pt.cut ω) :=
    congrArg (Option.map fun P : Partition => P.cut ω) hP
  have h2' : twoWired h ω { r.s with prog := (bitsProgs h.ms).wired ω keep } =
      some ({ d with part := d.part.cut ω } : Design.Two) :=
    congrArg (Option.map fun d : Design.Two => ({ d with part := d.part.cut ω } : Design.Two)) h2
  have hρ' : replayOf (Pt.cut ω) (d.part.cut ω) r.u = some r.ρ := replayOf_cut ω hin.2.2.1 hW'.stays hρ
  have hrb' : (d.part.cut ω).blocks[r.ρ]? = some (cutBlock ω r.rb) :=
    (Partition.cut_getElem? d.part ω r.ρ).trans (congrArg (Option.map (cutBlock ω)) hrb)
  have e := Reveal.rootsAt_eq (partitionWired h ω) (twoWired h ω) (iroots []) (v := r.seen) (g := g) hP' h2' hρ' hrb'
  have hp := parse_prog h hx
  unfold CheckedRun.rootsWired
  rw [hp]
  exact e

/-- **The decoded value at a gate of an accepted run's checked unit is the witness's bit at the layout's column.** -/
theorem decOf_checked (hr : r.Ok h) (i : ℕ) :
    decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i =
      _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h i) := by
  have hj := sts_pos hr
  have e0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hj
  have hβ : (layoutFor h).blockOf r.x r.m₁ (r.xS h) r.u = some (r.β h) := blockOf_some hr
  have hy : r.wire h i < 2 ^ r.sts[0].c.kLog := by
    rw [← e0]
    exact wire_lt hr i
  have hw : (layoutFor h).wire r.x r.m₁ (r.xS h) r.u i = 0 * 2 ^ r.sts[0].c.kLog + r.wire h i := by
    rw [Nat.zero_mul, Nat.zero_add]
    rfl
  obtain ⟨_, _, spec, hx, -, -, -, hok, hu, hck, -, -, -, hset⟩ := hr
  have hsx := sessionScoped h r.x r.m₁ (unitStmt h r.m₂ r.u) r.c₀ r.c₁ _ (stmtOkAt_unit h hx hok hu)
  cases hS : FlockDeploy.parse true _root_.Flock.Readable.relation (unitStmt h r.m₂ r.u) with
  | none => simp [Claims.Flock.check, hS] at hck
  | some S => exact decOf_bitAt hset hS (hsx r.ins spec r.sts hset).1 hβ r.wF hj hy hw

/-- **An opening that holds a session's row passes the deployed check under the roots the wired program opens its
gate under**: at an accepted run's checked unit and a frame that reads its ports off the roots, an opening at gate
`g` whose row, salt and path are those of port `S.port` of the unit's instance, at a gate whose slot is `S.slot`,
where the port reads `g`'s value as a registered read or off the interior record (`ReadsRegisteredWired`,
`ReadsInteriorWired`). The row opens the registered read (`rows_open`) or the published tree (`rows_rooted`), the
entry the slot names has that read's or that tree's root (`EntriesAt`), and the frame checks the opening there
(`port_frameAt`, `opens_frame`). -/
theorem opens_slot (hr : r.Ok h) {slot : ℕ → Option Slot} {c : Code} (hf : h.frame = frameAt slot c)
    (h64 : 64 ≤ depth c.own) {S : Address} {g : ℕ} {o : Opening Bool} (hpos : o.pos = g)
    (hsl : slot g = some S.slot) (hn : r.Counted h)
    (hrow : SamplingDeploy.row c.place c.bits o = _root_.Flock.InnerRows.rowBytes r.p r.wF 0 (r.n h) (S.port r.u))
    (hsalt : c.salt o = (_root_.Flock.InnerRows.salt r.p r.wF 0 (r.n h) (S.port r.u)).data.toList)
    (hπ : bytes (c.π o) = pathAt r.ins r.sts (r.n h) (S.port r.u))
    (hrd : r.ReadsRegisteredWired h ω keep S g ∨ r.ReadsInteriorWired h ω keep S g) :
    (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 (r.rootsWired ω keep g) o = true := by
  have hj := sts_pos hr
  have e0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hj
  have hn0 : r.n h < r.sts[0].pub.n := by
    rw [← e0]
    exact hn
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  have hsx := sessionScoped h r.x r.m₁ (unitStmt h r.m₂ r.u) r.c₀ r.c₁ _ (stmtOkAt_unit h hx hok hu)
  obtain ⟨hEr, hEp⟩ := entriesAt_of h (c := c) (depthsAt_of h h64 (rootedDepthsAt h h64)) r.x r.s Pt d r.m₀ r.m₁
    r.m₂ r.c₀ r.c₁ r.rest r.wF r.u r.b r.ρ r.rb r.ins spec r.sts hx hP h2 hin hok hu hck hb hρ hrb hset
  have hrow' : SamplingDeploy.row h.frame.place h.frame.bits o =
      _root_.Flock.InnerRows.rowBytes r.p r.wF 0 (r.n h) (S.port r.u) := by
    rw [hf]
    exact hrow
  have hsalt' : h.frame.salt o = (_root_.Flock.InnerRows.salt r.p r.wF 0 (r.n h) (S.port r.u)).data.toList := by
    rw [hf]
    exact hsalt
  have hπ' : bytes (h.frame.π o) = pathAt r.ins r.sts (r.n h) (S.port r.u) := by
    rw [hf]
    exact hπ
  have hbb : ∀ B : ByteArray, bytes B.data.toList = B := fun B => bytes_bs B
  rcases hrd with ⟨reg, hreg, hfind, hroots, hkey, hsch, hleaf⟩ | ⟨hnone, hc1, hq, hint, hkey, hI⟩
  · obtain ⟨own, hown, hfo⟩ := Option.bind_eq_some_iff.1 hreg
    have hmem : reg ∈ own := Array.mem_of_find?_eq_some hfo
    obtain ⟨e, he, hre, hle, hde, hcap⟩ := hEr own reg hown hmem
    rw [← hroots] at he
    have he' : entryAt (r.rootsWired ω keep g) S.slot = some e := hkey.trans he
    obtain ⟨hlv, hlf⟩ := hleaf e he' hle
    obtain ⟨P, hPr, hPl, hPd, hPs, hat, hfindP⟩ := port_frameAt (c := c) hsl he' hre hlv hde hsch hcap
    have hro := rows_open hset hown hsx hck 0 hj
    rw [← e0] at hro
    obtain ⟨tb, _idx, ht, -, hregs⟩ := hro
    obtain ⟨q', positions, paths, hq', hent, hrows⟩ := hregs reg hmem
    obtain rfl : q' = S.port r.u := Option.some.inj (hq'.symm.trans hfind)
    obtain ⟨-, -, hopens⟩ := hrows (r.n h) hn
    have hpath : pathAt r.ins r.sts (r.n h) (S.port r.u) =
        paths.getD (_root_.Flock.Registered.ref tb r.sts[0]!.c.ports.size (S.port r.u) (r.n h)) ByteArray.empty := by
      simp only [pathAt, hreg, ht, hent]
    refine opens_frame h (reg := P) (pos := leafAt (r.rootsWired ω keep g) S.slot e) ?_ ?_ ?_
    · rw [hf, hpos]
      exact hat
    · rw [hf, hpos]
      exact hfindP
    · rw [hrow', hsalt', _root_.Flock.InnerRows.commitString_data, hbb, hπ', hpath, hlf tb positions paths ht hent]
      unfold _root_.Flock.Registered.opens _root_.Flock.Registered.opensLeaf at hopens ⊢
      rw [hPr, hPl, hPd, hPs]
      exact hopens
  · have hR : r.rootsWired ω keep g = iroots r.x r.seen r.ρ := by
      unfold CheckedRun.rootsWired
      exact ite_eq_left hint
    have hq0 : S.port r.u < r.sts[0].c.ports.size := by
      rw [← e0]
      exact hq
    have hnr : ∀ own, r.ins.registered = some own → ∀ reg ∈ own,
        reg.name ≠ (r.sts[0]!.c.ports[S.port r.u]!).name := by
      intro own hown reg hreg
      have hnf : own.find? (fun reg => reg.name == (r.sts[0]!.c.ports[S.port r.u]!).name) = none := by
        simpa [regNamed, hown] using hnone
      simpa using Array.find?_eq_none.1 hnf reg hreg
    have hrr := rows_rooted hset hsx hck hc1 hj hn0 hq0
    rw [← e0] at hrr
    obtain ⟨pop, hpop, hopens⟩ := hrr
    obtain ⟨hsc, hI'⟩ := hI pop hpop
    have hEp0 := hEp 0 hj (S.port r.u) pop hc1 hq0
    rw [← e0] at hEp0
    obtain ⟨e, he, hre, hde, hcap⟩ := hEp0 hnr hpop
    have he' : entryAt (iroots r.x r.seen r.ρ) S.slot = some e := hkey.trans he
    obtain ⟨hlv, hlf⟩ := hI' e he'
    obtain ⟨P, hPr, hPl, hPd, hPs, hat, hfindP⟩ := port_frameAt (c := c) hsl he' hre hlv hde hsc hcap
    have hpath : pathAt r.ins r.sts (r.n h) (S.port r.u) =
        pubPath r.sts[0]!.c pop (S.port r.u)
          (rowAt r.sts[0]!.c pop (S.port r.u) (unitOf r.ins.read.draw (r.n h))) := by
      simp only [pathAt, hnone, hpop]
    rw [hR]
    refine opens_frame h (reg := P) (pos := leafAt (iroots r.x r.seen r.ρ) S.slot e) ?_ ?_ ?_
    · rw [hf, hpos]
      exact hat
    · rw [hf, hpos]
      exact hfindP
    · rw [hrow', hsalt', _root_.Flock.InnerRows.commitString_data, hbb, hπ', hpath, hlf]
      unfold _root_.Flock.Registered.opens _root_.Flock.Registered.opensLeaf at hopens ⊢
      rw [hPr, hPl, hPd, hPs]
      exact hopens

/-- **The opening of gate `i` at position `ω i` holds the row the unit's session holds `i` in**, read at `ω i`'s
place, which is that of `i`'s address (`row_opening`). -/
theorem row_wired (hr : r.Ok h) {addr : ℕ → Address} {i : ℕ} (hpl : (addr i).place = (addr (ω i)).place)
    (hq : (addr i).port r.u < r.sts[0]!.c.ports.size) (hR : r.RowAt h (addr i) i) :
    SamplingDeploy.row (addressPlace fun g => some (addr g)) bitsOf
        ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool) =
      _root_.Flock.InnerRows.rowBytes r.p r.wF 0 (r.n h) ((addr i).port r.u) := by
  obtain ⟨hv2, hseg⟩ := v2_of_v2Row (v2Row_port hr hq)
  have hp : addressPlace (fun g => some (addr g)) (ω i) = addressPlace (fun g => some (addr g)) i := hpl.symm
  have hat : SamplingDeploy.row (addressPlace fun g => some (addr g)) bitsOf
      ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool) =
        SamplingDeploy.row (addressPlace fun g => some (addr g)) bitsOf (r.opening h (addr i) i).2 :=
    row_at (place := addressPlace fun g => some (addr g)) (place' := addressPlace fun g => some (addr g))
      (bits := bitsOf) (o := ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool))
      (o' := (r.opening h (addr i) i).2) hp rfl rfl
  have hrow : SamplingDeploy.row (addressPlace fun g => some (addr g)) bitsOf (r.opening h (addr i) i).2 =
      _root_.Flock.InnerRows.rowBytes r.p r.wF 0 (r.n h) ((addr i).port r.u) :=
    row_opening (addr := fun g => some (addr g)) (c₀ := h.frame) (S := addr i) (g := i) rfl hv2 hseg hR
  exact hat.trans hrow

/-- **The opening of gate `i` at any position holds the salt of its address's port's row** (`salt_opening`). -/
theorem salt_wired {addr : ℕ → Address} (i p : ℕ) :
    saltOf ({ (r.opening h (addr i) i).2 with pos := p } : Opening Bool) =
      (_root_.Flock.InnerRows.salt r.p r.wF 0 (r.n h) ((addr i).port r.u)).data.toList := by
  have hsalt : saltOf (r.opening h (addr i) i).2 =
      (_root_.Flock.InnerRows.salt r.p r.wF 0 (r.n h) ((addr i).port r.u)).data.toList :=
    salt_opening (h := h) (r := r) (fun g => some (addr g)) h.frame (addr i) i
  exact (saltOf_pos (r.opening h (addr i) i).2 p).trans hsalt

/-- **The opening of gate `i` at any position holds the path of its address's port's row** (`path_opening`). -/
theorem path_wired {addr : ℕ → Address} (i p : ℕ) :
    bytes (pathOf ({ (r.opening h (addr i) i).2 with pos := p } : Opening Bool)) =
      pathAt r.ins r.sts (r.n h) ((addr i).port r.u) := by
  have hπ : bytes (pathOf (r.opening h (addr i) i).2) = pathAt r.ins r.sts (r.n h) ((addr i).port r.u) :=
    path_opening (h := h) (r := r) (fun g => some (addr g)) h.frame (addr i) i
  exact (congrArg bytes (pathOf_pos (r.opening h (addr i) i).2 p)).trans hπ

/-- **A port that reads a gate's value is one of its session's first table's.** -/
theorem port_lt_of_reads {S : Address} {g : ℕ}
    (hrd : r.ReadsRegisteredWired h ω keep S g ∨ r.ReadsInteriorWired h ω keep S g) :
    S.port r.u < r.sts[0]!.c.ports.size := by
  rcases hrd with ⟨_, -, hfind, -⟩ | ⟨-, -, hq, -⟩
  · exact List.mem_range.1 (List.mem_of_find?_eq_some hfind)
  · exact hq

/-- **`row_wired` at SPOT's own frame at the wired program**: the frame's place and bits are the base frame's. -/
theorem row_frame (hr : r.Ok h) {addr : ℕ → Address} {c₀ : Code} {i : ℕ}
    (hpl : (addr i).place = (addr (ω i)).place) (hq : (addr i).port r.u < r.sts[0]!.c.ports.size)
    (hR : r.RowAt h (addr i) i) :
    SamplingDeploy.row (revealCodeWired h ω keep (fun g => some (addr g)) c₀).place
        (revealCodeWired h ω keep (fun g => some (addr g)) c₀).bits
        ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool) =
      _root_.Flock.InnerRows.rowBytes r.p r.wF 0 (r.n h) ((addr i).port r.u) := by
  rw [place_revealCodeWired, bits_revealCodeWired]
  exact row_wired (ω := ω) (addr := addr) (i := i) hr hpl hq hR

/-- **`salt_wired` at SPOT's own frame at the wired program.** -/
theorem salt_frame {addr : ℕ → Address} (c₀ : Code) (i : ℕ) :
    (revealCodeWired h ω keep (fun g => some (addr g)) c₀).salt
        ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool) =
      (_root_.Flock.InnerRows.salt r.p r.wF 0 (r.n h) ((addr i).port r.u)).data.toList := by
  rw [salt_revealCodeWired]
  exact salt_wired (h := h) (r := r) (addr := addr) i (ω i)

/-- **`path_wired` at SPOT's own frame at the wired program.** -/
theorem path_frame {addr : ℕ → Address} (c₀ : Code) (i : ℕ) :
    bytes ((revealCodeWired h ω keep (fun g => some (addr g)) c₀).π
        ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool)) =
      pathAt r.ins r.sts (r.n h) ((addr i).port r.u) := by
  rw [path_revealCodeWired]
  exact path_wired (h := h) (r := r) (addr := addr) i (ω i)

/-- **The address's opening of gate `i`, at position `ω i`, passes** under the roots the wired program opens `ω i`
under, at an accepted run's checked unit and SPOT's own frame at the wired program: the opening holds the row the
unit's session holds `i` in (`row_opening`), read at `ω i`'s place and checked at `ω i`'s slot, which are those of
`i`'s address. -/
theorem opens_at (hr : r.Ok h) {addr : ℕ → Address} {c₀ : Code}
    (hf : h.frame =
      frameAt (slotsOf fun g => some (addr g)) (revealCodeWired h ω keep (fun g => some (addr g)) c₀))
    {i : ℕ} (hpl : (addr i).place = (addr (ω i)).place) (hsl : (addr i).slot = (addr (ω i)).slot)
    (hH : r.AddressHoldsWired h ω keep (addr i) i (ω i)) :
    (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 (r.rootsWired ω keep (ω i))
      ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool) = true := by
  obtain ⟨hn, hR, hrd⟩ := hH
  have hq : (addr i).port r.u < r.sts[0]!.c.ports.size := port_lt_of_reads hrd
  have hslot : slotsOf (fun g => some (addr g)) (ω i) = some (addr i).slot := congrArg some hsl.symm
  have h64 : 64 ≤ depth (revealCodeWired h ω keep (fun g => some (addr g)) c₀).own :=
    Nat.le_of_eq (depth_revealCodeWired h ω keep (fun g => some (addr g)) c₀).symm
  have hpos : ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool).pos = ω i := rfl
  exact opens_slot hr hf h64 (S := addr i) (g := ω i)
    (o := ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool)) hpos hslot hn
    (row_frame (ω := ω) (keep := keep) (addr := addr) (c₀ := c₀) (i := i) hr hpl hq hR)
    (salt_frame (h := h) (ω := ω) (keep := keep) (r := r) (addr := addr) c₀ i)
    (path_frame (h := h) (ω := ω) (keep := keep) (r := r) (addr := addr) c₀ i) hrd

/-- **The reveal at the wired program opens each decoded value a checked unit needs**, at addresses that hold
(`Reveals`'s shape, at one run): for each gate `i` an accepted run's checked unit needs side by side, the reveal at
the wired program holds an opening that passes, at position `ω i`, of the unit's decoded value at `i`, under the
roots the wired program opens `ω i` under. -/
theorem reveals_at (hr : r.Ok h) {addr : ℕ → Address} {c₀ : Code}
    (hf : h.frame =
      frameAt (slotsOf fun g => some (addr g)) (revealCodeWired h ω keep (fun g => some (addr g)) c₀))
    (hcons : ∀ i, (addr i).place = (addr (ω i)).place ∧ (addr i).slot = (addr (ω i)).slot)
    (hholds : ∀ i, r.s.prog.Needs r.b i → r.AddressHoldsWired h ω keep (addr i) i (ω i))
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) {i : ℕ} (hi : r.s.prog.Needs r.b i) :
    ∃ q ∈ revealWired h ω keep h.frame r.s r.u r.seen r.wF,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧ q.2.pos = ω i ∧
        q.2.val = decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i ∧ q.1 = r.rootsWired ω keep (ω i) := by
  have hs : ({ r.s with prog := bitsProgs h.ms } : AuditStatement) = r.s := by
    obtain ⟨_, _, _, hx, -⟩ := hr
    exact congrArg (fun p : Prog => ({ r.s with prog := p } : AuditStatement)) (parse_prog h hx).symm
  have hG : h.frame.reveal { r.s with prog := bitsProgs h.ms } r.u r.seen r.wF = (needsSide h r.s r.u).map fun j =>
      (rootsAtWired h ω keep r.s r.u r.seen (ω j),
        ({ addressOp h (fun g => some (addr g)) r.s r.u r.seen r.wF j [] with pos := ω j } : Opening Bool)) :=
    (congrArg (fun t : AuditStatement => h.frame.reveal t r.u r.seen r.wF) hs).trans
      (congrArg (fun f : Code => f.reveal r.s r.u r.seen r.wF) hf)
  have hm := mem_revealWired h ω keep (R := fun j => rootsAtWired h ω keep r.s r.u r.seen (ω j))
    (op := fun j => addressOp h (fun g => some (addr g)) r.s r.u r.seen r.wF j []) hG ((mem_needsSide hr).2 hi)
  obtain ⟨spec, hset⟩ : ∃ spec, FlockDeploy.setupOf true (unitStmt h r.m₂ r.u) = some (r.ins, spec, r.sts) := by
    obtain ⟨_, _, spec, -, -, -, -, -, -, -, -, -, -, hset⟩ := hr
    exact ⟨spec, hset⟩
  have hop : addressOp h (fun g => some (addr g)) r.s r.u r.seen r.wF i [] = (r.opening h (addr i) i).2 :=
    addressOp_seenAt h (addr := fun g => some (addr g)) (S := addr i) (g := i) rfl hset []
  have hmem : (r.rootsWired ω keep (ω i), ({ (r.opening h (addr i) i).2 with pos := ω i } : Opening Bool)) ∈
      revealWired h ω keep h.frame r.s r.u r.seen r.wF := by
    rw [← hop, ← rootsAtWired_eq hr hW (ω i)]
    exact hm
  exact ⟨_, hmem, opens_at hr hf (hcons i).1 (hcons i).2 (hholds i hi), rfl, (decOf_checked hr i).symm, rfl⟩

/-- **The reveal at the wired program opens each decoded value a checked unit needs**, at SPOT's own frame there
(`BuiltWired`) and under `AddressesWired` (`reveals_at` at the addresses it names). -/
theorem reveals_wired (hr : r.Ok h) (hB : BuiltWired h ω keep) (hA : AddressesWired h ω keep)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) {i : ℕ} (hi : r.s.prog.Needs r.b i) :
    ∃ q ∈ revealWired h ω keep h.frame r.s r.u r.seen r.wF,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true ∧ q.2.pos = ω i ∧
        q.2.val = decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i ∧ q.1 = r.rootsWired ω keep (ω i) := by
  obtain ⟨c₀, hf⟩ := hB
  rw [addressMapWired_eq h ω keep hA] at hf
  exact reveals_at hr hf (Exists.choose_spec hA).1 (fun j hj => (Exists.choose_spec hA).2 r hr j hj) hW hi

/-- **Values that agree with the openings that pass make a checked unit's proof block correct at the wired
program** (the module's docstring), at one accepted run's checked unit. `hdec` is what the unit's session statement
gives (`decoded_rewired`): values, equal through `ω` to the unit's decoded values at every gate its proof block holds
or reads, on which the block, without the gates `ω` moves, is correct at the wired program, where two such gates
that `ω` sends to one gate decode to one value. That agreement is proved here from the values' own agreement with
the openings: each of the two gates is needed side by side, so the reveal holds an opening of each at the one
position under the one set of roots. -/
theorem correct_wired (hr : r.Ok h) (hB : BuiltWired h ω keep) (hA : AddressesWired h ω keep)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hdec : (∀ i j, r.s.prog.Touches r.b i → r.s.prog.Touches r.b j → ω i = ω j →
        decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i = decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u j) →
      ∃ v : ℕ → Bool,
        (∀ i, r.s.prog.Touches r.b i → v (ω i) = decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i) ∧
          (r.s.prog.wired ω keep).Correct (fun g => some (v g)) (cutBlock ω r.b))
    (val : ℕ → Option Bool) (ι : ℕ → Bool)
    (hv : ∀ q ∈ revealWired h ω keep h.frame r.s r.u r.seen r.wF,
      (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true →
        (q.1 = r.s.roots → val q.2.pos = some q.2.val) ∧ (q.1 = iroots r.x r.seen r.ρ → ι q.2.pos = q.2.val)) :
    (r.s.prog.wired ω keep).Correct ((r.s.prog.wired ω keep).refIn (cutBlock ω r.rb) val ι) (cutBlock ω r.b) := by
  obtain ⟨Pt, hW'⟩ : ∃ Pt, r.s.prog.Rewires Pt ω keep := by
    obtain ⟨Pt, _, _, hx, hP, -⟩ := hr
    exact ⟨Pt, rewires_stmt h hx hP hW⟩
  have hne : ∀ i, r.s.prog.Needs r.b i →
      (r.s.prog.wired ω keep).refIn (cutBlock ω r.rb) val ι (ω i) =
        some (decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i) := by
    intro i hi
    obtain ⟨q, hq, hop, hpos, hval, hroot⟩ := reveals_wired hr hB hA hW hi
    obtain ⟨h1, h2⟩ := hv q hq hop
    unfold Prog.refIn
    by_cases hc : ω i ∈ cutBlock ω r.rb ∧ (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) (ω i) = false
    · have hR : q.1 = iroots r.x r.seen r.ρ := by
        rw [hroot]
        unfold CheckedRun.rootsWired
        exact ite_eq_left hc
      have e := h2 hR
      rw [hpos, hval] at e
      rw [ite_eq_left hc]
      exact congrArg some e
    · have hR : q.1 = r.s.roots := by
        rw [hroot]
        unfold CheckedRun.rootsWired
        exact ite_eq_right hc
      have e := h1 hR
      rw [hpos, hval] at e
      rw [ite_eq_right hc]
      exact e
  have hag : ∀ i j, r.s.prog.Touches r.b i → r.s.prog.Touches r.b j → ω i = ω j →
      decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u i = decOf (layoutFor h) r.x r.m₁ (r.xS h) r.wF r.u j := by
    intro i j hi hj hij
    by_cases hii : ω i = i
    · by_cases hjj : ω j = j
      · rw [(hii.symm.trans hij).trans hjj]
      · have e1 := hne i (hW'.needs_of_target hi hjj (hij.symm.trans hii))
        have e2 := hne j (hW'.needs_of_moved hj hjj)
        rw [hij, e2] at e1
        exact (Option.some.inj e1).symm
    · have e1 := hne i (hW'.needs_of_moved hi hii)
      by_cases hjj : ω j = j
      · have e2 := hne j (hW'.needs_of_target hj hii (hij.trans hjj))
        rw [hij, e2] at e1
        exact (Option.some.inj e1).symm
      · have e2 := hne j (hW'.needs_of_moved hj hjj)
        rw [hij, e2] at e1
        exact (Option.some.inj e1).symm
  obtain ⟨v, hvt, hc⟩ := hdec hag
  refine (r.s.prog.wired ω keep).correct_of_needs (fun g hg => ?_) hc
  obtain ⟨i, hi, rfl⟩ := Prog.needs_wired hW'.input hW'.out hg
  rw [hne i hi]
  exact congrArg some (hvt i (Prog.touches_of_needs hi))

end CheckedRun

/-! ## Every checked unit of an accepted run -/

/-- **The session statements open every unit they check, at the wired program** (`StmtOpens`'s shape there): at a
run's statement in the two-stage scope, session statements `m₂` that the verifier's check passes and a Flock witness
in the relation of unit `u`'s statement, each unit `u` the two stages check has a proof block `b` in a replay block
`ρ` with gates `rb`, and any values `val` and interior values `ι` that agree with the openings the witness reveals at
the wired program and that pass (under the run's roots, and the interior roots `m₁` states for `ρ`) make `b`, without
the gates `ω` moves, correct at the wired program on the values of `ρ` without them. -/
def StmtOpensWired : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    ∃ b ρ rb, Pt.blocks[u]? = some b ∧ replayOf Pt d.part u = some ρ ∧ d.part.blocks[ρ]? = some rb ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ revealWired h ω keep h.frame s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF,
          (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (seenAt m₀ m₁ m₂ c₀ c₁ rest) ρ → ι q.2.pos = q.2.val)) →
        (s.prog.wired ω keep).Correct ((s.prog.wired ω keep).refIn (cutBlock ω rb) val ι) (cutBlock ω b)

/-- **`StmtOpensWired` at SPOT's own frame from `AddressesWired`**, where `ω` rewires the members' programs
(`Prog.Rewires`): each checked unit's decoded values make its proof block correct at the wired program
(`decoded_rewired`), and the reveal opens them (`CheckedRun.correct_wired`). -/
theorem stmtOpensWired (hB : BuiltWired h ω keep) (hA : AddressesWired h ω keep)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) : StmtOpensWired h ω keep := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u hx hP hd hin hok hu hck
  have hlt : u < Pt.blocks.length := by
    have := mem_checked_lt hu
    rwa [units_base h hx hP] at this
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hin.2.2.2.1 u hlt)
  obtain ⟨rb, hrb⟩ := replay_block hρ
  refine ⟨Pt.blocks[u], ρ, rb, List.getElem?_eq_getElem hlt, hρ, hrb, fun val ι hv => ?_⟩
  obtain ⟨ins, spec, sts, hset⟩ :
      ∃ ins spec sts, FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) := by
    have hs := sessionOk_of_stmtOkAt h (stmtOkAt_unit h hx hok hu)
    simp only [sessionOk, Bool.and_eq_true] at hs
    obtain ⟨ins, spec, sts, hset, -⟩ := zkOk_sound hs.1.1.1
    exact ⟨ins, spec, sts, hset⟩
  have hr : CheckedRun.Ok h ⟨x, s, m₀, m₁, m₂, c₀, c₁, rest, wF, u, Pt.blocks[u], ρ, rb, ins, sts⟩ :=
    ⟨Pt, d, spec, hx, hP, hd, hin, hok, hu, hck, List.getElem?_eq_getElem hlt, hρ, hrb, hset⟩
  exact CheckedRun.correct_wired hr hB hA hW
    (fun hag => decoded_rewired h hx hP hd hin hok hu hck (List.getElem?_eq_getElem hlt) hρ hrb hW hag) val ι hv

/-- **Every held code whose frame is set to SPOT's own at the wired program is built there**: the address map
doesn't read the frame. -/
theorem builtWired_spot (c₀ : Code) :
    BuiltWired
      { h with
        frame := frameAt (slotsOf (addressMapWired h ω keep))
          (revealCodeWired h ω keep (addressMapWired h ω keep) c₀) } ω keep :=
  ⟨c₀, rfl⟩

end SpotFlock
