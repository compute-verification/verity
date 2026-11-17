module

public import Proofs.Flock.SpotFlock.AddressesWired

/-! The five facts of the address premise at the wired program, each about one address map and nothing else, so each is
proved on its own. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

namespace CheckedRun

variable (r : CheckedRun h.stage)

def WidthAt (S : Address) : Prop :=
  S.place.1 = 16 * (r.sts[0]!.c.ports[S.port r.u]!).words

def CopyAt (S : Address) (i : ℕ) : Prop :=
  _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.n h / r.sts[0]!.c.g)
      (FlockSoundness.Typed.msgCol r.sts[0]!.c (r.n h % r.sts[0]!.c.g) (S.port r.u) (S.place.2 / 8) (S.place.2 % 8)) =
    _root_.Flock.InnerRows.bitAt r.p r.wF 0 (r.β h) (r.wire h i)

def RowHolds (S : Address) (g : ℕ) : Prop :=
  r.Counted h ∧
    (∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      ∀ e, entryAt (r.rootsWired ω keep g) S.slot = some e → e.leaves = reg.leaves →
        leavesAt (r.rootsWired ω keep g) S.slot e = reg.leaves ∧
        ∀ t positions paths, r.sts[0]!.pub.tables = some t →
          _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths) →
          leafAt (r.rootsWired ω keep g) S.slot e =
            positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size (S.port r.u) (r.n h)) 0) ∧
    (regNamed r.ins r.sts (S.port r.u) = none →
      r.ins.read.I.count ≤ 1 ∧
      ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
        ∀ e, entryAt (iroots r.x r.seen r.ρ) S.slot = some e →
          leavesAt (iroots r.x r.seen r.ρ) S.slot e = (pubPort r.sts[0]!.c pop (S.port r.u)).leaves ∧
          leafAt (iroots r.x r.seen r.ρ) S.slot e =
            rowAt r.sts[0]!.c pop (S.port r.u) (unitOf r.ins.read.draw (r.n h)))

def EntryHolds (S : Address) (g : ℕ) : Prop :=
  (∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      (List.range r.sts[0]!.c.ports.size).find? (fun q => r.sts[0]!.c.ports[q]!.name == reg.name) =
          some (S.port r.u) ∧
        entryAt (r.rootsWired ω keep g) S.slot =
          keyAt (r.rootsWired ω keep g)
            (readKeyAt h r.x r.m₁ r.c₀ r.c₁ (templateOf h.ms r.u) reg.name reg.value) ∧
        S.slot.schema = reg.schema) ∧
    (regNamed r.ins r.sts (S.port r.u) = none →
      S.port r.u < r.sts[0]!.c.ports.size ∧
        entryAt (iroots r.x r.seen r.ρ) S.slot =
          keyAt (iroots r.x r.seen r.ρ)
            (.interior (h.names.getD (templateOf h.ms r.u) "") (r.sts[0]!.c.ports[S.port r.u]!).name) ∧
        ∀ pop, _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0]!.c r.ins.read.I.publicFile = .ok pop →
          S.slot.schema = (pubPort r.sts[0]!.c pop (S.port r.u)).schema)

def RootsHold (S : Address) (g : ℕ) : Prop :=
  (∀ reg, regNamed r.ins r.sts (S.port r.u) = some reg →
      r.rootsWired ω keep g = readRoots h r.x r.m₁ r.c₀ r.c₁ r.ρ (templateOf h.ms r.u) reg.name reg.value) ∧
    (regNamed r.ins r.sts (S.port r.u) = none →
      g ∈ cutBlock ω r.rb ∧ (r.s.prog.wired ω keep).Boundary (cutBlock ω r.rb) g = false)

end CheckedRun

variable (addr : ℕ → Address)

def WidthFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.WidthAt h (addr i)

def CopyFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.CopyAt h (addr i) i

def RowFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RowHolds h ω keep (addr i) (ω i)

def EntryFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.EntryHolds h ω keep (addr i) (ω i)

def OneTreeFact : Prop :=
  (∀ i, (addr i).place = (addr (ω i)).place ∧ (addr i).slot = (addr (ω i)).slot) ∧
    ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RootsHold h ω keep (addr i) (ω i)

end SpotFlock
