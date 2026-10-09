import PouwCheck

/-!
# PoUW's window check: what its Lean code (`PouwCheck.protocol`) does

The properties of the executable check (`verity/core/protocols/pouw/lean/check/`, the port of
`verity/core/protocols/pouw/core_check.py`), stated against `protocol.check` as it runs, for every input of
the types its decoders produce. Each is a `Prop`, and `Properties.Proofs.Pouw.Check` proves every one.

`AcceptMeans` is the check's whole meaning: what it accepts is a registration that keeps PoUW's rule
(`RegistrationKeeps`), which the service accepted with the profile and draw the rule fixes (`ServiceAccepted`), and it
then publishes the window's items at the verifier's values and the service's draw. The rest are its consequences a reader
checks first, and what the other verdicts mean. That the check accepts every such registration is completeness,
`Properties.Proofs.Pouw.Check.acceptMeans_complete`.

What the window's draw guarantees (`Properties.Pouw.Window.DrawMisses`) is the window's property, not the check's: the
check establishes that the service drew exactly `K′ = kPrime ε δ` distinct units of the registered `N`, all passed.
-/

namespace Properties.Pouw.Check

open PouwCheck
open PouwWindow (Work point level floorPoint)

/-! ## Acceptance -/

/-- The registration keeps PoUW's rule, with `f` the level, sampling item and `root_W` it fixes: a served window in the
verifier's context registering exactly `window.REGISTERED`; a work level whose credited tiles are at least the range's
low end and whose next level is at most the floor point of its high end; each item the verifier's own at that level
(`expected`, compared canonically); the population `N = point(level + 1)`; the law `subset:K′` with `K′` the window's
exact `kPrime`; and the check's query and scheme. -/
def RegistrationKeeps (p : Params) (c : Registration) (f : Fixed) : Prop :=
  (c.window.kind = "served" ∧ c.window.context = p.context) ∧
  (keys c.window.items).getD [] = registeredSorted ∧
  levelOf c.window.items = some f.work.level ∧
  f.work.level + 1 ≤ level p.gridBits (floorPoint p.gridBits p.hi) ∧ p.lo ≤ point p.gridBits f.work.level ∧
  f.work = Work.at p.gridBits f.work.level ∧
  PouwWindow.sampling (toRat p.eps) (toRat p.delta) f.work p.check.tileWork p.gridBits = .ok f.sampling ∧
  f.rootW = (c.roots.find? (·.port = "W")).map (·.root) ∧
  (∀ kv ∈ expected p f, cs (get c.window.items kv.1) = cs kv.2) ∧
  c.population = f.work.n ∧
  cs c.law = cs (subsetLaw f.sampling.k) ∧
  c.query = p.query ∧ c.scheme = p.check.scheme

/-- The service accepted this registration (`e` is one-stage's answer for its digest) with `prof` and `draw`: the profile
of the registered Program over one level, the registered partition's `N` units drawn `subset:K′`, at the window's δ,
with `K′` units drawn; and a draw `subset:K′` over `N` of `K′` distinct units below `N`, ascending. -/
def ServiceAccepted (p : Params) (c : Registration) (f : Fixed) (e : Evidence) (prof : Core.Integrity)
    (draw : UnitDraw) : Prop :=
  e.protocol = oneStage ∧ e.commitment = some (Flock.hexOf c.digest) ∧ e.said = .accept prof draw ∧
  prof.programs = #[c.program] ∧
  prof.levels = #[{ name := unitLevel, query := some c.partition, size := f.work.n, draw := some (.subset f.sampling.k) }] ∧
  prof.delta = p.delta ∧ prof.drawn = #[(unitLevel, f.sampling.k)] ∧
  draw.law = .subset f.sampling.k ∧ draw.population = f.work.n ∧
  draw.units.size = f.sampling.k ∧ Ascending draw.units.toList ∧ ∀ u ∈ draw.units.toList, u < f.work.n

/-- What the check accepts is a registration that keeps PoUW's rule, which the service accepted as the rule fixes; it
then publishes the check item, the level's work, the sampling item, `root_W` and the drawn units, and the profile is the
service's. -/
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : Registration) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    protocol.check p c e = .accept pub prof →
      ∃ f draw, RegistrationKeeps p c f ∧ ServiceAccepted p c f e prof draw ∧
        pub = ⟨p.check, f.work, f.sampling, f.rootW, draw.units⟩

/-- An accepted window is credited a grid level inside the declared range: its credited tiles are at least `lo`, and
the tiles it pads to at most `hi`. -/
def AcceptedInRange : Prop :=
  ∀ (p : Params) (c : Registration) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    protocol.check p c e = .accept pub prof → p.lo ≤ pub.work.credited ∧ pub.work.n ≤ p.hi

/-- An accepted window's draw is `K′` distinct units of its `N`, `K′` the window's exact `kPrime` at its ε and δ, and the
registration was for that law over that population. -/
def AcceptedDraw : Prop :=
  ∀ (p : Params) (c : Registration) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    protocol.check p c e = .accept pub prof →
      pub.work = Work.at p.gridBits pub.work.level ∧
      PouwWindow.kPrime (toRat p.eps) (toRat p.delta) ((pub.work.credited : Int) * p.check.tileWork) pub.work.n
        p.check.tileWork = .ok pub.sampling.k ∧
      c.population = pub.work.n ∧ cs c.law = cs (subsetLaw pub.sampling.k) ∧
      pub.drawn.size = pub.sampling.k ∧ Ascending pub.drawn.toList ∧ ∀ u ∈ pub.drawn.toList, u < pub.work.n

/-- An accepted window's outcome is the service's, for this registration: one-stage's answer citing its digest, and the
profile published is the one the service gave. -/
def AcceptedIsTheService's : Prop :=
  ∀ (p : Params) (c : Registration) (e : Evidence) (pub : Public) (prof : Core.Integrity),
    protocol.check p c e = .accept pub prof →
      e.protocol = oneStage ∧ e.commitment = some (Flock.hexOf (protocol.digest c)) ∧
      ∃ draw, e.said = .accept prof draw ∧ pub.drawn = draw.units

/-- What acceptance publishes is `window.PUBLIC`, item for item in its order. -/
def PublishesPublic : Prop := ∀ pub : Public, pub.entries.map (·.1) = PouwWindow.publicItems.map (·.item)

/-- The public outputs and the profile are never `null`, so `Core.Answer.ofJson` reads an accepted answer back. -/
def NeverNull : Prop :=
  ∀ (pub : Public) (prof : Core.Integrity), protocol.publicJson pub ≠ .null ∧ protocol.profileJson prof ≠ .null

/-! ## The other verdicts -/

/-- The check is undecided only when the service is, on a registration that keeps PoUW's rule and an answer for it;
the reason is the service's. -/
def UndecidedOnlyFromService : Prop :=
  ∀ (p : Params) (c : Registration) (e : Evidence) (r : String),
    protocol.check p c e = .undecided r →
      (∃ f, RegistrationKeeps p c f) ∧ e.protocol = oneStage ∧ e.commitment = some (Flock.hexOf c.digest) ∧
        ∃ r₀, e.said = .undecided r₀ ∧ r = "service: " ++ r₀

/-! ## The commitment -/

/-- The digest is one-stage's registration digest, SHA-512 of the format, a zero byte and the canonical record. -/
def DigestIsOneStage's : Prop :=
  ∀ c : Registration, protocol.digest c = Flock.Sha512.hash (c.format.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

/-- A decoded registration keeps the bytes it was read from, so its digest is of exactly those bytes. -/
def CommitmentKeepsBytes : Prop := ∀ (b : ByteArray) (c : Registration), protocol.commitment b = .ok c → c.bytes = b

end Properties.Pouw.Check
