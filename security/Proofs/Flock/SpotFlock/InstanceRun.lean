module

public import Proofs.Flock.SpotFlock.InstanceBoundary
public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.Bridge.Members
public import Proofs.Flock.Claims.Sampling.Select.SecondWitness
public import Proofs.Flock.Claims.Sampling.Select.Prefix
public import Proofs.Flock.Claims.Sampling.Select.Steps
public import Proofs.Flock.Claims.Sampling.Select.Take
public import Proofs.Flock.Claims.Complete.Encode
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Init.Data.Range.Basic
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Lean.Data.Json.Basic

/-!
# A run the two-stage instance accepts, given an honest session

`SpotFlock.Instance.ok_held S`: at `held R bnd0 desc q c₀`, some run is accepted with a checked unit (`CheckedRun.Ok`)
wherever `S : HonestSession R desc q c₀`, the facts about the run that read `R`'s Flock statement, holds.

What the registration decides of the run holds for every `R`, at the zero coin `coin0` (every element `0`) in rounds 0
and 1:
* the first stage samples replay block `0` (`sample_coin0`), the boundary's one unit;
* the second stage samples unit `0` of its two (`selectSecond_coin0`), so its sample is some (`secondDraw_held`) and
  unit `0` is checked (`checked_coin0`).

Each stage reads 82 bytes of its coin (`2·k·width + 64`, a width of 9 bytes). The kernel evaluates each on 82 zero bytes
(`select_zeros`, `second_zeros`), with its sorts, canonical JSON, `repeat` and map their fuelled twins (`qsort_fuel`,
`canon_fuel`, `onWith_fuel`, `subset_fn`), and a stage reads only those bytes (`select_append_subset`,
`selectSecond_append`). The rest of the run's acceptance follows from `Ref`: the design (`two_held`) and its scope
(`scopeOk_sound` at `Ref.scope`), the checked unit's proof block and replay block, and round 2's message stating one
session statement (`stmtsOf_m₂`).

`HonestSession` holds the facts that read `R`'s Flock statement: the run's statement `x` and round-1 record `m₁`, the
session statement `xS` and its witness `wF`; `xS` has `R.x`'s rounds, is in scope (`sessionOk`) and reads the run's
(`readsOk`, at the zero coins), Flock's check accepts it at `wF`, and it sets up.

## What this does not show

That an `HonestSession` exists. Each of its facts is about `R`'s Flock statement or about round 2's statement check
against it, and none is proved for any statement: a concrete `Ref` would supply them, the honest prover's session at the
sample `selectSecond_coin0` fixes. Until then the two-stage guarantee's non-vacuity at this instance is not proved.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private Spot.Draw.STRATUM_KEYS Spot.Draw.WORK_STRATUM_KEYS Spot.Draw.lawKeys Spot.Draw.stratum? from Spot.Draw

namespace SpotFlock.Instance

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Properties.Spot.SecurityProofs (bind_eq_ok throw_bind_ne onWith_fuel canon_fuel qsort_fuel subset_fn)
open Lean (Json)

/-! ## The zero coin -/

/-- **The zero coin**: every element `0`. -/
def coin0 : FlockDeploy.Coin := fun _ => 0

theorem toBytes_zero_mem {b : UInt8} (hb : b ∈ (0 : _root_.Flock.F128).toBytes.data.toList) : b = 0 := by
  have h16 : ∀ i : Fin 16, (if i.val < 8 then ((0 : UInt64) >>> (8 * i.val).toUInt64).toUInt8
      else ((0 : UInt64) >>> (8 * (i.val - 8)).toUInt64).toUInt8) = 0 := by decide +kernel
  simp only [_root_.Flock.F128.toBytes, Array.mem_toList_iff, Array.mem_ofFn] at hb
  obtain ⟨i, rfl⟩ := hb
  exact h16 i

theorem bytes_coin0_mem {b : UInt8} (hb : b ∈ FlockDeploy.Coin.bytes coin0) : b = 0 := by
  unfold FlockDeploy.Coin.bytes at hb
  obtain ⟨i, -, hb⟩ := List.mem_flatMap.1 hb
  exact toBytes_zero_mem hb

theorem coinByte_coin0 (i : Fin SpotFlock.L) : SpotFlock.coinByte coin0 i = 0 := by
  have h : (FlockDeploy.Coin.bytes coin0).getD i.val 0 = 0 := by
    rw [List.getD_eq_getElem?_getD]
    cases h : (FlockDeploy.Coin.bytes coin0)[i.val]? with
    | none => rfl
    | some b => rw [Option.getD_some, bytes_coin0_mem (List.mem_of_getElem? h)]
  unfold SpotFlock.coinByte
  simp only [h]
  rfl

/-- **The zero coin's bytes**, as `Select` reads them: `L` zeros. -/
theorem src_coin0 : SpotFlock.src coin0 = ⟨Array.replicate SpotFlock.L 0⟩ := by
  unfold SpotFlock.src Properties.Spot.bytesOf
  refine congrArg ByteArray.mk (Array.ext (by simp) fun i h1 h2 => ?_)
  simp only [Array.getElem_ofFn, Array.getElem_replicate, coinByte_coin0]
  rfl

/-- **The zero coin's bytes are 82 zeros, then more.** -/
theorem src_coin0_split :
    SpotFlock.src coin0 = ⟨Array.replicate 82 0⟩ ++ ⟨Array.replicate (SpotFlock.L - 82) 0⟩ := by
  rw [src_coin0]
  apply ByteArray.ext
  rw [ByteArray.data_append, Array.replicate_append_replicate]
  rfl

theorem toOption_eq {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

/-! ## The two stages at the zero coin -/

/-- **The first stage's sample on 82 zero bytes**: `subset:1` over one unit samples it, reading the 82 bytes. -/
theorem drawFirst_zeros :
    (Spot.Draw.drawUnits (.subset 1) 1 ⟨Array.replicate 82 0⟩).toOption = some (#[0], 82) := by
  unfold Spot.Draw.drawUnits _root_.Flock.Draw.drawWith
  simp only [onWith_fuel _ _ _ 2, subset_fn]
  decide +kernel

theorem drawFirst_coin0 : Spot.Draw.drawUnits (.subset 1) 1 (SpotFlock.src coin0) = .ok (#[0], 82) := by
  rw [src_coin0_split]
  exact Properties.Spot.SecurityProofs.drawUnits_append (toOption_eq drawFirst_zeros)

/-- **The boundary's first stage on 82 zero bytes** proves its one unit. -/
theorem select_zeros :
    (Spot.Select.select (.law "subset:1") program0 query0 none none ⟨Array.replicate 82 0⟩).toOption.map (·.2.1) =
      some #[0] := by
  simp only [Spot.Select.select, units0, bind, Except.bind, designOver0, deriveDesign0, Option.map_none,
    Option.getD_none, flockDesign0, toOption_eq drawFirst_zeros]
  unfold Spot.Draw.checkDraw Spot.Draw.checkDrawFull Spot.Draw.sameLaw Spot.Draw.lawOf Spot.Draw.lawKeys Spot.keys?
    Spot.sortStrings Spot.jeq
  simp only [canon_fuel 16, qsort_fuel]
  decide +kernel

/-- **The boundary's first stage reads only the bytes its sample reads.** -/
theorem select_append_subset {ξ t : ByteArray} {r : Array ℕ × ℕ}
    (h : Spot.Draw.drawUnits (.subset 1) 1 ξ = .ok r) :
    Spot.Select.select (.law "subset:1") program0 query0 none none (ξ ++ t) =
      Spot.Select.select (.law "subset:1") program0 query0 none none ξ := by
  simp only [Spot.Select.select, units0, bind, Except.bind, designOver0, deriveDesign0, Option.map_none,
    Option.getD_none, flockDesign0, h, Properties.Spot.SecurityProofs.drawUnits_append h]

/-- **The boundary's first stage at the zero coin** proves its one unit. -/
theorem select_coin0 :
    (Spot.Select.select (reg bnd0).first.rule (reg bnd0).first.program (reg bnd0).first.query
      (reg bnd0).first.table (reg bnd0).first.closure (SpotFlock.src coin0)).toOption.map (·.2.1) = some #[0] := by
  show (Spot.Select.select (.law "subset:1") program0 query0 none none (SpotFlock.src coin0)).toOption.map (·.2.1) =
    some #[0]
  rw [src_coin0_split, select_append_subset (toOption_eq drawFirst_zeros)]
  exact select_zeros

/-- **The replay sample at the zero coin**: replay block `0`. -/
theorem sample_coin0 : (reg bnd0).sample (SpotFlock.src coin0) = #[0] := by
  have h := select_coin0
  unfold RegTwo.sample
  revert h
  cases Spot.Select.select (reg bnd0).first.rule (reg bnd0).first.program (reg bnd0).first.query
    (reg bnd0).first.table (reg bnd0).first.closure (SpotFlock.src coin0) with
  | error e => intro h; simp [Except.toOption] at h
  | ok r =>
    obtain ⟨d, p, rd⟩ := r
    intro h
    simpa [Except.toOption] using h

/-- **The second stage reads only the bytes it needs**: an accepted second stage on `ξ` is the same on `ξ ++ t`. -/
theorem selectSecond_append {sample ns ks : Array ℕ} {ξ t : ByteArray} {r : Json × ℕ}
    (h : Spot.Select.selectSecond sample ns ks ξ = .ok r) : Spot.Select.selectSecond sample ns ks (ξ ++ t) = .ok r := by
  unfold Spot.Select.selectSecond Spot.Draw.drawFirstRead at h ⊢
  cases hq : Spot.Select.secondProblem sample ns ks with
  | some e => rw [hq] at h; dsimp only at h; exact absurd h (throw_bind_ne _ _ _)
  | none =>
    rw [hq] at h
    dsimp only at h ⊢
    obtain ⟨⟨draw, read⟩, hd, h⟩ := bind_eq_ok.1 h
    refine bind_eq_ok.2 ⟨(draw, read), ?_, h⟩
    obtain ⟨parsed, hp, hd⟩ := bind_eq_ok.1 hd
    obtain ⟨⟨units, read'⟩, hu, hd⟩ := bind_eq_ok.1 hd
    exact bind_eq_ok.2 ⟨parsed, hp, bind_eq_ok.2 ⟨(units, read'),
      Properties.Spot.SecurityProofs.drawUnits_append hu, hd⟩⟩

/-- **The second stage on 82 zero bytes**: over one sampled replay block of two proof blocks sampling one, it proves
the first. -/
theorem second_zeros :
    ((Spot.Select.selectSecond #[0] #[2] #[1] ⟨Array.replicate 82 0⟩).toOption.map fun r => Spot.Draw.proved r.1) =
      some #[0] := by
  unfold Spot.Draw.proved Spot.Select.selectSecond Spot.Draw.drawFirstRead Spot.Draw.drawUnits
    _root_.Flock.Draw.drawWith _root_.Flock.Draw.stratifiedWith Spot.Draw.flockLaw Spot.Draw.strataProblem
    _root_.Flock.Draw.workProblem Spot.Draw.checkDraw Spot.Draw.checkDrawFull Spot.Draw.strataMessage
    Spot.Draw.sameLaw Spot.Draw.lawOf Spot.Draw.lawKeys Spot.Draw.stratum? Spot.Draw.STRATUM_KEYS
    Spot.Draw.WORK_STRATUM_KEYS Spot.keys? Spot.sortStrings Spot.jeq
  simp only [onWith_fuel _ _ _ 2, canon_fuel 16, qsort_fuel, subset_fn]
  decide +kernel

/-- **The second stage at the zero coins** proves interior unit `0`. -/
theorem selectSecond_coin0 : ∃ d rd, Spot.Select.selectSecond #[0] #[2] #[1] (SpotFlock.src coin0) = .ok (d, rd) ∧
    Spot.Draw.proved d = #[0] := by
  have h := second_zeros
  cases hs : Spot.Select.selectSecond #[0] #[2] #[1] ⟨Array.replicate 82 0⟩ with
  | error e => rw [hs] at h; simp [Except.toOption] at h
  | ok r =>
    obtain ⟨d, rd⟩ := r
    rw [hs] at h
    refine ⟨d, rd, ?_, by simpa [Except.toOption] using h⟩
    rw [src_coin0_split]
    exact selectSecond_append hs

/-- **Unit `0` is checked** at the zero coins, of the two units of the one replay block. -/
theorem checked_coin0 : 0 ∈ (reg bnd0).checked 2 (SpotFlock.src coin0) (SpotFlock.src coin0) := by
  obtain ⟨d, rd, hs, hp⟩ := selectSecond_coin0
  unfold RegTwo.checked
  rw [sample_coin0, show (reg bnd0).ns = #[2] from rfl, show (reg bnd0).ks = #[1] from rfl, hs]
  dsimp only
  rw [hp]
  decide +kernel

variable (R : Ref) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The second draw at the zero coins is some.** -/
theorem secondDraw_held : ∃ d, secondDraw (held R bnd0 desc q c₀) coin0 coin0 = some d := by
  obtain ⟨d, rd, hs, -⟩ := selectSecond_coin0
  have hs' : Spot.Select.selectSecond ((held R bnd0 desc q c₀).reg.sample (SpotFlock.src coin0))
      (held R bnd0 desc q c₀).reg.ns (held R bnd0 desc q c₀).reg.ks (SpotFlock.src coin0) = .ok (d, rd) := by
    rw [show (held R bnd0 desc q c₀).reg.sample (SpotFlock.src coin0) = #[0] from sample_coin0]
    exact hs
  exact ⟨d, by simp only [secondDraw, hs']⟩

/-! ## The run -/

/-- **The round-2 message stating one session statement** `xS`. -/
def m₂Of (xS : Msg) : Msg := FlockDeploy.encList (fun r => FlockDeploy.encBytes ⟨r.toArray⟩) [xS]

theorem stmtsOf_m₂ {xS : Msg} (hx : xS.length < 2 ^ 64) : stmtsOf (m₂Of xS) = some [xS] := by
  have hr : (FlockDeploy.rdList FlockDeploy.rdBytes).run (m₂Of xS) = some ([⟨xS.toArray⟩], []) := by
    have := FlockComplete.rdList_encList (r := FlockDeploy.rdBytes) (e := FlockDeploy.encBytes)
      (l := [⟨xS.toArray⟩]) (fun a ha rest => by
        rw [List.mem_singleton] at ha
        subst ha
        exact FlockComplete.rdBytes_encBytes (by simpa [ByteArray.size] using hx) rest) (by simp) []
    rw [List.append_nil] at this
    exact this
  unfold stmtsOf
  rw [hr]
  dsimp only
  split
  · rfl
  · rename_i hc
    exact (hc rfl).elim

/-- **An honest session of `held R bnd0 desc q c₀` at the zero coins**: what a run's acceptance reads of `R`'s Flock
statement. The run's statement `x` and round-1 record `m₁`; the session statement `xS`, with `R.x`'s rounds, in scope
(`sessionOk`), reading the run's (`readsOk`) and setting up; and the witness `wF` Flock's check accepts at it. -/
structure HonestSession where
  x : Msg
  m₁ : Msg
  xS : Msg
  wF : Claims.Flock.Wit
  small : xS.length < 2 ^ 64
  rounds : (FlockDeploy.V true).k xS = (FlockDeploy.V true).k R.x
  sess : sessionOk (held R bnd0 desc q c₀) 0 xS = true
  reads : readsOk (held R bnd0 desc q c₀) 0 x m₁ xS coin0 coin0 = true
  check : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) xS wF = true
  setup : (FlockDeploy.setupOf true xS).isSome = true

variable {R desc q c₀}

theorem decode_session (S : HonestSession R desc q c₀) : ∃ st, decode S.x = some st := by
  cases hx : decode S.x with
  | none =>
    have h := S.reads
    unfold readsOk at h
    rw [hx] at h
    simp at h
  | some st => exact ⟨st, rfl⟩

/-- **Unit `0`'s session statement** is the one `m₂Of xS` states. -/
theorem unitStmt_held {xS : Msg} (hx : xS.length < 2 ^ 64) : unitStmt (held R bnd0 desc q c₀) (m₂Of xS) 0 = xS := by
  unfold unitStmt stmtAt
  rw [stmtsOf_m₂ hx]
  rfl

/-- **Round 2's check passes** the honest session's statement at the zero coins. -/
theorem stmtOk_held (S : HonestSession R desc q c₀) {st : Roots} (hst : decode S.x = some st) :
    stmtOk (held R bnd0 desc q c₀) S.x S.m₁ (m₂Of S.xS) coin0 coin0 = true := by
  have hparse : parse (held R bnd0 desc q c₀) S.x = some
      { desc := desc, prog := bitsProgs (ms2 R.x), query := q,
        design := (replayDesign (held R bnd0 desc q c₀)).getD ⟨[]⟩, roots := st.roots } := by
    unfold parse
    rw [hst]
    rfl
  unfold stmtOk
  rw [hparse, stmtsOf_m₂ S.small]
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
  refine ⟨⟨⟨rfl, two_held_isSome R bnd0 desc q c₀ _⟩, rfl⟩, fun t ht => ?_⟩
  obtain rfl : t = 0 := by
    have : t < 1 := ht
    omega
  exact ⟨⟨S.rounds, S.sess⟩, S.reads⟩

/-- **A run the instance accepts**: at an honest session `S`, the run of `S`'s messages at the zero coins is accepted
and its unit `0` is checked. -/
theorem ok_held (S : HonestSession R desc q c₀) : ∃ r : CheckedRun, r.Ok (held R bnd0 desc q c₀) := by
  obtain ⟨st, hst⟩ := decode_session S
  obtain ⟨⟨ins, spec, sts⟩, hsetup⟩ := Option.isSome_iff_exists.1 S.setup
  have hparse : parse (held R bnd0 desc q c₀) S.x = some
      { desc := desc, prog := bitsProgs (ms2 R.x), query := q,
        design := (replayDesign (held R bnd0 desc q c₀)).getD ⟨[]⟩, roots := st.roots } := by
    unfold parse
    rw [hst]
    rfl
  have hscope : InScopeTwo (membersPart (ms2 R.x))
      ⟨filesReplay (ms2 R.x) #[2] (oneLayout bnd0.gs), design1, [1]⟩
      { desc := desc, prog := bitsProgs (ms2 R.x), query := q,
        design := (replayDesign (held R bnd0 desc q c₀)).getD ⟨[]⟩, roots := st.roots } :=
    scopeOk_sound (by rw [filesReplay_oneLayout]; exact R.scope) _ rfl
  have hlen : (membersPart (ms2 R.x)).blocks.length = 2 :=
    (Flock.Bridge.Members.length_blocks_membersPart (ms := ms2 R.x)).trans rfl
  have h0 : 0 < (membersPart (ms2 R.x)).blocks.length := by rw [hlen]; omega
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hscope.2.2.2.1 0 h0)
  have hρlt : ρ < (filesReplay (ms2 R.x) #[2] (oneLayout bnd0.gs)).blocks.length := by
    unfold replayOf at hρ
    exact List.mem_range.1 (List.mem_of_find?_eq_some hρ)
  have hcount : unitCount (held R bnd0 desc q c₀) S.x = 2 := by
    unfold unitCount
    rw [hparse]
    exact hlen
  refine ⟨⟨S.x, _, [], S.m₁, m₂Of S.xS, coin0, coin0, [], S.wF, 0, _, ρ, _, ins, sts⟩, membersPart (ms2 R.x),
    ⟨filesReplay (ms2 R.x) #[2] (oneLayout bnd0.gs), design1, [1]⟩, spec,
    hparse, rfl, two_held R bnd0 desc q c₀ _, hscope, stmtOk_held S hst, ?_, ?_,
    List.getElem?_eq_getElem h0, hρ, List.getElem?_eq_getElem hρlt, ?_⟩
  · show 0 ∈ (reg bnd0).checked (unitCount (held R bnd0 desc q c₀) S.x) (SpotFlock.src coin0) (SpotFlock.src coin0)
    rw [hcount]
    exact checked_coin0
  · show Claims.Flock.check _ (unitStmt (held R bnd0 desc q c₀) (m₂Of S.xS) 0) S.wF = true
    rw [unitStmt_held S.small]
    exact S.check
  · show FlockDeploy.setupOf true (unitStmt (held R bnd0 desc q c₀) (m₂Of S.xS) 0) = some (ins, spec, sts)
    rw [unitStmt_held S.small]
    exact hsetup

end SpotFlock.Instance
