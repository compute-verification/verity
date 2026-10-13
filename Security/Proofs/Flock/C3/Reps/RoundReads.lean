import Proofs.Flock.C3.Compose.RsFacts
import Std.Data.String.ToNat

/-!
# `RoundReads` and `WRoundReads` at the verifier's message-round ports

At `own := fun j => ownAt ss j`, the verifier's message-round ports of each statement, the reads-file entries
`RoundReads` and `WRoundReads` ask for are the statement's own (`RecVStar.ownPorts`): R's opening part reads `m<i>`
under `rec-opening-c<i>` at row 0 (`mem_ownPorts_opening`), and a residuals' part reads `m<m>` under `rec-c<m>` at rows
0 and 1 (`mem_ownPorts_inner`), wherever the statement has the port (`mIndex_m`). The accepted sessions' reads hold them
(`OwnIn`, `ownIn_checked`, from `checkOwn`). The rest is taken as hypotheses, U's links (`C3/Reps/RoundLinks.lean`):
the statement at a copy or part is an opening part or a residuals' part (`hform`), the circuit's port `m<k>` is one of
the statement's (`hport`, a `checkStmt` refusal), the unit a copy or rep reads at (`hun`) and the stage selector there
(`hsel`).

* `roundReads_ownAt`: `RoundReads` at `ownAt ss`.
* `wRoundReads_ownAt`: `WRoundReads` at `ownAt ss`.
-/

namespace FlockC3.Reps.RoundReads

open Flock FlockSoundness FlockSoundness.Discharge FlockVBridge.Compiled FlockC3.PubRows
open FlockC3.Compose (OpenCopy RoundReads WRoundReads)
open FlockVBridge.Family (partsOf)

/-- **`m<k>`'s index is `k`.** -/
theorem mIndex_m (k : ℕ) : RecVStar.mIndex s!"m{k}" = some k := by
  have hd : ((s!"m{k}").drop 1).copy = k.repr := by
    apply String.toList_injective
    rw [String.toList_copy_drop]
    simp [toString]
  have hs : (s!"m{k}").startsWith "m" = true := by simp [toString]
  unfold RecVStar.mIndex
  simp only [hs, ↓reduceIte]
  exact (congrArg String.toNat? hd).trans (Nat.toNat?_repr k)

/-- **An opening part's port `m<i>` is its message-round port** under `rec-opening-c<i>`, read at row 0. -/
theorem mem_ownPorts_opening {m c j nv i n : ℕ} {S : RecOpen.GfS} {ports : List (String × ℕ)}
    (h : (s!"m{i}", n) ∈ ports) :
    (s!"m{i}", s!"rec-opening-c{i}", #[some 0]) ∈ RecVStar.ownPorts (.opening m c j S ports nv) :=
  List.mem_filterMap.2 ⟨_, h, by simp only [mIndex_m]⟩

/-- **A residuals' part's port `m<m>` is its message-round port** under `rec-c<m>`, read at rows 0 and 1. -/
theorem mem_ownPorts_inner {j m n : ℕ} {nm : String} {P : VStar.Part} (h : (s!"m{m}", n) ∈ P.S.ports) :
    (s!"m{m}", s!"rec-c{m}", #[some 0, some 1]) ∈ RecVStar.ownPorts (.inner j nm P) :=
  List.mem_filterMap.2 ⟨_, h, by simp only [mIndex_m, Option.map_some]⟩

theorem mem_ownAt {ss : Array VStar.Stmt} {j : ℕ} {s : VStar.Stmt} (hj : ss[j]? = some s)
    {q : String × String × Array (Option ℕ)} (hq : q ∈ RecVStar.ownPorts s) :
    { (default : Flock.Registered.Port) with name := q.1, value := q.2.1, reads := q.2.2 } ∈ ownAt ss j := by
  simp only [ownAt, hj, Option.map_some, Option.getD_some, List.map_toArray, List.mem_toArray]
  exact List.mem_map_of_mem (List.mem_append_left _ hq)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`RoundReads` at the verifier's message-round ports**: at a copy, the session's statement is an opening part
(`hform`) whose ports hold the circuit's `m<i>` (`hport`), so `ownAt` has `m<i>` under `rec-opening-c<i>` at row 0,
the copy's unit's (`hun`), where stage `c₀ + i` reads (`hsel`). -/
theorem roundReads_ownAt {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {ss : Array VStar.Stmt} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hform : ∀ gs s un j p, OpenCopy zs dg N k Rw sh w gs s un j p →
      ∃ m c j' S₁ ports nv, ss[(s : ℕ)]? = some (.opening m c j' S₁ ports nv))
    (hport : ∀ (s : Fin S) m c j' S₁ ports nv, ss[(s : ℕ)]? = some (.opening m c j' S₁ ports nv) →
      ∀ (q : Fin (Vs s).c.ports.size) (i : ℕ), (Vs s).c.ports[q].name = s!"m{i}" → ∃ n, (s!"m{i}", n) ∈ ports)
    (hun : ∀ gs s un j p, OpenCopy zs dg N k Rw sh w gs s un j p → un.val = 0)
    (hsel : ∀ i < 2, sel (c₀ + i) s!"rec-opening-c{i}" 0) :
    RoundReads zs dg N k Rw sh w (fun j => ownAt ss j) sel c₀ := by
  intro i hi gs s un j p hO q hq
  obtain ⟨m, c, j', S₁, ports, nv, hs⟩ := hform gs s un j p hO
  obtain ⟨n, hn⟩ := hport s m c j' S₁ ports nv hs q i hq
  refine ⟨_, mem_ownAt hs (mem_ownPorts_opening hn), rfl, rfl, 0, ?_, hsel i hi⟩
  rw [hun gs s un j p hO]
  rfl

/-- **`WRoundReads` at the verifier's message-round ports**: part `p`'s session's statement is a residuals' part
(`hform`) whose ports hold the circuit's `m<m>` (`hport`), so `ownAt` has `m<m>` under `rec-c<m>` at rows 0 and 1,
rep `ρ`'s unit's row `ρ` (`hun`), where rep `ρ`'s stage of round `m` reads (`hsel`). -/
theorem wRoundReads_ownAt {sh : Flock.Shape} {ss : Array VStar.Stmt} {sel : ℕ → String → ℕ → Prop}
    {stage : Fin 2 → ℕ → ℕ} {sess : ℕ → Fin S} {un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv}
    (hform : ∀ p < (partsOf sh).size, ∃ j nm P, ss[(sess p : ℕ)]? = some (.inner j nm P))
    (hport : ∀ p < (partsOf sh).size, ∀ j nm P, ss[(sess p : ℕ)]? = some (.inner j nm P) →
      ∀ (q : Fin (Vs (sess p)).c.ports.size) (m : ℕ), (Vs (sess p)).c.ports[q].name = s!"m{m}" →
        ∃ n, (s!"m{m}", n) ∈ P.S.ports)
    (hun : ∀ p < (partsOf sh).size, ∀ ρ, (un p ρ).val = ρ.val)
    (hsel : ∀ p < (partsOf sh).size, ∀ (m : ℕ) (ρ : Fin 2) (q : Fin (Vs (sess p)).c.ports.size),
      (Vs (sess p)).c.ports[q].name = s!"m{m}" → sel (stage ρ m) s!"rec-c{m}" ρ.val) :
    WRoundReads sh (fun j => ownAt ss j) sel stage sess un := by
  intro p hp m ρ q hq
  obtain ⟨j, nm, P, hs⟩ := hform p hp
  obtain ⟨n, hn⟩ := hport p hp j nm P hs q m hq
  refine ⟨_, mem_ownAt hs (mem_ownPorts_inner hn), rfl, rfl, ρ.val, ?_, hsel p hp m ρ q hq⟩
  rw [hun p hp ρ]
  match ρ with
  | 0 => rfl
  | 1 => rfl

end FlockC3.Reps.RoundReads
