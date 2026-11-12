module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SpotFlock.FrameIsBuilt
public import Proofs.Flock.SamplingGeneric.WiredScope

/-!
# A checked unit's decoded values at a wired program

At SPOT[Flock]'s code the session statements hold each checked unit's circuit gates at SPOT's own layout
(`embeds_layoutFor`), so the values the layout decodes make the unit's proof block correct in the statement's program
(`decodes_of_embeds`). `decoded_wired` reads that through any map `ω` on the program's gates: the block, cut to the
gates `ω` leaves alone, is correct at the wired program (`Prog.wired`) on values that are the unit's decoded values
through `ω`.

The conditions on the map, the statement's program and the block name no run: `ω` moves only input gates, each to a
returned gate that is no constant and is outside the block. One condition is on the run: the unit's decoded values
agree at any two of the block's gates and the wires they read that `ω` sends to one gate. It is where the wired
program says more than the units side by side: a unit that reads one gate of its writer at two of its own input
gates is correct there only if its session holds one value at both, which Flock's relation for the unit's own
circuit does not say and the commitments the session opens do.

`scope_rewired` and `decoded_rewired` state the scope and the decoded values at the members' programs under the one
proposition `Prog.Rewires` on the map, which names no run.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)

variable (h : Held)

/-- **A checked unit's decoded values make its cut block correct at the wired program**: at a run's statement in the
two-stage scope, session statements `m₂` the verifier's check passes and a Flock witness in the relation of unit
`u`'s statement, there are values, equal through `ω` to the unit's decoded values at every gate of `S`, on which
`u`'s proof block, cut, is correct at the statement's program wired by `ω`. `S` holds the block's gates and the wires
they read, and the unit's decoded values agree at any two gates of `S` that `ω` sends to one gate. -/
theorem decoded_wired {ω : ℕ → ℕ} {keep : ℕ → Bool} {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} {m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {wF : Claims.Flock.Wit} {u ρ : ℕ}
    {b rb : List ℕ}
    (hx : parse h x = some s) (hP : partition h s = some Pt) (h2 : two h s = some d) (hsc : InScopeTwo Pt d s)
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true)
    (hb : Pt.blocks[u]? = some b) (hρ : replayOf Pt d.part u = some ρ) (hrb : d.part.blocks[ρ]? = some rb)
    (hin : ∀ i, ω i ≠ i → s.prog.gates[i]? = some .input)
    (hnc : ∀ i, ω i ≠ i → ∀ c, s.prog.gates[ω i]? ≠ some (.const c))
    (hw : ∀ i, ω i ≠ i → ω i ∈ s.prog.outs)
    (htgt : ∀ g ∈ b, ∀ y : BGate, s.prog.gates[g]? = some y → ∀ i ∈ y.reads, ω i ≠ i → ω i ∉ b)
    (S : ℕ → Prop) (hSb : ∀ g ∈ b, S g)
    (hSr : ∀ g ∈ b, ∀ y : BGate, s.prog.gates[g]? = some y → ∀ i ∈ y.reads, S i)
    (hagree : ∀ i j, S i → S j → ω i = ω j →
      decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u i = decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u j) :
    ∃ v : ℕ → Bool, (∀ i, S i → v (ω i) = decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u i) ∧
      (s.prog.wired ω keep).Correct (fun g => some (v g)) (cutBlock ω b) := by
  have hd := decodes_of_embeds h (layoutFor h) (embeds_layoutFor h) x s Pt d m₁ m₂ c₀ c₁ wF u b ρ rb hx hP h2 hsc
    hok hu hck hb hρ hrb
  have hc : s.prog.Correct (fun g => some (decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u g)) b :=
    s.prog.correct_of_congr (fun g _ => by simp only [Prog.refIn, ite_self])
      (fun g _ y _ i _ _ => by simp only [Prog.refIn, ite_self]) hd
  exact Prog.exists_correct_wired hin hnc hw htgt S hSb hSr hagree hc

/-- **The conditions on the map, at a run's statement**: a map that rewires the members' programs over the members'
partition (`Prog.Rewires`) rewires the statement's program over its partition. -/
theorem rewires_stmt {ω : ℕ → ℕ} {keep : ℕ → Bool} {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) : s.prog.Rewires Pt ω keep := by
  rw [parse_prog h hx, partition_eq h hP]
  exact hW

/-- **The two-stage scope at the code's wired program**: a run's statement in the two-stage scope is in it at the
members' programs wired by `ω`, its partition and its design's replay partition cut, where `ω` rewires the members'
programs (`Prog.Rewires`). The scope check that ran on the units side by side is the only one read. -/
theorem scope_rewired {ω : ℕ → ℕ} {keep : ℕ → Bool} {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) (hsc : InScopeTwo Pt d s) :
    InScopeTwo (Pt.cut ω) { d with part := d.part.cut ω } { s with prog := s.prog.wired ω keep } :=
  (rewires_stmt h hx hP hW).inScopeTwo hsc

/-- **A checked unit's decoded values at the code's wired program** (`decoded_wired`), where `ω` rewires the members'
programs (`Prog.Rewires`) and the unit's decoded values agree at any two gates its proof block holds or reads that
`ω` sends to one gate: there are values, equal through `ω` to the unit's decoded values at every gate the block holds
or reads, on which the block, cut, is correct at the wired program. -/
theorem decoded_rewired {ω : ℕ → ℕ} {keep : ℕ → Bool} {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} {m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {wF : Claims.Flock.Wit} {u ρ : ℕ}
    {b rb : List ℕ}
    (hx : parse h x = some s) (hP : partition h s = some Pt) (h2 : two h s = some d) (hsc : InScopeTwo Pt d s)
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hck : Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true)
    (hb : Pt.blocks[u]? = some b) (hρ : replayOf Pt d.part u = some ρ) (hrb : d.part.blocks[ρ]? = some rb)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep)
    (hagree : ∀ i j, s.prog.Touches b i → s.prog.Touches b j → ω i = ω j →
      decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u i = decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u j) :
    ∃ v : ℕ → Bool,
      (∀ i, s.prog.Touches b i → v (ω i) = decOf (layoutFor h) x m₁ (unitStmt h m₂ u) wF u i) ∧
        (s.prog.wired ω keep).Correct (fun g => some (v g)) (cutBlock ω b) := by
  have hW' := rewires_stmt h hx hP hW
  have hbm : b ∈ Pt.blocks := List.mem_of_getElem? hb
  exact decoded_wired h hx hP h2 hsc hok hu hck hb hρ hrb hW'.input hW'.noConst hW'.out (hW'.off b hbm)
    (s.prog.Touches b) (fun _ hg => Or.inl hg) (fun g hg y hy _ hi => Or.inr ⟨g, hg, y, hy, hi⟩) hagree

end SpotFlock
