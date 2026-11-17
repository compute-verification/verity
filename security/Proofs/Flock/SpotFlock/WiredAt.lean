module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.SpotFlock.FrameIsBuilt
public import Proofs.Flock.SamplingGeneric.WiredScope

/-! A checked unit's decoded values make its cut proof block correct at a wired program (`decoded_wired`, `decoded_rewired`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)

variable (h : Held)

/-- The one condition on the run: the unit's decoded values agree at any two gates of `S` that `ω` sends to one gate,
which Flock's relation for the unit's own circuit does not say and the commitments the session opens do. -/
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

theorem rewires_stmt {ω : ℕ → ℕ} {keep : ℕ → Bool} {x : Msg} {s : AuditStatement} {Pt : Partition}
    (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) : s.prog.Rewires Pt ω keep := by
  rw [parse_prog h hx, partition_eq h hP]
  exact hW

theorem scope_rewired {ω : ℕ → ℕ} {keep : ℕ → Bool} {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : parse h x = some s) (hP : partition h s = some Pt)
    (hW : (bitsProgs h.ms).Rewires (membersPart h.ms) ω keep) (hsc : InScopeTwo Pt d s) :
    InScopeTwo (Pt.cut ω) { d with part := d.part.cut ω } { s with prog := s.prog.wired ω keep } :=
  (rewires_stmt h hx hP hW).inScopeTwo hsc

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
