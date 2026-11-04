import Proofs.NetworkCertifier.KSWired
import Proofs.NetworkCertifier.SampledAuditAccepted

/-!
# The wires a drawn proof block's session reveals

`SessionWires` (`Proofs.NetworkCertifier.KSWired`) is `SessionOpens` with each wired gate a drawn proof block holds or reads opened to its
writer's value (`WireRight`). Where the session proves the wire itself, its statement reading the writer's committed
leaf as the wired input, a Flock witness in its relation reveals it: revealed in the session's Flock witness, so nothing
reaches the verifier. `RevealsWires H D opens wires` says so at an accepted run, at an opening check `opens`: for each
wired gate of a checked unit's proof block, the reveal holds an opening at the gate, under the statement's roots, that
`opens` accepts and whose value is its writer's committed one (`WriterRevealed`): the writer is no constant gate, and
an opening in the same reveal at the writer, of the same value, is one `opens` accepts.

* `WiresRevealed H D wires`: `RevealsWires` at the commitment's own check, `D.commit.Opens H`.
* `RevealsWires.mono`: a check that accepts more keeps the wires revealed.
* `sessionWires_of`: `SessionOpens` and `WiresRevealed` give `SessionWires`. Values agreeing with the passing openings
  hold the opened value at the wired gate, and the same value at its writer (`val_of_writerRevealed`).
* `ksWired_of_revealed`: sampling at the incorrect set (`KSBy` at the wired check) from `KS_of`'s hypotheses with
  `WiresRevealed` beside `SessionOpens`, by `KSWired_of`.
* `wiresRevealed_none`: with no wires there is nothing to reveal.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The writer `w` is revealed at the value `v`** among the openings `os`, in the session's Flock witness; nothing
reaches the verifier. `w` is no constant gate, and an opening among `os` under `roots`, at `w` and of value `v`, is one
`opens` accepts: the writer's committed leaf, which the session reads. -/
def WriterRevealed (opens : List Msg → Opening Bool → Prop) (C : Prog) (roots : List Msg)
    (os : List (List Msg × Opening Bool)) (w : ℕ) (v : Bool) : Prop :=
  (∀ c, C.gates[w]? ≠ some (.const c)) ∧ ∃ q ∈ os, q.1 = roots ∧ q.2.pos = w ∧ q.2.val = v ∧ opens q.1 q.2

theorem WriterRevealed.mono {opens opens' : List Msg → Opening Bool → Prop} (h : ∀ r o, opens r o → opens' r o)
    {C : Prog} {roots : List Msg} {os : List (List Msg × Opening Bool)} {w : ℕ} {v : Bool}
    (hw : WriterRevealed opens C roots os w v) : WriterRevealed opens' C roots os w v := by
  obtain ⟨hnc, q, hq, hq1, hqw, hqv, hqo⟩ := hw
  exact ⟨hnc, q, hq, hq1, hqw, hqv, h _ _ hqo⟩

/-- Values agreeing with the openings `opens` accepts under `roots` hold a revealed writer, no constant gate, at its
revealed value. -/
theorem val_of_writerRevealed {opens : List Msg → Opening Bool → Prop} {C : Prog} {roots : List Msg}
    {os : List (List Msg × Opening Bool)} {w : ℕ} {v : Bool} {val : ℕ → Option Bool}
    (hcons : ∀ q ∈ os, opens q.1 q.2 → q.1 = roots → val q.2.pos = some q.2.val)
    (h : WriterRevealed opens C roots os w v) : (∀ c, C.gates[w]? ≠ some (.const c)) ∧ val w = some v := by
  obtain ⟨hnc, q, hq, hq1, hqw, hqv, hqo⟩ := h
  refine ⟨hnc, ?_⟩
  rw [← hqw, ← hqv]
  exact hcons q hq hqo hq1

section Hyps

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **The wires are revealed at the check `opens`**, in the session's Flock witness; nothing reaches the verifier. On a
statement in scope, at an accepted run that checks slot `u` and shows its session, a Flock witness in the session's
relation reveals, for each wired gate `g` the proof block at `u` holds or reads, an opening at `g` under the statement's
roots that `opens` accepts, whose value is its writer's, revealed there (`WriterRevealed`). -/
def RevealsWires (H : Msg → Msg) (D : Deploy Coin Col T) (opens : List Msg → Opening Bool → Prop)
    (wires : AuditStatement → ℕ → Option ℕ) : Prop :=
  ∀ x s Pt (Ω : Type) (P : Prover Ω) ω r u xS vS wF,
    D.parse x = some s → D.partition s = some Pt → InScope Pt s →
    (run H P D.V x ω r).out = .acc → (∃ p : Fin (nPU D x), (D.checked x r)[p.val]? = some u) →
    D.session x u (run H P D.V x ω r).seen = some (xS, vS) → Claims.Flock.check D.flock.parse xS wF = true →
    ∀ b, Pt.blocks[u]? = some b → ∀ g, (g ∈ b ∨ ∃ y ∈ b, g ∈ (s.prog.gates[y]?.map BGate.reads).getD []) →
      ∀ w, wires s g = some w →
        ∃ q ∈ D.reveal s u (run H P D.V x ω r).seen wF, q.1 = s.roots ∧ q.2.pos = g ∧ opens q.1 q.2 ∧
          WriterRevealed opens s.prog s.roots (D.reveal s u (run H P D.V x ω r).seen wF) w q.2.val

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **A check that accepts more keeps the wires revealed.** -/
theorem RevealsWires.mono {H : Msg → Msg} {D : Deploy Coin Col T} {opens opens' : List Msg → Opening Bool → Prop}
    {wires : AuditStatement → ℕ → Option ℕ} (h : ∀ r o, opens r o → opens' r o) (hw : RevealsWires H D opens wires) :
    RevealsWires H D opens' wires := by
  intro x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck b hb g hg w hgw
  obtain ⟨q, hq, hq1, hqg, hqo, hwr⟩ := hw x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck b hb g hg w hgw
  exact ⟨q, hq, hq1, hqg, h _ _ hqo, hwr.mono h⟩

/-- **The wires are revealed** at the commitment's own check (`RevealsWires` at `D.commit.Opens H`). -/
def WiresRevealed (H : Msg → Msg) (D : Deploy Coin Col T) (wires : AuditStatement → ℕ → Option ℕ) : Prop :=
  RevealsWires H D (fun roots o => D.commit.Opens H roots o = true) wires

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **`SessionWires` from `SessionOpens` and the revealed wires**: values agreeing with the passing openings make the
proof block correct (`SessionOpens`), and hold at each wired gate it holds or reads its writer's committed value. -/
theorem sessionWires_of {H : Msg → Msg} {D : Deploy Coin Col T} {iroots : Msg → Seen → ℕ → List Msg}
    {wires : AuditStatement → ℕ → Option ℕ} (hopens : SessionOpens H D iroots) (hw : WiresRevealed H D wires) :
    SessionWires H D iroots wires := by
  intro x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck
  obtain ⟨b, hb, hok⟩ := hopens x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck
  refine ⟨b, hb, fun val ι hcons => ⟨hok val ι hcons, fun g hg => ?_⟩⟩
  intro w hgw
  obtain ⟨q, hq, hq1, hqg, hqo, hwr⟩ := hw x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck b hb g hg w hgw
  obtain ⟨hnc, hwv⟩ := val_of_writerRevealed (fun q' hq' ho h1 => (hcons q' hq' ho).1 h1) hwr
  refine ⟨hnc, q.2.val, ?_, hwv⟩
  rw [← hqg]
  exact (hcons q hq hqo).1 hq1

set_option synthInstance.maxSize 1024 in
/-- **Sampling at the incorrect set from sampling's hypotheses and the revealed wires**: `KSWired_of`, with
`SessionWires` from `SessionOpens` and `WiresRevealed` (`sessionWires_of`). -/
theorem ksWired_of_revealed (wires : AuditStatement → ℕ → Option ℕ) {H : Msg → Msg} {D : Deploy Coin Col T}
    (iroots : Msg → Seen → ℕ → List Msg) (hbind : D.commit.Binds H) (hsess : SessionKS H D)
    (hopens : SessionOpens H D iroots) (hw : WiresRevealed H D wires) (hfix : InteriorFixed H D iroots)
    (hdraw : DrawLaw D) (hrefuse : Refuses H D) (hε : ∀ x, 0 ≤ D.εtable x) : KSBy H D (wiredOk wires) :=
  KSWired_of wires iroots hbind hsess (sessionWires_of hopens hw) hfix hdraw hrefuse hε

omit [Fintype Coin] [Nonempty Coin] [DecidableEq Col] [Zero Col] in
/-- **With no wires there is nothing to reveal.** -/
theorem wiresRevealed_none (H : Msg → Msg) (D : Deploy Coin Col T) : WiresRevealed H D fun _ _ => none := by
  intro _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ w hw
  simp at hw

end Hyps

end NetTiming
