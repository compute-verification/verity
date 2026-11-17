module

public import Proofs.Flock.SpotFlock.RowsOpen
public import Proofs.Flock.Claims.Bridge.Transports

/-! The frame opens the session's rows: `FrameIs` gives `FrameRows` and so `SessionOpensTwo` with `Embeds`
(`sessionOpensTwo_of_frameIs`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Certifier.Bytes (bs)

theorem leaf_commitString (r y : List UInt8) :
    bytes (FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 (FlockSoundness.Binding.hm96Input r y))) =
      _root_.Flock.Hm96.Default512.treeLeaf (bytes (FlockSoundness.Binding.commitString r y)) := by
  rw [bytes_digBytes, FlockSoundness.Refine.H512_val]
  show _root_.Flock.Sha512.hash ⟨(FlockSoundness.Refine.K512.leafPrefix ++
      FlockSoundness.Binding.commitString r y).toArray⟩ =
    _root_.Flock.Sha512.hash
      (_root_.Flock.Hm96.Default512.leafPrefix ++ bytes (FlockSoundness.Binding.commitString r y))
  congr 1

variable (h : Held)

theorem opens_frame {roots : List Msg} {o : Opening Bool} {reg : _root_.Flock.Registered.Port} {pos : ℕ}
    (hat : h.frame.at_ roots o.pos = some (bs reg.root, pos))
    (hfind : (h.frame.ownAt roots o.pos).find? (fun P => decide (bs P.root = bs reg.root)) = some reg)
    (hop : _root_.Flock.Registered.opens reg
      (bytes (FlockSoundness.Binding.commitString (SamplingDeploy.row h.frame.place h.frame.bits o) (h.frame.salt o)))
      pos (bytes (h.frame.π o)) = true) :
    (SamplingDeploy.commit (code h)).Opens FlockDeploy.sha512 roots o = true :=
  (opens_sha512 (code h) roots o).2 ⟨(bs reg.root, pos), reg, hat, hfind, by rw [leaf_commitString]; exact hop⟩

/-- Either port is the frame's own, read off the roots, so it need agree with `reg` or `pubPort` only on the four fields
a registered read reads. A session of more than one table needs only the first branch: round 2 refuses a session
split across Flock tables (`second-tables`). -/
def FrameIs (L : Layout) : Prop :=
  ∀ x s Pt d (m₀ m₁ m₂ : Msg) (c₀ : FlockDeploy.Coin) (c₁ : Stage h) (rest : List (Msg × Out)) wF u b ρ rb ins spec sts,
    parse h x = some s → partition h s = some Pt → two h s = some d → InScopeTwo Pt d s →
    stmtOk h x m₁ m₂ c₀ c₁ = true → u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁) →
    Claims.Flock.check (FlockDeploy.parse true _root_.Flock.Readable.relation) (unitStmt h m₂ u) wF = true →
    Pt.blocks[u]? = some b → replayOf Pt d.part u = some ρ → d.part.blocks[ρ]? = some rb →
    FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts) →
    ∀ g, s.prog.Needs b g → ∃ β j y n q, L.blockOf x m₁ (unitStmt h m₂ u) u = some β ∧ ∃ hj : j < sts.size,
      y < 2 ^ sts[j].c.kLog ∧ L.wire x m₁ (unitStmt h m₂ u) u g = j * 2 ^ sts[j].c.kLog + y ∧ n < sts[j].pub.n ∧
      let o := Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u
        (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g
      o.2.val = _root_.Flock.InnerRows.bitAt { ins.read with sts } wF j β y ∧
        SamplingDeploy.row h.frame.place h.frame.bits o.2 =
          _root_.Flock.InnerRows.rowBytes { ins.read with sts } wF j n q ∧
        h.frame.salt o.2 = (_root_.Flock.InnerRows.salt { ins.read with sts } wF j n q).data.toList ∧
        ((∃ own reg, ins.registered = some own ∧ reg ∈ own ∧
          (List.range sts[j].c.ports.size).find? (fun q => sts[j].c.ports[q]!.name == reg.name) = some q ∧
          ∀ t positions paths, sts[j].pub.tables = some t →
            _root_.Flock.Registered.entry sts[j].pub reg = .ok (positions, paths) →
            ∃ P : _root_.Flock.Registered.Port, P.root = reg.root ∧ P.leaves = reg.leaves ∧
              P.domain = reg.domain ∧ P.schema = reg.schema ∧
              h.frame.at_ o.1 g =
                some (bs P.root, positions.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) 0) ∧
              (h.frame.ownAt o.1 g).find? (fun P' => decide (bs P'.root = bs P.root)) = some P ∧
              bytes (h.frame.π o.2) =
                paths.getD (_root_.Flock.Registered.ref t sts[j].c.ports.size q n) ByteArray.empty) ∨
        (ins.read.I.count ≤ 1 ∧ q < sts[j].c.ports.size ∧ ∀ pop,
          _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop →
          ∃ P : _root_.Flock.Registered.Port, P.root = (pubPort sts[j].c pop q).root ∧
            P.leaves = (pubPort sts[j].c pop q).leaves ∧ P.domain = (pubPort sts[j].c pop q).domain ∧
            P.schema = (pubPort sts[j].c pop q).schema ∧
            h.frame.at_ o.1 g = some (bs P.root, rowAt sts[j].c pop q (unitOf ins.read.draw n)) ∧
            (h.frame.ownAt o.1 g).find? (fun P' => decide (bs P'.root = bs P.root)) = some P ∧
            bytes (h.frame.π o.2) = pubPath sts[j].c pop q (rowAt sts[j].c pop q (unitOf ins.read.draw n))))

theorem frameRows_of_frameIs (L : Layout) (hF : FrameIs h L) : FrameRows h L := by
  intro x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  obtain ⟨β, j, y, n, q, hβ, hj, hy, hw, hn, hval, hrow, hsalt, hchk⟩ :=
    hF x s Pt d m₀ m₁ m₂ c₀ c₁ rest wF u b ρ rb ins spec sts hx hP h2 hin hok hu hck hb hρ hrb hset g hg
  refine ⟨β, j, y, n, q, hβ, hj, hy, hw, hn,
    Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g,
    Reveal.openAt_mem (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) hP hb hg, rfl, hval,
    Reveal.rootsAt_eq (partition h) (two h) (iroots []) hP h2 hρ hrb, ?_⟩
  have hb : ∀ B : ByteArray, bytes B.data.toList = B := fun B => bytes_bs B
  rcases hchk with ⟨own, reg, hown, hreg, hq, hread⟩ | ⟨hc1, hq, hrooted⟩
  · refine Or.inl ⟨own, reg, hown, hreg, hq, fun t positions paths ht hent hop => ?_⟩
    obtain ⟨P, hPr, hPl, hPd, hPs, hat, hfind, hπ⟩ := hread t positions paths ht hent
    refine opens_frame h (o := (Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u
      (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g).2) hat hfind ?_
    rw [hrow, hsalt, _root_.Flock.InnerRows.commitString_data, hb, hπ]
    unfold _root_.Flock.Registered.opens _root_.Flock.Registered.opensLeaf at hop ⊢
    rw [hPr, hPl, hPd, hPs]
    exact hop
  · refine Or.inr ⟨hc1, hq, fun pop hpop hop => ?_⟩
    obtain ⟨P, hPr, hPl, hPd, hPs, hat, hfind, hπ⟩ := hrooted pop hpop
    refine opens_frame h (o := (Reveal.openAt (partition h) (two h) (iroots []) (Reveal.frameAt h.frame) s u
      (seenAt m₀ m₁ m₂ c₀ c₁ rest) wF g).2) hat hfind ?_
    rw [hrow, hsalt, _root_.Flock.InnerRows.commitString_data, hb, hπ]
    unfold _root_.Flock.Registered.opens _root_.Flock.Registered.opensLeaf at hop ⊢
    rw [hPr, hPl, hPd, hPs]
    exact hop

theorem sessionOpensTwo_of_frameIs (L : Layout) (he : Embeds h L) (hF : FrameIs h L) :
    Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation (code h))
      (two h) iroots :=
  sessionOpensTwo_of_rows h L he (frameRows_of_frameIs h L hF)

end SpotFlock
