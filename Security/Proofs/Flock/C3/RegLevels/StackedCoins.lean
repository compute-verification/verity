import Proofs.Flock.C3.RegLevels.TopHold
import Batteries.Recycling.MonadSatisfying.Array

/-!
# R's held stream has a round of coins for each round of R's schedule

`RecVStar.candidate3` holds R's stream (`stacked/rep0`) to the shape staging committed R at (`checkStackedShape`: R's
two claims, and as many rounds as `stackedRounds` at it), and keeps each of the stream's rounds' coins
(`Array.mapM` over the rounds). So the stream the verifier holds has a round of coins for each round of R's schedule
(`candidate3_held`), at the shape `RecVStar.innerCommits` was given (`innerCommits_held`). The walks are
`TopHold.candidate3_top`'s and `TopHold.innerCommits_top`'s.
-/

namespace FlockC3.RegLevels.StackedCoins

open Lean Flock.RecVStar
open FlockC3.RegLevels.TopHold (eok_bind forIn_inv)

/-- **R's stream is held to R's schedule**: the shape `shape` the verifier holds R at is some `sh` at R's two claims,
and the stream `x` holds one round of coins for each round of `stackedRounds sh`. -/
def Held (shape : Option StackedShape) (x : InnerStream) : Prop :=
  ∃ sh, shape = some sh ∧ sh.claims = STACKED_CLAIMS ∧ ∃ rs, stackedRounds sh = some rs ∧ x.coins.size = rs.size

/-- **`checkStackedShape` passes only at R's two claims and R's round count**: its rounds `rs` are as many as
`stackedRounds sh`'s. -/
theorem shape_of_check {s : String} {sh : StackedShape} {rs : Array Json} (h : checkStackedShape s sh rs = .ok ()) :
    sh.claims = STACKED_CLAIMS ∧ ∃ want, stackedRounds sh = some want ∧ rs.size = want.size := by
  unfold checkStackedShape at h
  extract_lets at_ s0 jp at h
  split at h
  · cases h
  rename_i hc
  simp (config := { zeta := false }) only [jp] at h
  split at h
  · rename_i want hw
    extract_lets jp2 at h
    split at h
    · cases h
    rename_i hsz
    refine ⟨by simpa using hc, want, hw, by simpa using hsz⟩
  · cases h

/-- **Candidate 3's session holds R's stream to R's schedule** (`candidate3_top`'s walk, at the coins): the stream it
builds named `stacked/rep0` passed `checkStackedShape` at the shape it was given, and its coins are its rounds'
(`Array.mapM` keeps the size). -/
theorem candidate3_held {tr : Json} {rounds : Array Json} {t : String} {names : List String}
    {msgs : Array (String × Nat × ByteArray)} {regR : Option (String × Json)} {shape : Option StackedShape}
    {out : Array InnerStream} (h : candidate3 tr rounds t names msgs regR shape = .ok out) :
    ∀ x ∈ out, x.name = "stacked/rep0" → Held shape x := by
  unfold candidate3 at h
  extract_lets stacked want tbl same tables wantTables rootOf0 out0 seen0 jpRoots jpTables jpDecl jpStart at h
  split at h
  · cases h
  simp only [jpStart] at h
  split at h
  rotate_left
  · cases h
  obtain ⟨declared, -, h⟩ := eok_bind h
  simp (config := { zeta := false }) only [jpDecl] at h
  extract_lets decl count at h
  split at h
  · cases h
  obtain ⟨u, -, h⟩ := eok_bind h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpTables] at h
  split at h
  rotate_left
  · cases h
  obtain ⟨roots, -, h⟩ := eok_bind h
  simp (config := { zeta := false }) only [jpRoots] at h
  extract_lets jpR at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpR] at h
  obtain ⟨rootOf, -, h⟩ := eok_bind h
  extract_lets rootOf' out1 at h
  obtain ⟨st, hmain, h⟩ := eok_bind h
  extract_lets o a at h
  obtain ⟨_, -, h⟩ := eok_bind h
  have hout : o = out := Except.ok.inj h
  subst hout
  refine forIn_inv (fun st : Array InnerStream × Array ByteArray => ∀ x ∈ st.1, x.name = "stacked/rep0" →
    Held shape x) want ?_ (fun x hx => absurd hx (by simp [out0])) hmain
  intro s hs ⟨o, a⟩ r hinv hr
  extract_lets o' a' table committed tops rs cs at hr
  obtain ⟨topsF, -, hr⟩ := eok_bind hr
  extract_lets topsF' allN jpEnd at hr
  have hend : jpEnd () = .ok r ∧
      (table = some STACKED → ∃ sh, shape = some sh ∧ checkStackedShape s sh rs = .ok ()) := by
    split at hr
    · split at hr
      · obtain ⟨⟨⟩, hc, hr⟩ := eok_bind hr
        exact ⟨hr, fun _ => ⟨_, rfl, hc⟩⟩
      · cases hr
    · rename_i htab
      exact ⟨hr, fun e => absurd (by simp [e]) htab⟩
  obtain ⟨coins, hco, hend'⟩ := eok_bind hend.1
  refine ⟨_, (Except.ok.inj hend').symm, ?_⟩
  intro y hy hname
  rcases Array.mem_push.1 hy with hy | rfl
  · exact hinv y hy hname
  · simp only at hname
    subst hname
    have htable : table = some STACKED := by
      have h1 : "stacked/rep0".startsWith (toString SPARSE ++ toString "/") = false := by decide +kernel
      have h2 : "stacked/rep0".startsWith (toString STACKED ++ toString "/") = true := by decide +kernel
      simp only [table, h1, h2]
      rfl
    obtain ⟨sh, hsh, hc⟩ := hend.2 htable
    obtain ⟨hcl, want', hw, hsz⟩ := shape_of_check hc
    refine ⟨sh, hsh, hcl, want', hw, ?_⟩
    rw [← hsz]
    exact SatisfiesM_Except_eq.1 (Array.size_mapM _ rs) coins hco

/-- **The verifier's hold of the inner transcript holds R's stream to R's schedule** (`innerCommits_top`'s walk): its
`stacked/rep0` stream is `candidate3`'s, so `Held` at the shape `innerCommits` was given, which is the shape the hold
records (`InnerCommits.stacked`). -/
theorem innerCommits_held {tr : Json} {record : String} {reg : Option Json} {bound : Option String}
    {shape : Option StackedShape} {cs : InnerCommits} {x : InnerStream}
    (h : innerCommits tr record reg bound shape = .ok cs) (hs : cs.stream? "stacked/rep0" = some x) :
    Held shape x ∧ cs.stacked = shape := by
  unfold innerCommits at h
  extract_lets msgs0 extra stackedS declaredR jpMain at h
  split at h
  · cases h
  simp (config := { zeta := false }) only [jpMain] at h
  obtain ⟨held, -, h⟩ := eok_bind h
  split at h
  rotate_left
  · cases h
  rename_i rounds _
  extract_lets regR jpRec at h
  split at h
  rotate_left
  · cases h
  obtain ⟨j, -, h⟩ := eok_bind h
  simp (config := { zeta := false }) only [jpRec] at h
  split at h
  rotate_left
  · cases h
  rename_i streams _
  obtain ⟨msgs, -, h⟩ := eok_bind h
  extract_lets msgs' names c3 recNames jpT at h
  split at h
  rotate_left
  · cases h
  rename_i t0 _
  have key : ∀ t', jpT t' = .ok cs → c3 = true ∧
      ∃ out, candidate3 tr rounds t' names msgs' regR shape = .ok out ∧ cs.streams = out ∧ cs.stacked = shape := by
    intro t' hT
    simp (config := { zeta := false }) only [jpT] at hT
    extract_lets repNames reps jpC3 jpShape jpLoop at hT
    split at hT
    · cases hT
    obtain ⟨u, -, hT⟩ := eok_bind hT
    have hS : jpShape () = .ok cs := by
      split at hT
      · cases hT
      · cases hT
      · split at hT
        · cases hT
        · exact hT
      · exact hT
    simp (config := { zeta := false }) only [jpShape] at hS
    split at hS
    · cases hS
    simp (config := { zeta := false }) only [jpC3] at hS
    split at hS
    · rename_i hc3
      obtain ⟨out, hout, hS⟩ := eok_bind hS
      have := Except.ok.inj hS
      subst this
      exact ⟨hc3, out, hout, rfl, rfl⟩
    · have := Except.ok.inj hS
      subst this
      simp [InnerCommits.stream?] at hs
  split at h
  · split at h
    · rename_i t' _
      obtain ⟨w, hw, h⟩ := eok_bind h
      have : t' = w := Except.ok.inj hw
      subst this
      obtain ⟨-, out, hout, hstreams, hstk⟩ := key t' h
      have hs' : out.find? (·.name == "stacked/rep0") = some x := hstreams ▸ hs
      have hq := Array.find?_some hs'
      exact ⟨candidate3_held hout x (Array.mem_of_find?_eq_some hs') (beq_iff_eq.1 hq), hstk⟩
    · cases h
  · split at h
    · cases h
    · obtain ⟨_, -, h⟩ := eok_bind h
      exact absurd (key _ h).1 ‹_›

end FlockC3.RegLevels.StackedCoins
