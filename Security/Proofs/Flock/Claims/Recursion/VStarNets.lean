import Proofs.Flock.Claims.Recursion.VStarTraceOk
import Proofs.Flock.VBridge.Family.Names
import Std.Data.String.ToNat

/-!
# A staged part's circuit has three nets (`nets_three`)

`traceOk_deployed` takes `c.nets.size = 3`. It holds at every staged part: the setup's `checkStmt` holds the circuit
file to V\*'s (`checkCircuit` runs `checkFile`, so `circuitFile sv f holo = .ok circuitB`: `checkFile_ok`) and refuses
lookup tables (`tables = []`: `checkStmt_tables`), and the staged circuit is `HmRow.parse` of that file
(`table0`, `circuitOf`).

V\*'s file is fixed text (`VStar.emitCircuitOf`, `VStar/Holo.lean:109`; `VStar.emitSections`, `VStar/Compose.lean:233`):
its two head lines, then exactly three sections, `CIRCUIT unit <n>`, `CIRCUIT sha512x3 <n>` and `CIRCUIT hm96 <n>`, each
`<n>` the count of its net's lines (`netLines`), the last ending the file (`file_text`, `fileLs_secs`). `HmRow.parse`'s
section loop (`HmRow.lean:640-649`) reads one net per section, jumping `1 + n` lines, and with no tables its lookup loop
adds none (`parse_nets`), so the circuit has three nets (`nets_of_file`). The lines split as written because none holds
a newline: the unit's name is the parsed unit's name, a token (`unit_name`, `tokens_noWs`), and META is `pyDumps`'s one
line (`pyDumps_noNl`). No proof reads through `Lean.Json.parse`.
-/

namespace RecDeploy.VStarNets

open _root_.Flock
open Flock.HmNets (Laid shaLaid hmLaid)
open FlockVBridge.Names
open Claims (Msg)
open RecDeploy.VStarBuild (partAt progOf)
open RecDeploy.VStarTrace (TraceOk)
open RecDeploy.VStarTraceOk (RowsOk)

/-! ## V\*'s text -/

/-- Lines, each with its newline. -/
def joinNl (L : List String) : String := String.join (L.map (· ++ "\n"))

theorem joinNl_nil : joinNl [] = "" := rfl

theorem joinNl_cons (a : String) (L : List String) : joinNl (a :: L) = a ++ "\n" ++ joinNl L := by
  simp [joinNl, String.join_cons, String.append_assoc]

theorem joinNl_append (L M : List String) : joinNl (L ++ M) = joinNl L ++ joinNl M := by
  induction L with
  | nil => simp [joinNl_nil]
  | cons a L ih => rw [List.cons_append, joinNl_cons, joinNl_cons, ih]; simp [String.append_assoc]

/-- A net's row lines (`VStar.emitNet`'s loop). -/
def rowsLs (l : Laid) : List String :=
  (List.range' 0 l.useful).map fun i => VStar.rowLine (l.ra.getD i #[]) (l.rb.getD i #[])

/-- A net's lines: its header, then its rows. -/
def netLs (name : String) (l : Laid) : List String := VStar.header name l :: rowsLs l

/-- A section's lines: `CIRCUIT <nm> <n>`, then its net's. -/
def secLs (nm net : String) (l : Laid) : List String :=
  ("CIRCUIT " ++ nm ++ " " ++ toString (VStar.netLines l)) :: netLs net l

/-- V\*'s file's lines: the head's two, then the unit's, `sha512x3`'s and `hm96`'s sections. The two fixed nets'
layouts are parameters, so that no proof term over these lines makes the kernel evaluate `shaLaid ()` or `hmLaid ()`. -/
def fileLs (name d : String) (l ls lh : Laid) : List String :=
  ["flock-circuit " ++ name, "META " ++ d] ++ secLs "unit" name l ++ secLs VStar.SHA_NET VStar.SHA_NET ls ++
    secLs VStar.HM_NET VStar.HM_NET lh

theorem idRun {α : Type} (x : α) : Id.run x = x := rfl

theorem run_put {E : StateM String PUnit} (xs s : String) :
    ((do (fun x => (modify (· ++ x) : StateM String PUnit)) xs; E : StateM String PUnit).run s) = E.run (s ++ xs) := by
  rw [StateT.run_bind]
  first | rfl | simp [StateT.run_modify]

theorem forIn_put (g : ℕ → String) : ∀ (xs : List ℕ) (s : String),
    (forIn (m := StateM String) xs PUnit.unit (fun i _ => do
      (fun x => (modify (· ++ x) : StateM String PUnit)) (g i ++ "\n"); pure (ForInStep.yield PUnit.unit))).run s =
      (PUnit.unit, s ++ joinNl (xs.map g))
  | [], s => by simp [joinNl_nil]; rfl
  | i :: xs, s => by
    rw [List.forIn_cons, StateT.run_bind]
    show (StateT.run (forIn xs PUnit.unit _) (s ++ (g i ++ "\n"))) = _
    rw [forIn_put g xs, List.map_cons, joinNl_cons, String.append_assoc]

theorem emitNet_run (name : String) (l : Laid) (s : String) :
    (VStar.emitNet (m := StateM String) (fun x => modify (· ++ x)) name l none).run s =
      (PUnit.unit, s ++ joinNl (netLs name l)) := by
  unfold VStar.emitNet
  rw [StateT.run_bind]
  show (StateT.run (forIn _ PUnit.unit _ >>= fun _ => pure PUnit.unit) (s ++ (VStar.header name l ++ "\n"))) = _
  rw [Std.Legacy.Range.forIn_eq_forIn_range', StateT.run_bind, forIn_put]
  have hsz : [:l.useful].size = l.useful := by simp [Std.Legacy.Range.size]
  rw [hsz]
  simp only [netLs, rowsLs, joinNl_cons, String.append_assoc]
  rfl

set_option maxHeartbeats 1000000 in
theorem emitSections_run (u : VStar.UnitC) (l : Laid) (s : String) :
    (VStar.emitSections (m := StateM String) (fun x => modify (· ++ x)) u l).run s =
      (PUnit.unit, s ++ joinNl (secLs "unit" u.name l ++ secLs VStar.SHA_NET VStar.SHA_NET (shaLaid ()) ++
        secLs VStar.HM_NET VStar.HM_NET (hmLaid ()))) := by
  unfold VStar.emitSections
  generalize shaLaid () = ls
  generalize hmLaid () = lh
  simp [List.forIn_cons, List.forIn_nil, StateT.run_bind, StateT.run_modify, emitNet_run, secLs, joinNl_cons,
    joinNl_append, String.append_assoc]
  refine Prod.ext rfl ?_
  change s ++ _ ++ _ ++ _ = _
  simp only [String.append_assoc]
  rfl

set_option maxHeartbeats 1000000 in
/-- **V\*'s file is its lines** (`RecVStar.circuitFile`): the head `flock-circuit <unit>`, `META <d>`, then the three
sections. -/
theorem file_text (u : VStar.UnitC) (l : Laid) (d : String) :
    ((do (fun x => (modify (· ++ x) : StateM String PUnit)) (s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n"); VStar.emitSections (m := StateM String) (fun x => modify (· ++ x)) u l : StateM String PUnit).run "").2 =
      joinNl (fileLs u.name d l (shaLaid ()) (hmLaid ())) := by
  rw [run_put, emitSections_run]
  generalize shaLaid () = ls
  generalize hmLaid () = lh
  try dsimp only
  simp only [fileLs, List.append_assoc, joinNl_append, joinNl_cons, joinNl_nil]
  simp only [String.empty_append, String.append_empty, String.append_assoc]
  have h0 : ∀ t : String, toString t = t := fun _ => rfl
  have h1 : ∀ t : String, VStar.FORMAT_CIRCUIT ++ (" " ++ t) = "flock-circuit " ++ t := fun t => by
    rw [← String.append_assoc]; rfl
  have h2 : ∀ t : String, "\nMETA " ++ t = "\n" ++ ("META " ++ t) := fun t => by
    rw [← String.append_assoc]; rfl
  simp only [h0, h1, h2]

/-- `withMeta`'s head is `compose`'s first two lines, META a `pyDumps` line. -/
theorem withMeta_ok {u : VStar.UnitC} {extra : List (String × Lean.Json)} {head h : String}
    (hw : VStar.withMeta u extra head = .ok h) :
    ∃ d, NoNl d ∧ h = s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n" := by
  unfold VStar.withMeta at hw
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at hw
  repeat' (first | split at hw | (injection hw with hw; subst hw))
  all_goals first | exact absurd hw (by simp) | (refine ⟨_, ?_, rfl⟩; exact pyDumps_noNl ‹_›)

/-- **The emitted file's program**: put the head, `flock-circuit <unit>` and one META line, then the sections. -/
theorem emit_head {holo : Bool} {u : VStar.UnitC} {l : Laid} {sub : VStar.Sub} {b : VStar.Bindings}
    {e : StateM String Unit}
    (h : VStar.emitCircuitOf (m := StateM String) holo (fun x => modify (· ++ x)) u l sub b = .ok e) :
    ∃ d, NoNl d ∧ e = (do (fun x => (modify (· ++ x) : StateM String PUnit)) (s!"{VStar.FORMAT_CIRCUIT} {u.name}\nMETA " ++ d ++ "\n"); VStar.emitSections (m := StateM String) (fun x => modify (· ++ x)) u l) := by
  unfold VStar.emitCircuitOf at h
  split at h
  · unfold VStar.emitCircuitHolo at h
    obtain ⟨_, -, h⟩ := FlockSoundness.except_bind_ok h
    obtain ⟨h0, -, h⟩ := FlockSoundness.except_bind_ok h
    obtain ⟨head, hw, h⟩ := FlockSoundness.except_bind_ok h
    cases h
    obtain ⟨d, hd, rfl⟩ := withMeta_ok hw
    exact ⟨d, hd, rfl⟩
  · obtain ⟨sha, head, hh, rfl⟩ := emitCircuit_ok h
    obtain ⟨j, d, hd, rfl⟩ := metaHead_ok hh
    exact ⟨d, pyDumps_noNl hd, rfl⟩

/-- **A statement's unit is named as the statement** (`VStar.Stmt.unit`, `Stmt.name`). -/
theorem unit_name {s : VStar.Stmt} {u : VStar.UnitC} (h : s.unit = .ok u) : u.name = s.name := by
  cases s with
  | recOpen _ L H _ => exact recOpenUnit_name h
  | inner _ nm pt => exact innerUnit_name h
  | _ =>
    simp only [VStar.Stmt.unit, VStar.sparseUnit] at h
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    repeat' (first | split at h | (injection h with h; subst h))
    all_goals first | exact absurd h (by simp) | rfl

/-! ## The lines -/

theorem splitOn_join : ∀ L : List String, (∀ x ∈ L, NoNl x) → (joinNl L).splitOn "\n" = L ++ [""]
  | [], _ => by rw [joinNl_nil, splitOn_nl]; rfl
  | a :: L, h => by
    rw [joinNl_cons, splitOn_line _ _ (h a List.mem_cons_self),
      splitOn_join L fun x hx => h x (List.mem_cons_of_mem _ hx), List.cons_append]

/-- `HmRow.parse`'s lines of newline-free lines, each with its newline, are those lines. -/
theorem linesOf_join (L : List String) (h : ∀ x ∈ L, NoNl x) : linesOf (joinNl L) = L.toArray := by
  unfold linesOf
  rw [splitOn_join L h]
  simp

theorem rowLine_noNl (a b : Array ℕ) : NoNl (VStar.rowLine a b) := by
  unfold VStar.rowLine
  simp only [Id.run, ← Array.forIn_toList, List.forIn_pure_yield_eq_foldl, pure_bind]
  have hg : ∀ (o : String) (c : ℕ), NoNl o → NoNl (o.push ' ' ++ toString c) := fun o c ho => by
    rw [String.push_eq_append]
    exact noNl_append.2 ⟨noNl_append.2 ⟨ho, by simp [NoNl]⟩, nat_noNl c⟩
  exact foldl_noNl _ hg _ _ (hg _ _ (foldl_noNl _ hg _ _ (nat_noNl _)))

theorem netLs_noNl (name : String) (l : Laid) (h : NoNl name) : ∀ x ∈ netLs name l, NoNl x := by
  intro x hx
  simp only [netLs, rowsLs, List.mem_cons, List.mem_map] at hx
  rcases hx with rfl | ⟨i, -, rfl⟩
  · exact header_noNl _ _ h
  · exact rowLine_noNl _ _

theorem secLs_noNl (nm net : String) (l : Laid) (hm : NoNl nm) (h : NoNl net) : ∀ x ∈ secLs nm net l, NoNl x := by
  intro x hx
  simp only [secLs, List.mem_cons] at hx
  rcases hx with rfl | hx
  · exact noNl_append.2 ⟨noNl_append.2 ⟨noNl_append.2 ⟨by simp [NoNl], hm⟩, by simp [NoNl]⟩, nat_noNl _⟩
  · exact netLs_noNl net l h x hx

theorem fileLs_noNl (name d : String) (l ls lh : Laid) (hn : NoNl name) (hd : NoNl d) :
    ∀ x ∈ fileLs name d l ls lh, NoNl x := by
  intro x hx
  simp only [fileLs, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with (((rfl | rfl) | hx) | hx) | hx
  · exact noNl_append.2 ⟨by simp [NoNl], hn⟩
  · exact noNl_append.2 ⟨by simp [NoNl], hd⟩
  · exact secLs_noNl _ _ _ (by simp [NoNl]) hn x hx
  · exact secLs_noNl _ _ _ (by simp [NoNl, VStar.SHA_NET]) (by simp [NoNl, VStar.SHA_NET]) x hx
  · exact secLs_noNl _ _ _ (by simp [NoNl, VStar.HM_NET]) (by simp [NoNl, VStar.HM_NET]) x hx

/-! ## Tokens -/

theorem tokens_three (a b c : String) (ha : a ≠ "") (hb : b ≠ "") (hc : c ≠ "") (na : NoWs a) (nb : NoWs b)
    (nc : NoWs c) : tokens (a ++ " " ++ b ++ " " ++ c) = #[a, b, c] := by
  have hF : (a ++ " " ++ b ++ " " ++ c).toList.foldl tstep (#[], "") = (#[a, b], c) := by
    simp only [String.toList_append, List.foldl_append]
    rw [foldl_word _ na, String.empty_append, String.ofList_toList]
    rw [show " ".toList = [' '] from rfl]
    simp only [List.foldl_cons, List.foldl_nil]
    rw [show tstep (#[], a) ' ' = (#[a], "") by simp [tstep, ha]]
    rw [foldl_word _ nb, String.empty_append, String.ofList_toList]
    rw [show tstep (#[a], b) ' ' = (#[a, b], "") by simp [tstep, hb]]
    rw [foldl_word _ nc, String.empty_append, String.ofList_toList]
  rw [tokens_eq]
  simp only [hF]
  simp [hc]

theorem tstep_noWs (p : Array String × String) (c : Char) (h : (∀ x ∈ p.1.toList, NoWs x) ∧ NoWs p.2) :
    (∀ x ∈ (tstep p c).1.toList, NoWs x) ∧ NoWs (tstep p c).2 := by
  unfold tstep
  by_cases hw : c.isWhitespace = true
  · rw [if_pos hw]
    split
    · refine ⟨fun x hx => ?_, by simp [NoWs]⟩
      simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hx
      rcases hx with hx | rfl
      · exact h.1 x hx
      · exact h.2
    · exact h
  · rw [if_neg hw]
    refine ⟨h.1, ?_⟩
    rw [String.push_eq_append]
    exact noWs_append.2 ⟨h.2, by simpa [NoWs] using hw⟩

theorem foldl_tstep_noWs : ∀ (l : List Char) (p : Array String × String), ((∀ x ∈ p.1.toList, NoWs x) ∧ NoWs p.2) →
    (∀ x ∈ (l.foldl tstep p).1.toList, NoWs x) ∧ NoWs (l.foldl tstep p).2
  | [], _, h => h
  | c :: l, p, h => foldl_tstep_noWs l _ (tstep_noWs p c h)

/-- **Every token is one word.** -/
theorem tokens_noWs (s : String) : ∀ x ∈ (tokens s).toList, NoWs x := by
  rw [tokens_eq]
  have h := foldl_tstep_noWs s.toList (#[], "") ⟨by simp, by simp [NoWs]⟩
  generalize s.toList.foldl tstep (#[], "") = p at h ⊢
  dsimp only
  split
  · intro x hx
    simp only [Array.toList_push, List.mem_append, List.mem_singleton] at hx
    rcases hx with hx | rfl
    · exact h.1 x hx
    · exact h.2
  · exact h.1

theorem tokens_getD_noWs (s : String) (i : ℕ) : NoWs ((tokens s).getD i "") := by
  rw [Array.getD_eq_getD_getElem?]
  cases hi : (tokens s)[i]? with
  | none => simp [NoWs]
  | some x =>
    simp only [Option.getD_some]
    obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.1 hi
    exact tokens_noWs s _ (Array.mem_toList_iff.mpr (Array.getElem_mem hlt))

/-! ## The sections -/

/-- Line `a` is a section line `<kw> <name> <n>` whose net runs to line `b`. -/
def Sec (lines : Array String) (a b : ℕ) : Prop :=
  a < lines.size ∧ ∃ n, ((tokens (lines.getD a "")).getD 2 "").toNat? = some n ∧ b = a + 1 + n

theorem nat_ne_empty (n : ℕ) : toString n ≠ "" := Nat.repr_ne_empty

theorem secLs_length (nm net : String) (l : Laid) : (secLs nm net l).length = 1 + VStar.netLines l := by
  simp [secLs, netLs, rowsLs, VStar.netLines]
  omega

theorem sec_at (pre post : List String) (nm net : String) (l : Laid) (hn : NoWs nm) (he : nm ≠ "") :
    Sec (pre ++ secLs nm net l ++ post).toArray pre.length (pre.length + (secLs nm net l).length) := by
  refine ⟨by simp [secLs_length] <;> omega, VStar.netLines l, ?_, by rw [secLs_length]; omega⟩
  have hl : (pre ++ secLs nm net l ++ post).toArray.getD pre.length "" =
      "CIRCUIT" ++ " " ++ nm ++ " " ++ toString (VStar.netLines l) := by
    rw [Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.append_assoc,
      List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]
    simp only [secLs, List.cons_append, List.getElem?_cons_zero, Option.getD_some]
    simp only [String.append_assoc]
    rfl
  rw [hl, tokens_three _ _ _ (by decide) he (nat_ne_empty _) (by simp [NoWs]) hn (nat_noWs _)]
  exact Nat.toNat?_repr _

/-- **V\*'s file's three sections**: at line 2 the unit's, then `sha512x3`'s, then `hm96`'s, which ends the file. -/
theorem fileLs_secs (name d : String) (l ls lh : Laid) :
    Sec (fileLs name d l ls lh).toArray 2 (2 + (secLs "unit" name l).length) ∧
      Sec (fileLs name d l ls lh).toArray (2 + (secLs "unit" name l).length)
        (2 + (secLs "unit" name l).length + (secLs VStar.SHA_NET VStar.SHA_NET ls).length) ∧
      Sec (fileLs name d l ls lh).toArray
        (2 + (secLs "unit" name l).length + (secLs VStar.SHA_NET VStar.SHA_NET ls).length)
        (fileLs name d l ls lh).toArray.size := by
  have h1 := sec_at ["flock-circuit " ++ name, "META " ++ d]
    (secLs VStar.SHA_NET VStar.SHA_NET ls ++ secLs VStar.HM_NET VStar.HM_NET lh) "unit" name l
    (by simp [NoWs]) (by decide)
  have h2 := sec_at (["flock-circuit " ++ name, "META " ++ d] ++ secLs "unit" name l)
    (secLs VStar.HM_NET VStar.HM_NET lh) VStar.SHA_NET VStar.SHA_NET ls
    (by simp [NoWs, VStar.SHA_NET]) (by decide)
  have h3 := sec_at (["flock-circuit " ++ name, "META " ++ d] ++ secLs "unit" name l ++
    secLs VStar.SHA_NET VStar.SHA_NET ls) [] VStar.HM_NET VStar.HM_NET lh
    (by simp [NoWs, VStar.HM_NET]) (by decide)
  simp only [List.append_assoc, List.length_append, List.append_nil, List.length_cons, List.length_nil] at h1 h2 h3
  simp only [fileLs, List.append_assoc, List.size_toArray, List.length_append, List.length_cons, List.length_nil]
  refine ⟨?_, ?_, ?_⟩
  · simpa using h1
  · simpa [Nat.add_assoc] using h2
  · simpa [Nat.add_assoc] using h3

/-! ## `HmRow.parse`'s nets -/

set_option maxHeartbeats 4000000 in
/-- **`HmRow.parse`'s nets** without tables: each net's name is one word (a header line's token, `parse_name`), and where
lines 2, `p1` and `p2` are section lines running to `p1`, `p2` and the last line, the section loop (`HmRow.lean:640-649`)
reads exactly three nets and the lookup loop none. -/
theorem parse_nets (p1 p2 : ℕ) {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    {text : String} (h : HmRow.parse tags bytes tables = .ok c) (htext : String.fromUTF8? bytes = some text)
    (htb : tables = []) :
    (∀ e ∈ c.nets.toList, NoWs e.2.name) ∧
      (Sec (linesOf text) 2 p1 → Sec (linesOf text) p1 p2 → Sec (linesOf text) p2 (linesOf text).size →
        c.nets.size = 3) := by
  unfold HmRow.parse at h
  simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h)
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := FlockSoundness.except_bind_ok h; cases hte
  rename_i txt htxt
  rw [htext] at htxt
  cases htxt
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨text', hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp0] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h)
  replace h := FlockSoundness.ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h)
  replace h := FlockSoundness.ite_throw_ok h
  simp (config := { zeta := false }) only [jp2] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h)
  split at h
  swap
  · obtain ⟨_, hte, -⟩ := FlockSoundness.except_bind_ok h; cases hte
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
  cases hte
  simp (config := { zeta := false }) only [jp3] at h
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨f1, hf1, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨f2, hf2, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h)
  replace h := FlockSoundness.ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h)
  obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
    split at h
    · replace h := FlockSoundness.ite_throw_ok h
      simp (config := { zeta := false }) only [jp6] at h
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h)
      replace h := FlockSoundness.ite_throw_ok h
      simp (config := { zeta := false }) only [jp7] at h
      obtain ⟨cs, -, h⟩ := FlockSoundness.except_bind_ok h
      exact ⟨cs, h⟩
    · obtain ⟨cs, -, h⟩ := FlockSoundness.except_bind_ok h
      exact ⟨cs, h⟩
  simp (config := { zeta := false }) only [jp5] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨s1, hwhile, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h)
  simp (config := { zeta := false }) only [jpT] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h)
  obtain ⟨lks, h⟩ : ∃ lks, jpL lks = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := FlockSoundness.except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpL] at h
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨s2, hlook, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets2 at h)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨ports, hrp, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpK at h)
  replace h := FlockSoundness.ite_throw_ok h
  simp (config := { zeta := false }) only [jpK] at h
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨g, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨upv, -, h⟩ := hh)
  split at h
  swap
  · cases h
  rename_i unitNet hunit
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) netOf at h)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨ranges, hranges, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨leavesIn, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨leavesOut, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨dummy, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨dummyDigests, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨dummyOutputs, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨outPorts, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpR at h)
  obtain ⟨outRowPorts, h⟩ : ∃ r, jpR r = .ok c := by
    split at h
    · exact ⟨_, h⟩
    · obtain ⟨_, -, h⟩ := FlockSoundness.except_bind_ok h
      obtain ⟨r, -, h⟩ := FlockSoundness.except_bind_ok h
      exact ⟨r, FlockSoundness.ite_throw_ok h⟩
  simp (config := { zeta := false }) only [jpR] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) perVu0 jpP at h)
  obtain ⟨pv, h⟩ : ∃ pv, jpP () pv = .ok c := by
    split at h
    · obtain ⟨p', -, h⟩ := FlockSoundness.except_bind_ok h
      exact ⟨_, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jpP] at h
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨shaNet, hsha, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨hmNet, hhm, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨outNet, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpO at h)
  obtain ⟨onp, h⟩ : ∃ x, jpO x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := FlockSoundness.except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpO] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at h)
  obtain ⟨lc, h⟩ : ∃ x, jpC x = .ok c := by
    split at h <;> (obtain ⟨l, -, h⟩ := FlockSoundness.except_bind_ok h; exact ⟨l, h⟩)
  simp (config := { zeta := false }) only [jpC] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpW at h)
  obtain ⟨wires, h⟩ : ∃ x, jpW x = .ok c := by
    split at h
    · obtain ⟨_, -, h⟩ := FlockSoundness.except_bind_ok h
      obtain ⟨l, -, h⟩ := FlockSoundness.except_bind_ok h
      exact ⟨l, h⟩
    · obtain ⟨l, -, h⟩ := FlockSoundness.except_bind_ok h
      exact ⟨l, h⟩
  simp (config := { zeta := false }) only [jpW] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) derived wiresT rowNet at h)
  obtain ⟨hwires, h⟩ := FlockSoundness.ite_throw_ok' h
  simp (config := { zeta := false }) only at h
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨ls, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨leafId, -, h⟩ := hh)
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpF at h)
  replace h := FlockSoundness.ite_throw_ok h
  simp (config := { zeta := false }) only [jpF] at h
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨kLog, -, h⟩ := hh)
  simp (config := { zeta := false }) only [pure_bind] at h
  ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) c0 at h)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w, hcheck, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w', hrows, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w'', hrec, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w''', hres, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
  have hc : c0 = c := Except.ok.inj h
  subst hc
  have hL : lines = linesOf text := rfl
  rw [← hL]
  have hQ : (∀ e ∈ s1.1.toList, NoWs e.2.name) ∧
      (Sec lines 2 p1 → Sec lines p1 p2 → Sec lines p2 lines.size → s1.1.size = 3) := by
    refine loop_inv' (fun b : Array (String × Net) × ℕ => (∀ e ∈ b.1.toList, NoWs e.2.name) ∧
        (Sec lines 2 p1 → Sec lines p1 p2 → Sec lines p2 lines.size →
          (b.1.size = 0 ∧ b.2 = 2) ∨ (b.1.size = 1 ∧ b.2 = p1) ∨ (b.1.size = 2 ∧ b.2 = p2) ∨
            (b.1.size = 3 ∧ b.2 = lines.size)))
      (fun b => (∀ e ∈ b.1.toList, NoWs e.2.name) ∧
        (Sec lines 2 p1 → Sec lines p1 p2 → Sec lines p2 lines.size → b.1.size = 3))
      (fun b => lines.size - b.2) ?_ _ (nets0, at0) s1 le_rfl
      ⟨fun e he => by have h00 : nets0 = #[] := rfl; simp [h00] at he,
        fun _ _ _ => Or.inl ⟨rfl, rfl⟩⟩ hwhile
    intro b hb s h
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at_ at h)
    split at h
    · rename_i hlt
      have hlt' : b.2 < lines.size := hlt
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) hh jp at h)
      replace h := FlockSoundness.ite_throw_ok h
      simp (config := { zeta := false }) only [jp] at h
      split at h
      · rename_i n hn
        have hn' : ((tokens (lines.getD b.2 "")).getD 2 "").toNat? = some n := by
          simpa [hh, at_, getElem!_str] using hn
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp' at h)
        obtain ⟨-, h⟩ := FlockSoundness.ite_throw_ok' h
        simp (config := { zeta := false }) only [jp'] at h
        ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp'' at h)
        split at h
        · rename_i v hv
          (have hh' := FlockSoundness.except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh')
          cases hnet
          simp (config := { zeta := false }) only [jp''] at h
          ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at' at h)
          have hs := Except.ok.inj h
          subst hs
          have hsz : nets'.size = b.1.size + 1 := by simp [nets', nets]
          have hat : at' = b.2 + 1 + n := rfl
          refine Or.inr ⟨_, rfl, ⟨?_, fun h1 h2 h3 => ?_⟩, ?_⟩
          · intro e he
            simp only [nets', nets, Array.toList_push, List.mem_append, List.mem_singleton] at he
            rcases he with he | rfl
            · exact hb.1 e he
            · rw [(parse_name hv).2]
              exact tokens_getD_noWs _ _
          · show (nets'.size = 0 ∧ at' = 2) ∨ (nets'.size = 1 ∧ at' = p1) ∨ (nets'.size = 2 ∧ at' = p2) ∨
              (nets'.size = 3 ∧ at' = lines.size)
            rw [hsz, hat]
            rcases hb.2 h1 h2 h3 with ⟨hs0, ha⟩ | ⟨hs0, ha⟩ | ⟨hs0, ha⟩ | ⟨hs0, ha⟩
            · obtain ⟨-, n1, hn1, he1⟩ := h1
              rw [ha, hn1] at hn'
              have e1 := Option.some.inj hn'
              exact Or.inr (Or.inl ⟨by omega, by omega⟩)
            · obtain ⟨-, n1, hn1, he1⟩ := h2
              rw [ha, hn1] at hn'
              have e1 := Option.some.inj hn'
              exact Or.inr (Or.inr (Or.inl ⟨by omega, by omega⟩))
            · obtain ⟨-, n1, hn1, he1⟩ := h3
              rw [ha, hn1] at hn'
              have e1 := Option.some.inj hn'
              exact Or.inr (Or.inr (Or.inr ⟨by omega, by omega⟩))
            · exact absurd hlt' (by omega)
          · show lines.size - at' < lines.size - b.2
            rw [hat]
            omega
        · (have hh' := FlockSoundness.except_bind_ok h; clear h; obtain ⟨net, hnet, h⟩ := hh')
          cases hnet
      · (have hh' := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh')
        cases hw
    · rename_i hge
      have hge' : ¬ b.2 < lines.size := hge
      have hs := Except.ok.inj h
      subst hs
      refine Or.inl ⟨_, rfl, hb.1, fun h1 h2 h3 => ?_⟩
      rcases hb.2 h1 h2 h3 with ⟨-, ha⟩ | ⟨-, ha⟩ | ⟨-, ha⟩ | ⟨hs0, -⟩
      · exact (hge' (by rw [ha]; exact h1.1)).elim
      · exact (hge' (by rw [ha]; exact h2.1)).elim
      · exact (hge' (by rw [ha]; exact h3.1)).elim
      · exact hs0
  have hn2 : nets2 = nets1 := by
    rw [← Array.forIn_toList] at hlook
    refine FlockSoundness.forIn_except_inv (fun nets : Array (String × Net) => nets = nets1) _ ?_ nets1 s2 rfl hlook
    intro a _ b hb s h
    obtain ⟨lname, spec⟩ := a
    ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at h)
    split at h
    · ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) size at h)
      (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨_, -, h⟩ := hh)
      ((try beta_reduce at h); extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at h)
      replace h := FlockSoundness.ite_throw_ok h
      simp (config := { zeta := false }) only [jpA] at h
      (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨n, -, h⟩ := hh)
      (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨lo, -, h⟩ := hh)
      (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at h)
      replace h := FlockSoundness.ite_throw_ok h
      simp (config := { zeta := false }) only [jpB] at h
      split at h
      · exfalso
        rename_i heq
        rw [htb] at heq
        first | (simp at heq; done) | cases heq
      · (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
        cases hw
    · (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨w, hw, h⟩ := hh)
      cases hw
  have hc0 : c0.nets = s1.1 := hn2
  rw [hc0]
  exact hQ

/-! ## Three nets -/

/-- **The circuit parsed from V\*'s file has three nets**: `circuitFile` writes the head and three sections
(`emit_head`, `file_text`), none of whose lines holds a newline (`fileLs_noNl`: the unit's name is the parsed unit's,
one word), so `HmRow.parse` without tables reads one net per section (`parse_nets`, `fileLs_secs`). -/
theorem nets_of_lines {tags : Tags} {c : Circuit} {name d : String} {l ls lh : Laid}
    (hp : HmRow.parse tags (joinNl (fileLs name d l ls lh)).toUTF8 [] = .ok c) (hd : NoNl d)
    (hn : name = c.unit.name) (hun : c.unitNet < c.nets.size) : c.nets.size = 3 := by
  have hP := parse_nets (2 + (secLs "unit" name l).length)
    (2 + (secLs "unit" name l).length + (secLs VStar.SHA_NET VStar.SHA_NET ls).length) hp
    (fromUTF8?_toUTF8 _) rfl
  have hcu : NoWs c.unit.name := by
    have hm : c.nets[c.unitNet] ∈ c.nets.toList := Array.mem_toList_iff.mpr (Array.getElem_mem hun)
    rw [show c.unit = (c.nets.getD c.unitNet default).2 from rfl, Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem hun, Option.getD_some]
    exact hP.1 _ hm
  have hnm : NoNl name := by
    rw [hn]
    exact noNl_of_noWs hcu
  rw [linesOf_join _ (fileLs_noNl _ _ _ _ _ hnm hd)] at hP
  obtain ⟨s1, s2, s3⟩ := fileLs_secs name d l ls lh
  exact hP.2 s1 s2 s3

/-- **The circuit parsed from V\*'s file has three nets**: `circuitFile` writes the head and three sections
(`emit_head`, `file_text`), none of whose lines holds a newline (`fileLs_noNl`: the unit's name is the parsed unit's,
one word), so `HmRow.parse` without tables reads one net per section (`parse_nets`, `fileLs_secs`). -/
theorem nets_of_file {sv : VStar.Stmt} {f : VStar.Bindings} {holo : Bool} {b : ByteArray} {tags : Tags} {c : Circuit}
    (hf : RecVStar.circuitFile sv f holo = .ok b) (hp : HmRow.parse tags b [] = .ok c) (hn : sv.name = c.unit.name)
    (hun : c.unitNet < c.nets.size) : c.nets.size = 3 := by
  unfold RecVStar.circuitFile at hf
  obtain ⟨u, hu, hf⟩ := FlockSoundness.except_bind_ok hf
  obtain ⟨l, -, hf⟩ := FlockSoundness.except_bind_ok hf
  obtain ⟨e, he, hf⟩ := FlockSoundness.except_bind_ok hf
  cases hf
  obtain ⟨d, hd, he'⟩ := emit_head he
  rw [idRun, he', file_text u l d] at hp
  exact nets_of_lines hp hd (by rw [unit_name hu, hn]) hun

/-- `checkStmt` at an inner shape refuses lookup tables. -/
theorem checkStmt_tables {i : Flock.RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    (h : Flock.RecVStar.checkStmt (some i) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) : tables = [] := by
  unfold Flock.RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := FlockSoundness.Refine.ebind_ok h
      cases h1
    · rename_i ht
      simpa using ht

/-- **A staged part's circuit has three nets**: the unit, `sha512x3` and `hm96`. The setup's `checkStmt` holds the
part's circuit file to V\*'s (`checkFile_ok`) and refuses tables (`checkStmt_tables`), and its circuit is
`HmRow.parse`'s of that file (`table0`), the circuit `partAt` reads (`partCircuit_staged`). -/
theorem nets_three {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : VStar.Stmt} {c : Circuit}
    (hp : partAt x s = some (sv, c)) : c.nets.size = 3 := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  obtain ⟨p, r', _, hx, ht, hpi, hr', _, h0, _⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  obtain ⟨tags', P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  rw [← hpc]
  obtain ⟨ms, i, spec, sts, rfl, hsu, hck⟩ := hS
  have hpx := parse_of_setupO hsu
  rw [hx] at hpx
  cases hpx
  have h0' : 0 < sts.size := h0
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain ⟨r0, -, hinner, -⟩ := insO_staged hins
  have hC := RecIns.setup_checks hset
  rw [hinner, insO_staged_part hins] at hC
  have hC0 := checks_stmt hC 0 h0'
  obtain ⟨j, f, sv0, hj, -, hcc⟩ := checkStmt_circuit hC0
  cases hj
  obtain ⟨hck', hfile, -, -⟩ := FlockSoundness.Discharge.Recursion.checkCircuit_ok hcc
  obtain ⟨_, _, _, hname, _⟩ := FlockSoundness.Discharge.Recursion.check_ok hck'
  have hcf := FlockSoundness.Discharge.Recursion.checkFile_ok hfile
  have htb := checkStmt_tables hC0
  have hty : i.base.read.tags.typed = false := ((Flock.Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1.2.1
  have hpi' := publicInputs_none_of_scope hck
  have htab : i.base.read.tablesOf = some sts := by
    unfold Flock.Readable.Read.tablesOf
    rw [RecIns.setup_read hset]
    cases hz : i.base.read.zk
    · simp
    · simp [RecIns.setup_shapeOk hset hz]
  obtain ⟨st0, hst0, hc0, -⟩ := Flock.Readable.table0 htab hty hpi'
  rw [Array.getElem?_eq_getElem h0', Option.some.injEq] at hst0
  subst hst0
  have hparse : HmRow.parse i.base.read.tags i.base.read.I.circuitFile [] = .ok sts[0].c := by
    rw [← htb]
    unfold ProvedScope.circuitOf at hc0
    split at hc0
    · exact (FlockSoundness.Discharge.HiddenExec.parse_untyped hty hc0).1
    · exact hc0
  exact nets_of_file hcf hparse hname P.unitNet

/-- **`TraceOk` at a staged part, given the port inputs' rows** (`traceOk_deployed` with `nets_three`). -/
theorem traceOk_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {src : VStar.In.Src} {pin : VStar.In.PartIn}
    (hi : VStar.In.partIn src (progOf c) sv = .ok pin) (hR : RowsOk c pin) : TraceOk c pin :=
  VStarTraceOk.traceOk_deployed hS hp hi (nets_three hS hp) hR

theorem partAt_lt {x : Msg} {s : ℕ} {sv : VStar.Stmt} {c : Circuit} (hp : partAt x s = some (sv, c)) :
    s < RecDeploy.nS x := by
  obtain ⟨r, hr, -, hc⟩ := VStarTraceOk.partAt_some hp
  unfold RecDeploy.nS
  rw [hr]
  cases hq : r.w.parts[s]? with
  | none => rw [hq] at hc; simp at hc
  | some q => exact (List.getElem?_eq_some_iff.1 hq).1

/-- **`TraceOk` at every part of a staged statement** (`HpEarly.TraceOkAt`'s body): each part `s < nS x` staged at
`xOf s`, and the port inputs' rows sized (`RowsOk`). -/
theorem traceOkAt_staged {I : VStarBuild.InnerRec} {x : Msg} {w : Claims.Flock.Wit} {ρ : I.Tape}
    {h : List (Msg × Msg)} {stg : Stage} {xOf : ℕ → Msg} (hS : ∀ s < RecDeploy.nS x, StagedPart stg s x (xOf s))
    (hR : ∀ s st c src pin, partAt x s = some (st, c) → VStarBuild.srcOf I x w ρ h = some src →
      VStar.In.partIn src (progOf c) st = .ok pin → RowsOk c pin) :
    ∀ s st c src pin, partAt x s = some (st, c) → VStarBuild.srcOf I x w ρ h = some src →
      VStar.In.partIn src (progOf c) st = .ok pin → TraceOk c pin :=
  fun s st c src pin hp hsrc hi => traceOk_staged (hS s (partAt_lt hp)) hp hi (hR s st c src pin hp hsrc hi)

end RecDeploy.VStarNets
