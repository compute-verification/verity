module

public import Proofs.Flock.VBridge.Family.Text
public import VStar.Compose
public import Std.Data.String.ToNat
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.List.SplitOn.Basic

/-!
# V\*'s circuit file as lines

V\*'s file is fixed text: two head lines, `flock-circuit <unit>` and `META <d>`, then three sections,
`CIRCUIT unit <n>`, `CIRCUIT sha512x3 <n>` and `CIRCUIT hm96 <n>`, each `<n>` the count of its net's lines
(`VStar.netLines`) and each net its header line and its rows. `fileLs` is those lines and `joinNl` writes lines out,
each with its newline. No line holds a newline (`fileLs_noNl`), so the text splits back into its lines
(`splitOn_join`), and a line of three words has them as its tokens (`tokens_three`).

This module imports the verifier's code and `Family/Text.lean`, and nothing else of the analysis: the concrete
statement's file (`SessionLoop/ConcreteNet.lean`, `SessionLoop/Concrete/File.lean`) is built from these lines, so it
is rebuilt only when the verifier's code or these few modules change. That V\*'s emitted file is these lines, and
that `HmRow.parse` reads three nets from it, is `VStarNetsThree.lean`.
-/

@[expose] public section

namespace RecDeploy.VStarNets

open _root_.Flock
open Flock.HmNets (Laid shaLaid hmLaid)
open FlockVBridge.Names

/-! ## V\*'s text -/

/-- Lines, each with its newline. -/
def joinNl (L : List String) : String := String.join (L.map (· ++ "\n"))

theorem joinNl_nil : joinNl [] = "" := (rfl)

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

/-! ## The lines -/

theorem splitOn_join : ∀ L : List String, (∀ x ∈ L, NoNl x) → (joinNl L).splitOn "\n" = L ++ [""]
  | [], _ => by rw [joinNl_nil, splitOn_nl]; rfl
  | a :: L, h => by
    rw [joinNl_cons, splitOn_line _ _ (h a List.mem_cons_self),
      splitOn_join L fun x hx => h x (List.mem_cons_of_mem _ hx), List.cons_append]

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

theorem secLs_length (nm net : String) (l : Laid) : (secLs nm net l).length = 1 + VStar.netLines l := by
  simp [secLs, netLs, rowsLs, VStar.netLines]
  omega

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

end RecDeploy.VStarNets
