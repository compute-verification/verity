module

public import Proofs.Properties.Flock.Nets.Build
import all Init.Data.String.Defs
import all Init.Data.Repr

/-!
# A built circuit's lines fit

A row's line (`VStar.rowLine`) takes at most 29 bytes per entry and per size when each is below `10 ^ 7`
(`rowLine_size`), so a layout whose ANDs' rows read at most `W` bits and whose outputs read at most `Wo` has every row's
line under `2 ^ 20` bytes when `29 (W + Wo + 4)` is (`layout_lines`). A header reads only the layout's header fields
(`header_eq`), so the two fixed nets' headers are evaluated in the kernel at their fields (`sha_header`, `hm_header`),
which needs the core's string and digit definitions whole. `builds_layout` puts these together with `layout_builds`:
a built circuit's layout builds its net and its lines fit.
-/

@[expose] public section

open Flock Properties.Flock.Nets
open Flock.HmNets hiding and

namespace FlockDeploy.Concrete.Nets

theorem utf8ByteSize_ofList_le : ∀ l : List Char, (String.ofList l).utf8ByteSize ≤ 4 * l.length
  | [] => by simp
  | c :: l => by
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, List.length_cons]
    have := Char.utf8Size_le_four c
    have := utf8ByteSize_ofList_le l
    omega

theorem toString_size {n : Nat} (h : n < 10000000) : (toString n).utf8ByteSize ≤ 28 := by
  rw [Nat.toString_eq_ofList_toDigits]
  have h1 := utf8ByteSize_ofList_le (Nat.toDigits 10 n)
  have h2 := (Nat.length_toDigits_le_iff (b := 10) (k := 7) (by decide) (by decide)).2 h
  omega

theorem foldl_size : ∀ (l : List Nat) (s : String), (∀ x ∈ l, x < 10000000) →
    (l.foldl (fun s x => (s.push ' ') ++ toString x) s).utf8ByteSize ≤ s.utf8ByteSize + 29 * l.length
  | [], s, _ => by simp
  | x :: l, s, h => by
    rw [List.foldl_cons]
    have ih := foldl_size l ((s.push ' ') ++ toString x) fun y hy => h y (List.mem_cons_of_mem _ hy)
    have hx := toString_size (h x List.mem_cons_self)
    rw [String.utf8ByteSize_append, String.utf8ByteSize_push] at ih
    have : ' '.utf8Size = 1 := rfl
    rw [List.length_cons]
    omega

/-- **A row's line takes at most 29 bytes per entry and per size**, each entry and size below `10 ^ 7`. -/
theorem rowLine_size (a b : Array Nat) (ha : ∀ x ∈ a, x < 10000000) (hb : ∀ x ∈ b, x < 10000000)
    (hsa : a.size < 10000000) (hsb : b.size < 10000000) :
    (VStar.rowLine a b).utf8ByteSize ≤ 29 * (a.size + b.size + 2) := by
  unfold VStar.rowLine
  simp only [Id.run, Array.forIn_pure_yield_eq_foldl, pure_bind, ← Array.foldl_toList]
  have h1 := foldl_size a.toList (toString a.size) fun x hx => ha x (Array.mem_toList_iff.1 hx)
  have h2 := foldl_size b.toList ((a.toList.foldl (fun s x => (s.push ' ') ++ toString x) (toString a.size)).push ' ' ++
    toString b.size) fun x hx => hb x (Array.mem_toList_iff.1 hx)
  rw [String.utf8ByteSize_append, String.utf8ByteSize_push] at h2
  have := toString_size hsa
  have := toString_size hsb
  have : ' '.utf8Size = 1 := rfl
  simp only [Array.length_toList] at h1 h2
  refine Nat.le_trans (Nat.le_of_eq (rfl : _ = (List.foldl (fun s x => s.push ' ' ++ toString x)
    ((List.foldl (fun s x => s.push ' ' ++ toString x) (toString a.size) a.toList).push ' ' ++ toString b.size)
      b.toList).utf8ByteSize)) ?_
  omega

theorem size_cols (nIn inRows cp : Nat) (f : Lin) : (cols nIn inRows cp f).size ≤ f.s.length + 1 := by
  unfold cols
  split <;> simp

/-- **Every row's line of a built circuit's layout is under `2 ^ 20` bytes** when its ANDs' rows read at most `W` bits,
its outputs at most `Wo`, its columns are below `10 ^ 7` and `29 (W + Wo + 4) < 2 ^ 20`. -/
theorem layout_lines (st : St) (inBits outBits : Array Nat) (outs : Array Lin) (nIn inW outW start W Wo : Nat)
    (hnIn : inBits.foldl (· + ·) 0 = nIn) (hinW : groupWords nIn = inW)
    (houtW : groupWords (outBits.foldl (· + ·) 0) = outW)
    (hstart : (inW * WORD + st.rows.length + outW * WORD - 1) / (outW * WORD) * (outW * WORD) = start)
    (hn : st.next = nIn + st.rows.length)
    (hout : ∀ k (hk : k < st.rows.reverse.length), st.rows.reverse[k].out = nIn + k)
    (hrow : ∀ r ∈ st.rows, (∀ v ∈ r.a.s, v < r.out) ∧ ∀ v ∈ r.b.s, v < r.out)
    (ho : ∀ f ∈ outs.toList, ∀ v ∈ f.s, v < st.next) (hsz : outs.size ≤ outW * WORD)
    (hw : ∀ r ∈ st.rows, r.a.s.length + r.b.s.length ≤ W) (hwo : ∀ f ∈ outs.toList, f.s.length ≤ Wo)
    (hcp : start + outW * WORD < 10000000) (hW : 29 * (W + Wo + 4) < 1048576) :
    ∀ i < (layout st inBits outBits outs).useful, (VStar.rowLine ((layout st inBits outBits outs).ra.getD i #[])
      ((layout st inBits outBits outs).rb.getD i #[])).utf8ByteSize < 2 ^ 20 := by
  obtain ⟨hsa, hsb, -, -, -, -, -, hst, hIn, hins, hpI, hpM, hpO, hands, houts, hcon, hord⟩ :=
    layout_facts st inBits outBits outs nIn inW outW start hnIn hinW houtW hstart hn hout hrow ho hsz _ rfl
  have hle := rows_le _ _ nIn (inW * WORD) _ hsa hsb hIn (by omega) hins hpI hcon hord
  generalize layout st inBits outBits outs = l at *
  intro i _
  have hs : (l.ra.getD i #[]).size + (l.rb.getD i #[]).size ≤ W + Wo + 2 := by
    by_cases h1 : i < nIn
    · rw [(hins i h1).1, (hins i h1).2]
      simp only [List.size_toArray, List.length_singleton]
      omega
    by_cases h2 : i < inW * WORD
    · rw [(hpI i (by omega) h2).1, (hpI i (by omega) h2).2]
      simp
    by_cases h3 : i < inW * WORD + st.rows.length
    · obtain ⟨k, rfl⟩ : ∃ k, i = inW * WORD + k := ⟨i - inW * WORD, by omega⟩
      have hk : k < st.rows.reverse.length := by simp; omega
      rw [(hands k hk).1, (hands k hk).2]
      have := size_cols nIn (inW * WORD) (start + outW * WORD) st.rows.reverse[k].a
      have := size_cols nIn (inW * WORD) (start + outW * WORD) st.rows.reverse[k].b
      have := hw _ (List.mem_reverse.1 (List.getElem_mem hk))
      omega
    by_cases h4 : i < start
    · rw [(hpM i (by omega) h4).1, (hpM i (by omega) h4).2]
      simp
    by_cases h5 : i < start + outs.size
    · obtain ⟨j, rfl⟩ : ∃ j, i = start + j := ⟨i - start, by omega⟩
      have hj : j < outs.size := by omega
      rw [(houts j hj).1, (houts j hj).2]
      have := size_cols nIn (inW * WORD) (start + outW * WORD) outs[j]
      have := hwo _ (Array.getElem_mem_toList hj)
      simp only [List.size_toArray, List.length_singleton]
      omega
    by_cases h6 : i < start + outW * WORD
    · rw [(hpO i (by omega) h6).1, (hpO i (by omega) h6).2]
      simp
    by_cases h7 : i = start + outW * WORD
    · subst h7
      rw [hcon.1, hcon.2]
      simp only [List.size_toArray, List.length_singleton]
      omega
    · have ha : l.ra.getD i #[] = #[] := by
        simp [Array.getD_eq_getD_getElem?, hsa, show ¬ i < start + outW * WORD + 1 by omega]
      have hb : l.rb.getD i #[] = #[] := by
        simp [Array.getD_eq_getD_getElem?, hsb, show ¬ i < start + outW * WORD + 1 by omega]
      rw [ha, hb]
      simp
  have hlt : ∀ c ∈ l.ra.getD i #[] ++ l.rb.getD i #[], c < 10000000 := fun c hc => by
    have := hle i c hc
    omega
  have := rowLine_size (l.ra.getD i #[]) (l.rb.getD i #[]) (fun c hc => hlt c (Array.mem_append_left _ hc))
    (fun c hc => hlt c (Array.mem_append_right _ hc)) (by omega) (by omega)
  show _ < 1048576
  omega

/-- A header reads only the layout's header fields. -/
theorem header_eq (name : String) (l : Laid) :
    VStar.header name l = VStar.header name ⟨l.useful, l.constPos, l.inGroups, l.outGroups, #[], #[]⟩ := rfl

/-- `sha512x3`'s header at its fields fits. -/
theorem sha_header : (VStar.header "sha512x3"
    ⟨253952 + 16 * WORD + 1, 253952 + 16 * WORD, #[⟨0, 64, SHA_IN⟩], #[⟨253952 / WORD, 16, SHA_OUT⟩], #[], #[]⟩).utf8ByteSize
      < 2 ^ 20 := by
  decide +kernel

/-- `hm96`'s header at its fields fits. -/
theorem hm_header : (VStar.header "hm96"
    ⟨245760 + 8 * WORD + 1, 245760 + 8 * WORD, #[⟨0, 32, HM_IN⟩], #[⟨245760 / WORD, 8, HM_OUT⟩], #[], #[]⟩).utf8ByteSize
      < 2 ^ 20 := by
  decide +kernel

/-- **A built circuit's layout builds its net and its lines fit**: `layout_builds` and `layout_lines`, with the
header's length at its fields. -/
theorem builds_layout (name : String) (st : St) (inBits outBits : Array Nat) (outs : Array Lin)
    (nIn inW outW start W Wo : Nat) (hnIn : inBits.foldl (· + ·) 0 = nIn) (hinW : groupWords nIn = inW)
    (houtW : groupWords (outBits.foldl (· + ·) 0) = outW)
    (hstart : (inW * WORD + st.rows.length + outW * WORD - 1) / (outW * WORD) * (outW * WORD) = start)
    (hn : st.next = nIn + st.rows.length)
    (hout : ∀ k (hk : k < st.rows.reverse.length), st.rows.reverse[k].out = nIn + k)
    (hrow : ∀ r ∈ st.rows, (∀ v ∈ r.a.s, v < r.out) ∧ ∀ v ∈ r.b.s, v < r.out)
    (ho : ∀ f ∈ outs.toList, ∀ v ∈ f.s, v < st.next) (hsz : outs.size ≤ outW * WORD)
    (hU : max 7 (log2ceil (start + outW * WORD + 1)) = 18) (hD : start / WORD + outW ≤ 2 ^ 18 / WORD)
    (hw : ∀ r ∈ st.rows, r.a.s.length + r.b.s.length ≤ W) (hwo : ∀ f ∈ outs.toList, f.s.length ≤ Wo)
    (hcp : start + outW * WORD < 10000000) (hW : 29 * (W + Wo + 4) < 1048576)
    (hhdr : (VStar.header name ⟨start + outW * WORD + 1, start + outW * WORD, #[⟨0, inW, inBits⟩],
      #[⟨start / WORD, outW, outBits⟩], #[], #[]⟩).utf8ByteSize < 2 ^ 20) :
    Builds name (layout st inBits outBits outs) (start + outW * WORD + 1) ⟨0, inW, inBits⟩
      ⟨start / WORD, outW, outBits⟩ inW := by
  obtain ⟨hsa, hsb, hu, hc, hig, hog, -⟩ :=
    layout_facts st inBits outBits outs nIn inW outW start hnIn hinW houtW hstart hn hout hrow ho hsz _ rfl
  refine ⟨hsa, hsb, layout_builds name st inBits outBits outs nIn inW outW start hnIn hinW houtW hstart hn hout hrow
    ho hsz hU hD, ?_, layout_lines st inBits outBits outs nIn inW outW start W Wo hnIn hinW houtW hstart hn hout hrow
    ho hsz hw hwo hcp hW⟩
  rw [header_eq, hu, hc, hig, hog]
  exact hhdr

end FlockDeploy.Concrete.Nets
