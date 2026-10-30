import Proofs.Flock.Recursive.LinksBind
import Proofs.Flock.Verifier.RecLinks
import Proofs.Flock.Verifier.Hash
import Proofs.Flock.Claims.Recursion.InnerSetupSame
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Claims.Recursion.DeployChecked
import Proofs.Flock.Claims.KS.TableBound

/-!
# The link clause's inner half: an accepted link row binds the inner witness's rows

A link row's unit (`RecLinkRef.linkRef`) outputs the roots of the trees its reader's and its writer's openings climb.
Where those are the roots of the verifier's two trees over the inner statement's rows (`RecLinkPart.innerRows`, the
digests its files publish), the row holds at every witness of the inner relation (`LinkHolds`), or SHA-512 collides on
two of a short list of strings (`innerMsgs`), each a function of the row's words, the inner statement and the witness.

* `commitString_binding`: two `hm96` commit strings of 64-byte digests that are one have one digest, or the salts'
  digest inputs collide;
* `mem_innerRows_in`, `mem_innerRows_out`: what a row of the verifier's trees is: unit `u`'s input row `q`, or its
  output row `o` at port `nIn + o`, with the digest the files publish there;
* `inner_of_R`: at a statement the reader parses, a witness of the inner relation is a witness of the inner statement
  the statement's bytes parse to (`innerParsed`), which the verifier set up as one table at the files' rows;
* `inner_link`: the chain, `RecLinkBind.link_binding` (the openings), `innerSetup_rows` (the files' rows are the
  setup's), `InnerRows.rows_commit_check` (the witness commits them) and `commitString_binding`.
-/

namespace RecDeploy.Links

open Flock
open Claims (Msg)
open RecLinkRef (bytesOf field expected linkRef)

/-- **The `hm96` commit string binds its digest**: two commit strings of 64-byte digests that are one have one
digest, or the salts' digest inputs are distinct with one SHA-512. -/
theorem commitString_binding {x x' y y' : ByteArray} (hx : x.size = 64) (hx' : x'.size = 64)
    (h : Hm96.Default512.commitString x y = Hm96.Default512.commitString x' y') :
    x = x' ∨ RecLinkBind.Collides (Hm96.Default512.saltPrefix ++ y, Hm96.Default512.saltPrefix ++ y') := by
  have h' : Hm96.xor64 x (Hm96.Default512.mask y) ++ Sha512.hash (Hm96.Default512.saltPrefix ++ y) =
      Hm96.xor64 x' (Hm96.Default512.mask y') ++ Sha512.hash (Hm96.Default512.saltPrefix ++ y') := h
  have hb := ByteArray.append_inj_left h' (by rw [Hm96.xor64_size, Hm96.xor64_size])
  by_cases hy : y = y'
  · subst hy
    exact .inl (Hm96.xor64_cancel hx hx' hb)
  · rw [hb] at h'
    exact .inr ⟨fun he => hy ((ByteArray.append_right_inj _).mp he), (ByteArray.append_right_inj _).mp h'⟩

/-- A row of the verifier's input-rows tree is unit `u`'s input row `q`, with the digest the statement publishes. -/
theorem mem_innerRows_in {st : Stmt} {u q : ℕ} {b : ByteArray} (h : ((u, q), b) ∈ (RecLinkPart.innerRows st).1) :
    u < st.pub.n ∧ q < HmOut.nIn st.c ∧ b = st.pub.digest u q st.c.ports.size := by
  simp only [RecLinkPart.innerRows, List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq] at h
  obtain ⟨u', hu, q', hq, ⟨rfl, rfl⟩, rfl⟩ := h
  exact ⟨hu, hq, rfl⟩

/-- A row of the verifier's output-rows tree is unit `u`'s output row `o`, at port `nIn + o`, with the digest the
statement publishes. -/
theorem mem_innerRows_out {st : Stmt} {u o : ℕ} {b : ByteArray} (h : ((u, o), b) ∈ (RecLinkPart.innerRows st).2) :
    u < st.pub.n ∧ o < st.c.outRowPorts.size ∧ b = st.pub.digest u (HmOut.nIn st.c + o) st.c.ports.size := by
  simp only [RecLinkPart.innerRows, List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq] at h
  obtain ⟨u', hu, o', ho, ⟨rfl, rfl⟩, rfl⟩ := h
  exact ⟨hu, ho, rfl⟩

/-- **The inner witness's statement**: at a statement the reader parses, a witness of the inner relation at its inner
bytes is a witness of the inner statement they parse to (`innerParsed`), which parses, passes the statement check, has
hidden outputs and is one table, set up at the files' circuit, instance count and rows (`innerSetup_rows`). -/
theorem inner_of_R {x : Msg} {cs : Claims.Recursion.CompiledStatement} (hx : parse x = some cs)
    {w : Claims.Flock.Wit} (hR : Claims.Flock.R (FlockDeploy.parse false Readable.relation) cs.inner w) :
    ∃ r p f spec' st' st, readX x = some r ∧ innerParsed r = some p ∧ Readable.Parses p ∧ p.check = .ok () ∧
      p.tags.hiddenOutputs = true ∧ Claims.Flock.RelS (Readable.toReadable Readable.relation p) w ∧ p.sts = #[st] ∧
      filesOf r = some (.files f) ∧ f.stmt "the inner session" = .ok (spec', st') ∧ st.c = st'.c ∧
      st.pub.n = st'.pub.n ∧ ∀ u < st'.pub.n, ∀ q < st'.c.ports.size,
        st.pub.digest u q st'.c.ports.size = st'.pub.digest u q st'.c.ports.size := by
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hx
  obtain ⟨r', hr', hxi'⟩ := readX_of_parse hx
  have hxi : xIOf x = r.w.inner := by rw [hxi', Option.some.inj (hr.symm.trans hr')]
  have e : xIOf x = (RecScope.toStmt (readOf r)).inner := by unfold xIOf; rw [hx]; rfl
  have hin : (RecScope.toStmt (readOf r)).inner = r.w.inner := e.symm.trans hxi
  obtain ⟨s, hs, hRS⟩ := hR
  rw [hin] at hs
  obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 hs
  obtain ⟨f, spec', st', si, spec, st, hfs, hstm, hset, htags, e1, e2, e3⟩ := innerSetup_rows hr ha
  obtain ⟨hps, hpr, -⟩ := FlockDeploy.KS.parseBytes_setup hp hset
  obtain ⟨si', spec₂, sts₂, hset₂, hck, -⟩ := parse_innerOk hx
  rw [hxi, hset] at hset₂
  cases hset₂
  obtain ⟨rd, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
  refine ⟨r, p, f, spec', st', st, hr, hp, Readable.parses_of_parse hrp, ?_, ?_, hRS, hps, hfs, hstm, e1, e2, e3⟩
  · show Readable.Read.check p.toRead = _
    rw [hpr]
    exact hck
  · show p.toRead.tags.hiddenOutputs = true
    rw [hpr, htags]
    rfl

/-- **The strings a link row's inner half names a collision among**: the reader's and the writer's tree pairs
(`RecLinkBind.readerPair`, `writerPair`), the salt digest inputs at each end (the witness's salt and the opening's),
and the two rows' bytes, at the units and rows the row names (`field`). -/
def innerMsgs (p : Readable.Parsed) (w : Claims.Flock.Wit) (UB PB : ℕ)
    (rows : List ((ℕ × ℕ) × ByteArray) × List ((ℕ × ℕ) × ByteArray)) (trow xr yr raux xw yw waux : List ℕ) :
    List ByteArray :=
  let f := field (bytesOf trow)
  let qw := HmOut.nIn p.sts[0]!.c + f 4
  [(RecLinkBind.readerPair UB PB rows.1 trow xr yr raux).1, (RecLinkBind.readerPair UB PB rows.1 trow xr yr raux).2,
    (RecLinkBind.writerPair UB PB rows.2 trow xw yw waux).1, (RecLinkBind.writerPair UB PB rows.2 trow xw yw waux).2,
    Hm96.Default512.saltPrefix ++ InnerRows.salt p w 0 (f 1) (f 2), Hm96.Default512.saltPrefix ++ bytesOf yr,
    Hm96.Default512.saltPrefix ++ InnerRows.salt p w 0 (f 3) qw, Hm96.Default512.saltPrefix ++ bytesOf yw,
    ⟨(InnerRows.rowBytes p w 0 (f 1) (f 2)).toArray⟩, ⟨(InnerRows.rowBytes p w 0 (f 3) qw).toArray⟩]

theorem collides_mem {ms : List ByteArray} {pr : ByteArray × ByteArray} (h1 : pr.1 ∈ ms) (h2 : pr.2 ∈ ms)
    (h : RecLinkBind.Collides pr) : ∃ a ∈ ms, ∃ b ∈ ms, RecLinkBind.Collides (a, b) :=
  ⟨pr.1, h1, pr.2, h2, h⟩

/-- **An accepted link row holds at the inner witness**: at a witness of the deployed relation at an inner statement
that parses, passes the statement check and has hidden outputs, set up as one table `st` at the files' statement
`st'`'s circuit, instance count and rows, a link row whose unit outputs the roots of the verifier's two trees over
`st'`'s rows (`RecLinkPart.innerRows`) holds (`LinkHolds`), or two of `innerMsgs` are distinct with one SHA-512. -/
theorem inner_link {p : Readable.Parsed} {w : Claims.Flock.Wit} (hP : Readable.Parses p) (hc : p.check = .ok ())
    (hh : p.tags.hiddenOutputs = true) (hR : Claims.Flock.RelS (Readable.toReadable Readable.relation p) w)
    {st st' : Stmt} (hst : p.sts = #[st]) (e1 : st.c = st'.c) (e2 : st.pub.n = st'.pub.n)
    (e3 : ∀ u < st'.pub.n, ∀ q < st'.c.ports.size,
      st.pub.digest u q st'.c.ports.size = st'.pub.digest u q st'.c.ports.size)
    {UB PB : ℕ} {R W : ByteArray} {row : List UInt8} {trow xr yr raux xw yw waux : List ℕ}
    (hRr : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows st').1 = .ok R)
    (hWr : RecLinkTree.rowsRoot UB PB (RecLinkPart.innerRows st').2 = .ok W)
    (hout : linkRef UB PB trow xr yr raux xw yw waux = .ok (expected R, expected W))
    (hrow : bytesOf trow = ⟨row.toArray⟩) :
    LinkHolds p w row ∨ ∃ a ∈ innerMsgs p w UB PB (RecLinkPart.innerRows st') trow xr yr raux xw yw waux,
      ∃ b ∈ innerMsgs p w UB PB (RecLinkPart.innerRows st') trow xr yr raux xw yw waux,
        RecLinkBind.Collides (a, b) := by
  obtain ⟨hj, h0⟩ : ∃ hj : 0 < p.sts.size, p.sts[0] = st := by
    rw [hst]
    exact ⟨by simp, rfl⟩
  have h0b : p.sts[0]! = st := by rw [getElem!_pos p.sts 0 hj, h0]
  have hq : HmOut.nIn p.sts[0]!.c = HmOut.nIn st'.c := by rw [h0b, e1]
  simp only [LinkHolds, ← hrow, h0b]
  rw [e1, e2]
  by_cases hk : field (bytesOf trow) 0 = RecLinks.KIND_LINK
  swap
  · exact .inl fun hk' => absurd hk' hk
  obtain ⟨hlen, -⟩ := RecLinkBind.linkRef_ok hout
  simp only [List.map_cons, List.map_nil, RecLinkRef.portWords, List.cons.injEq] at hlen
  obtain ⟨-, hxr, -, -, hxw, -, -, -⟩ := hlen
  have hxr64 : (bytesOf xr).size = 64 := by rw [RecLinkBind.bytesOf_size, hxr]
  have hxw64 : (bytesOf xw).size = 64 := by rw [RecLinkBind.bytesOf_size, hxw]
  have hle : st'.c.outRowPorts.size ≤ st'.c.ports.size := by
    have := InnerRows.outRows_le hP hc hh hj
    rwa [h0, e1] at this
  have hnIn : HmOut.nIn st'.c ≤ st'.c.ports.size := Nat.sub_le _ _
  rcases RecLinkBind.link_binding hRr hWr hout hk with ⟨hin, hon, hx⟩ | hcr | hcw
  rotate_left
  · exact .inr (collides_mem (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true])
      (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true]) hcr)
  · exact .inr (collides_mem (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true])
      (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true]) hcw)
  obtain ⟨hu1, hq2, hd1⟩ := mem_innerRows_in hin
  obtain ⟨hu3, ho4, hd3⟩ := mem_innerRows_out hon
  have hq2' : field (bytesOf trow) 2 < st'.c.ports.size := by omega
  have hq4' : HmOut.nIn st'.c + field (bytesOf trow) 4 < st'.c.ports.size := by
    unfold HmOut.nIn
    omega
  have hr1 := InnerRows.rows_commit_check hP hc hh hR hj (i := field (bytesOf trow) 1) (q := field (bytesOf trow) 2)
    (by rw [h0, e2]; exact hu1) (by rw [h0, e1]; exact hq2')
  have hr3 := InnerRows.rows_commit_check hP hc hh hR hj (i := field (bytesOf trow) 3)
    (q := HmOut.nIn st'.c + field (bytesOf trow) 4) (by rw [h0, e2]; exact hu3) (by rw [h0, e1]; exact hq4')
  rw [h0, e1, e3 _ hu1 _ hq2', ← hd1] at hr1
  rw [h0, e1, e3 _ hu3 _ hq4', ← hd3] at hr3
  rcases commitString_binding (Sha512.hash_size _) hxr64 hr1 with hb1 | hs1
  swap
  · exact .inr (collides_mem (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true])
      (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true]) hs1)
  rcases commitString_binding (Sha512.hash_size _) hxw64 hr3 with hb3 | hs3
  swap
  · exact .inr (collides_mem (by simp only [innerMsgs, hq, List.mem_cons, eq_self, true_or, or_true])
      (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true]) hs3)
  by_cases he : (⟨(InnerRows.rowBytes p w 0 (field (bytesOf trow) 1) (field (bytesOf trow) 2)).toArray⟩ : ByteArray) =
      ⟨(InnerRows.rowBytes p w 0 (field (bytesOf trow) 3) (HmOut.nIn st'.c + field (bytesOf trow) 4)).toArray⟩
  · exact .inl fun _ => ⟨hu1, hq2, hu3, ho4, congrArg (fun b : ByteArray => b.data.toList) he⟩
  · have hcol : RecLinkBind.Collides
        (⟨(InnerRows.rowBytes p w 0 (field (bytesOf trow) 1) (field (bytesOf trow) 2)).toArray⟩,
          ⟨(InnerRows.rowBytes p w 0 (field (bytesOf trow) 3) (HmOut.nIn st'.c + field (bytesOf trow) 4)).toArray⟩) :=
      ⟨he, hb1.trans (hx.trans hb3.symm)⟩
    exact .inr (collides_mem (by simp only [innerMsgs, List.mem_cons, eq_self, true_or, or_true])
      (by simp only [innerMsgs, hq, List.mem_cons, eq_self, true_or, or_true]) hcol)

end RecDeploy.Links
