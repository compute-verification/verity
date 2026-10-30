import Proofs.Flock.Claims.Recursion.Deploy
import Proofs.Flock.Recursive.InnerRows
import Proofs.Flock.Verifier.Registered
import Proofs.Flock.Soundness.Discharge.Hm.Row
import Flock.RecLinkPart

/-!
# The link clause at the deployment: what the link table's rows say, and the table they are

`RecursiveKSLinkAt` takes a link clause `Q`, the rows `rows` reads off the outer witnesses, and the link finder's inputs.
This file defines the first two at the deployment.

* `tableRows`: where the verifier's inner shape names the link table, row `n` is the link part's instance `n`'s `trow`,
  read off that part's witness at its circuit's message columns (`trowAt`, as `InnerRows.rowBytes` reads a row, behind no
  prefix). The link part is V\*'s statement last but one (`linkAt`), and its circuit is the one the statement's bytes
  hold (`partCircuit`), so the rows are a function of the statement and the witnesses alone.
* `LinksHold`, the link clause: at a statement carrying the table (its own root `T` and row count `N`, `--rec-links`),
  the rows are `N`, each opens its position of the registered table at `T` (`RowOpens`, under the domain the link part's
  program registers it at, `tableDomain`), and every link row holds at the inner witness (`LinkHolds`: the reader's and
  the writer's rows hash one string, `InnerRows.rowBytes`, at the inner statement the statement's bytes parse to).

`Q` holds the rows to `T`, not to a table the prover picks: two rows that open one position of the table at `T` are one,
or SHA-512 collides (`rowOpens_binding`, from the opening's binding and the `hm96` leaf's; `linksHold_rows`).
-/

namespace RecDeploy.Links

open Flock
open Claims (Msg)

/-! ## The rows, off the outer witnesses -/

/-- **The link table's part in V\*'s list** at the verifier's own inner shape: its index, the list's last but one, and
its row count `N`. -/
def linkAt (r : XR) : Option (ℕ × ℕ) :=
  (RecVStar.statements r.shape).toOption.bind fun ss =>
    match ss[ss.size - 2]? with
    | some (VStar.Stmt.link _ _ N) => some (ss.size - 2, N)
    | _ => none

/-- The link part's index, row count and circuit, as the setup parses the part (`partCircuit`). -/
def linkCircuit (r : XR) : Option (ℕ × ℕ × Circuit) := do
  let (s, N) ← linkAt r
  let c ← r.w.parts[s]?.bind partCircuit
  pure (s, N, c)

/-- The link part's port `trow` among a circuit's ports. -/
def trowPort (c : Circuit) : Option ℕ := c.ports.findIdx? (·.name == "trow")

/-- **Instance `n`'s port `q`, as a part's witness holds it**: the bytes at the port's message columns (`Typed.msgCol`)
of template instance `n % g` of block `n / g`, in the part's one table, as `InnerRows.rowBytes` reads a row behind its
prefix. -/
def trowAt (c : Circuit) (q : ℕ) (w : Claims.Flock.Wit) (n : ℕ) : List UInt8 :=
  FlockSoundness.Discharge.Hm.bytesOf (fun y => w.hid.getD (n / c.g * 2 ^ c.kLog + y) false) (2 * c.ports[q]!.words)
    (FlockSoundness.Typed.msgCol c (n % c.g) q)

/-- **The link table's rows, off the outer witnesses**: where the verifier's inner shape names the table, the link
part's `N` instances' `trow` in order (`trowAt`), in the link part's witness; none elsewhere. -/
def tableRows (x : Msg) (ws : List Claims.Flock.Wit) : List (List UInt8) :=
  match readX x with
  | none => []
  | some r => match linkCircuit r with
    | none => []
    | some (s, N, c) => match trowPort c with
      | none => []
      | some q => (List.range N).map (trowAt c q (ws.getD s ⟨[]⟩))

/-! ## The table at `T` -/

/-- The registered domain the link part's program puts a table of `N` rows under: `Registered.domain` at the program its
circuit's META names, the value `rec-links`, rows of `LINK_ROW_WORDS` u16 words. -/
def tableDomain (c : Circuit) (N : ℕ) : Option ByteArray := do
  let prog ← (c.metaJ.getObjValAs? String "program_sha512").toOption
  (Registered.domain prog RecLinkPart.LINKS_VALUE VStar.LINK_ROW_WORDS N 16).toOption

/-- The registered link table: root `T`, `N` `hm96-sha512/row/v1` rows of `LINK_ROW_WORDS` u16 words, under `dom`.
`Registered.opens` reads only its root, leaves, domain and schema. -/
def tablePort (dom T : ByteArray) (N : ℕ) : Registered.Port where
  name := "trow"
  value := RecLinkPart.LINKS_VALUE
  root := T
  leaves := N
  words := some VStar.LINK_ROW_WORDS
  wordBits := 16
  domain := dom
  readsRel := #[]

/-- A table row's bytes as `sha512/row/v1` hashes them. -/
def rowMsgOf (row : List UInt8) : ByteArray := FlockSoundness.Discharge.Hm.rowMsg VStar.LINK_ROW_WORDS row

/-- **Row `n` of the table at `T` is `row`**: under some salt, the commit string of the SHA-512 of the row's
`sha512/row/v1` message opens the registered table at `T` (`tablePort`) at position `n`, through some path. -/
def RowOpens (dom T : ByteArray) (N n : ℕ) (row : List UInt8) : Prop :=
  ∃ salt path : ByteArray,
    Registered.opens (tablePort dom T N) (Hm96.Default512.commitString (Sha512.hash (rowMsgOf row)) salt) n path = true

theorem rowMsgOf_inj {r r' : List UInt8} (h : rowMsgOf r = rowMsgOf r') : r = r' := by
  have h := congrArg (fun b : ByteArray => b.data.toList) h
  simp only [rowMsgOf, FlockSoundness.Discharge.Hm.rowMsg, Array.toList_toArray] at h
  exact List.append_cancel_left h

/-- **The table at `T` binds its rows**: two rows that open one position of the registered table at `T` under one
domain are one, or SHA-512 collides: the two commit strings' tree leaves differ (the opening's binding,
`Registered.opensLeaf_collision`), or the leaves are one and the digests or the salts differ (the `hm96` leaf's binding,
`Hm96.Default512.leaf_binding`), or the digests are one and the rows' `sha512/row/v1` messages differ. -/
theorem rowOpens_binding {dom T : ByteArray} {N n : ℕ} {row row' : List UInt8} (h : RowOpens dom T N n row)
    (h' : RowOpens dom T N n row') : row = row' ∨ Sha512.Collision := by
  obtain ⟨salt, path, ho⟩ := h
  obtain ⟨salt', path', ho'⟩ := h'
  by_cases hrow : row = row'
  · exact .inl hrow
  refine .inr ?_
  by_cases hm : rowMsgOf row = rowMsgOf row'
  · exact absurd (rowMsgOf_inj hm) hrow
  by_cases hd : Sha512.hash (rowMsgOf row) = Sha512.hash (rowMsgOf row')
  · exact ⟨_, _, hm, hd⟩
  have hne : (Sha512.hash (rowMsgOf row), salt) ≠ (Sha512.hash (rowMsgOf row'), salt') :=
    fun he => hd (Prod.mk.inj he).1
  by_cases hl : Hm96.Default512.leaf (Sha512.hash (rowMsgOf row)) salt =
      Hm96.Default512.leaf (Sha512.hash (rowMsgOf row')) salt'
  · exact Hm96.Default512.leaf_binding _ _ _ _ (Sha512.hash_size _) (Sha512.hash_size _) hne hl
  · exact Registered.opensLeaf_collision (tablePort dom T N) n (Hm96.Default512.leaf (Sha512.hash (rowMsgOf row)) salt)
      path (Hm96.Default512.leaf (Sha512.hash (rowMsgOf row')) salt') path' (Hm96.Default512.leaf_size _ _)
      (Hm96.Default512.leaf_size _ _) hl ho ho'

/-! ## The link clause -/

/-- The inner statement the statement's bytes parse to, as the deployed relation reads it (`FlockDeploy.parse false`:
`Readable.parseBytes` at the plain reader). -/
def innerParsed (r : XR) : Option _root_.Flock.Readable.Parsed :=
  _root_.Flock.Readable.parseBytes (FlockDeploy.readOf false) r.w.inner

/-- **Link row `row` holds at the inner witness `w`** of the inner statement `p`: where the row is a link
(`KIND_LINK`), its reader, unit `f₁`'s input row `f₂`, and its writer, unit `f₃`'s output row `f₄`, are rows of `p`'s
table, and `w`'s row hashes read one string at both (`InnerRows.rowBytes`; the writer's output row at port
`nIn + f₄`). Any other row says nothing. -/
def LinkHolds (p : _root_.Flock.Readable.Parsed) (w : Claims.Flock.Wit) (row : List UInt8) : Prop :=
  let f := RecLinkRef.field ⟨row.toArray⟩
  f 0 = RecLinks.KIND_LINK →
    f 1 < p.sts[0]!.pub.n ∧ f 2 < HmOut.nIn p.sts[0]!.c ∧ f 3 < p.sts[0]!.pub.n ∧ f 4 < p.sts[0]!.c.outRowPorts.size ∧
      InnerRows.rowBytes p w 0 (f 1) (f 2) = InnerRows.rowBytes p w 0 (f 3) (HmOut.nIn p.sts[0]!.c + f 4)

/-- **The link clause**: at a statement carrying the link table (its own root `T` and row count `N`, `--rec-links`),
the rows are `N`, row `n` opens position `n` of the registered table at `T` under the domain the link part's program
registers it at (`RowOpens`, `tableDomain`), and every row holds at the inner witness, at the inner statement the
statement's bytes parse to (`LinkHolds`). It says nothing at a statement carrying no table. -/
def LinksHold (x : Msg) (w : Claims.Flock.Wit) (rs : List (List UInt8)) : Prop :=
  ∀ r, readX x = some r → ∀ T N, r.w.links = some (T, N) →
    rs.length = N ∧
      (∀ s N' c dom, linkCircuit r = some (s, N', c) → tableDomain c N = some dom →
        ∀ n (hn : n < rs.length), RowOpens dom T N n rs[n]) ∧
      ∀ p, innerParsed r = some p → ∀ row ∈ rs, LinkHolds p w row

/-- The link clause says nothing at a statement carrying no table. -/
theorem linksHold_none {x : Msg} {r : XR} (hr : readX x = some r) (hl : r.w.links = none) (w : Claims.Flock.Wit)
    (rs : List (List UInt8)) : LinksHold x w rs := by
  intro r' hr' T N hl'
  rw [hr] at hr'
  cases hr'
  rw [hl] at hl'
  cases hl'

/-- **The clause holds the rows to `T`**: two row lists the clause takes at one statement carrying the table, where its
link part's program registers a domain, are one, or SHA-512 collides. -/
theorem linksHold_rows {x : Msg} {r : XR} (hr : readX x = some r) {T : ByteArray} {N : ℕ}
    (hl : r.w.links = some (T, N)) {s N' : ℕ} {c : Circuit} {dom : ByteArray} (hc : linkCircuit r = some (s, N', c))
    (hd : tableDomain c N = some dom) {w w' : Claims.Flock.Wit} {rs rs' : List (List UInt8)} (h : LinksHold x w rs)
    (h' : LinksHold x w' rs') : rs = rs' ∨ Sha512.Collision := by
  obtain ⟨hn, ho, -⟩ := h r hr T N hl
  obtain ⟨hn', ho', -⟩ := h' r hr T N hl
  by_cases hc' : Sha512.Collision
  · exact .inr hc'
  refine .inl (List.ext_getElem (hn.trans hn'.symm) fun n h1 h2 => ?_)
  rcases rowOpens_binding (ho s N' c dom hc hd n h1) (ho' s N' c dom hc hd n h2) with he | he
  · exact he
  · exact absurd he hc'

end RecDeploy.Links
