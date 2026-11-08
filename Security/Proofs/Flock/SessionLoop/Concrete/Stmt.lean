module

public import Proofs.Flock.SessionLoop.Concrete
public import Proofs.Flock.SessionLoop.Concrete.Inputs
public import Proofs.Flock.SessionLoop.Concrete.Sizes
import all Init.Data.String.Defs

/-! # A concrete statement the deployed `--zk` setup parses

`concreteStmt` is the encoding of `concreteRaw`: the `verity/flock-circuit@eb90718f` statement (`Concrete.wC`) on the
circuit file holding the unit `copy16` and the verifier's two fixed nets (`Concrete.bytesOf`), and the public file naming
that file's SHA-512 (`concreteHash`). At the open guarantee that the two nets build
(`Properties.Flock.Nets.FixedNetsBuild`):

* `concreteStmt_decode`: it decodes back to `concreteRaw`, since every length it writes fits its word (`concreteRaw_fits`);
* `concreteStmt_setup`: `FlockDeploy.setupOf true` parses it, to the inputs `concreteIns` and the setup `concreteSetup`
  (`concreteStmt_setupOf`), whose session has one table (`concreteStmt_table`).
-/

@[expose] public section

namespace FlockDeploy

open Flock
open Properties.Flock.Nets (Builds built sparseOf FixedNetsBuild)

/-! ## The statement and its round trip -/

/-- The SHA-512 of the concrete statement's circuit file, which its public file names. -/
def concreteHash : ByteArray := Sha512.hash (Concrete.bytesOf (HmNets.shaLaid ()) (HmNets.hmLaid ()))

/-- The concrete statement, decoded. -/
def concreteRaw : Raw := Concrete.wC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash

/-- **A concrete `--zk` statement**: `concreteRaw`'s encoding. -/
def concreteStmt : Claims.Msg := concreteRaw.encode

theorem name_size : Concrete.NAME.toUTF8.size < 2 ^ 64 := by decide +kernel

theorem Concrete.fits_wC (ls lh : HmNets.Laid) (H : ByteArray) (hc : (Concrete.bytesOf ls lh).size < 2 ^ 64)
    (hp : (Concrete.pubOf (hexOf H)).size < 2 ^ 64) : (Concrete.wC ls lh H).Fits where
  statement := name_size
  circuit := hc
  publicFile := hp
  tables := by simp [Concrete.wC]
  tableItems := by simp [Concrete.wC]
  count := by simp [Concrete.wC]
  draw := by intro _ h; cases h
  partition := by intro _ h; cases h
  program := by intro _ h; cases h
  work := by intro _ h; cases h
  workTable := by intro _ h; cases h
  stratified := by intro _ h; cases h
  closure := by intro _ h; cases h
  registered := by intro _ h; cases h
  publicInputs := by intro _ h; cases h

/-- A net that builds has a line per useful column and its header's (`VStar.netLines`). -/
theorem netLines_of {name : String} {l : HmNets.Laid} {useful : Nat} {ig og : Group} {w : Nat}
    (hb : Builds name l useful ig og w) : VStar.netLines l = 1 + useful := by
  obtain ⟨-, -, h, -⟩ := hb
  have h' : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb =
      .ok (built name useful ig og w (sparseOf l.ra) (sparseOf l.rb)) := h
  have hu : useful = l.useful := (Net.ofRows_ok h').2.2.2.1
  unfold VStar.netLines
  omega

theorem sha_lines (hn : FixedNetsBuild) : VStar.netLines (HmNets.shaLaid ()) = 256002 := netLines_of hn.1

theorem hm_lines (hn : FixedNetsBuild) : VStar.netLines (HmNets.hmLaid ()) = 246786 := netLines_of hn.2

theorem concreteRaw_fits (hn : FixedNetsBuild) : concreteRaw.Fits :=
  Concrete.fits_wC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash
    (Concrete.circuit_size _ _ hn.1.2.2.2 hn.2.2.2.2 (sha_lines hn) (hm_lines hn))
    (Concrete.public_size concreteHash (Concrete.hash_size _))

/-- **The concrete statement decodes** to the `Raw` it encodes. -/
theorem concreteStmt_decode (hn : FixedNetsBuild) : decode concreteStmt = some concreteRaw :=
  decode_encode_of_fits (concreteRaw_fits hn)

/-! ## Its setup -/

theorem insOf_of {zk : Bool} {x : Claims.Msg} {w : Raw} {s : Ins} (hd : decode x = some w) (hz : w.zk = zk)
    (hi : w.ins = .ok s) : insOf zk x = some s := by
  simp [insOf, hd, hz, hi, Except.toOption]

theorem setupOf_of {zk : Bool} {x : Claims.Msg} {s : Ins} {p : SessionSpec × Array Stmt} (hi : insOf zk x = some s)
    (hs : s.setup = .ok p) : setupOf zk x = some (s, p) := by
  obtain ⟨spec, sts⟩ := p
  simp [setupOf, hi, hs, Except.toOption]

/-- The concrete statement's inputs (`Concrete.ins_W`). -/
def concreteIns : Ins := Concrete.insC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash

/-- The concrete statement's session and tables (`Concrete.setup_W`), at the canonical forms the setup computes. -/
noncomputable def concreteSetup : SessionSpec × Array Stmt :=
  Concrete.setW (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash (sparseOf (HmNets.shaLaid ()).ra)
    (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C
    Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C Concrete.helloC concreteIns

theorem concreteStmt_ins (hn : FixedNetsBuild) : insOf true concreteStmt = some concreteIns :=
  insOf_of (concreteStmt_decode hn) rfl (Concrete.ins_W _ _ _)

theorem concreteIns_setup (hn : FixedNetsBuild) : concreteIns.setup = .ok concreteSetup :=
  have hp := fixedNets_parse hn
  Concrete.setup_W (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash (sparseOf (HmNets.shaLaid ()).ra)
    (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C
    Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C Concrete.helloC concreteIns rfl rfl rfl
    (Concrete.tags_rC _ _ _)
    (Concrete.setup_rC _ _ _ _ _ _ _ _ _ _ _ _ rfl (sha_lines hn) (hm_lines hn) hp.1.2 hp.2.2 Concrete.hc1_of
      Concrete.hc2_of (fun c h1 h2 => Concrete.check_of hp.1.1 hp.2.1 c h1 h2) Concrete.hu1_of
      (Concrete.hu2_of _ _ _ _ _ _ _ _) (Concrete.hd_of _) Concrete.hid_of Concrete.hhello_of)

theorem concreteStmt_setupOf (hn : FixedNetsBuild) :
    setupOf true concreteStmt = some (concreteIns, concreteSetup) :=
  setupOf_of (concreteStmt_ins hn) (concreteIns_setup hn)

/-- **The deployed `--zk` setup parses the concrete statement**, at the open guarantee that the verifier's two fixed
nets build. -/
theorem concreteStmt_setup (hn : FixedNetsBuild) : (setupOf true concreteStmt).isSome = true := by
  rw [concreteStmt_setupOf hn]
  rfl

theorem concreteSetup_tables : concreteSetup.2.size = 1 := rfl

/-- **The concrete statement's session has one table.** -/
theorem concreteStmt_table (hn : FixedNetsBuild) :
    ∃ s spec sts, setupOf true concreteStmt = some (s, spec, sts) ∧ sts.size = 1 :=
  ⟨concreteIns, concreteSetup.1, concreteSetup.2, concreteStmt_setupOf hn, concreteSetup_tables⟩

end FlockDeploy
