module

public import Proofs.Flock.SpotFlock.InstanceStmt.Inputs
public import Proofs.Flock.SpotFlock.InstanceStmt.Setup
public import Proofs.Flock.SpotFlock.InstanceStmt.Texts

/-! # The two-instance statement's setup at the fixed nets

`setup_rX` is the reading's setup (`Read.setup`) at any layouts of the fixed nets, any circuit SHA-512 and any class,
under `setupTables_X`'s premises, as `FlockDeploy.Concrete.setup_rC` is the concrete statement's. `setup_x` is it at the
statement's own: where the verifier's two fixed nets build (`FixedNetsBuild`), every premise holds, the nets' from their
build (`FlockDeploy.fixedNets_parse`, `sha_lines`, `hm_lines`), the class from `classOf_eq`, and the texts and the hello
from their values holding only integers (`Texts`). Its one table is `tabAt r`.
-/

@[expose] public section
open Flock RecDeploy.VStarNets
open Flock.Readable (Read)
open Properties.Flock.Nets (sparseOf FixedNetsBuild)
open FlockDeploy.Concrete (shaB hmB)

namespace SpotFlock.InstanceStmt

theorem setup_rX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse)
    (s1 s3 s4 s5 s6 : String) (hello : ByteArray → Nat → String)
    (hr : r.Plain)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hcl : HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok C)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hCe : C.isEmpty = false)
    (hu1 : canon (unitsJ (hexOf C)) = .ok s3)
    (hu2 : canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok s3)
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4)
    (hd2 : canonUtf8 (hdrD r (cO ls lh H C A B A' B' s1) (hexOf H) (hexOf C)) = .ok s5)
    (hid : canonUtf8 (Tags.rowV2Identity tz.identity) = .ok s6)
    (hhello : ∀ σ m, helloOf tz σ m coinsX = .ok (hello σ m)) :
    Read.setup (rX r ls lh H C) = .ok (tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello) := by
  unfold Read.setup
  rw [tags_rX]
  exact setupTables_X r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello hr hH hcl hls hlh hps hph hc1 hc2 hck hCe hu1 hu2 hd
    hd2 hid hhello

/-! ## At the fixed nets -/

abbrev AF : Sparse := sparseOf (HmNets.shaLaid ()).ra
abbrev BF : Sparse := sparseOf (HmNets.shaLaid ()).rb
abbrev AF' : Sparse := sparseOf (HmNets.hmLaid ()).ra
abbrev BF' : Sparse := sparseOf (HmNets.hmLaid ()).rb

/-- The statement's circuit at the fixed nets. -/
noncomputable abbrev cF : Circuit :=
  cO (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X

/-- The statement's session and its one table at the fixed nets, each text and the hello named. -/
noncomputable abbrev tabAt (r : Run) : SessionSpec × Array Stmt :=
  tabX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X (s3X (hexOf classHash))
    (s4X r (hexOf circuitHash) (hexOf classHash)) (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX

/-- **The statement's setup**, where the verifier's two fixed nets build. -/
theorem setup_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : Read.setup (ins r).read = .ok (tabAt r) := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  exact setup_rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of

end SpotFlock.InstanceStmt
