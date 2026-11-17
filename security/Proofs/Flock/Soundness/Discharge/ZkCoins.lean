module

public import Proofs.Flock.Soundness.Discharge.ZkCoins.Basic
public import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Statements
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Weights
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Lazy
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Leaf
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Hybrid
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Erase
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Reads
import Mathlib.Combinatorics.Matroid.Init

/-! The coin step in the `--zk` session: its soundness on committed coins (`SessionCoinsHm96`) is stated and no
theorem proves it, while the coin step (`committed_le`, `committed_le_hm96`) and `er_sessionZKC` are proved. -/

@[expose] public section
