import FlockSoundness.Discharge.ZkTree.Defs
import FlockSoundness.Discharge.ZkTree.Lazy
import FlockSoundness.Discharge.ZkTree.Hybrid
import FlockSoundness.Discharge.ZkTree.Audit
import FlockSoundness.Discharge.ZkTree.Sound
import FlockSoundness.Discharge.ZkTree.Real
import FlockSoundness.Discharge.ZkTree.Play
import FlockSoundness.Discharge.ZkTree.Layout

/-!
# The tree-coin hybrid (A6 under `--zk`, `coin-tree/hm96-sha512/v2`)

The model's session draws each round's coins when the round is reached (A6). Under `--zk` the coin server draws every
round's coins, and a salt for each, at `Hello`, and sends the root of a tree of keyed hm96 leaves with the session key;
a round's leaf is opened with the answer to the round's message. The model's coins are still the run's up to
`N·2^-256`, `N` the leaves under the key: for every event, the tree-coin run's probability is at most a mixture of model
provers' plus `N·2^-256` (`treeCoins_le_model`, `treeAudit_le_model`), and so `zk_session_sound` carries over with
`N·2^-256` added (`zk_session_sound_tree`). The argument is statistical: hm96's two-vector hiding, a leaf at a time.
-/
