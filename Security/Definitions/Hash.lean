import Definitions.Hash.Basic
import Definitions.Hash.Tree
import Definitions.Hash.RowSeg
import Definitions.Hash.RowV2
import Definitions.Hash.Hm96
import Definitions.Hash.RowLeaf
import Definitions.Hash.EmbeddingRead

/-! The hash area's definitions: messages, digests and collisions among a run's inputs; frame-v3 trees,
`sha512/row-seg/v1`, `sha512/row/v2` and `hm96-sha512/v1` over a hash; the served bit-row leaf they compose; and
`EmbeddingRead_v1`, a read of one such leaf at a hidden rank. -/
