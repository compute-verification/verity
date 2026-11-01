module

public import Definitions.Hash.Basic
public import Definitions.Hash.Tree
public import Definitions.Hash.RowSeg
public import Definitions.Hash.RowV2
public import Definitions.Hash.Hm96
public import Definitions.Hash.RowLeaf
public import Definitions.Hash.EmbeddingRead

/-! The hash area's definitions: messages, digests and collisions among a run's inputs; frame-v3 trees,
`sha512/row-seg/v1`, `sha512/row/v2` and `hm96-sha512/v1` over a hash; the served bit-row leaf they compose; and
`EmbeddingRead_v1`, a read of one such leaf at a hidden rank. -/

@[expose] public section
