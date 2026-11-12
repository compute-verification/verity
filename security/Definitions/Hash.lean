module

public import Definitions.Hash.Basic
public import Definitions.Hash.Tree
public import Definitions.Hash.RowSeg
public import Definitions.Hash.RowV1
public import Definitions.Hash.RowV2
public import Definitions.Hash.Hm96
public import Definitions.Hash.RowLeaf
public import Definitions.Hash.EmbeddingRead
public import Definitions.Hash.HistoryTable

/-! The hash area's definitions: messages, digests and collisions among a run's inputs; frame-v3 trees,
`sha512/row-seg/v1`, `sha512/row/v2`, `sha512/row/v1` and `hm96-sha512/v1` over a hash; the served row leaf they
compose;
`EmbeddingRead_v1`, a read of one such leaf at a hidden rank; and the keyed leaf, a frame-v3 subtree whose top message
is hashed under a window's secret key. -/

@[expose] public section
