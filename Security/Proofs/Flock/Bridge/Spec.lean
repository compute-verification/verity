module

public import Proofs.Flock.Bridge.Spec.Tree
public import Proofs.Flock.Bridge.Spec.Row

/-! C-Flock's commitment code as the trusted hash text states it, the bridge an opening of the deployed commitment
(`Flock.Bridge.commitDepL`) needs to be one of `Hash.RowLeaf`'s. `Tree`: the registered path check
(`Flock.Registered.opensLeaf`) is `Hash.Tree.accepts`'s, at a 64-byte domain (`opensLeaf_accepts`). `Row`: the
`sha512/row/v2` message is `Hash.RowV2.msg` (`enc_eq`), and `Hm96.Default512`'s prefixes are `Hash.Hm96`'s
(`saltPrefix_eq`, `leafPrefix_eq`). -/
