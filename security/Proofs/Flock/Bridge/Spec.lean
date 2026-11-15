module

public import Proofs.Flock.Bridge.Spec.Tree
public import Proofs.Flock.Bridge.Spec.Row
public import Proofs.Flock.Bridge.Spec.Hm96
public import Proofs.Flock.Bridge.Spec.RowSeg

/-! Flock's commitment code as the trusted hash text states it, the bridge an opening of the deployed commitment
(`Flock.Bridge.commitDepL`) needs to be one of `Hash.RowLeaf`'s. `Tree`: the registered path check
(`Flock.Registered.opensLeaf`) is `Hash.Tree.accepts`'s, at a 64-byte domain (`opensLeaf_accepts`). `Row`: the
`sha512/row/v2` message is `Hash.RowV2.msg` (`enc_eq`), and `Hm96.Default512`'s prefixes are `Hash.Hm96`'s
(`saltPrefix_eq`, `leafPrefix_eq`). `Hm96`: the executable's mask is `Hash.Hm96.mask` (`bs_mask`), and the Bridge's hm96
leaf is `Hash.Hm96.leaf` (`sha512Msg_hm96Outer`). `RowSeg`: the `sha512/row-seg/v1` outer message is
`Hash.RowSeg.outerMsg` (`segEnc_eq`), and a registered schema is a `Hash.RowLeaf` format (`formatOf`). -/
