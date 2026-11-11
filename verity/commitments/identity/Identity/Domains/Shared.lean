module

/-!
The shared domain separators: the entries of the one table (`Identity.Domains`, whose docstring has its rules) of the
identity hash, the commitments, randomness, SPOT, Flock and the served path, which more than one protocol reads.
-/

@[expose] public section

namespace Domains

-- Shared: the identity hash, frame-v3 trees and indexed domains (`verity.ml.commitments`)
abbrev TAGGED_SHA512 : String := "verity/tagged-sha512/v1"
abbrev TAGGED_SHA256 : String := "veritor/tagged-sha256/v1"
abbrev FRAME_V3 : String := "veritor/protocol/merkle/frame/v3"
abbrev FRAME_V3_SHA512_ROOT : String := "frame-v3-sha512/root/v1"
abbrev INDEXED_RANGE : String := "veritor/indexed-domain/range/v1"
abbrev INDEXED_EXPLICIT : String := "veritor/indexed-domain/explicit/v1"

-- Shared: hm96 leaves and row digests (`verity.ml.commitments`, `Flock.HmRow`)
abbrev HM96_SHA512 : String := "hm96-sha512/v1"
abbrev HM96_SHA512_NODE_ROW : String := "hm96-sha512/v1/node-row"
abbrev HM96_SHA512_ROW_V1 : String := "hm96-sha512/row/v1"
abbrev HM96_SHA512_ROW : String := "hm96-sha512/row/v2"
abbrev HM96_SHA512_ROW_SEG : String := "hm96-sha512/row-seg/v1"
abbrev HM96_SHA512_KEY : String := "verity/hm96-sha512/key/v1"
abbrev HM96_SHA512_LEAF : String := "verity/hm96-sha512/leaf/v1"
abbrev HM96_SHA512_SALT : String := "verity/hm96-sha512/salt/v1"
abbrev HM96_SHA512_SCHEME : String := "verity/hm96-sha512/scheme/v1"
abbrev HM96_SHA256 : String := "hm96-sha256/v1"
abbrev HM96_SHA256_KEY : String := "verity/hm96-sha256/key/v1"
abbrev HM96_SHA256_LEAF : String := "verity/hm96-sha256/leaf/v1"
abbrev HM96_SHA256_SALT : String := "verity/hm96-sha256/salt/v1"
abbrev HM96_SHA256_SCHEME : String := "verity/hm96-sha256/scheme/v1"
abbrev SHA512_ROW_DIGEST_V1 : String := "sha512/row/v1"
abbrev SHA512_ROW_DIGEST : String := "sha512/row/v2"
abbrev SHA512_ROW_SEG_DIGEST : String := "sha512/row-seg/v1"
abbrev SHA512_ROW_V1 : String := "verity/sha512-row/v1"
abbrev SHA512_ROW : String := "verity/sha512-row/v2"
abbrev SHA512_ROW_SEG : String := "verity/sha512-row-seg/v1"
abbrev SHA512_ROW_SEG_SEGMENT : String := "verity/sha512-row-seg/v1/segment"

-- Shared: randomness (`verity.ml.randomness`, `Randomness`)
abbrev RANDOMNESS_DERIVE : String := "verity/randomness/derive/v2"
abbrev RANDOMNESS_STREAM : String := "verity/randomness/stream/v2"
abbrev RANDOMNESS_SHARD : String := "verity/randomness/shard/v2"
abbrev RANDOMNESS_DERIVE_V1 : String := "verity/randomness/derive/v1"
abbrev RANDOMNESS_STREAM_V1 : String := "verity/randomness/stream/v1"

-- Shared: sampled proofs' one-stage statement, registrations and partitions; Flock's public inputs and descriptors
abbrev ONE_STAGE_PROTOCOL : String := "verity/sampled-proofs/one-stage/v1"
abbrev ONE_STAGE_PROTOCOL_V2 : String := "verity/sampled-proofs/one-stage/v2"
abbrev ONE_STAGE_SERVED_DOMAIN : String := "verity/one-stage/served-domain/v1"
abbrev REGISTRATION_V1 : String := "verity/registration/v1"
abbrev REGISTRATION : String := "verity/registration/v2"
abbrev PARTITION : String := "verity/partition/v1"
abbrev PARTITION_OWNERS : String := "verity/partition/v1/owners"
abbrev FLOCK_PUBLIC_INPUTS : String := "verity/flock-public-inputs/v1"
abbrev IR_DESCRIPTOR : String := "verity-ir/descriptor/v1"

-- Shared: the served path's PoUW manifests (`verity_vllm.protocols.pouw`)
abbrev VLLM_PEARL_C_RUN_V1 : String := "verity-vllm/pouw-pearl-c-run/v1"
abbrev VLLM_PEARL_C_RUN : String := "verity-vllm/pouw-pearl-c-run/v2"
abbrev VLLM_PEARL_C_RUN_CERTIFIED : String := "verity-vllm/pouw-pearl-c-run/v3"

end Domains
