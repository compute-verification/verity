//! RELATION-ONLY DIAGNOSTIC (feature `unsound-relation-only`): never sound, never a result.
//!
//! A guest built with this feature runs nothing but the relation of one
//! `GemmCoordinate<K>` obligation in `veritor.tc-ampere-bf16@1`: no statement,
//! no witness, no opening table, no leaf hashing.  It is the arithmetic
//! baseline of the zkVM path (`archive/AGENTS.md`, "SP1 is the system-level
//! reference"): the sound K=1536 guest spends 82 % of its cycles parsing the
//! statement and building the opening table, 4.2 % in the relation
//! (`r20260921-224405-22dc`), and `unsound-skip-authentication` removes only
//! the 1.3 % of leaf hashing and path folding.
//!
//! The **compact input** is
//!
//! ```text
//! K: u32 LE || x[0..K]: u16 LE (ascending k) || W[0..K]: u16 LE (ascending k)
//! ```
//!
//! -- the x row then the W row of the one obligation's boundary, exactly the
//! `values` the sound guest decodes from the openings -- and the guest commits
//!
//! ```text
//! sha256(compact input) || sha256(output word: u16 LE) || 0xEE        (65 bytes)
//! ```
//!
//! the shape of the sound guest's `h_O || h_L || verdict`, so the host's
//! plumbing is unchanged.  The relation is [`evaluate`]: the lowered body of
//! `GemmCoordinate<K>` gate by gate -- `tc_dot16_0` over block 0, `tc_dot16`
//! chained over blocks `1..K/16`, `f32_to_bf16` -- through the same
//! [`GateSet::evaluate`] the kernel's `check_relations` calls, argument
//! frame and width checks included; nothing about the arithmetic is
//! reimplemented.  A malformed input (K not a positive multiple of 16, a
//! length other than `4 + 4K`) or a chain that rejects (a non-finite word)
//! commits `[0; 32]` in the output slot and the reject byte `0xEF`.
//!
//! Cycle scopes: `relation_only.parse` (decoding the words), `gates` (the
//! chain; the same name as the sound guest's relation pass, so the two are
//! read side by side) and `relation_only.digest` (both hashes, through the
//! `sha2` crate the sound guest's `digest` phase uses -- the SP1 precompile
//! in the guest); the guest adds `relation_only.io` around `read_vec`.

use crate::frame::sha256;
use crate::gateset::{GateSet, Op, TC_AMPERE_BF16_ID, TC_K};
use crate::{PUBLIC_VALUES_WITH_OBJECT_LEN, VERDICT_ACCEPT, VERDICT_REJECT};

/// Why a compact input is refused (the guest then commits the reject byte).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompactError {
    /// Fewer than the four bytes of `K`.
    Short,
    /// `K` is zero or not a multiple of 16 (a `GemmCoordinate<K>` body has `K/16` step gates).
    K,
    /// The byte length is not `4 + 4K`.
    Length,
}

/// Encode a compact input from the two rows (the test-side and Python-side layout reference).
/// `None` when the rows differ in length, are empty or not a multiple of 16, or hold a word above 16 bits.
pub fn encode_compact(x: &[u64], w: &[u64]) -> Option<Vec<u8>> {
    if x.len() != w.len() || x.is_empty() || x.len() % TC_K != 0 || x.len() > u32::MAX as usize {
        return None;
    }
    let mut out = Vec::with_capacity(4 + 4 * x.len());
    out.extend_from_slice(&(x.len() as u32).to_le_bytes());
    for &word in x.iter().chain(w) {
        if word > 0xFFFF {
            return None;
        }
        out.extend_from_slice(&(word as u16).to_le_bytes());
    }
    Some(out)
}

/// Decode a compact input into its two rows of `K` words (each below `2^16`).
pub fn parse_compact(bytes: &[u8]) -> Result<(Vec<u64>, Vec<u64>), CompactError> {
    if bytes.len() < 4 {
        return Err(CompactError::Short);
    }
    let k = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if k == 0 || k % TC_K != 0 {
        return Err(CompactError::K);
    }
    if bytes.len() != 4 + 4 * k {
        return Err(CompactError::Length);
    }
    let words = |row: &[u8]| -> Vec<u64> {
        row.chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]) as u64)
            .collect()
    };
    Ok((words(&bytes[4..4 + 2 * k]), words(&bytes[4 + 2 * k..])))
}

/// The relation of `GemmCoordinate<K>` on one row pair, as the kernel evaluates it: the BF16 output
/// word, or `None` where the kernel's `check_relations` would reject (a word outside its gate's
/// width, a non-finite tensor-core operand).  `x` and `w` must be `K` words each, `K` a positive
/// multiple of 16.
pub fn evaluate(x: &[u64], w: &[u64]) -> Option<u64> {
    if x.len() != w.len() || x.is_empty() || x.len() % TC_K != 0 {
        return None;
    }
    let set = GateSet::resolve(TC_AMPERE_BF16_ID, 32).expect("the pinned family resolves at width 32");
    // the interpreter's argument frame, `[acc, a0..a15, b0..b15]` (`relations::gather_args`)
    let mut args: Vec<u64> = Vec::with_capacity(1 + 2 * TC_K);
    args.extend_from_slice(&x[..TC_K]);
    args.extend_from_slice(&w[..TC_K]);
    let mut acc = set.evaluate(Op::TcDot16_0, &args)?;
    for block in 1..x.len() / TC_K {
        args.clear();
        args.push(acc);
        args.extend_from_slice(&x[block * TC_K..(block + 1) * TC_K]);
        args.extend_from_slice(&w[block * TC_K..(block + 1) * TC_K]);
        acc = set.evaluate(Op::TcDot16, &args)?;
    }
    args.clear();
    args.push(acc);
    set.evaluate(Op::F32ToBf16, &args)
}

/// The output word as hashed into the public values: two little-endian bytes, like the input words.
pub fn output_word_bytes(word: u64) -> [u8; 2] {
    (word as u16).to_le_bytes()
}

/// The 65 public bytes for a compact input: `sha256(input) || sha256(output word) || verdict`, with
/// the output slot `[0; 32]` and the reject byte when the input is malformed or the chain rejects.
pub fn decide(input: &[u8]) -> [u8; PUBLIC_VALUES_WITH_OBJECT_LEN] {
    scope_start!("relation_only.parse");
    let rows = parse_compact(input);
    scope_end!("relation_only.parse");

    scope_start!("gates");
    let output = rows.ok().and_then(|(x, w)| evaluate(&x, &w));
    scope_end!("gates");

    scope_start!("relation_only.digest");
    let input_digest = sha256(input);
    let output_digest = output.map(|word| sha256(&output_word_bytes(word)));
    scope_end!("relation_only.digest");

    let mut out = [0u8; PUBLIC_VALUES_WITH_OBJECT_LEN];
    out[..32].copy_from_slice(&input_digest);
    match output_digest {
        Some(digest) => {
            out[32..64].copy_from_slice(&digest);
            out[64] = VERDICT_ACCEPT;
        }
        None => out[64] = VERDICT_REJECT,
    }
    out
}

#[cfg(test)]
mod tests {
    //! The compact-input relation is the gate-set evaluation of the recorded vectors: for the K=16
    //! and K=1536 `GemmCoordinate` records under `fixtures/typed-obligation-v0`, [`evaluate`] on
    //! the boundary rows gives the committed output word, the kernel's own `check_relations` on the
    //! recorded statement accepts the same values, and [`decide`] commits the documented 65 bytes.

    use super::*;
    use crate::codec::parse_statement;
    use crate::kernel::gate_set;
    use crate::kernel::relations::check_relations;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/typed-obligation-v0");

    /// `(K, x row, W row, committed output word, statement bytes)` of a one-obligation GemmCoordinate vector.
    fn vector(name: &str) -> (usize, Vec<u64>, Vec<u64>, u64, Vec<u8>) {
        let text = std::fs::read_to_string(format!("{DIR}/{name}.json")).expect("the recorded vector");
        let record: serde_json::Value = serde_json::from_str(&text).unwrap();
        let sub = &record["object"]["subcircuits"][0];
        assert_eq!(sub["definition"]["name"], "GemmCoordinate", "{name}");
        let k = sub["definition"]["static"]["K"].as_u64().unwrap() as usize;
        assert_eq!(record["object"]["obligations"].as_array().unwrap().len(), 1, "{name}: one obligation");
        let values: Vec<u64> = record["values"][0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(values.len(), 2 * k + 1, "{name}: x row, W row, one output");
        let statement = record["statement"].as_str().unwrap();
        let statement = (0..statement.len() / 2).map(|i| u8::from_str_radix(&statement[2 * i..2 * i + 2], 16).unwrap()).collect();
        (k, values[..k].to_vec(), values[k..2 * k].to_vec(), values[2 * k], statement)
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// `input_digest` pins the compact bytes across languages: the Python builder
    /// (`verity_sp1.relation_only.compact_input`) gives the same sha256 for the same vector.
    fn check(name: &str, expected_k: usize, input_digest: &str) {
        let (k, x, w, out, statement) = vector(name);
        assert_eq!(k, expected_k);
        // the kernel's relation pass on the recorded statement accepts the recorded values ...
        let statement = parse_statement(&statement).unwrap();
        let set = gate_set(&statement).unwrap();
        let mut row = x.clone();
        row.extend(&w);
        row.push(out);
        assert_eq!(check_relations(&statement, &[row.clone()], &set), Ok(()), "{name}");
        // ... and rejects a wrong output word, so agreement below is not vacuous
        *row.last_mut().unwrap() ^= 1;
        assert!(check_relations(&statement, &[row], &set).is_err(), "{name}");
        // the bare chain gives the committed output word
        assert_eq!(evaluate(&x, &w), Some(out), "{name}: relation-only evaluation");
        // and the public values are as documented
        let compact = encode_compact(&x, &w).unwrap();
        assert_eq!(compact.len(), 4 + 4 * k);
        assert_eq!(hex(&sha256(&compact)), input_digest, "{name}: the compact bytes the Python side builds");
        assert_eq!(parse_compact(&compact), Ok((x.clone(), w.clone())));
        let public = decide(&compact);
        assert_eq!(&public[..32], &sha256(&compact));
        assert_eq!(&public[32..64], &sha256(&output_word_bytes(out)));
        assert_eq!(public[64], VERDICT_ACCEPT);
        #[cfg(feature = "unsound-relation-only")]
        assert_eq!(public[64], 0xEE);
    }

    #[test]
    fn k16_vector_matches_the_gate_set_evaluation() {
        check(
            "accept--packed-word-row",
            16,
            "f72d64523d28995b114189993eb4c419bd095c23a35d575510ef68c562943656",
        );
    }

    #[test]
    fn k1536_vector_matches_the_gate_set_evaluation() {
        check(
            "stage2--qkv-proj-decode-0-k1536",
            1536,
            "2ab508b9d12b9bfe7933c95edc0ae26e64f29b99f745867ce50f6e4344fd8561",
        );
    }

    #[test]
    fn malformed_inputs_and_rejecting_chains_commit_the_reject_byte() {
        let (_, x, w, _, _) = vector("accept--packed-word-row");
        let compact = encode_compact(&x, &w).unwrap();
        for bad in [&compact[..3], &compact[..compact.len() - 1], &[compact.as_slice(), &[0u8]].concat()] {
            let public = decide(bad);
            assert_eq!(&public[..32], &sha256(bad));
            assert_eq!(&public[32..64], &[0u8; 32]);
            assert_eq!(public[64], VERDICT_REJECT);
        }
        let mut k15 = compact.clone();
        k15[..4].copy_from_slice(&15u32.to_le_bytes());
        assert_eq!(parse_compact(&k15), Err(CompactError::K));
        assert_eq!(parse_compact(&compact[..3]), Err(CompactError::Short));
        assert_eq!(parse_compact(&compact[..compact.len() - 2]), Err(CompactError::Length));
        // a non-finite operand rejects the step gate, as the kernel does
        let mut nan = x.clone();
        nan[0] = 0x7FC0;
        assert_eq!(evaluate(&nan, &w), None);
        assert_eq!(decide(&encode_compact(&nan, &w).unwrap())[64], VERDICT_REJECT);
        // rows of the wrong shape are not a compact input
        assert!(encode_compact(&x[..15], &w[..15]).is_none());
        assert!(encode_compact(&[0x1_0000; 16], &w).is_none());
        assert_eq!(evaluate(&x[..15], &w[..15]), None);
    }
}
