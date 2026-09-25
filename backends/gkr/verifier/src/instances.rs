//! The frozen instance sets an operands-bound statement may name (`instances.txt`): per (relation, lo, vus) the
//! model whose circuits prove it and the SHA-256 of the operand words (`x.bin`, `w.bin`, little-endian, VU-major)
//! and of the public words at their width.  The verifier compares the statement's files against this table, so the
//! prover's harness cannot substitute operands.  `vu-k1536` is the committed frozen tier: its digests are those of
//! `fixtures/bench-instances/v1/manifest.json` (checked by the test below).  The recipe sets are
//! `backends/direct/ligero/relchain.py instances()` / `fp4/chain.py instances_fp4()` at their fixed seeds,
//! re-derivable with `python -m gpu.bind pin`.

pub struct Pin {
    pub relation: &'static str,
    pub lo: usize,
    pub vus: usize,
    pub model: &'static str,
    pub x: &'static str,
    pub w: &'static str,
    pub y: &'static str,
    pub source: &'static str,
}

pub const PINS: &[Pin] = &[
    Pin {
        relation: "vu-k1536", lo: 0, vus: 4096, model: "ampere_bf16_m16n8k16",
        x: "ebef04e94f90248cc86e1c7d157a9739a979d1cf2c4a98cec90cc501e6f6f3bd",
        w: "da301f94aa96ab9da7b285891400db3877d2947fa325f5c58783a52231725588",
        y: "16dcf48cc74c25d54375b49b2f33a85b7615a7321eab494521b797e74ce518ab",
        source: "fixtures/bench-instances/v1/manifest.json tier vu-k1536 (.x.u16, .w.u16, .y.u16)",
    },
];

pub fn find(relation: &str, lo: usize, vus: usize) -> Option<&'static Pin> {
    PINS.iter().find(|p| p.relation == relation && p.lo == lo && p.vus == vus)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_sha(text: &str, file: &str) -> String {
        let i = text.find(&format!("\"{file}\"")).unwrap_or_else(|| panic!("manifest: {file}"));
        let key = "\"sha256\": \"";
        let j = i + text[i..].find(key).unwrap() + key.len();
        text[j..j + 64].to_string()
    }

    #[test]
    fn frozen_tier_matches_the_committed_manifest() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/bench-instances/v1/manifest.json");
        let text = std::fs::read_to_string(path).expect("the committed frozen manifest");
        let p = find("vu-k1536", 0, 4096).unwrap();
        assert_eq!(p.x, manifest_sha(&text, "vu-k1536.x.u16"));
        assert_eq!(p.w, manifest_sha(&text, "vu-k1536.w.u16"));
        assert_eq!(p.y, manifest_sha(&text, "vu-k1536.y.u16"));
    }

    #[test]
    fn pins_are_unique_and_hex() {
        for (i, a) in PINS.iter().enumerate() {
            for d in [a.x, a.w, a.y] { assert!(d.len() == 64 && d.bytes().all(|c| c.is_ascii_hexdigit())); }
            for b in &PINS[i + 1..] { assert!(!(a.relation == b.relation && a.lo == b.lo && a.vus == b.vus)); }
        }
    }
}
