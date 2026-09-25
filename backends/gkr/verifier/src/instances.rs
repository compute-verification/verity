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
    Pin {
        relation: "bf16-hopper", lo: 0, vus: 4096, model: "hopper_bf16_m16n8k16",
        x: "b237bb51c52bb43c4227b3a4b77dca6b7ff9038e3e97acdbd02a738f4fe76dd9",
        w: "af3ec99a25d24ae29a9c8f518d2d182c9fcc066d9eff6f55e0e19e5287432b3e",
        y: "ab8c806788d3d099e42e496926b42706b11f104ac6dedf8811f0cd4734728cac",
        source: "backends/direct/ligero/relchain.py instances() seed 20260922 (python -m gpu.bind pin bf16-hopper)",
    },
    Pin {
        relation: "fp8-hopper", lo: 0, vus: 4096, model: "hopper_e4m3_wgmma_k32",
        x: "bdd722c87eeb036e4fd6dcee1b00069702b18748131cb9d8850370f3b0308b83",
        w: "2695c71e2df413fd2a564ae55990884b37dfc50f9c860b4ce3a4831ec1569a05",
        y: "4d7cd59074702ca06ce052be3eee783d2e637f2576edd270e7f226533afba929",
        source: "backends/direct/ligero/relchain.py instances() seed 20260922 (python -m gpu.bind pin fp8-hopper)",
    },
    Pin {
        relation: "fp8-ada", lo: 0, vus: 4096, model: "ada_e4m3_m16n8k32",
        x: "bdd722c87eeb036e4fd6dcee1b00069702b18748131cb9d8850370f3b0308b83",
        w: "2695c71e2df413fd2a564ae55990884b37dfc50f9c860b4ce3a4831ec1569a05",
        y: "9b94a8b82e9e978587e072af6311546b1e1940ea54e67a9f266043008905f283",
        source: "backends/direct/ligero/relchain.py instances() seed 20260922 (python -m gpu.bind pin fp8-ada)",
    },
    Pin {
        relation: "fp4-nvf4", lo: 0, vus: 4096, model: "blackwell_sm120_nvf4",
        x: "be9ddb4ceeff7f5a8f687a9a0bf4a6db54804c71964627a06b17f258f66bf3b5",
        w: "2a544b5886bbdc5e0d7c1a4965cc45568bb784c2e77c23948f4a4f2796f67e3c",
        y: "cf5a6564f9006056429b2d9871acd475e99080210f9654daeaa63dd257b1341b",
        source: "backends/direct/ligero/fp4/chain.py instances_fp4() seed 20260922 (python -m gpu.bind pin fp4-nvf4)",
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
