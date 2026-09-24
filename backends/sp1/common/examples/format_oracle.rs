//! The native oracle of the per-format arithmetic over a frozen instance set:
//! every VU's public word recomputed from its x and W rows must equal the
//! model-recorded y of the `sp1-format-instances/v1` file
//! (`verity_sp1.format_instances`), bit for bit.
//!
//!     cargo run --release -p veritor-zk-common --example format_oracle -- FILE.bin...
//!
//! Negatives, per file: every VU's y with one bit flipped is a mismatch; a NaN
//! (or otherwise off-domain) operand in the first element of every VU is
//! refused.  Prints one JSON line per file; exits 1 if any VU mismatches or
//! any negative is accepted.

use std::time::Instant;

use veritor_zk_common::{nvfp4, tc_fp8, tc_hopper_bf16};

struct Set {
    format: String,
    target: String,
    instances: serde_json::Value,
    n: usize,
    row_bytes: usize,
    y_bytes: usize,
    x: Vec<u8>,
    w: Vec<u8>,
    y: Vec<u8>,
}

fn read(path: &str) -> Set {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&data[..8], b"VYSP1FI\x01", "{path}: not an sp1-format-instances/v1 file");
    let hl = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[12..12 + hl]).unwrap();
    let n = header["n"].as_u64().unwrap() as usize;
    let row_bytes = header["row_bytes"].as_u64().unwrap() as usize;
    let y_bytes = header["y_bytes"].as_u64().unwrap() as usize;
    let o = 12 + hl;
    assert_eq!(data.len(), o + 2 * n * row_bytes + n * y_bytes, "{path}: size");
    Set {
        format: header["format"].as_str().unwrap().to_string(),
        target: header["target"].as_str().unwrap().to_string(),
        instances: header["instances"].clone(),
        n,
        row_bytes,
        y_bytes,
        x: data[o..o + n * row_bytes].to_vec(),
        w: data[o + n * row_bytes..o + 2 * n * row_bytes].to_vec(),
        y: data[o + 2 * n * row_bytes..].to_vec(),
    }
}

fn words16(row: &[u8]) -> Vec<u16> {
    row.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect()
}

/// The format's public word of one VU (`None` outside the model).
fn vu(format: &str, x: &[u8], w: &[u8]) -> Option<u32> {
    match format {
        "fp8-ada" => tc_fp8::vu_ada(x, w),
        "fp8-hopper" => tc_fp8::vu_hopper(x, w),
        "bf16-hopper" => tc_hopper_bf16::vu(&words16(x), &words16(w)).map(u32::from),
        "fp4-nvf4" => nvfp4::vu(x, w),
        other => panic!("unknown format {other}"),
    }
}

/// The first operand element of a row replaced by an off-domain word: a NaN for BF16 / E4M3, a scale byte with
/// the padding bit for NVFP4 (its elements are all finite, so the scale is what can be off-domain).
fn off_domain(format: &str, row: &[u8]) -> Vec<u8> {
    let mut row = row.to_vec();
    match format {
        "bf16-hopper" => row[..2].copy_from_slice(&0x7FC0u16.to_le_bytes()),
        "fp8-ada" | "fp8-hopper" => row[0] = 0x7F,
        "fp4-nvf4" => row[nvfp4::K] = 0x80,
        _ => unreachable!(),
    }
    row
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "usage: format_oracle FILE.bin...");
    let mut ok = true;
    for path in &files {
        let set = read(path);
        let t0 = Instant::now();
        let mut matches = 0usize;
        let mut first_mismatch: Option<(usize, u32, Option<u32>)> = None;
        let mut perturbed_accepted = 0usize;
        let mut off_domain_accepted = 0usize;
        for i in 0..set.n {
            let x = &set.x[i * set.row_bytes..(i + 1) * set.row_bytes];
            let w = &set.w[i * set.row_bytes..(i + 1) * set.row_bytes];
            let mut yb = [0u8; 4];
            yb[..set.y_bytes].copy_from_slice(&set.y[i * set.y_bytes..(i + 1) * set.y_bytes]);
            let expected = u32::from_le_bytes(yb);
            let got = vu(&set.format, x, w);
            if got == Some(expected) {
                matches += 1;
            } else if first_mismatch.is_none() {
                first_mismatch = Some((i, expected, got));
            }
            let perturbed = expected ^ (1 << (i % (8 * set.y_bytes)));
            if got == Some(perturbed) {
                perturbed_accepted += 1;
            }
            if vu(&set.format, &off_domain(&set.format, x), w).is_some() {
                off_domain_accepted += 1;
            }
        }
        let seconds = t0.elapsed().as_secs_f64();
        let pass = matches == set.n && perturbed_accepted == 0 && off_domain_accepted == 0;
        ok &= pass;
        println!(
            "{}",
            serde_json::json!({
                "file": path, "format": set.format, "target": set.target, "instances": set.instances,
                "n": set.n, "y_match": matches, "first_mismatch": first_mismatch.map(|(i, e, g)| serde_json::json!({
                    "vu": i, "expected": format!("{e:#010x}"), "got": g.map(|g| format!("{g:#010x}")),
                })),
                "perturbed_y_accepted": perturbed_accepted, "off_domain_accepted": off_domain_accepted,
                "seconds": (seconds * 1000.0).round() / 1000.0, "pass": pass,
            })
        );
    }
    std::process::exit(if ok { 0 } else { 1 });
}
