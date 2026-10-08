//! Verity conformance vectors (not part of Pearl): reads `$VERITY_VEC_IN` (JSON inputs written by
//! verity's `verity/core/protocols/pouw/tests/pearl_vectors.py`) and writes Pearl's outputs to `$VERITY_VEC_OUT`.
#![cfg(test)]

use serde_json::{Value, json};

use crate::api::fp8::compute::{bf16_div, bf16_fma, bf16_mul};
use crate::api::fp8::jackpot_policy::{JackpotPolicy, OperandStrip};
use crate::api::fp8::noise::{NoiseFactor, OperandNoise, Side, sample_line};
use crate::api::fp8::prequant::{exact_norms, open_prequant};
use crate::api::fp8::public_params::Device;
use crate::api::fp8::quantization::Fp8E4M3Quant;
use crate::api::fp8::transcript::{compute_jackpot_ticket, hash_labelled};
use crate::api::fp8::utils::{B200, H100, xor_fold_extract};
use crate::api::layout::{AxisPattern, DimType, lane_assignment};
use crate::circuit::fp8::unpredictability::{budget, grid_exponent, lambda, skip};

fn hex(s: &Value) -> Vec<u8> {
    let s = s.as_str().unwrap();
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}
fn to_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn u16s(s: &Value) -> Vec<u16> {
    hex(s).chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}
fn u16s_hex(v: &[u16]) -> String {
    to_hex(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}
fn f32s_hex(v: &[f32]) -> String {
    to_hex(&v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect::<Vec<u8>>())
}
fn label(name: &str) -> &'static [u8] {
    match name {
        "key-A" => b"pearl/v4/FP8/key-A",
        "key-B" => b"pearl/v4/FP8/key-B",
        "seed-A" => b"pearl/v4/FP8/seed-A",
        "seed-B" => b"pearl/v4/FP8/seed-B",
        "noise-line" => b"pearl/v4/FP8/noise-line",
        "jackpot" => b"pearl/v4/FP8/jackpot",
        _ => panic!("label {name}"),
    }
}
fn seed32(v: &Value) -> [u8; 32] {
    hex(v).try_into().unwrap()
}

/// One operand side: prequant planes -> BF16 rows, exact norms, noise, noisy quantization.
fn side(v: &Value, quant: &Fp8E4M3Quant, k: usize) -> (Vec<u16>, crate::api::fp8::quantization::BuiltRows, Value) {
    let rows = v["rows"].as_u64().unwrap() as usize;
    let ints: Vec<i8> = hex(&v["ints"]).into_iter().map(|b| b as i8).collect();
    let scales = u16s(&v["scales"]);
    let clean = open_prequant(&ints, &scales, rows, k, 8).unwrap();
    let norms = exact_norms(&ints, &scales, rows, k, 8).unwrap();
    let seed = seed32(&v["seed"]);
    let side_id = v["side"].as_u64().unwrap();
    let lines: Vec<u32> = v["lines"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let (s_e, s_f) = if side_id == 0 { (Side::A, Side::A) } else { (Side::B, Side::B) };
    let _ = s_f;
    let mut e = Vec::new();
    for &l in &lines {
        e.extend(sample_line(&seed, if side_id == 0 { Side::A } else { Side::B }, NoiseFactor::E, l, 32));
    }
    let f_seed = seed32(&v["f_seed"]);
    let mut f = Vec::new();
    for u in 0..k as u32 {
        f.extend(sample_line(&f_seed, if side_id == 0 { Side::A } else { Side::B }, NoiseFactor::F, u, 32));
    }
    let _ = s_e;
    let built = quant.noisy_quantize(&clean, &OperandNoise { e: e.clone(), f: f.clone() }, &norms).unwrap();
    let out = json!({
        "clean": u16s_hex(&clean),
        "norms": norms.iter().map(|&(a, b)| json!([a, b])).collect::<Vec<_>>(),
        "e": to_hex(&e), "f_sha": to_hex(blake3::hash(&f).as_bytes()),
        "codes": to_hex(&built.noised_part),
        "alpha": u16s_hex(&built.alpha), "beta": u16s_hex(&built.beta), "l2": u16s_hex(&built.l2),
    });
    (clean, built, out)
}

#[test]
fn verity_vectors() {
    let Ok(path_in) = std::env::var("VERITY_VEC_IN") else { return };
    let input: Value = serde_json::from_str(&std::fs::read_to_string(path_in).unwrap()).unwrap();
    let mut out = serde_json::Map::new();

    out.insert("hash_labelled".into(), Value::Array(input["hash_labelled"].as_array().unwrap().iter().map(|c| {
        let key = if c["key"].is_null() { None } else { Some(seed32(&c["key"])) };
        json!(to_hex(&hash_labelled(&hex(&c["msg"]), label(c["label"].as_str().unwrap()), key.as_ref())))
    }).collect()));

    out.insert("lines".into(), Value::Array(input["lines"].as_array().unwrap().iter().map(|c| {
        let side = if c["side"].as_u64().unwrap() == 0 { Side::A } else { Side::B };
        let factor = if c["factor"].as_u64().unwrap() == 0 { NoiseFactor::E } else { NoiseFactor::F };
        json!(to_hex(&sample_line(&seed32(&c["seed"]), side, factor, c["line"].as_u64().unwrap() as u32, 32)))
    }).collect()));

    out.insert("bf16_ops".into(), Value::Array(input["bf16_ops"].as_array().unwrap().iter().map(|c| {
        let (a, b, d) = (c[0].as_u64().unwrap() as u16, c[1].as_u64().unwrap() as u16, c[2].as_u64().unwrap() as u16);
        let r = |x: anyhow::Result<u16>| x.map(|v| json!(v)).unwrap_or(Value::Null);
        json!([r(bf16_mul(a, b)), r(bf16_div(a, b)), r(bf16_fma(a, b, d))])
    }).collect()));

    out.insert("matmul".into(), Value::Array(input["matmul"].as_array().unwrap().iter().map(|c| {
        let (m, n, k) = (c["m"].as_u64().unwrap() as usize, c["n"].as_u64().unwrap() as usize, c["k"].as_u64().unwrap() as usize);
        let (a, b) = (hex(&c["a"]), hex(&c["b"]));
        let res = B200 {}.matmul_fp8(&a, &b, None, m, n, k).unwrap();
        let parts = B200 {}.matmul_fp8_partials(&a, &b, m, n, k).unwrap();
        let h = if k % 32 == 0 { json!(f32s_hex(&H100 {}.matmul_fp8(&a, &b, None, m, n, k).unwrap())) } else { Value::Null };
        json!({"out": f32s_hex(&res), "partials": parts.iter().map(|p| f32s_hex(p)).collect::<Vec<_>>(), "h100": h})
    }).collect()));

    out.insert("census".into(), Value::Array(input["census"].as_array().unwrap().iter().map(|c| {
        let dev = if c["device"].as_u64().unwrap() == 0 { Device::H100 } else { Device::B200 };
        let l = lambda(dev, c["alpha"].as_u64().unwrap() as u16, c["x"].as_u64().unwrap() as u16, c["l2"].as_u64().unwrap() as u16);
        let g = grid_exponent(dev, c["z"].as_f64().unwrap(), c["m"].as_f64().unwrap());
        json!([l, g, skip(l, c["lb"].as_u64().unwrap(), g)])
    }).collect()));

    let rows_p = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)]).unwrap();
    let cols_p = AxisPattern::new(&[(4, DimType::Fold), (4, DimType::Blake)]).unwrap();
    let lanes = lane_assignment(&rows_p, &cols_p);
    out.insert("lanes".into(), json!(lanes));
    out.insert("pattern".into(), json!(to_hex(&rows_p.to_bytes())));
    out.insert("budget".into(), json!(budget(1024, 16, 16)));

    for (key, dev) in [("tiles", Device::B200), ("tiles_h100", Device::H100)] {
        let quant = Fp8E4M3Quant::new(dev);
        out.insert(key.into(), Value::Array(input["tiles"].as_array().unwrap().iter().map(|c| {
            let k = c["k"].as_u64().unwrap() as usize;
            let (clean_a, built_a, side_a) = side(&c["a"], &quant, k);
            let (clean_b, built_b, side_b) = side(&c["b"], &quant, k);
            let acc = dev.matmul_fp8(&built_a.noised_part, &built_b.noised_part, None, 16, 16, k).unwrap();
            let msg = xor_fold_extract(&acc, &lanes);
            let seed_a = seed32(&c["seed_a"]);
            let ticket = compute_jackpot_ticket(&seed_a, &msg);
            let a = OperandStrip { clean: clean_a, built: built_a };
            let b = OperandStrip { clean: clean_b, built: built_b };
            let verdict = JackpotPolicy::for_device(dev).evaluate(&a, &b, k, &rows_p, &cols_p).unwrap();
            json!({"a": side_a, "b": side_b, "acc": f32s_hex(&acc), "msg": to_hex(&msg), "jackpot": to_hex(&ticket.jackpot),
                   "approved": verdict.is_some(), "policy_msg": verdict.map(|m| to_hex(&m))})
        }).collect()));
    }

    std::fs::write(std::env::var("VERITY_VEC_OUT").unwrap(), serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
