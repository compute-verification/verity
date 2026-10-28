//! flock-trace: the witness `flock-circuit prove` assigns a staged statement (M0's zero mask slot, no `--zk`), as digests, for
//! the Lean trace assigner's difftest (`VStar.Trace`, `vstar-trace`; `verity/core/service/vstar/tests/test_vstar_trace_difftest.py`).
//!
//!   flock-trace --instances INST --circuit FILE
//!
//! Prints one JSON line: `m`, `k_log`, `instances`, per block the SHA-512 of its `z` (`blocks`) and of each range's slots
//! (`ranges`, in the circuit's range order), each F128 word `lo` then `hi`, little-endian. The witness is the CPU prover's:
//! `Stmt::row_inputs` and `Stmt::unit_inputs` from the instance file's rows and salts, `Stmt::converge`, `Stmt::witness`.

use std::collections::BTreeMap;

use flock_field::F128;
use flock_live::circuit::{Composite, Instances, RowTamper, Stmt};
use serde_json::json;
use sha2::{Digest, Sha512};

fn refuse(e: String) -> ! {
    eprintln!("flock-trace: {e}");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| match args.iter().position(|a| a == k) {
        Some(i) if i + 1 < args.len() => args[i + 1].clone(),
        _ => refuse(format!("usage: flock-trace --instances INST --circuit FILE (no {k})")),
    };
    let text = std::fs::read_to_string(arg("--circuit")).unwrap_or_else(|e| refuse(format!("circuit: {e}")));
    let c = Composite::parse_with(&text, None, None).unwrap_or_else(|e| refuse(e));
    let inst = Instances::load(&arg("--instances"), &c).unwrap_or_else(|e| refuse(e));
    let st = Stmt::new(c, inst.n, serde_json::Value::Null).unwrap_or_else(|e| refuse(e));
    let rows = inst.instance_rows(&st.c).unwrap_or_else(|| refuse("the instance file holds no rows (a verifier's file)".into()));
    let salts = inst.instance_salts(st.c.ports.len()).unwrap_or_else(|| refuse("the instance file holds no salts".into()));
    let rowin = |b: usize| st.row_inputs(Some(&rows), Some(&salts), b, RowTamper::default());
    let inputs = |b: usize, k: usize| st.unit_inputs(Some(&rows), b, k);
    let mask_words = 1usize << (st.c.meta["mask_log"].as_u64().unwrap_or(14) as usize - 7);
    let mask = |_: usize| vec![F128::ZERO; mask_words];
    let nets = BTreeMap::new();
    let ins = st.converge(&inputs, &rowin, &nets);
    let (z, _, _, _) = st.witness(&ins, &mask, &nets);
    let wpb = 1usize << (st.c.k_log - 7);
    let digest = |ws: &[F128]| {
        let mut h = Sha512::new();
        for w in ws {
            h.update(w.lo.to_le_bytes());
            h.update(w.hi.to_le_bytes());
        }
        flock_live::hex(&h.finalize())
    };
    let blocks: Vec<String> = (0..st.blocks()).map(|b| digest(&z[b * wpb..(b + 1) * wpb])).collect();
    let ranges: Vec<Vec<String>> = (0..st.blocks())
        .map(|b| st.c.ranges.iter().map(|r| {
            let at = b * wpb + r.start() / 128;
            digest(&z[at..at + ((r.count << r.slot_log) / 128)])
        }).collect())
        .collect();
    println!("{}", json!({"m": st.m, "k_log": st.c.k_log, "instances": st.n, "blocks": blocks, "ranges": ranges}));
}
