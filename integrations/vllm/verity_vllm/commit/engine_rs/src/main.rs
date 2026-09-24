//! C1 pilot — `vce`: Merkle roots over packed Verity values with the production framing.
//!
//! Input: a plan file written by `vllm-poc/scripts/cprof_c1_plan.py`:
//!   "VCE1" u32 ntrees { u32 nleaves { u32 hlen, header[hlen], u64 offset, u64 nbytes } } and a pack file of the values
//!   in tree order.  Leaf = SHA-256(header || u64be(nbytes) || value) (= verity_vllm.commit.hashing.leaf_hash when header is
//!   fasttree.leaf_header); node = SHA-256(NODE_PREFIX || left || u64be(32) || right); lift = SHA-256(LIFT_PREFIX || child).
//!   `--pos` replaces every header by the constant "verity/pos-leaf/v0" (position-identified leaves, C0).
//! Output: one line per tree `<index> <root hex>`, then `TIME <mode> <threads> <leaves> <bytes> <seconds>` on stderr.

use memmap2::Mmap;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const NODE_TAG: &[u8] = b"verity-vllm/node/v1";
const LIFT_TAG: &[u8] = b"verity-vllm/lift/v1";
const POS_TAG: &[u8] = b"verity/pos-leaf/v0";

fn tag_prefix(tag: &[u8]) -> Vec<u8> {
    let mut v = (tag.len() as u32).to_be_bytes().to_vec();
    v.extend_from_slice(tag);
    v
}

struct Leaf<'a> {
    header: &'a [u8],
    value: &'a [u8],
}

fn leaf_hash(l: &Leaf, pos: bool) -> [u8; 32] {
    let mut h = Sha256::new();
    if pos {
        h.update(POS_TAG);
    } else {
        h.update(l.header);
    }
    h.update((l.value.len() as u64).to_be_bytes());
    h.update(l.value);
    h.finalize().into()
}

fn fold(mut level: Vec<[u8; 32]>, node_prefix: &[u8], lift_prefix: &[u8], par: bool) -> [u8; 32] {
    let u64_32 = 32u64.to_be_bytes();
    while level.len() > 1 {
        let pairs = level.len() / 2;
        let node = |i: usize| -> [u8; 32] {
            let mut h = Sha256::new();
            h.update(node_prefix);
            h.update(level[2 * i]);
            h.update(u64_32);
            h.update(level[2 * i + 1]);
            h.finalize().into()
        };
        let mut next: Vec<[u8; 32]> = if par && pairs >= 512 {
            (0..pairs).into_par_iter().map(node).collect()
        } else {
            (0..pairs).map(node).collect()
        };
        if level.len() % 2 == 1 {
            let mut h = Sha256::new();
            h.update(lift_prefix);
            h.update(level[level.len() - 1]);
            next.push(h.finalize().into());
        }
        level = next;
    }
    level.pop().unwrap_or([0u8; 32])
}

fn rd_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_be_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    v
}
fn rd_u64(b: &[u8], p: &mut usize) -> u64 {
    let v = u64::from_be_bytes(b[*p..*p + 8].try_into().unwrap());
    *p += 8;
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut plan_path = None;
    let mut pack_path = None;
    let mut pos = false;
    let mut threads = 1usize;
    let mut reps = 1usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--plan" => { plan_path = Some(args[i + 1].clone()); i += 1; }
            "--pack" => { pack_path = Some(args[i + 1].clone()); i += 1; }
            "--pos" => pos = true,
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 1; }
            "--reps" => { reps = args[i + 1].parse().unwrap(); i += 1; }
            _ => panic!("unknown arg {}", args[i]),
        }
        i += 1;
    }
    let plan = unsafe { Mmap::map(&File::open(plan_path.expect("--plan")).unwrap()).unwrap() };
    let pack = unsafe { Mmap::map(&File::open(pack_path.expect("--pack")).unwrap()).unwrap() };
    assert_eq!(&plan[..4], b"VCE1");
    let mut p = 4usize;
    let ntrees = rd_u32(&plan, &mut p) as usize;
    let mut trees: Vec<Vec<Leaf>> = Vec::with_capacity(ntrees);
    let mut total_bytes = 0u64;
    let mut total_leaves = 0usize;
    for _ in 0..ntrees {
        let n = rd_u32(&plan, &mut p) as usize;
        let mut leaves = Vec::with_capacity(n);
        for _ in 0..n {
            let hl = rd_u32(&plan, &mut p) as usize;
            let header = &plan[p..p + hl];
            p += hl;
            let off = rd_u64(&plan, &mut p) as usize;
            let nb = rd_u64(&plan, &mut p) as usize;
            total_bytes += nb as u64;
            leaves.push(Leaf { header, value: &pack[off..off + nb] });
        }
        total_leaves += n;
        trees.push(leaves);
    }
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let node_prefix = { let mut v = tag_prefix(NODE_TAG); v.extend_from_slice(&32u64.to_be_bytes()); v };
    let lift_prefix = { let mut v = tag_prefix(LIFT_TAG); v.extend_from_slice(&32u64.to_be_bytes()); v };
    // touch the pack once so the first rep does not pay page faults for the file read (that cost is reported separately)
    let t_touch = Instant::now();
    let mut acc = 0u64;
    for chunk in pack.chunks(4096) { acc = acc.wrapping_add(chunk[0] as u64); }
    let touch_s = t_touch.elapsed().as_secs_f64();
    let mut roots: Vec<[u8; 32]> = Vec::new();
    let mut best = f64::MAX;
    let mut times = Vec::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        roots = if threads > 1 {
            trees.par_iter().map(|leaves| {
                let digests: Vec<[u8; 32]> = if leaves.len() >= 2048 {
                    leaves.par_iter().map(|l| leaf_hash(l, pos)).collect()
                } else {
                    leaves.iter().map(|l| leaf_hash(l, pos)).collect()
                };
                fold(digests, &node_prefix, &lift_prefix, true)
            }).collect()
        } else {
            trees.iter().map(|leaves| {
                let digests: Vec<[u8; 32]> = leaves.iter().map(|l| leaf_hash(l, pos)).collect();
                fold(digests, &node_prefix, &lift_prefix, false)
            }).collect()
        };
        let dt = t0.elapsed().as_secs_f64();
        times.push(dt);
        if dt < best { best = dt; }
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for (i, r) in roots.iter().enumerate() {
        writeln!(out, "{} {}", i, hex::encode(r)).unwrap();
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    eprintln!("TIME mode={} threads={} trees={} leaves={} bytes={} best_s={:.6} median_s={:.6} touch_s={:.6} acc={}",
        if pos { "pos" } else { "prod" }, threads, ntrees, total_leaves, total_bytes, best, median, touch_s, acc % 7);
}
