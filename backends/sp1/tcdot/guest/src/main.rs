//! The modified-SP1 (TC_DOT chip) guest for the Table 2 bare statement `relation-bare/v1`
//! (`veritor_zk_common::bare`): the stock-SP1 guest's statement and public values, byte for byte, with each
//! 16-element step of every VU's chain proved by the `TC_DOT_BF16` precompile.
//!
//! Input (`SP1Stdin` buffers, in order):
//! 1. the `bare` header, unchanged (`bare::encode_header`: id, K, B, `vus_per_read`, the claimed words y);
//! 2. the routing hint: `B * 12` bytes, VU i's 96-bit little-endian mask with bit t set when step t runs in software;
//! 3. `ceil(B / vus_per_read)` chunks of VUs, 6144 bytes each.  A VU whose mask is all ones is in the stock layout
//!    (`bare::encode_chunk`: x row then W row) and runs `bare::check_vu`; any other VU is 96 step blocks, block t =
//!    `x[16t..16t+16] || W[16t..16t+16]` (u16le words, 64 bytes), so the precompile reads its operands in place.
//!
//! The hint is untrusted prover advice outside the statement, which is existential in the private rows.  A step
//! routed to the chip is proved by the `TC_DOT_BF16` AIR, which equals `tc::tc_dot` bit for bit inside the chip's
//! operand contract (finite operands and accumulator, both GroupSum outputs zero or fp32-normal) and has no satisfying
//! row outside it; a step routed to software runs `tc::tc_dot` itself, and a whole VU routed to software runs the
//! stock guest's kernel.  Whatever the hint, the proof is of the reference chain on some rows or there is no proof.

#![no_main]
sp1_zkvm::entrypoint!(main);

use veritor_zk_common::{bare, tc};

/// Steps per VU.
const STEPS: usize = bare::K / 16;
/// Bytes per step block: 16 x words, then 16 W words.
const BLOCK_BYTES: usize = 64;
/// Bytes of one VU's routing mask.
const ROUTE_BYTES: usize = STEPS / 8;

fn main() {
    let header = sp1_zkvm::io::read_vec();
    let public_values = match bare::parse_header(&header) {
        None => bare::public_values(&[0u8; 32], false),
        Some(h) => {
            println!("cycle-tracker-report-start: bare.digest");
            let digest = bare::statement_digest_le(&h.id, h.k, h.b, h.y_le);
            println!("cycle-tracker-report-end: bare.digest");
            let (b, per) = (h.b as usize, h.vus_per_read as usize);
            let routes = sp1_zkvm::io::read_vec();
            let mut ok = h.k as usize == bare::K && per > 0 && routes.len() == b * ROUTE_BYTES;
            let mut i = 0;
            println!("cycle-tracker-report-start: bare.check");
            while ok && i < b {
                let n = per.min(b - i);
                let chunk = sp1_zkvm::io::read_vec();
                ok = chunk.len() == n * bare::VU_BYTES && chunk.as_ptr() as usize % 8 == 0;
                let mut v = 0;
                while ok && v < n {
                    let j = i + v;
                    let claimed = u16::from_le_bytes([h.y_le[2 * j], h.y_le[2 * j + 1]]);
                    let blocks = &chunk[v * bare::VU_BYTES..(v + 1) * bare::VU_BYTES];
                    ok = check_vu(blocks, &routes[j * ROUTE_BYTES..(j + 1) * ROUTE_BYTES]) == Some(claimed);
                    v += 1;
                }
                i += n;
            }
            println!("cycle-tracker-report-end: bare.check");
            bare::public_values(&digest, ok)
        }
    };
    sp1_zkvm::io::commit_slice(&public_values);
    sp1_zkvm::syscalls::syscall_halt(0);
}

/// One VU (8-byte aligned): the 96-step chain from `+0` and the BF16 epilogue; `None` outside the relation's domain.
fn check_vu(vu: &[u8], route: &[u8]) -> Option<u16> {
    let mut acc: u64 = 0;
    if route.iter().all(|&byte| byte == 0) {
        let mut ab = vu.as_ptr() as *const u16;
        for _ in 0..STEPS {
            sp1_zkvm::syscalls::syscall_tc_dot_bf16(ab, &mut acc);
            ab = ab.wrapping_add(BLOCK_BYTES / 2);
        }
    } else if route.iter().all(|&byte| byte == 0xFF) {
        // SAFETY: `vu` is 8-byte aligned and holds exactly VU_WORDS little-endian u64 words.
        let words = unsafe { core::slice::from_raw_parts(vu.as_ptr() as *const u64, bare::VU_WORDS) };
        let (x, w) = words.split_at(bare::ROW_WORDS);
        return bare::check_vu(x, w);
    } else {
        for t in 0..STEPS {
            let block = &vu[BLOCK_BYTES * t..BLOCK_BYTES * (t + 1)];
            if (route[t / 8] >> (t % 8)) & 1 == 0 {
                sp1_zkvm::syscalls::syscall_tc_dot_bf16(block.as_ptr() as *const u16, &mut acc);
            } else {
                acc = software_step(acc, block)?;
            }
        }
    }
    Some(tc::f32_to_bf16(acc) as u16)
}

/// One step through the relation's own step function, `tc::tc_dot`.
fn software_step(acc: u64, block: &[u8]) -> Option<u64> {
    let word = |k: usize| u64::from(u16::from_le_bytes([block[2 * k], block[2 * k + 1]]));
    let a: [u64; 16] = core::array::from_fn(word);
    let b: [u64; 16] = core::array::from_fn(|k| word(16 + k));
    tc::tc_dot(&tc::AMPERE_BF16_M16N8K16, acc, &a, &b)
}
