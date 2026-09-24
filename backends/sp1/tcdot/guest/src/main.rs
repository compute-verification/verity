//! The modified-SP1 (TC_DOT chip) guest for the Table 2 bare statement `relation-bare/v2`
//! (`veritor_zk_common::bare`): the stock-SP1 guest's statement and public values, byte for byte, with each
//! 16-element step of a `bf16-ampere` VU's chain proved by the `TC_DOT_BF16` precompile.
//!
//! Input (`SP1Stdin` buffers, in order):
//! 1. the `bare` header, unchanged (`bare::encode_header`: format, id, K, B, `vus_per_read`, the claimed words y);
//! 2. the routing hint: `B * 16` bytes, VU i's mask as two little-endian u64 words with bit t set when step t runs in
//!    software (bits 96..128 are set only together with all the others: the whole VU runs in software);
//! 3. `ceil(B / vus_per_read)` chunks of VUs.  For `bf16-ampere`, a VU whose mask is all ones is in the stock layout
//!    (`bare::encode_chunk`: x row then W row) and runs `bare::check_one`; any other VU is 96 step blocks, block t =
//!    `x[16t..16t+16] || W[16t..16t+16]` (u16le words, 64 bytes), so the precompile reads its operands in place.
//!    The other formats have no chip: their chunks are in the stock layout and run `bare::check_words`, as the stock
//!    guest does.
//!
//!    With the `stream-operands` feature (the witness-operands fork, whose `TC_DOT_BF16` takes each step's operands
//!    from the input stream), a `bf16-ampere` VU is its own buffers instead: one stock-layout buffer when its mask is
//!    all ones, else its 96 step blocks as 96 buffers, which the chip steps' `ecall`s consume and the software steps
//!    read.  `vus_per_read` is then unused.
//!
//! The hint is untrusted prover advice outside the statement, which is existential in the private rows.  A step
//! routed to the chip is proved by the `TC_DOT_BF16` AIR, which equals `tc::tc_dot` bit for bit inside the chip's
//! operand contract (finite operands and accumulator, both GroupSum outputs zero or fp32-normal) and has no satisfying
//! row outside it; a step routed to software runs `tc::tc_dot` itself, and a whole VU routed to software runs the
//! stock guest's kernel.  Whatever the hint, the proof is of the reference chain on some rows or there is no proof.

#![no_main]
sp1_zkvm::entrypoint!(main);

use veritor_zk_common::bare::{self, Format};
use veritor_zk_common::tc;

/// Steps per VU.
const STEPS: usize = bare::K / 16;
/// Bytes per step block: 16 x words, then 16 W words.
const BLOCK_BYTES: usize = 64;
/// u64 words of one VU's routing mask.
const ROUTE_WORDS: usize = 2;
/// Whether `TC_DOT_BF16` takes its operands from the input stream (the witness-operands fork).
const STREAM: bool = cfg!(feature = "stream-operands");

fn main() {
    let header = sp1_zkvm::io::read_vec();
    let public_values = match bare::parse_header(&header) {
        None => bare::public_values(&[0u8; 32], false),
        Some(h) => {
            println!("cycle-tracker-report-start: bare.digest");
            let digest = bare::statement_digest_le(h.format, &h.id, h.k, h.b, h.y_le);
            println!("cycle-tracker-report-end: bare.digest");
            let (b, per, yb) = (h.b as usize, h.vus_per_read as usize, h.format.y_bytes());
            let routes = sp1_zkvm::io::read_vec();
            // SAFETY: as for the chunks below
            let (route_head, masks, route_tail) = unsafe { routes.align_to::<u64>() };
            let chip = h.format == Format::Bf16Ampere;
            let mut ok = h.k as usize == bare::K
                && per > 0
                && (!chip || (route_head.is_empty() && route_tail.is_empty() && masks.len() == b * ROUTE_WORDS));
            let mut i = 0;
            println!("cycle-tracker-report-start: bare.check");
            while STREAM && chip && ok && i < b {
                let claimed = u32::from(u16::from_le_bytes([h.y_le[2 * i], h.y_le[2 * i + 1]]));
                ok = check_vu_stream([masks[ROUTE_WORDS * i], masks[ROUTE_WORDS * i + 1]], claimed);
                i += 1;
            }
            while ok && i < b {
                let n = per.min(b - i);
                let chunk = sp1_zkvm::io::read_vec();
                // SAFETY: every bit pattern is a u64; a misaligned head or a partial tail rejects
                let (head, words, tail) = unsafe { chunk.align_to::<u64>() };
                ok = head.is_empty() && tail.is_empty();
                if ok && chip {
                    ok = words.len() == n * bare::VU_WORDS;
                    let mut v = 0;
                    while ok && v < n {
                        let j = i + v;
                        let claimed = u32::from(u16::from_le_bytes([h.y_le[2 * j], h.y_le[2 * j + 1]]));
                        let (lo, hi) = (v * bare::VU_WORDS, (v + 1) * bare::VU_WORDS);
                        let mask = [masks[ROUTE_WORDS * j], masks[ROUTE_WORDS * j + 1]];
                        ok = check_vu(&chunk[8 * lo..8 * hi], &words[lo..hi], mask, claimed);
                        v += 1;
                    }
                } else if ok {
                    ok = bare::check_words(h.format, &chunk, words, &h.y_le[yb * i..yb * (i + n)]);
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

/// One `bf16-ampere` VU (8-byte aligned, as bytes and as the same bytes read as u64 words): whether the 96-step chain
/// from `+0` and the BF16 epilogue give the claimed word.
fn check_vu(vu_bytes: &[u8], vu_words: &[u64], mask: [u64; ROUTE_WORDS], claimed: u32) -> bool {
    if mask == [u64::MAX; ROUTE_WORDS] {
        return bare::check_one(Format::Bf16Ampere, vu_bytes, vu_words, claimed);
    }
    let mut acc: u64 = 0;
    if mask == [0; ROUTE_WORDS] {
        chip_chain(vu_bytes, &mut acc);
    } else {
        for t in 0..STEPS {
            let block = &vu_bytes[BLOCK_BYTES * t..BLOCK_BYTES * (t + 1)];
            if (mask[t / 64] >> (t % 64)) & 1 == 0 {
                sp1_zkvm::syscalls::syscall_tc_dot_bf16(block.as_ptr() as *const u16, &mut acc);
            } else {
                match software_step(acc, block) {
                    Some(next) => acc = next,
                    None => return false,
                }
            }
        }
    }
    u32::from(tc::f32_to_bf16(acc) as u16) == claimed
}

/// All 96 steps of one VU through `TC_DOT_BF16`: one `ecall` per 64-byte step block, unrolled.
#[allow(unused_variables)]
fn chip_chain(vu_bytes: &[u8], acc: &mut u64) {
    assert!(vu_bytes.len() == STEPS * BLOCK_BYTES);
    #[cfg(target_os = "zkvm")]
    // SAFETY: the blocks are 8-byte aligned (the caller's chunk passed `align_to`) and in bounds. TC_DOT_BF16 returns
    // no value, so every ecall writes its own code back to t0 and leaves a0 and a1 alone.
    unsafe {
        core::arch::asm!(
            ".rept {steps}",
            "ecall",
            "addi a0, a0, {block}",
            ".endr",
            steps = const STEPS,
            block = const BLOCK_BYTES,
            inout("t0") sp1_zkvm::syscalls::TC_DOT_BF16 => _,
            inout("a0") vu_bytes.as_ptr() => _,
            in("a1") acc as *mut u64,
            options(nostack),
        );
    }
    #[cfg(not(target_os = "zkvm"))]
    unreachable!()
}

/// One `bf16-ampere` VU from its own buffers (`stream-operands`): whether the 96-step chain from `+0` and the BF16
/// epilogue give the claimed word.
fn check_vu_stream(mask: [u64; ROUTE_WORDS], claimed: u32) -> bool {
    if mask == [u64::MAX; ROUTE_WORDS] {
        let vu = sp1_zkvm::io::read_vec();
        // SAFETY: every bit pattern is a u64; a misaligned head or a partial tail rejects
        let (head, words, tail) = unsafe { vu.align_to::<u64>() };
        return head.is_empty()
            && tail.is_empty()
            && words.len() == bare::VU_WORDS
            && bare::check_one(Format::Bf16Ampere, &vu, words, claimed);
    }
    let mut acc: u64 = 0;
    if mask == [0; ROUTE_WORDS] {
        stream_steps::<STEPS>(&mut acc);
    } else {
        for t in 0..STEPS {
            if (mask[t / 64] >> (t % 64)) & 1 == 0 {
                stream_steps::<1>(&mut acc);
            } else {
                let block = sp1_zkvm::io::read_vec();
                if block.len() != BLOCK_BYTES {
                    return false;
                }
                match software_step(acc, &block) {
                    Some(next) => acc = next,
                    None => return false,
                }
            }
        }
    }
    u32::from(tc::f32_to_bf16(acc) as u16) == claimed
}

/// `N` steps through `TC_DOT_BF16` with streamed operands (each takes the next input buffer): `N` unrolled `ecall`s.
#[allow(unused_variables)]
fn stream_steps<const N: usize>(acc: &mut u64) {
    let acc: *mut u64 = acc;
    #[cfg(target_os = "zkvm")]
    // SAFETY: TC_DOT_BF16 returns no value, so every ecall writes its own code back to t0 and leaves a0 and a1 alone.
    // a0 is only the chip's operand-address argument (aligned, at least 2^16, never read): the accumulator's address.
    unsafe {
        core::arch::asm!(
            ".rept {n}",
            "ecall",
            ".endr",
            n = const N,
            inout("t0") sp1_zkvm::syscalls::TC_DOT_BF16 => _,
            in("a0") acc,
            in("a1") acc,
            options(nostack),
        );
    }
    #[cfg(not(target_os = "zkvm"))]
    unreachable!()
}

/// One step through the relation's own step function, `tc::tc_dot`.
fn software_step(acc: u64, block: &[u8]) -> Option<u64> {
    let word = |k: usize| u64::from(u16::from_le_bytes([block[2 * k], block[2 * k + 1]]));
    let a: [u64; 16] = core::array::from_fn(word);
    let b: [u64; 16] = core::array::from_fn(|k| word(16 + k));
    tc::tc_dot(&tc::AMPERE_BF16_M16N8K16, acc, &a, &b)
}
