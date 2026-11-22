//! An injected seed's leaf salts (a test harness's, feature `seed-injection`, or unit tests): leaf `i` of tree `id` is ChaCha20
//! blocks `3i, 3i + 1, 3i + 2` under the seed's `STREAM_SALT` key with nonce `id`, so a test proof repeats. A proving build
//! has no such module, so its leaf salts can only be the OS's.

use crate::zk_hooks::{SALT_BYTES, STREAM_SALT};
use rand_chacha::{
    ChaCha20Rng,
    rand_core::{RngCore, SeedableRng},
};

/// The key an injected seed's salts are expanded under (on the heap, so moving it copies no key byte).
pub(crate) fn key(seed: &[u8; 32]) -> Box<[u8; 32]> {
    let mut key = Box::new([0u8; 32]);
    let mut r = ChaCha20Rng::from_seed(*seed);
    r.set_stream(STREAM_SALT);
    r.fill_bytes(&mut key[..]);
    key
}

/// Writes tree `id`'s salts under `key` into `b`, whole leaves, on the current rayon pool.
pub(crate) fn fill_salts(key: &[u8; 32], id: u64, b: &mut [u8]) {
    use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
    b.par_chunks_mut(SALT_BYTES).enumerate().for_each(|(i, s)| {
        for j in 0..3 {
            s[64 * j..64 * j + 64].copy_from_slice(&chacha20_block(key, 3 * i as u64 + j as u64, id));
        }
    });
}

/// One ChaCha20 block (20 rounds): state = "expand 32-byte k", the key, a 64-bit block counter (words 12-13) and a 64-bit
/// nonce (words 14-15), output little-endian.
pub(crate) fn chacha20_block(key: &[u8; 32], counter: u64, nonce: u64) -> [u8; 64] {
    let mut s = [0u32; 16];
    s[..4].copy_from_slice(&[0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]);
    for i in 0..8 {
        s[4 + i] = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
    }
    s[12] = counter as u32;
    s[13] = (counter >> 32) as u32;
    s[14] = nonce as u32;
    s[15] = (nonce >> 32) as u32;
    let mut x = s;
    let qr = |x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize| {
        x[a] = x[a].wrapping_add(x[b]);
        x[d] = (x[d] ^ x[a]).rotate_left(16);
        x[c] = x[c].wrapping_add(x[d]);
        x[b] = (x[b] ^ x[c]).rotate_left(12);
        x[a] = x[a].wrapping_add(x[b]);
        x[d] = (x[d] ^ x[a]).rotate_left(8);
        x[c] = x[c].wrapping_add(x[d]);
        x[b] = (x[b] ^ x[c]).rotate_left(7);
    };
    for _ in 0..10 {
        qr(&mut x, 0, 4, 8, 12);
        qr(&mut x, 1, 5, 9, 13);
        qr(&mut x, 2, 6, 10, 14);
        qr(&mut x, 3, 7, 11, 15);
        qr(&mut x, 0, 5, 10, 15);
        qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7, 8, 13);
        qr(&mut x, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        out[4 * i..4 * i + 4].copy_from_slice(&x[i].wrapping_add(s[i]).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk_hooks::{LeafSalts, ProverSeed};

    /// RFC 7539 section 2.3.2: key 00..1f, block counter 1, nonce 000000090000004a00000000 (words 13-15).
    #[test]
    fn chacha20_block_matches_rfc7539() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let b = chacha20_block(&key, 1 | (0x0900_0000u64 << 32), 0x4a00_0000);
        assert_eq!(crate::hex(&b), concat!("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e",
                                           "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"));
    }

    /// An injected seed's salts are its ChaCha20 tape: leaf `i` of tree `id` is blocks `3i..3i+2` under the `STREAM_SALT` key with
    /// nonce `id`, so a test proof repeats.
    #[test]
    fn an_injected_seed_expands_its_salts() {
        let seed = ProverSeed::injected([5; 32]);
        let mut key = [0u8; 32];
        let mut r = ChaCha20Rng::from_seed([5; 32]);
        r.set_stream(STREAM_SALT);
        r.fill_bytes(&mut key);
        let s = LeafSalts::for_seed(&seed);
        let all = s.expand(7, 5);
        for i in 0..5 {
            let want: Vec<u8> = (0..3).flat_map(|j| chacha20_block(&key, 3 * i as u64 + j, 7)).collect();
            assert_eq!(all[SALT_BYTES * i..SALT_BYTES * (i + 1)], want);
        }
        assert_eq!(LeafSalts::for_seed(&seed).expand(7, 5), all);
        assert_eq!(*s.level0(3), LeafSalts::for_seed(&seed).expand(0, 3));
    }
}
