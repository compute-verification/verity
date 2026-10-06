//! BLAKE3 (Merkle leaves / nodes), BLAKE2b-256 (transcript state chain) and SHAKE-256 (the challenge expander),
//! written from the specifications -- the ligero-verify `hash.rs` primitives, copied so the crates stay independent.
//! No dependency, no shared code with the Python prover (`blake3` package, `hashlib`).

// ------------------------------------------------------------------------------------------------------ BLAKE3 ----

const B3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const B3_PERM: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;

#[inline(always)]
fn g(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

fn b3_compress(cv: &[u32; 8], block: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
    let mut s = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], B3_IV[0], B3_IV[1], B3_IV[2], B3_IV[3],
        counter as u32, (counter >> 32) as u32, block_len, flags,
    ];
    let mut m = *block;
    for r in 0..7 {
        g(&mut s, 0, 4, 8, 12, m[0], m[1]);
        g(&mut s, 1, 5, 9, 13, m[2], m[3]);
        g(&mut s, 2, 6, 10, 14, m[4], m[5]);
        g(&mut s, 3, 7, 11, 15, m[6], m[7]);
        g(&mut s, 0, 5, 10, 15, m[8], m[9]);
        g(&mut s, 1, 6, 11, 12, m[10], m[11]);
        g(&mut s, 2, 7, 8, 13, m[12], m[13]);
        g(&mut s, 3, 4, 9, 14, m[14], m[15]);
        if r < 6 {
            let mut p = [0u32; 16];
            for i in 0..16 {
                p[i] = m[B3_PERM[i]];
            }
            m = p;
        }
    }
    for i in 0..8 {
        s[i] ^= s[i + 8];
        s[i + 8] ^= cv[i];
    }
    s
}

fn words_of(block: &[u8]) -> [u32; 16] {
    let mut w = [0u32; 16];
    let mut buf = [0u8; 64];
    buf[..block.len()].copy_from_slice(block);
    for i in 0..16 {
        w[i] = u32::from_le_bytes([buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]);
    }
    w
}

/// The node whose compression produces a chaining value / the hash: the XOF re-compresses the root node per 64-byte
/// output block with the output-block counter.
struct B3Output {
    cv: [u32; 8],
    block: [u32; 16],
    block_len: u32,
    flags: u32,
    counter: u64,
}

impl B3Output {
    fn chain(&self) -> [u32; 8] {
        let out = b3_compress(&self.cv, &self.block, self.counter, self.block_len, self.flags);
        let mut cv = [0u32; 8];
        cv.copy_from_slice(&out[..8]);
        cv
    }
    fn root_bytes(&self, t: u64) -> [u8; 64] {
        let out = b3_compress(&self.cv, &self.block, t, self.block_len, self.flags | ROOT);
        let mut b = [0u8; 64];
        for i in 0..16 {
            b[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
        }
        b
    }
}

/// One chunk (<= 1024 bytes) as an output node: its last block over the chaining value of the blocks before it.
fn b3_chunk(chunk: &[u8], counter: u64) -> B3Output {
    let mut cv = B3_IV;
    let nblocks = if chunk.is_empty() { 1 } else { (chunk.len() + 63) / 64 };
    for i in 0..nblocks {
        let lo = 64 * i;
        let hi = (lo + 64).min(chunk.len());
        let block = &chunk[lo..hi];
        let mut flags = 0;
        if i == 0 {
            flags |= CHUNK_START;
        }
        if i == nblocks - 1 {
            flags |= CHUNK_END;
            return B3Output { cv, block: words_of(block), block_len: block.len() as u32, flags, counter };
        }
        let out = b3_compress(&cv, &words_of(block), counter, block.len() as u32, flags);
        cv.copy_from_slice(&out[..8]);
    }
    unreachable!()
}

/// The output node of `data` whose first chunk has counter `first_chunk` (0 for the whole message).
fn b3_node(data: &[u8], first_chunk: u64) -> B3Output {
    let nchunks = (data.len() + 1023) / 1024;
    if nchunks <= 1 {
        return b3_chunk(data, first_chunk);
    }
    // left subtree: the largest power of two strictly below nchunks
    let mut left = 1usize;
    while left * 2 < nchunks {
        left *= 2;
    }
    let l = b3_node(&data[..left * 1024], first_chunk).chain();
    let r = b3_node(&data[left * 1024..], first_chunk + left as u64).chain();
    let mut block = [0u32; 16];
    block[..8].copy_from_slice(&l);
    block[8..].copy_from_slice(&r);
    B3Output { cv: B3_IV, block, block_len: 64, flags: PARENT, counter: 0 }
}

pub fn blake3(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&b3_node(data, 0).root_bytes(0)[..32]);
    out
}

/// BLAKE3 extended output: the first `out_len` bytes of the XOF (`blake3.blake3(data).digest(length=out_len)`).
pub fn blake3_xof(data: &[u8], out_len: usize) -> Vec<u8> {
    let node = b3_node(data, 0);
    let mut out = Vec::with_capacity(out_len + 64);
    let mut t = 0u64;
    while out.len() < out_len {
        out.extend_from_slice(&node.root_bytes(t));
        t += 1;
    }
    out.truncate(out_len);
    out
}

// ------------------------------------------------------------------------------------------------- BLAKE2b-256 ----

const B2_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];
const B2_SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn g2(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn b2_compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for i in 0..16 {
        m[i] = u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&B2_IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in B2_SIGMA.iter() {
        g2(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g2(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g2(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g2(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g2(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g2(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g2(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g2(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// BLAKE2b with a 32-byte digest, no key (`hashlib.blake2b(data, digest_size=32)`).
pub fn blake2b_256(data: &[u8]) -> [u8; 32] {
    let mut h = B2_IV;
    h[0] ^= 0x0101_0000 ^ 32;
    let n = data.len();
    let nblocks = if n == 0 { 1 } else { (n + 127) / 128 };
    for i in 0..nblocks {
        let lo = 128 * i;
        let hi = (lo + 128).min(n);
        let mut block = [0u8; 128];
        block[..hi - lo].copy_from_slice(&data[lo..hi]);
        b2_compress(&mut h, &block, hi as u128, i == nblocks - 1);
    }
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[8 * i..8 * i + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

// --------------------------------------------------------------------------------------------------- SHAKE-256 ----

const KECCAK_RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b,
    0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008a, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000a, 0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const KECCAK_ROT: [[u32; 5]; 5] = [
    // [x][y]
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

fn keccak_f(a: &mut [u64; 25]) {
    for rc in KECCAK_RC.iter() {
        // theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        // rho + pi
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(KECCAK_ROT[x][y]);
            }
        }
        // chi
        for x in 0..5 {
            for y in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        // iota
        a[0] ^= rc;
    }
}

/// SHAKE-256 of `data`, `out_len` output bytes (`hashlib.shake_256(data).digest(out_len)`).
pub fn shake256(data: &[u8], out_len: usize) -> Vec<u8> {
    const RATE: usize = 136;
    let mut st = [0u64; 25];
    let absorb = |st: &mut [u64; 25], block: &[u8; RATE]| {
        for i in 0..RATE / 8 {
            st[i] ^= u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
        }
        keccak_f(st);
    };
    let mut i = 0;
    while data.len() - i >= RATE {
        absorb(&mut st, data[i..i + RATE].try_into().unwrap());
        i += RATE;
    }
    let mut last = [0u8; RATE];
    let rem = data.len() - i;
    last[..rem].copy_from_slice(&data[i..]);
    last[rem] ^= 0x1F;
    last[RATE - 1] ^= 0x80;
    absorb(&mut st, &last);
    let mut out = Vec::with_capacity(out_len);
    loop {
        for i in 0..RATE / 8 {
            out.extend_from_slice(&st[i].to_le_bytes());
            if out.len() >= out_len {
                out.truncate(out_len);
                return out;
            }
        }
        keccak_f(&mut st);
    }
}

// ------------------------------------------------------------------------------------------------------ SHA-256 ----

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 (`hashlib.sha256(data).digest()`): the LGTO0001 statement and key digests.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes(block[4 * t..4 * t + 4].try_into().unwrap());
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for t in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[t]).wrapping_add(w[t]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// 64 hex digits -> 32 bytes (compile-time constants only: panics on anything else).
pub fn unhex32(s: &str) -> [u8; 32] {
    assert_eq!(s.len(), 64, "unhex32: {s}");
    let mut o = [0u8; 32];
    for i in 0..32 {
        o[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex digit");
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_empty_and_abc() {
        assert_eq!(hex(&blake3(b"")), "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");
        assert_eq!(hex(&blake3(b"abc")), "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85");
    }

    #[test]
    fn blake2b_256_empty() {
        // hashlib.blake2b(b"", digest_size=32)
        assert_eq!(hex(&blake2b_256(b"")), "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8");
    }

    #[test]
    fn blake2b_256_multiblock_vs_hashlib() {
        let mut d: Vec<u8> = (0..=255u8).collect::<Vec<_>>().repeat(3);
        d.extend_from_slice(b"xyz");
        assert_eq!(hex(&blake2b_256(&d)), "8ba45c58183d3582137af5b403aed5ae7108918d1f609befaaa567bc459144ee");
        let e: Vec<u8> = (0..81u8).collect();
        assert_eq!(hex(&blake2b_256(&e)), "9fbb33b95e79c5c1683ab09a6abff6612fffb4458543dc8bbc7723a6dc2bf2ed");
    }

    #[test]
    fn shake256_multiblock_vs_hashlib() {
        let mut d: Vec<u8> = (0..=255u8).collect::<Vec<_>>().repeat(3);
        d.extend_from_slice(b"xyz");
        let h = hex(&shake256(&d, 300));
        assert!(h.starts_with("3c07aabd5752a3e95ce297232c2628f53a43951e79b403b44d0b71d1e8202be2c059ddee37442ecb6c16123ac1dd5943"));
        assert!(h.ends_with("aeb3d4b7b4f363564d3d0e234f9"));
    }

    #[test]
    fn sha256_vectors() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        // 56 bytes: the padding spills into a second block
        assert_eq!(
            hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn shake256_empty() {
        // hashlib.shake_256(b"").digest(32)
        assert_eq!(hex(&shake256(b"", 32)), "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");
    }
}
