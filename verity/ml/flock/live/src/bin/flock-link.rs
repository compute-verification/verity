//! flock-link: flock-128-r2 live sessions whose statement is linkable to a prime-field relation proof.
//!
//! The statement (flock-link C4): `2 vus` operand rows of `k` 16-bit words (the x rows, then the W columns, word k
//! LSB first: position `(row k_total + k) 16 + i`), each hashed as the core row leaf `blake3-keyed/row/v2`
//! (`BLAKE3(row bytes, key = the x or w row key)`). Flock proves every chunk's 16 compressions as ONE circuit: the
//! chaining values are wired row to row, the chunk's start value is the public key of its role, every params word
//! (counter, block_len, flags with KEYED_HASH) is wired to a public constant, and each chunk's output chaining value is
//! public. The verifier checks those public chunk values against the leaf digests natively (the keyed parent
//! compressions), so no middle block, counter, flag, length or key is free.
//!
//! The link (L1, L2): after `Commit(root_F, root_B, public words)` the verifier draws two GF(2^128) points over the
//! linked positions, the prover sends `y_k = z^(r_k)`, and both reps open `z^(r_k) = y_k` in their merged opening
//! (a ring-switched claim with eq weights), so each rep's reduction is a fresh round-by-round bound on live coins.
//!
//!   flock-link serve    --listen ADDR --out DIR --vus V [--k K] --seed S   verifier (statement = the seed's rows)
//!   flock-link prove    --verifier ADDR --vus V [--k K] --seed S          prover, one session
//!   flock-link selftest --vus V [--k K]                                    negatives against an in-process verifier
//!
//! The route (a) cell (the statement is the prime proof's: its leaf digests, x rows then W columns, 32 bytes each):
//!   flock-link sigma      --vus V --digests FILE --prime-commitment HEX        print Σ's preimage (sigma.txt)
//!   flock-link cell-serve --listen ADDR --out DIR --vus V --digests FILE --prime-commitment HEX --operator LANE [--sessions N]
//!   flock-link cell-prove --verifier ADDR --vus V --digests FILE --prime-commitment HEX --operands FILE
//!                         --root-f FILE --link-out FILE [--start FILE] [--prime-sock PATH]
//!   flock-link replay     --session DIR --vus V --digests FILE --prime-commitment HEX [--json OUT]
//!                         G2: re-verify the Flock side of a finished session from DIR/session.json and the proofs kept
//!                         beside it (DIR/chain.rep{0,1}.bin), with no live verifier; exit 0 iff accepted
//!   cell-prove waits for the prime prover's root_F (hex in --root-f), commits it with root_B, and writes the session's
//!   points and y (and Σ, root_B) to --link-out for the prime prover to finish its proof against. cell-serve keeps each
//!   session's proofs beside its record (E1). cell-serve serves the prime proof's coins too (`Prime` rounds, the
//!   library's live prime coins); cell-prove forwards them from --prime-sock on the session's connection.
//!
//! Output lines are `LIVE\t{json}` / `NEG\t{json}` / `SELFTEST\t{json}`.

use std::{
    array::from_fn,
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
    time::Instant,
};

use flock_core::{
    circuit::builder::{CircuitShape, GateType, ShapeBuilder, SlotId, SlotWitness, Wire},
    pcs::{PcsParams, ligerito::{LigeritoProfile, embedded_initial_k_or_default}},
    proof::{LinkClaim, R1csProofCircuitMerged, eq_skip_weights},
    test_rng::Rng,
    union::publics_digest,
};
use flock_field::F128;
use flock_hash::blake3_compress;
use flock_live::*;
use flock_prover::{
    pcs::Commitment,
    prover::{UnionSlotProverInput, prove_fast_ligerito_union_circuit_linked},
    r1cs::BlockR1cs,
    r1cs_hashes::blake3::{Compression, build_block_r1cs, generate_witness_batch_major_partial_into, io_schema},
    schedule::TableType,
    union::UnionInstance,
    verifier::verify_ligerito_union_circuit_linked,
};
use rayon::prelude::*;
use serde_json::{Value, json};

const TABLE: &str = "chain";
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;
const KEYED_HASH: u32 = 16;
/// The core row leaf `blake3-keyed/row/v2` (`verity.ml.commitments.rowleaf`): `BLAKE3(row bytes, key = "verity/blake3-leaf/v1/"
/// + "x" | "w", zero padded to 32 bytes)`.
const SCHEME: &str = "blake3-keyed/row/v2";
const KEY_PREFIX: &[u8] = b"verity/blake3-leaf/v1/";

fn row_key(role: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..KEY_PREFIX.len()].copy_from_slice(KEY_PREFIX);
    k[KEY_PREFIX.len()..KEY_PREFIX.len() + role.len()].copy_from_slice(role);
    k
}

fn key_words(k: &[u8; 32]) -> [u32; 8] {
    from_fn(|i| u32::from_le_bytes(k[4 * i..4 * i + 4].try_into().unwrap()))
}

/// Leaf `l`'s key: x rows are leaves `0..vus`, W columns `vus..2 vus`.
fn leaf_key(vus: usize, l: usize) -> [u8; 32] {
    row_key(if l < vus { b"x" } else { b"w" })
}

fn arg(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()).unwrap_or_else(|| d.to_string())
}

fn pack4(w: &[u32]) -> F128 {
    F128::new(w[0] as u64 | ((w[1] as u64) << 32), w[2] as u64 | ((w[3] as u64) << 32))
}
fn unpack4(v: F128) -> [u32; 4] {
    [v.lo as u32, (v.lo >> 32) as u32, v.hi as u32, (v.hi >> 32) as u32]
}
fn pack8(w: &[u32; 8]) -> [F128; 2] {
    [pack4(&w[..4]), pack4(&w[4..])]
}
fn unpack8(a: F128, b: F128) -> [u32; 8] {
    let (x, y) = (unpack4(a), unpack4(b));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}
fn pack_params(counter: u64, block_len: u32, flags: u32) -> F128 {
    F128::new(counter, block_len as u64 | ((flags as u64) << 32))
}
fn f128_bytes(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())).collect()
}
fn bytes_f128(b: &[u8]) -> Vec<F128> {
    b.chunks(16).map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..16].try_into().unwrap()))).collect()
}
fn f128_hex(v: &F128) -> String {
    format!("{:016x}{:016x}", v.hi, v.lo)
}

/// The prime prover's root_F: `file` holds its hex once the prime proof reached the link (written atomically).
fn wait_root_f(file: &str) -> Vec<u8> {
    let t = Instant::now();
    loop {
        if let Ok(s) = std::fs::read_to_string(file) {
            let s = s.trim();
            if s.len() == 64 {
                return (0..32).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect();
            }
        }
        assert!(t.elapsed().as_secs() < 3600, "no root_F in {file} after 3600 s");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

fn log2ceil(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// One BLAKE3 compression as a circuit gate (schema: cv0, cv1, m0..m3, params → out_lo0, out_lo1, out_hi0, out_hi1).
struct Blake3Gate {
    nu: usize,
}

impl GateType for Blake3Gate {
    type Row = Compression;
    type Hint = ();
    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&build_block_r1cs(self.nu)).with_io_schema(io_schema())
    }
    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Compression {
        let cv = unpack8(inputs[0], inputs[1]);
        let mut m = [0u32; 16];
        for i in 0..4 {
            m[4 * i..4 * i + 4].copy_from_slice(&unpack4(inputs[2 + i]));
        }
        let (counter, block_len, flags) = (inputs[6].lo, inputs[6].hi as u32, (inputs[6].hi >> 32) as u32);
        let out = blake3_compress(&cv, &m, counter, block_len, flags);
        let lo = pack8(out[0..8].try_into().unwrap());
        let hi = pack8(out[8..16].try_into().unwrap());
        outputs.extend_from_slice(&[lo[0], lo[1], hi[0], hi[1]]);
        (cv, m, counter, block_len, flags)
    }
    fn witness(&self, _rows: &[Compression], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// The verifier-built statement shape. Row `leaf * blocks + block` hashes block `block` of operand row `leaf`, so the
/// linked position `row * 512 + u` (u = the block's message bit) is Flock bit `((4 + u/128) << nu | row) * 128 + u%128`.
struct Chain {
    vus: usize,
    k: usize,
    leaves: usize,
    chunks: usize,
    rows: usize,
    nu: usize,
    m_link: usize,
    n_fixed: usize,
    shape: CircuitShape,
    slot: SlotId,
    r1cs: BlockR1cs,
    /// The prime statement Σ names (`prime_commitment`: SHA-256 of its `commitment.txt`).
    prime: String,
}

impl Chain {
    fn new(vus: usize, k: usize) -> Self {
        let leaves = 2 * vus;
        let bytes = 2 * k;
        assert!(bytes % 1024 == 0 && bytes >= 2048, "a leaf is at least two whole BLAKE3 chunks");
        let chunks = bytes / 1024;
        let rows = leaves * (bytes / 64);
        let nu = log2ceil(rows).max(8);
        let mut b = ShapeBuilder::new(nu);
        let slot = b.slot(Blake3Gate { nu });
        let keys: Vec<[Wire; 2]> = [b"x", b"w"]
            .iter()
            .map(|r| pack8(&key_words(&row_key(*r))).map(|w| b.fixed_public_input(w)))
            .collect();
        let params: Vec<[Wire; 3]> = (0..chunks)
            .map(|c| [CHUNK_START, 0, CHUNK_END].map(|f| b.fixed_public_input(pack_params(c as u64, 64, f | KEYED_HASH))))
            .collect();
        let n_fixed = 4 + 3 * chunks;
        // No independence islands: the online phase gives each island a value buffer over every wire, which is
        // quadratic in the leaf count; the sequential run is a few hundred ms at 4,096 VUs.
        for l in 0..leaves {
            let key = keys[usize::from(l >= vus)];
            for pc in &params {
                let mut cv = key;
                for blk in 0..16 {
                    let m: [Wire; 4] = from_fn(|_| b.input());
                    let p = pc[if blk == 0 { 0 } else if blk == 15 { 2 } else { 1 }];
                    let o = b.gate(slot, &[cv[0], cv[1], m[0], m[1], m[2], m[3], p]);
                    cv = [o[0], o[1]];
                }
                b.publish(cv[0]);
                b.publish(cv[1]);
            }
        }
        let shape = b.finish().expect("chain circuit");
        assert_eq!(shape.counts, vec![rows]);
        Chain { vus, k, leaves, chunks, rows, nu, m_link: 9 + log2ceil(rows), n_fixed, shape, slot, r1cs: build_block_r1cs(nu), prime: "selftest".into() }
    }

    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.shape.registry, self.shape.counts.clone())
    }

    fn params(&self, lig: LigeritoProfile) -> PcsParams {
        let u = self.union();
        let m = u.dense_m();
        let batch = embedded_initial_k_or_default(m, lig);
        PcsParams { m, log_inv_rate: lig.log_inv_rate(), log_batch_size: batch, profile: lig, num_lanes: u.commit_lanes(batch), merkle_hash: Default::default() }
    }

    fn fixed(&self) -> Vec<F128> {
        let mut v: Vec<F128> = [b"x", b"w"].iter().flat_map(|r| pack8(&key_words(&row_key(*r)))).collect();
        for c in 0..self.chunks {
            for f in [CHUNK_START, 0, CHUNK_END] {
                v.push(pack_params(c as u64, 64, f | KEYED_HASH));
            }
        }
        v
    }

    fn spec(&self) -> TableSpec {
        self.spec_named(TABLE)
    }

    fn spec_named(&self, name: &str) -> TableSpec {
        let counts_le = self.shape.counts.iter().flat_map(|&n| (n as u64).to_le_bytes()).collect();
        TableSpec { name: name.into(), binding: Binding::Circuit { registry_digest: self.shape.registry.digest(), counts_le, circuit_digest: self.shape.circuit.digest() } }
    }

    /// Σ's preimage (`sigma.txt`): the scheme, Λ on both sides, N, m, the point choice (P2), the Flock statement, the
    /// prime statement (the SHA-256 of its `commitment.txt`) and the leaf digests (x rows, then W columns). The prime
    /// verifier (`archive/gkr/verifier/src/link.rs`) checks every line it can derive from its own statement.
    fn sigma_text(&self, digests: &[[u8; 32]], prime_commitment: &str) -> String {
        let cat: Vec<u8> = digests.iter().flatten().copied().collect();
        format!(
            "verity/link-L/v2\nscheme {SCHEME}\nlambda_f pos=((port*vus+v)*K+word)*bits+i\n\
             lambda_b pos=row*512+u->((4+u/128)<<nu|row)*128+u%128\npoints 2 gf2_128 x^128+x^7+x^2+x+1\n\
             vus {}\nk {}\nbits 16\nn {}\nm {}\nflock_nu {}\nflock_circuit {}\nflock_registry {}\n\
             flock_profile fast100 reps 2\nprime_commitment {prime_commitment}\nleaf_digests {} {}\n",
            self.vus,
            self.k,
            self.leaves * self.k * 16,
            self.m_link,
            self.nu,
            hex(&self.shape.circuit.digest()),
            hex(&self.shape.registry.digest()),
            digests.len(),
            hex(&sha256(&cat)),
        )
    }

    fn sigma(&self, digests: &[[u8; 32]]) -> [u8; 32] {
        sha256(self.sigma_text(digests, &self.prime).as_bytes())
    }

    fn inputs(&self, bytes: &[u8]) -> Vec<F128> {
        let mut v = self.fixed();
        v.reserve(self.rows * 4);
        for blk in bytes.chunks(64) {
            let w: Vec<u32> = blk.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            for j in 0..4 {
                v.push(pack4(&w[4 * j..4 * j + 4]));
            }
        }
        v
    }

    /// The link point (m coordinates over the linked positions) as a point of Flock's padded bit space.
    fn embed(&self, r: &[F128]) -> Vec<F128> {
        let u = self.union();
        let mm = u.m_total();
        let rowbits = self.m_link - 9;
        let mut p = Vec::with_capacity(mm);
        p.extend_from_slice(&r[..7]);
        p.extend_from_slice(&r[9..9 + rowbits]);
        p.resize(7 + self.nu, F128::ZERO);
        p.extend_from_slice(&[r[7], r[8], F128::ONE, F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO]);
        let start = u.slot_word_range(self.shape.registry_slot(self.slot)).start;
        assert_eq!(start % (1usize << (self.nu + 7)), 0, "slot block aligned above its word columns");
        let mut hi = start >> (self.nu + 7);
        while p.len() < mm {
            p.push(if hi & 1 == 1 { F128::ONE } else { F128::ZERO });
            hi >>= 1;
        }
        assert_eq!(hi, 0);
        p
    }

    /// `(rows, public)` of the honest circuit run on `bytes`.
    fn run(&self, bytes: &[u8]) -> (Vec<Compression>, Vec<F128>) {
        let w = self.shape.run(&self.inputs(bytes), &[]);
        (w.rows::<Blake3Gate>(self.slot).to_vec(), w.public)
    }

    fn prove(&self, rows: &[Compression], public: &[F128], lig: LigeritoProfile, hook: LinkHook<'_>, ch: &mut LiveChallenger) -> Vec<u8> {
        let circ = self.r1cs.csc_lincheck_circuit();
        let nu = self.nu;
        let slots = vec![UnionSlotProverInput::in_place(move |dst| generate_witness_batch_major_partial_into(rows, nu, dst), circ)];
        let (proof, commitment, _) =
            prove_fast_ligerito_union_circuit_linked(&self.union(), &self.shape.circuit, public, &self.params(lig), slots, Vec::new(), hook, ch);
        bincode::serialize(&(commitment, proof)).unwrap()
    }
}

type LinkHook<'a> = &'a mut dyn FnMut(&Commitment) -> Vec<LinkClaim>;

/// `sum over set bits p of eq(r, p)`: bit p is bit p%8 of byte p/8 (operand words little-endian, LSB first).
fn link_eval(bytes: &[u8], r: &[F128]) -> F128 {
    let m = r.len();
    assert!(m >= 3 && bytes.len() * 8 <= 1usize << m);
    let lo = eq_skip_weights(&r[..3]);
    let mut t0 = [F128::ZERO; 256];
    for x in 1..256usize {
        let rest = x & (x - 1);
        t0[x] = t0[rest] + lo[(x ^ rest).trailing_zeros() as usize];
    }
    let mid_n = (m - 3).min(11);
    let mid = eq_skip_weights(&r[3..3 + mid_n]);
    let hi = eq_skip_weights(&r[3 + mid_n..]);
    bytes
        .par_chunks(1 << mid_n)
        .enumerate()
        .map(|(h, c)| {
            let mut acc = F128::ZERO;
            for (i, &x) in c.iter().enumerate() {
                if x != 0 {
                    acc = acc + mid[i] * t0[x as usize];
                }
            }
            hi[h] * acc
        })
        .reduce(|| F128::ZERO, |a, b| a + b)
}

fn parent_cv(key: &[u32; 8], l: &[u32; 8], r: &[u32; 8], root: bool) -> [u32; 8] {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(l);
    m[8..].copy_from_slice(r);
    blake3_compress(key, &m, 0, 64, PARENT | KEYED_HASH | if root { ROOT } else { 0 })[..8].try_into().unwrap()
}

/// BLAKE3's keyed tree over at least two chunk chaining values.
fn tree(key: &[u32; 8], cvs: &[[u32; 8]], root: bool) -> [u32; 8] {
    if cvs.len() == 1 {
        return cvs[0];
    }
    let left = 1usize << (usize::BITS - 1 - (cvs.len() - 1).leading_zeros());
    parent_cv(key, &tree(key, &cvs[..left], false), &tree(key, &cvs[left..], false), root)
}

fn digest_of(key: &[u8; 32], cvs: &[[u32; 8]]) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, w) in tree(&key_words(key), cvs, true).iter().enumerate() {
        d[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    d
}

fn words(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n / 4).flat_map(|_| rng.next_u32().to_le_bytes()).collect()
}

fn leaf_digests(ch: &Chain, bytes: &[u8]) -> Vec<[u8; 32]> {
    bytes.chunks(2 * ch.k).enumerate().map(|(i, l)| *blake3::keyed_hash(&leaf_key(ch.vus, i), l).as_bytes()).collect()
}

fn domain(rep: usize) -> Vec<u8> {
    format!("flock-link/fast100x2/rep{rep}").into_bytes()
}

struct ChainVerifier {
    ch: Arc<Chain>,
    digests: BTreeMap<String, Vec<[u8; 32]>>,
}

impl ChainVerifier {
    fn public(&self, publics: &[u8]) -> Vec<F128> {
        let mut p = self.ch.fixed();
        p.extend(bytes_f128(publics));
        p
    }
}

impl StatementVerifier for ChainVerifier {
    fn domain(&self, _table: &str, rep: usize) -> Vec<u8> {
        domain(rep)
    }

    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        let c = &self.ch;
        let Some(digests) = self.digests.get(table) else {
            return Err(format!("C4: {table} is not in the statement"));
        };
        if publics.len() != 16 * 2 * c.chunks * c.leaves {
            return Err(format!("C4: {table}: {} bytes of public words, want {}", publics.len(), 32 * c.chunks * c.leaves));
        }
        let w = bytes_f128(publics);
        for (l, want) in digests.iter().enumerate() {
            let cvs: Vec<[u32; 8]> = (0..c.chunks).map(|j| unpack8(w[2 * (l * c.chunks + j)], w[2 * (l * c.chunks + j) + 1])).collect();
            if digest_of(&leaf_key(c.vus, l), &cvs) != *want {
                return Err(format!("C4: the public chunk values of leaf {l} do not hash to its digest"));
            }
        }
        Ok(Some(publics_digest(&self.public(publics))))
    }

    fn verify(&self, _table: &str, _rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
        let c = &self.ch;
        let (commitment, proof): (Commitment, R1csProofCircuitMerged) = bincode::deserialize(proof).map_err(|e| format!("decode: {e}"))?;
        let pcs = c.params(LigeritoProfile::Fast100);
        if commitment.params != pcs {
            return Err("R7/F2: proof's PCS parameters differ from the pinned Fast100 parameters".into());
        }
        if link.points.len() != 2 || link.y.len() != 2 {
            return Err("L2: the link claims are missing from the session".into());
        }
        let claims: Vec<LinkClaim> = link.points.iter().zip(link.y).map(|(r, y)| LinkClaim { point: c.embed(r), value: *y }).collect();
        let public = self.public(link.publics);
        let circ = c.r1cs.csc_lincheck_circuit();
        let circs: Vec<&dyn flock_core::lincheck::LincheckCircuit> = vec![circ];
        let u = c.union();
        catch_unwind(AssertUnwindSafe(|| {
            verify_ligerito_union_circuit_linked(&u, &c.shape.circuit, &public, &circs, &commitment, &proof, &pcs, ch, &claims)
        }))
        .map_err(|_| "verifier panicked".to_string())?
        .map_err(|e| format!("{e:?}").chars().take(160).collect::<String>())?;
        Ok(commitment.cap.as_flattened().to_vec())
    }
}

fn server(ch: &Arc<Chain>, digests: Vec<[u8; 32]>, prime_live: bool) -> Server {
    let (cfg, v) = server_cfg(ch, vec![(TABLE.to_string(), digests)], prime_live);
    Server::new(cfg, v)
}

/// One session over several chain tables (L3): one Σ over all of them, one `Commit` carrying every root_B before the
/// link points, one y per table and point.
fn server_tables(ch: &Arc<Chain>, tables: Vec<(String, Vec<[u8; 32]>)>) -> Server {
    let (cfg, v) = server_cfg(ch, tables, false);
    Server::new(cfg, v)
}

/// The verifier's configuration and statement verifier; `prime_live`: the route (a) cell's prime coins are served too.
fn server_cfg(ch: &Arc<Chain>, tables: Vec<(String, Vec<[u8; 32]>)>, prime_live: bool) -> (SessionConfig, Box<dyn StatementVerifier>) {
    let mut cfg = SessionConfig::flock_128_r2(tables.iter().map(|(n, _)| ch.spec_named(n)).collect());
    cfg.link = Some(LinkSpec { sigma: ch.sigma(&tables.iter().flat_map(|(_, d)| d.clone()).collect::<Vec<_>>()).to_vec(), m: ch.m_link, points: 2 });
    cfg.prime_live = prime_live;
    (cfg, Box::new(ChainVerifier { ch: ch.clone(), digests: tables.into_iter().collect() }))
}

/// G2: re-verify a finished single-table session from its record and the two proofs kept beside it.
fn replay(ch: &Arc<Chain>, digests: Vec<[u8; 32]>, rec: &Value, proofs: [Vec<u8>; 2]) -> Value {
    let prime_live = rec["config"]["prime_live"] == true;
    let (cfg, v) = server_cfg(ch, vec![(TABLE.to_string(), digests)], prime_live);
    let proofs: BTreeMap<(String, usize), Vec<u8>> = proofs.into_iter().enumerate().map(|(r, b)| ((TABLE.to_string(), r), b)).collect();
    match Server::from_record(cfg, v, rec, proofs) {
        Ok(mut s) => {
            let v = s.finish();
            json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "prime_live": prime_live,
                   "recorded_verdict_accepted": rec["verdict"]["accepted"]})
        }
        Err(e) => json!({"accepted": false, "why": e, "prime_live": prime_live, "recorded_verdict_accepted": rec["verdict"]["accepted"]}),
    }
}

/// The route (a) cell's prime prover reaches the session through `cell-prove`: `Prime` frames on a Unix socket are
/// forwarded on the session's own connection (one session, one record), after `Hello`; the session's proofs and
/// `Finish` wait until the prime prover closes the socket, so no prime round can follow the verdict.
struct PrimeProxy {
    hello: std::sync::atomic::AtomicBool,
    done: Mutex<Option<std::thread::JoinHandle<Value>>>,
}

impl PrimeProxy {
    fn start(path: &str, t: SharedTransport) -> Arc<Self> {
        use std::io::{BufReader, BufWriter};
        use std::sync::atomic::Ordering;
        let _ = std::fs::remove_file(path);
        let l = std::os::unix::net::UnixListener::bind(path).expect("--prime-sock");
        let px = Arc::new(PrimeProxy { hello: false.into(), done: Mutex::new(None) });
        let p2 = px.clone();
        let h = std::thread::spawn(move || {
            let Ok((conn, _)) = l.accept() else { return json!({"error": "no prime connection"}) };
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut w = BufWriter::new(conn);
            let (mut rounds, mut coins, mut up, mut down, mut wait, mut refused) = (0u64, 0u64, 0u64, 0u64, 0f64, None);
            while let Ok(b) = read_frame(&mut r) {
                while !p2.hello.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                let resp = match Req::decode(&b) {
                    Ok(req @ Req::Prime { .. }) => {
                        let tw = Instant::now();
                        let res = t.lock().unwrap().call(req).unwrap_or_else(Resp::Refused);
                        wait += tw.elapsed().as_secs_f64();
                        res
                    }
                    Ok(_) => Resp::Refused("the prime proxy forwards Prime rounds only".into()),
                    Err(e) => Resp::Refused(e),
                };
                let enc = resp.encode();
                rounds += 1;
                up += b.len() as u64 + 4;
                down += enc.len() as u64 + 4;
                match &resp {
                    Resp::Words(v) => coins += (v.len() / PRIME_DEG) as u64,
                    Resp::Refused(e) => { refused.get_or_insert(e.clone()); }
                    _ => {}
                }
                if write_frame(&mut w, &enc).is_err() {
                    break;
                }
            }
            json!({"rounds": rounds, "challenges": coins, "bytes_up": up, "bytes_down": down, "wait_s": wait, "refused": refused})
        });
        *px.done.lock().unwrap() = Some(h);
        px
    }

    fn join(&self) -> Value {
        self.done.lock().unwrap().take().map(|h| h.join().unwrap_or_else(|_| json!({"error": "prime proxy panicked"}))).unwrap_or(Value::Null)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum CommitMode {
    Honest,
    OtherRootB,
    Twice,
    NoRootF,
    YBeforePoints,
    NoY,
}

/// What a (possibly dishonest) prover does. `z` is what Flock commits, `b` the operand bits y is computed from
/// (the prime side's copy); `statement` is what the verifier's leaf digests are computed from.
#[derive(Clone)]
struct Plan {
    commit: CommitMode,
    y_delta: [u64; 2],
    rep1_y_delta: u64,
    flip_z_bit: bool,
    forge_middle: bool,
    bad_params: Option<usize>,
    publics_flip: bool,
    rep1_other_witness: bool,
    profile: [LigeritoProfile; 2],
    substitute: Option<Vec<Vec<u8>>>,
    /// The route (a) cell: the operand rows (x then W, LE words) instead of the seed's, root_F read from this file
    /// (hex, written by the prime prover) and the session's link context written to `link_out`.
    operands: Option<Arc<Vec<u8>>>,
    root_f_file: Option<String>,
    link_out: Option<String>,
    /// The cell: the real prime prover's rounds, through `cell-prove --prime-sock`.
    proxy: Option<Arc<PrimeProxy>>,
    /// The selftest's stand-in prime prover (states are arbitrary: only the server's order is exercised here; the
    /// prime verifier's replay is tested in archive/gkr).
    prime: PrimeSim,
    /// Override the server's `prime_live` (default: the plan's).
    server_prime: Option<bool>,
}

#[derive(Clone, Copy, PartialEq)]
enum PrimeSim {
    Off,
    /// Two rounds after `Hello`, one after y.
    Honest,
    /// A prime coin before `Hello`.
    BeforeHello,
    /// A prime coin between `Commit` (root_F) and y.
    AfterRootFBeforeY,
    /// `Commit` with no prime round before it (the prime proof's GKR coins were not live).
    NoneBeforeRootF,
    /// No prime round after y.
    NoneAfterY,
}

impl Plan {
    fn prime_live(&self) -> bool {
        self.proxy.is_some() || self.prime != PrimeSim::Off
    }

    fn honest() -> Self {
        Plan {
            commit: CommitMode::Honest,
            y_delta: [0, 0],
            rep1_y_delta: 0,
            flip_z_bit: false,
            forge_middle: false,
            bad_params: None,
            publics_flip: false,
            rep1_other_witness: false,
            profile: [LigeritoProfile::Fast100; 2],
            substitute: None,
            operands: None,
            root_f_file: None,
            link_out: None,
            proxy: None,
            prime: PrimeSim::Off,
            server_prime: None,
        }
    }
}

impl Plan {
    /// The server's configuration disagrees with the prover's about live prime coins.
    fn with_server_prime(mut self, on: bool) -> Self {
        self.server_prime = Some(on);
        self
    }
}

/// A forger's `link_sha256` over an edited record (root_F, every root_B, the points, y), as `record_json` computes it.
fn reseal(r: &mut Value) {
    let hb = |v: &Value| -> Vec<u8> {
        let s = v.as_str().unwrap();
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
    };
    let le = |v: &Value| -> Vec<u8> {
        let b = hb(v);
        b[8..].iter().rev().chain(b[..8].iter().rev()).copied().collect()
    };
    let l = &r["link"];
    let mut b = hb(&l["root_f"]);
    for v in l["root_b"].as_object().unwrap().values() {
        b.extend(hb(v));
    }
    for p in l["points"].as_array().unwrap() {
        for c in p.as_array().unwrap() {
            b.extend(le(c));
        }
    }
    for y in l["y"].as_array().unwrap() {
        b.extend(le(y));
    }
    r["link_sha256"] = json!(hex(&sha256(&b)));
}

/// The stand-in prime prover's round `k` (see [`PrimeSim`]).
fn sim_prime(t: &SharedTransport, k: u8) -> Result<(), String> {
    match call(t, Req::Prime { state: sha256(&[b"flock-link/selftest/prime-state".as_slice(), &[k]].concat()).to_vec(), n: 3 })? {
        Resp::Words(w) if w.len() == 3 * PRIME_DEG && w.iter().all(|&x| x < BABYBEAR_P) => Ok(()),
        r => Err(format!("prime round {k}: {r:?}")),
    }
}

fn call(t: &SharedTransport, r: Req) -> Result<Resp, String> {
    t.lock().unwrap().call(r)
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// The instance the plan proves: (statement bytes the verifier's digests come from, z bytes Flock commits, b bytes y
/// comes from).
fn instance(ch: &Chain, seed: u64, plan: &Plan) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let b = plan.operands.as_ref().map(|o| o.to_vec()).unwrap_or_else(|| words(seed, ch.leaves * ch.k * 2));
    let mut z = b.clone();
    if plan.flip_z_bit {
        z[ch.k * 2 + 77] ^= 0x10;
        return (z.clone(), z, b);
    }
    (b.clone(), z, b)
}

fn session(ch: &Chain, t: SharedTransport, seed: u64, plan: &Plan) -> (Value, Vec<Vec<u8>>) {
    let (hello, sigma) = {
        let (st, _, _) = instance(ch, seed, plan);
        let mut cfg = SessionConfig::flock_128_r2(vec![ch.spec()]);
        let sigma = ch.sigma(&leaf_digests(ch, &st));
        cfg.link = Some(LinkSpec { sigma: sigma.to_vec(), m: ch.m_link, points: 2 });
        cfg.prime_live = plan.prime_live();
        (cfg.hello(), sigma)
    };
    let (_, z, b) = instance(ch, seed, plan);
    let t0 = Instant::now();
    let (mut rows, mut public) = ch.run(&z);
    let mut b_link = b.clone();
    if plan.forge_middle {
        // Block 5 of leaf 0 carries another message; its output no longer feeds block 6, whose input is still the
        // honest chaining value, and the chunk's public output stays honest. y is computed from the forged bits.
        let at = 5 * 64 + 9;
        b_link[at] ^= 0x01;
        let (forged, _) = ch.run(&b_link);
        rows[5] = forged[5];
    }
    if let Some(what) = plan.bad_params {
        let r = &mut rows[15];
        match what {
            0 => r.4 &= !CHUNK_END,
            1 => r.2 = 7,
            _ => r.3 = 32,
        }
    }
    if plan.publics_flip {
        public[ch.n_fixed + 3].lo ^= 1;
    }
    let chunk_bytes = f128_bytes(&public[ch.n_fixed..]);
    let rows1 = if plan.rep1_other_witness { ch.run(&words(seed ^ 0xB, z.len())).0 } else { rows.clone() };
    let mut prove_s = Vec::new();
    let mut y_s = 0.0;
    let mut proofs: Vec<Vec<u8>> = Vec::new();
    let mut cached: Option<Vec<LinkClaim>> = None;
    let res: Result<(), String> = (|| {
        if plan.prime == PrimeSim::BeforeHello {
            sim_prime(&t, 0)?;
        }
        let hr = call(&t, Req::Hello(hello));
        if let Some(px) = &plan.proxy {
            px.hello.store(true, std::sync::atomic::Ordering::Release);
        }
        match hr? {
            Resp::Ok => {}
            r => return Err(format!("hello: {r:?}")),
        }
        if matches!(plan.prime, PrimeSim::Honest | PrimeSim::AfterRootFBeforeY | PrimeSim::NoneAfterY) {
            sim_prime(&t, 1)?;
            sim_prime(&t, 2)?;
        }
        if plan.commit == CommitMode::YBeforePoints {
            match call(&t, Req::Link(vec![0u8; 32]))? {
                Resp::Ok => {}
                r => return Err(format!("link: {r:?}")),
            }
        }
        for rep in 0..2 {
            let mut lch = LiveChallenger::new(t.clone(), TABLE, rep, &domain(rep))?;
            let tt = t.clone();
            let mut hook = |cm: &Commitment| -> Vec<LinkClaim> {
                if let Some(c) = &cached {
                    let mut c = c.clone();
                    if rep == 1 && plan.rep1_y_delta != 0 {
                        c[0].value.lo ^= plan.rep1_y_delta;
                    }
                    return c;
                }
                let mut root_b = cm.cap.as_flattened().to_vec();
                if plan.commit == CommitMode::OtherRootB {
                    root_b[0] ^= 1;
                }
                let root_f = if plan.commit == CommitMode::NoRootF {
                    Vec::new()
                } else if let Some(f) = &plan.root_f_file {
                    wait_root_f(f)
                } else {
                    sha256(&[b"flock-link/root-f-standin/v1".as_slice(), &seed.to_le_bytes()].concat()).to_vec()
                };
                let req = || Req::Commit { root_f: root_f.clone(), roots: vec![(TABLE.into(), root_b.clone())], publics: vec![(TABLE.into(), chunk_bytes.clone())] };
                let pts = match call(&tt, req()) {
                    Ok(Resp::Coins(c)) if c.len() == 2 * ch.m_link => c,
                    other => panic!("LIVE-REFUSED commit: {other:?}"),
                };
                if plan.commit == CommitMode::Twice {
                    match call(&tt, req()) {
                        Ok(Resp::Coins(_)) => {}
                        other => panic!("LIVE-REFUSED commit twice: {other:?}"),
                    }
                }
                if plan.prime == PrimeSim::AfterRootFBeforeY {
                    sim_prime(&tt, 3).unwrap_or_else(|e| panic!("LIVE-REFUSED {e}"));
                }
                let ty = Instant::now();
                let claims: Vec<LinkClaim> = pts
                    .chunks(ch.m_link)
                    .zip(plan.y_delta)
                    .map(|(r, d)| {
                        let mut y = link_eval(&b_link, r);
                        y.lo ^= d;
                        LinkClaim { point: ch.embed(r), value: y }
                    })
                    .collect();
                y_s = ty.elapsed().as_secs_f64();
                if plan.commit != CommitMode::NoY {
                    match call(&tt, Req::Link(f128_bytes(&claims.iter().map(|c| c.value).collect::<Vec<_>>()))) {
                        Ok(Resp::Ok) => {}
                        other => panic!("LIVE-REFUSED y: {other:?}"),
                    }
                }
                if plan.prime == PrimeSim::Honest {
                    sim_prime(&tt, 4).unwrap_or_else(|e| panic!("LIVE-REFUSED {e}"));
                }
                if let Some(out) = &plan.link_out {
                    let pts: Vec<Vec<String>> = pts.chunks(ch.m_link).map(|r| r.iter().map(f128_hex).collect()).collect();
                    let doc = json!({"kind": "flock-link-context/v1", "sigma": hex(&sigma), "root_f_sha256": hex(&sha256(&root_f)),
                                     "root_b_sha256": hex(&sha256(&root_b)), "points": pts,
                                     "y": claims.iter().map(|c| f128_hex(&c.value)).collect::<Vec<_>>()});
                    std::fs::write(format!("{out}.tmp"), doc.to_string()).unwrap();
                    std::fs::rename(format!("{out}.tmp"), out).unwrap();
                }
                cached = Some(claims.clone());
                claims
            };
            let tp = Instant::now();
            let r = if rep == 0 { &rows } else { &rows1 };
            let p = catch_unwind(AssertUnwindSafe(|| ch.prove(r, &public, plan.profile[rep], &mut hook, &mut lch))).map_err(panic_msg)?;
            prove_s.push(tp.elapsed().as_secs_f64());
            proofs.push(p);
        }
        Ok(())
    })();
    if let Err(e) = res {
        let v = match call(&t, Req::Finish) {
            Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
            other => json!({"accepted": false, "why": format!("{other:?}")}),
        };
        return (json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "prover_stopped": e.chars().take(200).collect::<String>()}), proofs);
    }
    let flock_done_s = t0.elapsed().as_secs_f64();
    let prime = plan.proxy.as_ref().map(|px| px.join());
    let sent = plan.substitute.clone().unwrap_or_else(|| proofs.clone());
    for (rep, p) in sent.iter().enumerate() {
        call(&t, Req::Proof { table: TABLE.into(), rep: rep as u32, bytes: p.clone() }).unwrap();
    }
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let (ws, wr, calls) = t.lock().unwrap().stats();
    let wait_s = t.lock().unwrap().wait_s();
    (
        json!({"accepted": v["accepted"], "why": v["why"], "verdict": v, "vus": ch.vus, "k": ch.k, "rows": ch.rows, "nu": ch.nu,
               "m_link": ch.m_link, "dense_m": ch.params(LigeritoProfile::Fast100).m, "prove_s": prove_s,
               "prove_total_s": prove_s.iter().sum::<f64>(), "y_s": y_s, "session_wall_s": t0.elapsed().as_secs_f64(),
               "proof_bytes": proofs.iter().map(|p| p.len()).collect::<Vec<_>>(), "wire_sent": ws, "wire_recv": wr,
               "calls": calls, "wait_s": wait_s, "flock_done_s": flock_done_s, "prime": prime}),
        proofs,
    )
}

#[derive(Clone, Copy, PartialEq)]
enum L3Case {
    Honest,
    OneRootOnly,
    SecondRootOther,
    SecondYTampered,
    ProofsSwapped,
}

/// Two chain tables in ONE session (L3): the second table's root is computed first (its prove runs up to the
/// binding, where the hook stops it: the commit is deterministic), so the single `Commit` carries both roots before
/// the link points are drawn; each table opens its own y at the shared points in both reps.
fn session_l3(ch: &Arc<Chain>, case: L3Case) -> Value {
    let names = ["chain0", "chain1"];
    let data: Vec<Vec<u8>> = (0..2).map(|i| words(0xA + 0x100 * i as u64, ch.leaves * ch.k * 2)).collect();
    let tables: Vec<(String, Vec<[u8; 32]>)> = (0..2).map(|i| (names[i].to_string(), leaf_digests(ch, &data[i]))).collect();
    let s = Arc::new(Mutex::new(server_tables(ch, tables)));
    let t = shared(InProc(s.clone()));
    let runs: Vec<(Vec<Compression>, Vec<F128>)> = data.iter().map(|d| ch.run(d)).collect();
    let root1: Vec<u8> = {
        let mut got = None;
        let mut stop = |cm: &Commitment| -> Vec<LinkClaim> {
            got = Some(cm.cap.as_flattened().to_vec());
            panic!("commit-only");
        };
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let mut fs = flock_prover::challenger::FsChallenger::with_hash(&domain(0), Default::default());
            let circ = ch.r1cs.csc_lincheck_circuit();
            let nu = ch.nu;
            let rows = &runs[1].0;
            let slots = vec![UnionSlotProverInput::in_place(move |dst| generate_witness_batch_major_partial_into(rows, nu, dst), circ)];
            prove_fast_ligerito_union_circuit_linked(&ch.union(), &ch.shape.circuit, &runs[1].1, &ch.params(LigeritoProfile::Fast100), slots, Vec::new(), &mut stop, &mut fs)
        }));
        got.expect("commit-only pass reached the binding")
    };
    let mut cached: Option<(Vec<Vec<F128>>, Vec<F128>)> = None;
    let res: Result<(), String> = (|| {
        let hello = s.lock().unwrap().cfg.hello();
        call(&t, Req::Hello(hello))?;
        let mut proofs = vec![Vec::new(), Vec::new()];
        for (ti, name) in names.iter().enumerate() {
            for rep in 0..2 {
                let mut lch = LiveChallenger::new(t.clone(), name, rep, &domain(rep))?;
                let tt = t.clone();
                let mut hook = |cm: &Commitment| -> Vec<LinkClaim> {
                    if cached.is_none() {
                        let mut r1 = root1.clone();
                        if case == L3Case::SecondRootOther {
                            r1[3] ^= 1;
                        }
                        let mut roots = vec![(names[0].to_string(), cm.cap.as_flattened().to_vec())];
                        if case != L3Case::OneRootOnly {
                            roots.push((names[1].to_string(), r1));
                        }
                        let publics = (0..2).map(|i| (names[i].to_string(), f128_bytes(&runs[i].1[ch.n_fixed..]))).collect();
                        let root_f = sha256(b"flock-link/root-f-standin/v1/l3").to_vec();
                        let pts = match call(&tt, Req::Commit { root_f, roots, publics }) {
                            Ok(Resp::Coins(c)) => c.chunks(ch.m_link).map(|r| r.to_vec()).collect::<Vec<_>>(),
                            other => panic!("LIVE-REFUSED commit: {other:?}"),
                        };
                        let mut ys = Vec::new();
                        for (i, d) in data.iter().enumerate() {
                            for r in &pts {
                                let mut y = link_eval(d, r);
                                if i == 1 && case == L3Case::SecondYTampered {
                                    y.hi ^= 1;
                                }
                                ys.push(y);
                            }
                        }
                        match call(&tt, Req::Link(f128_bytes(&ys))) {
                            Ok(Resp::Ok) => {}
                            other => panic!("LIVE-REFUSED y: {other:?}"),
                        }
                        cached = Some((pts, ys));
                    }
                    let (pts, ys) = cached.as_ref().unwrap();
                    pts.iter().zip(&ys[2 * ti..2 * ti + 2]).map(|(r, y)| LinkClaim { point: ch.embed(r), value: *y }).collect()
                };
                let (rows, public) = &runs[ti];
                let p = catch_unwind(AssertUnwindSafe(|| ch.prove(rows, public, LigeritoProfile::Fast100, &mut hook, &mut lch))).map_err(panic_msg)?;
                proofs[ti].push(p);
            }
        }
        if case == L3Case::ProofsSwapped {
            proofs.swap(0, 1);
        }
        for (ti, name) in names.iter().enumerate() {
            for rep in 0..2 {
                call(&t, Req::Proof { table: name.to_string(), rep: rep as u32, bytes: proofs[ti][rep].clone() })?;
            }
        }
        Ok(())
    })();
    let v: Value = match call(&t, Req::Finish) {
        Ok(Resp::Verdict(v)) => serde_json::from_str(&v).unwrap(),
        other => json!({"accepted": false, "why": format!("{other:?}")}),
    };
    let rec = s.lock().unwrap().record_json();
    json!({"accepted": v["accepted"], "why": v["why"], "prover_stopped": res.err(), "reps": v["reps"],
           "tables": rec["config"]["tables"].as_array().map(|a| a.len()), "link": rec["link"]["root_b_sha256"]})
}

fn inproc(ch: &Arc<Chain>, statement: &[u8], prime_live: bool) -> (SharedTransport, Arc<Mutex<Server>>) {
    let s = Arc::new(Mutex::new(server(ch, leaf_digests(ch, statement), prime_live)));
    (shared(InProc(s.clone())), s)
}

fn neg(ch: &Arc<Chain>, name: &str, maps_to: &str, plan: &Plan, want_accept: bool) -> (bool, Value) {
    let (ok, rec, _) = neg_proofs(ch, name, maps_to, plan, want_accept);
    (ok, rec)
}

fn neg_proofs(ch: &Arc<Chain>, name: &str, maps_to: &str, plan: &Plan, want_accept: bool) -> (bool, Value, Vec<Vec<u8>>) {
    let (st, _, _) = instance(ch, 0xA, plan);
    let (t, s) = inproc(ch, &st, plan.server_prime.unwrap_or(plan.prime_live()));
    let (v, proofs) = session(ch, t, 0xA, plan);
    let acc = v["accepted"].as_bool().unwrap_or(false);
    let rec = s.lock().unwrap().record_json();
    let reps: Vec<Value> = rec["verdict"]["reps"].as_array().cloned().unwrap_or_default().iter().map(|r| json!([r["stream"], r["ok"]])).collect();
    println!(
        "NEG\t{}",
        json!({"case": name, "maps_to": maps_to, "vus": ch.vus, "expect": if want_accept {"accept"} else {"reject"},
               "accepted": acc, "pass": acc == want_accept, "why": v["why"], "prover_stopped": v["prover_stopped"],
               "reps_ok": reps, "aborted": rec["aborted"], "link_mode": rec["link_mode"], "link_sha256": rec["link_sha256"]})
    );
    (acc == want_accept, rec, proofs)
}

fn selftest(ch: Arc<Chain>) {
    let h = Plan::honest;
    let all = std::cell::Cell::new(true);
    let t = |name: &str, maps: &str, p: Plan, acc: bool| -> Value {
        let (ok, rec) = neg(&ch, name, maps, &p, acc);
        all.set(all.get() && ok);
        rec
    };
    // Both reps must reject on their own for the L2 cases: check the per-rep verdicts.
    let both = |rec: &Value| -> bool {
        rec["verdict"]["reps"].as_array().map(|a| a.len() == 2 && a.iter().all(|r| r["ok"] == false)).unwrap_or(false)
    };
    let e0 = {
        let mut cfg = SessionConfig::flock_128_r2(vec![ch.spec()]);
        cfg.link = Some(LinkSpec { sigma: vec![0; 32], m: ch.m_link, points: 2 });
        cfg.require_link = false;
        Server::try_new(cfg, Box::new(ChainVerifier { ch: ch.clone(), digests: BTreeMap::new() })).err()
    };
    println!("NEG\t{}", json!({"case": "ungated_exchange_config_refused", "maps_to": "E0", "expect": "reject",
        "pass": e0.is_some(), "why": e0}));
    all.set(all.get() && e0.is_some());
    let honest = t("honest_linked_session", "honest", h(), true);
    let pts_a = honest["link"]["points"].clone();
    let honest2 = t("honest_linked_session_again", "honest (fresh points)", h(), true);
    let fresh = pts_a != honest2["link"]["points"];
    println!("NEG\t{}", json!({"case": "points_fresh_per_session_same_roots", "maps_to": "link neg 11", "pass": fresh,
        "same_roots": honest["link"]["root_b_sha256"] == honest2["link"]["root_b_sha256"]}));
    all.set(all.get() && fresh);
    t("root_b_other_than_bound_root", "L1 (root_B is the root R1 binds)", Plan { commit: CommitMode::OtherRootB, ..h() }, false);
    t("roots_committed_twice", "L1 / link neg 11 (points drawn once)", Plan { commit: CommitMode::Twice, ..h() }, false);
    t("no_root_f", "L1 / C2", Plan { commit: CommitMode::NoRootF, ..h() }, false);
    t("y_before_points", "L1 order", Plan { commit: CommitMode::YBeforePoints, ..h() }, false);
    t("coins_before_y", "L1 / R5", Plan { commit: CommitMode::NoY, ..h() }, false);
    let r = t("y1_tampered", "L2 / link neg 10", Plan { y_delta: [1, 0], ..h() }, false);
    let b1 = both(&r);
    let r = t("y1_y2_tampered_same_delta", "L2 / link neg 10 (compensation)", Plan { y_delta: [1 << 40, 1 << 40], ..h() }, false);
    let b2 = both(&r);
    let r = t("y2_tampered", "L2 / link neg 10", Plan { y_delta: [0, 1 << 63], ..h() }, false);
    let b3 = both(&r);
    println!("NEG\t{}", json!({"case": "link_neg10_rejected_by_each_rep", "maps_to": "L2", "pass": b1 && b2 && b3}));
    all.set(all.get() && b1 && b2 && b3);
    // The opening carries no y: each rep's ring switch is checked against the verifier's recorded y, whatever the
    // prover believed it was opening.
    t("rep1_prover_claims_other_y", "L2 (the recorded y is what both reps are checked against)", Plan { rep1_y_delta: 1, ..h() }, true);
    let r = t("z_bit_differs_from_y_source", "link neg 2 (bit flip in z only)", Plan { flip_z_bit: true, ..h() }, false);
    let b4 = both(&r);
    println!("NEG\t{}", json!({"case": "link_neg2_rejected_by_each_rep", "maps_to": "L2", "pass": b4}));
    all.set(all.get() && b4);
    t("forged_middle_block_honest_endpoints", "L4 / link neg 13", Plan { forge_middle: true, ..h() }, false);
    t("chunk_end_flag_dropped", "L4 / link neg 14 (flag)", Plan { bad_params: Some(0), ..h() }, false);
    t("wrong_counter", "L4 / link neg 14 (counter)", Plan { bad_params: Some(1), ..h() }, false);
    t("wrong_block_len", "L4 / link neg 14 (block_len)", Plan { bad_params: Some(2), ..h() }, false);
    t("public_chunk_value_forged", "L4 / C4 endpoints", Plan { publics_flip: true, ..h() }, false);
    t("reps_with_different_witness", "R1", Plan { rep1_other_witness: true, ..h() }, false);
    t("fast_profile_proofs", "R7 / F2", Plan { profile: [LigeritoProfile::Fast; 2], ..h() }, false);
    let (st, _, _) = instance(&ch, 0xA, &h());
    let (_, other) = session(&ch, inproc(&ch, &st, false).0, 0xA, &h());
    t("cross_session_replay_both_reps", "link neg 15", Plan { substitute: Some(other), ..h() }, false);
    // live prime coins: the server's order (the prime verifier's replay of the rounds is archive/gkr's tests)
    let prime = |p: PrimeSim| Plan { prime: p, ..h() };
    let (ok, prec, pproofs) = neg_proofs(&ch, "prime_live_honest", "P-live honest", &prime(PrimeSim::Honest), true);
    all.set(all.get() && ok);
    let shape = prec["prime"]["before_commit"] == 2 && prec["prime"]["rounds"].as_array().map(|a| a.len()) == Some(3)
        && prec["hello"].as_str().is_some_and(|x| x.contains(PRIME_MODE));
    println!("NEG\t{}", json!({"case": "prime_live_record_shape", "maps_to": "P-live / G2", "pass": shape}));
    all.set(all.get() && shape);
    t("prime_coin_before_hello", "P-live coins before commitments (R7)", prime(PrimeSim::BeforeHello), false);
    t("prime_coin_after_root_f_before_y", "P-live coins before commitments (R5)", prime(PrimeSim::AfterRootFBeforeY), false);
    t("root_f_without_live_prime_coins", "P-live (the GKR coins were not live)", prime(PrimeSim::NoneBeforeRootF), false);
    t("no_prime_coin_after_y", "P-live (rho and the queries were not live)", prime(PrimeSim::NoneAfterY), false);
    t("prime_rounds_to_a_flock_only_session", "P-live (R7: the configuration pins prime_live)",
      Plan { prime: PrimeSim::Honest, ..h() }.with_server_prime(false), false);
    // G2: the offline replay from the record alone, and tampered records
    let d = leaf_digests(&ch, &st);
    let pp: [Vec<u8>; 2] = [pproofs[0].clone(), pproofs[1].clone()];
    let rp = |name: &str, maps: &str, rec: &Value, proofs: &[Vec<u8>; 2], want: bool| {
        let v = replay(&ch, d.clone(), rec, proofs.clone());
        let acc = v["accepted"] == true;
        all.set(all.get() && acc == want);
        println!("NEG\t{}", json!({"case": name, "maps_to": maps, "expect": if want {"accept"} else {"reject"}, "accepted": acc,
            "pass": acc == want, "why": v["why"]}));
    };
    rp("replay_honest_record", "G2 honest", &prec, &pp, true);
    let edit = |f: &dyn Fn(&mut Value)| { let mut r = prec.clone(); f(&mut r); r };
    let flip_hex = |s: &Value| -> Value {
        let mut b = s.as_str().unwrap().to_string().into_bytes();
        let i = b.len() - 1;
        b[i] = if b[i] == b'0' { b'1' } else { b'0' };
        json!(String::from_utf8(b).unwrap())
    };
    let rep0 = |r: &Value| r["streams"].as_array().unwrap().iter().position(|s| s["stream"] == "chain/rep0").unwrap();
    rp("replay_tampered_flock_coin", "G2 tampered record (a coin)", &edit(&|r| {
        let i = rep0(r);
        let c = &mut r["streams"][i]["rounds"][3]["coins"][0];
        *c = flip_hex(c);
    }), &pp, false);
    rp("replay_tampered_msg_sha", "G2 tampered record (a committed round)", &edit(&|r| {
        let i = rep0(r);
        let c = &mut r["streams"][i]["rounds"][5]["msg_sha256"];
        *c = flip_hex(c);
    }), &pp, false);
    rp("replay_tampered_publics_resealed", "G2 tampered record (a public word, sha256 recomputed)", &edit(&|r| {
        let p = flip_hex(&r["link"]["publics"][TABLE]);
        let b: Vec<u8> = (0..p.as_str().unwrap().len() / 2).map(|i| u8::from_str_radix(&p.as_str().unwrap()[2 * i..2 * i + 2], 16).unwrap()).collect();
        r["link"]["publics_sha256"][TABLE] = json!(hex(&sha256(&b)));
        r["link"]["publics"][TABLE] = p;
    }), &pp, false);
    rp("replay_tampered_y_unsealed", "G2 tampered record (y; link_sha256 left)", &edit(&|r| {
        let y = flip_hex(&r["link"]["y"][0]);
        r["link"]["y"][0] = y;
    }), &pp, false);
    rp("replay_tampered_y_resealed", "G2 tampered record (y; link_sha256 recomputed)", &edit(&|r| {
        let y = flip_hex(&r["link"]["y"][0]);
        r["link"]["y"][0] = y;
        reseal(r);
    }), &pp, false);
    rp("replay_tampered_point_resealed", "G2 tampered record (a link point; link_sha256 recomputed)", &edit(&|r| {
        let c = flip_hex(&r["link"]["points"][1][4]);
        r["link"]["points"][1][4] = c;
        reseal(r);
    }), &pp, false);
    rp("replay_prime_before_commit_zeroed", "G2 tampered record (prime order)", &edit(&|r| r["prime"]["before_commit"] = json!(0)), &pp, false);
    rp("replay_prime_rounds_dropped", "G2 tampered record (prime rounds)", &edit(&|r| r["prime"] = Value::Null), &pp, false);
    rp("replay_hello_edited", "G2 tampered record (session parameters)", &edit(&|r| {
        r["hello"] = json!(r["hello"].as_str().unwrap().replace("\"reps\":2", "\"reps\":1"));
    }), &pp, false);
    rp("replay_proof_swapped", "G2 (the proofs beside the record)", &prec, &[pp[1].clone(), pp[0].clone()], false);
    let (_, bad, bproofs) = neg_proofs(&ch, "prime_live_y1_tampered", "L2 (for the replay below)", &Plan { y_delta: [1, 0], ..prime(PrimeSim::Honest) }, false);
    rp("replay_rejected_session_verdict_edited", "G2 (the replay recomputes the verdict)", &{
        let mut r = bad.clone();
        r["verdict"]["accepted"] = json!(true);
        r
    }, &[bproofs[0].clone(), bproofs[1].clone()], false);
    for (case, name, want) in [
        (L3Case::Honest, "l3_two_tables_one_session", true),
        (L3Case::OneRootOnly, "l3_points_before_second_root", false),
        (L3Case::SecondRootOther, "l3_second_table_binds_other_root", false),
        (L3Case::SecondYTampered, "l3_second_table_y_tampered", false),
        (L3Case::ProofsSwapped, "l3_tables_proofs_swapped", false),
    ] {
        let v = session_l3(&ch, case);
        let acc = v["accepted"].as_bool().unwrap_or(false);
        all.set(all.get() && acc == want);
        println!("NEG\t{}", json!({"case": name, "maps_to": "L3", "expect": if want {"accept"} else {"reject"}, "accepted": acc,
            "pass": acc == want, "why": v["why"], "prover_stopped": v["prover_stopped"], "reps": v["reps"], "roots": v["link"]}));
    }
    println!("SELFTEST\t{}", json!({"vus": ch.vus, "k": ch.k, "rows": ch.rows, "m_link": ch.m_link, "all_pass": all.get()}));
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    let vus: usize = arg("--vus", "8").parse().unwrap();
    let k: usize = arg("--k", "1536").parse().unwrap();
    let seed: u64 = arg("--seed", "10").parse().unwrap();
    let tb = Instant::now();
    let mut chain = Chain::new(vus, k);
    if cmd.starts_with("cell") || cmd == "sigma" || cmd == "replay" {
        chain.prime = arg("--prime-commitment", "");
        assert!(chain.prime.len() == 64, "--prime-commitment: the SHA-256 (hex) of the prime statement's commitment.txt");
    }
    let ch = Arc::new(chain);
    eprintln!("statement vus={vus} k={k} rows={} nu={} m_link={} built in {:.2}s", ch.rows, ch.nu, ch.m_link, tb.elapsed().as_secs_f64());
    match cmd.as_str() {
        "selftest" => selftest(ch),
        "serve" => {
            let st = words(seed, ch.leaves * ch.k * 2);
            let d = leaf_digests(&ch, &st);
            serve_loop(move || server(&ch, d.clone(), false));
        }
        "prove" => {
            let runs: usize = arg("--runs", "1").parse().unwrap();
            for run in 0..runs {
                let t = shared(TcpTransport::connect(&arg("--verifier", "127.0.0.1:7200")).unwrap());
                let (mut v, _) = session(&ch, t, seed, &Plan::honest());
                v["run"] = json!(run);
                println!("LIVE\t{v}");
            }
        }
        "sigma" => print!("{}", ch.sigma_text(&read_digests(&ch), &ch.prime)),
        "cell-serve" => {
            let d = read_digests(&ch);
            serve_loop(move || server(&ch, d.clone(), true));
        }
        "replay" => {
            let dir = std::path::PathBuf::from(arg("--session", ""));
            let rec: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("session.json")).expect("--session DIR/session.json"))
                .expect("session.json");
            let proof = |r: usize| std::fs::read(dir.join(format!("{TABLE}.rep{r}.bin"))).unwrap_or_default();
            let tr = Instant::now();
            let mut v = replay(&ch, read_digests(&ch), &rec, [proof(0), proof(1)]);
            v["replay_s"] = json!(tr.elapsed().as_secs_f64());
            v["session"] = json!(dir.display().to_string());
            println!("REPLAY\t{v}");
            if let Some(out) = std::env::args().position(|a| a == "--json").and_then(|i| std::env::args().nth(i + 1)) {
                std::fs::write(out, serde_json::to_string_pretty(&v).unwrap()).unwrap();
            }
            std::process::exit(if v["accepted"] == true { 0 } else { 1 });
        }
        "cell-prove" => {
            let d = read_digests(&ch);
            let ops = std::fs::read(arg("--operands", "")).expect("--operands: the x rows then the W columns, LE 16-bit words");
            assert_eq!(leaf_digests(&ch, &ops), d, "the operands do not hash to the statement's leaf digests");
            let plan = Plan {
                operands: Some(Arc::new(ops)),
                root_f_file: Some(arg("--root-f", "")),
                link_out: Some(arg("--link-out", "")),
                ..Plan::honest()
            };
            // `--start FILE`: the statement is built; the session (and its clock) starts when FILE appears
            eprintln!("READY");
            let start = arg("--start", "");
            while !start.is_empty() && !std::path::Path::new(&start).exists() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // the verifier may be re-binding its port between the sweep's sizes
            let addr = arg("--verifier", "127.0.0.1:7200");
            let tc = Instant::now();
            let conn = loop {
                match TcpTransport::connect(&addr) {
                    Ok(c) => break c,
                    Err(e) if tc.elapsed().as_secs() < 60 => { eprintln!("connect {addr}: {e}; retrying"); std::thread::sleep(std::time::Duration::from_millis(500)) }
                    Err(e) => panic!("connect {addr}: {e}"),
                }
            };
            let t = shared(conn);
            let sock = arg("--prime-sock", "");
            let plan = Plan { proxy: (!sock.is_empty()).then(|| PrimeProxy::start(&sock, t.clone())), ..plan };
            let (v, _) = session(&ch, t, seed, &plan);
            println!("LIVE\t{v}");
        }
        _ => eprintln!("usage: flock-link serve|prove|selftest|sigma|cell-serve|cell-prove|replay [--vus V] [--k K] [--seed S] ..."),
    }
}

/// `--digests`: the prime statement's leaf digests, x rows then W columns, 32 bytes each.
fn read_digests(ch: &Chain) -> Vec<[u8; 32]> {
    let b = std::fs::read(arg("--digests", "")).expect("--digests");
    assert_eq!(b.len(), 32 * ch.leaves, "--digests: {} leaves of 32 bytes", ch.leaves);
    b.chunks(32).map(|c| c.try_into().unwrap()).collect()
}

fn serve_loop(make: impl Fn() -> Server) {
    use std::io::{BufReader, BufWriter, Write};
    let listen = arg("--listen", "127.0.0.1:7200");
    let out = std::path::PathBuf::from(arg("--out", "sessions"));
    std::fs::create_dir_all(&out).unwrap();
    let l = std::net::TcpListener::bind(&listen).unwrap();
    let max: Option<usize> = arg("--sessions", "").parse().ok();
    println!("SERVING {listen}");
    for (i, conn) in l.incoming().enumerate() {
        let conn = conn.unwrap();
        conn.set_nodelay(true).unwrap();
        let mut r = BufReader::new(conn.try_clone().unwrap());
        let mut w = BufWriter::new(conn);
        let mut s = make();
        let id = format!("l{:04}-{}", i, std::process::id());
        loop {
            let Ok(b) = read_frame(&mut r) else { break };
            let resp = match Req::decode(&b) {
                Ok(req) => s.handle(req),
                Err(e) => Resp::Refused(e),
            };
            let done = matches!(resp, Resp::Verdict(_));
            if write_frame(&mut w, &resp.encode()).is_err() || done {
                break;
            }
        }
        let rec = s.record_json();
        let d = out.join(&id);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("session.json"), serde_json::to_string_pretty(&rec).unwrap()).unwrap();
        for ((t, r), b) in s.proofs() {
            std::fs::write(d.join(format!("{}.bin", rep_stream(t, *r).replace('/', "."))), b).unwrap();
        }
        let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
        std::fs::write(d.join("verifier.json"), json!({"operator": arg("--operator", ""), "host": host.trim(),
            "runpod_pod_id": std::env::var("RUNPOD_POD_ID").ok(), "research_run": std::env::var("RESEARCH_RUN_ID").ok(),
            "listen": listen}).to_string()).unwrap();
        let line = json!({"session": id, "accepted": rec["verdict"]["accepted"], "why": rec["verdict"]["why"], "aborted": rec["aborted"],
                          "link_mode": rec["link_mode"], "link_sha256": rec["link_sha256"], "require_link": rec["config"]["require_link"],
                          "verify_s": rec["verdict"]["verify_s"], "rounds": rec["verdict"]["rounds"]});
        let mut idx = std::fs::OpenOptions::new().create(true).append(true).open(out.join("index.jsonl")).unwrap();
        writeln!(idx, "{line}").unwrap();
        println!("SESSION\t{line}");
        if Some(i + 1) == max {
            break;
        }
    }
}
