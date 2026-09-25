//! Live verifier coins for Flock (the flock-128-r2 profile: two `Fast100` runs, soundness squared only with live coins).
//!
//! * [`LiveChallenger`] is the prover's `Challenger`. It never derives a challenge: every squeeze sends the bytes the
//!   prover absorbed since its previous squeeze (the "round", framed exactly like `FsChallenger`'s SHA-256
//!   transcript-v2 absorb stream, so the CUDA prover's host challenger produces identical rounds) and waits for the
//!   verifier's coins. Forks open live child streams.
//! * [`Server`] is the verifier's side of a session. It issues a coin only after it holds the round that precedes it,
//!   draws coins from the OS, pins the session parameters (profile, reps, flavour, statement), refuses a rep-1 coin
//!   unless rep 1 committed rep 0's root, and refuses every coin until the link context (root_F, points, y) arrived.
//!   With a [`LinkSpec`] the link context is a real exchange (flock-link L1): `Commit` carries root_F, every table's
//!   root_B and its public words; the server checks the public words against its statement, then draws the link
//!   points from the OS as its own coin slot; `Link` carries y; only then are Flock's coins issued, and each rep's
//!   first round must bind exactly the committed root_B. The replay hands the recorded points and y to the statement
//!   verifier, which opens them in every rep (L2).
//! * [`ReplayChallenger`] runs the ordinary Flock verifier over the recorded session: at every squeeze it checks that the
//!   proof's absorbed bytes hash to the round the server recorded before it issued that coin, then returns that coin.
//!   The session record ([`Server::record_json`]) is the evidence; a proof file alone verifies nothing.

pub mod chunk;
pub mod pure_block;
#[cfg(feature = "gpu")]
pub mod gpu;

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    net::TcpStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use flock_field::F128;
use flock_transcript::challenger::Challenger;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const OP_DOMAIN: u8 = 0x01;
pub const OP_LABEL: u8 = 0x02;
pub const OP_OBSERVE: u8 = 0x03;
pub const OP_SQUEEZE: u8 = 0x04;
pub const OP_BYTES: u8 = 0x05;
pub const KIND_SCALAR: u8 = 0x01;
pub const KIND_SLICE: u8 = 0x02;

pub const LABEL_MIXED: &[u8] = b"flock-mixed-v1";
pub const LABEL_R1CS: &[u8] = b"flock-r1cs-v0";

// ---------------------------------------------------------------------------------------------------------------
// Framing (transcript-v2: 16-byte header [op][kind][0;6][len u64 LE], payload zero-padded to 16).
// ---------------------------------------------------------------------------------------------------------------

pub fn push_header(out: &mut Vec<u8>, op: u8, kind: u8, len: u64) {
    let mut h = [0u8; 16];
    h[0] = op;
    h[1] = kind;
    h[8..].copy_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&h);
}

pub fn push_padded(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(b);
    let rem = b.len() % 16;
    if rem != 0 {
        out.resize(out.len() + 16 - rem, 0);
    }
}

fn push_f128(out: &mut Vec<u8>, v: F128) {
    out.extend_from_slice(&v.lo.to_le_bytes());
    out.extend_from_slice(&v.hi.to_le_bytes());
}

pub fn frame_domain(domain: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    push_header(&mut v, OP_DOMAIN, 0, domain.len() as u64);
    push_padded(&mut v, domain);
    v
}

pub fn sha256(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn f128_hex(v: &F128) -> String {
    format!("{:016x}{:016x}", v.hi, v.lo)
}

/// Stream names: rep roots are `<table>/rep<r>`, children `<parent>/f<i>`.
pub fn rep_stream(table: &str, rep: usize) -> String {
    format!("{table}/rep{rep}")
}

fn child_stream(parent: &str, idx: u32) -> String {
    format!("{parent}/f{idx}")
}

// ---------------------------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Req {
    Hello(String),
    Link(Vec<u8>),
    /// flock-link L1: root_F (the prime side's first commitment), each table's root_B, each table's public words.
    Commit { root_f: Vec<u8>, roots: Vec<(String, Vec<u8>)>, publics: Vec<(String, Vec<u8>)> },
    Open { stream: String, parent: String, label: Vec<u8>, at: u64 },
    Round { stream: String, content: Vec<u8>, n: u32 },
    Proof { table: String, rep: u32, bytes: Vec<u8> },
    Finish,
}

#[derive(Clone, Debug)]
pub enum Resp {
    Ok,
    Coins(Vec<F128>),
    Refused(String),
    Verdict(String),
}

fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn put_bytes(w: &mut Vec<u8>, b: &[u8]) {
    put_u32(w, b.len() as u32);
    w.extend_from_slice(b);
}

struct Cur<'a>(&'a [u8], usize);
impl Cur<'_> {
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.0.get(self.1..self.1 + 4).ok_or("short frame")?;
        self.1 += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.0.get(self.1..self.1 + 8).ok_or("short frame")?;
        self.1 += 8;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let n = self.u32()? as usize;
        let b = self.0.get(self.1..self.1 + n).ok_or("short frame")?;
        self.1 += n;
        Ok(b.to_vec())
    }
    fn string(&mut self) -> Result<String, String> {
        String::from_utf8(self.bytes()?).map_err(|e| e.to_string())
    }
}

impl Req {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Vec::new();
        match self {
            Req::Hello(s) => {
                w.push(1);
                put_bytes(&mut w, s.as_bytes());
            }
            Req::Link(b) => {
                w.push(2);
                put_bytes(&mut w, b);
            }
            Req::Open { stream, parent, label, at } => {
                w.push(3);
                put_bytes(&mut w, stream.as_bytes());
                put_bytes(&mut w, parent.as_bytes());
                put_bytes(&mut w, label);
                w.extend_from_slice(&at.to_le_bytes());
            }
            Req::Round { stream, content, n } => {
                w.push(4);
                put_bytes(&mut w, stream.as_bytes());
                put_bytes(&mut w, content);
                put_u32(&mut w, *n);
            }
            Req::Proof { table, rep, bytes } => {
                w.push(5);
                put_bytes(&mut w, table.as_bytes());
                put_u32(&mut w, *rep);
                put_bytes(&mut w, bytes);
            }
            Req::Finish => w.push(6),
            Req::Commit { root_f, roots, publics } => {
                w.push(7);
                put_bytes(&mut w, root_f);
                for list in [roots, publics] {
                    put_u32(&mut w, list.len() as u32);
                    for (t, b) in list {
                        put_bytes(&mut w, t.as_bytes());
                        put_bytes(&mut w, b);
                    }
                }
            }
        }
        w
    }

    pub fn decode(b: &[u8]) -> Result<Req, String> {
        let mut c = Cur(b, 1);
        Ok(match b.first().ok_or("empty frame")? {
            1 => Req::Hello(c.string()?),
            2 => Req::Link(c.bytes()?),
            3 => Req::Open { stream: c.string()?, parent: c.string()?, label: c.bytes()?, at: c.u64()? },
            4 => Req::Round { stream: c.string()?, content: c.bytes()?, n: c.u32()? },
            5 => Req::Proof { table: c.string()?, rep: c.u32()?, bytes: c.bytes()? },
            6 => Req::Finish,
            7 => {
                let root_f = c.bytes()?;
                let mut lists = Vec::new();
                for _ in 0..2 {
                    let n = c.u32()? as usize;
                    let mut l = Vec::new();
                    for _ in 0..n.min(1 << 10) {
                        l.push((c.string()?, c.bytes()?));
                    }
                    lists.push(l);
                }
                let publics = lists.pop().unwrap();
                let roots = lists.pop().unwrap();
                Req::Commit { root_f, roots, publics }
            }
            t => return Err(format!("unknown request tag {t}")),
        })
    }
}

impl Resp {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Vec::new();
        match self {
            Resp::Ok => w.push(1),
            Resp::Coins(v) => {
                w.push(2);
                put_u32(&mut w, v.len() as u32);
                for c in v {
                    push_f128(&mut w, *c);
                }
            }
            Resp::Refused(s) => {
                w.push(3);
                put_bytes(&mut w, s.as_bytes());
            }
            Resp::Verdict(s) => {
                w.push(4);
                put_bytes(&mut w, s.as_bytes());
            }
        }
        w
    }

    pub fn decode(b: &[u8]) -> Result<Resp, String> {
        let mut c = Cur(b, 1);
        Ok(match b.first().ok_or("empty frame")? {
            1 => Resp::Ok,
            2 => {
                let n = c.u32()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let lo = c.u64()?;
                    let hi = c.u64()?;
                    v.push(F128 { lo, hi });
                }
                Resp::Coins(v)
            }
            3 => Resp::Refused(c.string()?),
            4 => Resp::Verdict(c.string()?),
            t => return Err(format!("unknown response tag {t}")),
        })
    }
}

pub fn write_frame(w: &mut impl Write, b: &[u8]) -> std::io::Result<()> {
    w.write_all(&(b.len() as u32).to_le_bytes())?;
    w.write_all(b)?;
    w.flush()
}

pub fn read_frame(r: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut n = [0u8; 4];
    r.read_exact(&mut n)?;
    let mut b = vec![0u8; u32::from_le_bytes(n) as usize];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// How the prover reaches the verifier.
pub trait Transport: Send {
    fn call(&mut self, req: Req) -> Result<Resp, String>;
    /// Bytes sent and received so far, and the number of calls.
    fn stats(&self) -> (u64, u64, u64) {
        (0, 0, 0)
    }
    /// Seconds the prover spent waiting inside `call` (the latency live coins add).
    fn wait_s(&self) -> f64 {
        0.0
    }
}

pub struct TcpTransport {
    r: BufReader<TcpStream>,
    w: BufWriter<TcpStream>,
    sent: u64,
    recv: u64,
    calls: u64,
    wait: std::time::Duration,
}

impl TcpTransport {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let s = TcpStream::connect(addr)?;
        s.set_nodelay(true)?;
        Ok(Self { r: BufReader::new(s.try_clone()?), w: BufWriter::new(s), sent: 0, recv: 0, calls: 0, wait: Default::default() })
    }
}

impl Transport for TcpTransport {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let t = std::time::Instant::now();
        let b = req.encode();
        self.sent += b.len() as u64 + 4;
        self.calls += 1;
        write_frame(&mut self.w, &b).map_err(|e| format!("send: {e}"))?;
        let r = read_frame(&mut self.r).map_err(|e| format!("recv: {e}"))?;
        self.recv += r.len() as u64 + 4;
        self.wait += t.elapsed();
        Resp::decode(&r)
    }
    fn stats(&self) -> (u64, u64, u64) {
        (self.sent, self.recv, self.calls)
    }
    fn wait_s(&self) -> f64 {
        self.wait.as_secs_f64()
    }
}

/// The server in the same process (tests and negatives); the messages still go through encode/decode.
pub struct InProc(pub Arc<Mutex<Server>>);

impl Transport for InProc {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let req = Req::decode(&req.encode())?;
        let resp = self.0.lock().unwrap().handle(req);
        Resp::decode(&resp.encode())
    }
}

pub type SharedTransport = Arc<Mutex<Box<dyn Transport>>>;

pub fn shared(t: impl Transport + 'static) -> SharedTransport {
    Arc::new(Mutex::new(Box::new(t)))
}

// ---------------------------------------------------------------------------------------------------------------
// Prover side
// ---------------------------------------------------------------------------------------------------------------

/// Adversarial provers for the negatives: what the prover commits to the verifier differs from what it puts in the
/// proof. `round` counts the stream's coin-issuing rounds from 0.
#[derive(Clone, Debug)]
pub enum Tamper {
    /// Flip a byte of round `round`'s committed content: the proof's message at that point is not the one the coin
    /// was issued against (a message altered after its coin).
    Alter { stream: String, round: usize },
    /// Commit round `round` without its messages and send them at the start of the next round (a message sent after
    /// its coin).
    Defer { stream: String, round: usize },
    /// Absorb `pow_nonce` at every PoW site but put a different nonce in the proof.
    Nonce,
}

/// The prover's challenger in a live session. See the crate docs.
pub struct LiveChallenger {
    stream: String,
    link: SharedTransport,
    pending: Vec<u8>,
    rounds: u64,
    forks: AtomicU32,
    /// The value `grind_pow` returns and absorbs. Live coins need no grinding; a nonzero value models a prover that
    /// chooses its nonce (negative 4): the coin that follows is drawn by the verifier regardless.
    pub pow_nonce: u64,
    pub tamper: Option<Tamper>,
    deferred: Vec<u8>,
}

impl LiveChallenger {
    /// A rep's root stream. The server refuses its coins until the stream is opened (`open_rep`).
    pub fn new(link: SharedTransport, table: &str, rep: usize, domain: &[u8]) -> Result<Self, String> {
        let stream = rep_stream(table, rep);
        match link.lock().unwrap().call(Req::Open {
            stream: stream.clone(),
            parent: String::new(),
            label: domain.to_vec(),
            at: 0,
        })? {
            Resp::Ok => {}
            Resp::Refused(e) => return Err(e),
            r => return Err(format!("unexpected {r:?}")),
        }
        Ok(Self::stream(link, stream, frame_domain(domain)))
    }

    fn stream(link: SharedTransport, stream: String, pending: Vec<u8>) -> Self {
        Self {
            stream,
            link,
            pending,
            rounds: 0,
            forks: AtomicU32::new(0),
            pow_nonce: 0,
            tamper: None,
            deferred: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.stream
    }

    /// One round: `content` is everything absorbed since the previous squeeze, squeeze header included.
    pub fn round(&mut self, mut content: Vec<u8>, n: usize) -> Vec<F128> {
        let k = self.rounds as usize;
        if !self.deferred.is_empty() {
            let mut c = std::mem::take(&mut self.deferred);
            c.extend_from_slice(&content);
            content = c;
        }
        match &self.tamper {
            Some(Tamper::Alter { stream, round }) if *stream == self.stream && *round == k => {
                content[0] ^= 1;
            }
            Some(Tamper::Defer { stream, round }) if *stream == self.stream && *round == k => {
                let tail = content.split_off(content.len() - 16);
                self.deferred = content;
                content = tail;
            }
            _ => {}
        }
        self.rounds += 1;
        let resp = self.link.lock().unwrap().call(Req::Round { stream: self.stream.clone(), content, n: n as u32 });
        match resp {
            Ok(Resp::Coins(c)) if c.len() == n => c,
            Ok(Resp::Refused(e)) => panic!("LIVE-REFUSED {}: {e}", self.stream),
            other => panic!("LIVE-ERROR {}: {other:?}", self.stream),
        }
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let content = std::mem::take(&mut self.pending);
        self.round(content, n)
    }

    fn open_child(&self, label: &[u8]) -> String {
        let idx = self.forks.fetch_add(1, Ordering::SeqCst);
        let child = child_stream(&self.stream, idx);
        let resp = self.link.lock().unwrap().call(Req::Open {
            stream: child.clone(),
            parent: self.stream.clone(),
            label: label.to_vec(),
            at: self.rounds,
        });
        match resp {
            Ok(Resp::Ok) => child,
            other => panic!("LIVE-REFUSED open {child}: {other:?}"),
        }
    }
}

/// Absorb and squeeze methods shared by the live and the replay challenger: both must frame identically.
macro_rules! framed_challenger {
    () => {
        fn observe_label(&mut self, label: &[u8]) {
            push_header(&mut self.pending, OP_LABEL, 0, label.len() as u64);
            push_padded(&mut self.pending, label);
        }
        fn observe_f128(&mut self, v: F128) {
            push_header(&mut self.pending, OP_OBSERVE, KIND_SCALAR, 1);
            push_f128(&mut self.pending, v);
        }
        fn observe_f128_slice(&mut self, vs: &[F128]) {
            push_header(&mut self.pending, OP_OBSERVE, KIND_SLICE, vs.len() as u64);
            for v in vs {
                push_f128(&mut self.pending, *v);
            }
        }
        fn observe_bytes(&mut self, b: &[u8]) {
            push_header(&mut self.pending, OP_BYTES, 0, b.len() as u64);
            push_padded(&mut self.pending, b);
        }
        fn sample_f128(&mut self) -> F128 {
            self.squeeze(KIND_SCALAR, 1)[0]
        }
        fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
            self.squeeze(KIND_SLICE, n)
        }
    };
}

impl Challenger for LiveChallenger {
    framed_challenger!();

    /// R4: no grinding. The nonce is absorbed like any message; the coin after it is the verifier's.
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        let n = self.pow_nonce;
        self.observe_bytes(&n.to_le_bytes());
        if matches!(self.tamper, Some(Tamper::Nonce)) { n ^ 1 } else { n }
    }

    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        unreachable!("the prover's challenger never verifies")
    }

    /// R3: the child is a live stream of its own, opened at this position of the parent (after the parent's seed
    /// round, so the parent's messages before the fork are committed first); its coins come from the verifier.
    fn fork(&mut self, label: &'static [u8]) -> Self {
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }

    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        let name = self.open_child(label);
        let mut c = LiveChallenger::stream(self.link.clone(), name, frame_domain(label));
        c.observe_f128_slice(&seed);
        c.pow_nonce = self.pow_nonce;
        c.tamper = self.tamper.clone();
        c
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Verifier side: coin server
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Binding {
    /// `UnionInstance::bind_statement`: label, registry digest, counts (u64 LE each), root.
    Mixed { registry_digest: [u8; 32], counts_le: Vec<u8> },
    /// `verify_ligerito` / the CUDA prover: label, statement digest, root.
    R1cs { statement_digest: [u8; 32] },
    /// `UnionInstance::bind_statement_circuit`: the Mixed binding, then `flock-circuit-stmt-v2`, the circuit digest
    /// and the digest of the public words (pinned from the committed public words, checked by the statement).
    Circuit { registry_digest: [u8; 32], counts_le: Vec<u8>, circuit_digest: [u8; 32] },
}

pub const LABEL_CIRCUIT: &[u8] = b"flock-circuit-stmt-v2";

/// flock-link: the link context the verifier requires before Flock's first coin.
#[derive(Clone, Debug)]
pub struct LinkSpec {
    /// Σ, the link statement digest (scheme, Λ, N, m, the two sub-proof statements): recorded, and sent in `Hello`.
    pub sigma: [u8; 32],
    /// Coordinates per link point (the linked position space is `{0,1}^m`).
    pub m: usize,
    /// Independent GF(2^128) points (2: the Schwartz–Zippel term squares).
    pub points: usize,
}

#[derive(Clone, Debug)]
pub struct TableSpec {
    pub name: String,
    pub binding: Binding,
}

/// Fixed by the verifier's configuration, never by the prover (R7).
#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub profile: String,
    pub reps: usize,
    pub flavor: String,
    pub tables: Vec<TableSpec>,
    pub require_link: bool,
    /// `None`: the pre-flock-link stub (opaque `Link` bytes). `Some`: the real exchange.
    pub link: Option<LinkSpec>,
}

impl SessionConfig {
    pub fn flock_128_r2(tables: Vec<TableSpec>) -> Self {
        Self { profile: "fast100".into(), reps: 2, flavor: "rs".into(), tables, require_link: true, link: None }
    }

    /// E0: a link exchange is only sound with the R5 gate (no coin before y), so `link: Some` implies `require_link`.
    pub fn check(&self) -> Result<(), String> {
        if self.link.is_some() && !self.require_link {
            return Err("E0: a link exchange (link: Some) without the R5 gate (require_link: false) is refused: y could follow Flock's coins".into());
        }
        Ok(())
    }

    pub fn hello(&self) -> String {
        let mut h = json!({"profile": self.profile, "reps": self.reps, "flavor": self.flavor,
               "tables": self.tables.iter().map(|t| t.name.clone()).collect::<Vec<_>>()});
        if let Some(l) = &self.link {
            h["link"] = json!({"sigma": hex(&l.sigma), "m": l.m, "points": l.points});
        }
        h.to_string()
    }
}

#[derive(Clone, Debug)]
pub struct RoundRec {
    pub sha: [u8; 32],
    pub len: usize,
    pub n: u32,
    pub coins: Vec<F128>,
}

#[derive(Clone, Debug, Default)]
pub struct StreamRec {
    pub parent: String,
    pub label: Vec<u8>,
    pub at: u64,
    pub rounds: Vec<RoundRec>,
    pub cap: Option<Vec<u8>>,
}

/// The statement-specific verifier: decode `proof`, pin the flavour and the PCS parameters from the session
/// configuration, run the Flock verifier with `ch`, and return the flattened root the proof carries.
pub trait StatementVerifier: Send {
    fn verify(&self, table: &str, rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>)
        -> Result<Vec<u8>, String>;
    fn domain(&self, table: &str, rep: usize) -> Vec<u8>;
    /// Check a table's committed public words against the verifier's own statement (the endpoints: flock-link C4)
    /// and return the public-words digest its first round must bind (`None`: the binding carries no public words).
    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        if publics.is_empty() { Ok(None) } else { Err(format!("{table}: this statement takes no public words")) }
    }
}

/// What the verifier's record holds for one table when its proofs are replayed: the link points it drew, the y the
/// prover sent for this table (one per point), and the table's committed public words. Empty without a [`LinkSpec`].
pub struct LinkCtx<'a> {
    pub points: &'a [Vec<F128>],
    pub y: &'a [F128],
    pub publics: &'a [u8],
}

#[derive(Clone, Debug)]
struct CommitRec {
    root_f: Vec<u8>,
    roots: BTreeMap<String, Vec<u8>>,
    publics: BTreeMap<String, Vec<u8>>,
    publics_digest: BTreeMap<String, Option<[u8; 32]>>,
}

pub struct Server {
    pub cfg: SessionConfig,
    verifier: Box<dyn StatementVerifier>,
    hello: Option<String>,
    link: Option<Vec<u8>>,
    commit: Option<CommitRec>,
    points: Vec<Vec<F128>>,
    y: Vec<F128>,
    streams: BTreeMap<String, StreamRec>,
    proofs: BTreeMap<(String, usize), Vec<u8>>,
    aborted: Option<String>,
    verdict: Option<Value>,
    urandom: File,
    t0: std::time::Instant,
    /// Seconds spent handling everything but `Finish` (the verifier's compute inside the prover's coin waits).
    handle_s: f64,
}

fn parse_frame<'a>(c: &'a [u8], at: &mut usize, op: u8) -> Result<&'a [u8], String> {
    let h = c.get(*at..*at + 16).ok_or("binding: short round")?;
    if h[0] != op || h[1] != 0 || h[2..8] != [0u8; 6] {
        return Err(format!("binding: expected op {op} at byte {at}, found op {} kind {}", h[0], h[1]));
    }
    let len = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
    let p = c.get(*at + 16..*at + 16 + len).ok_or("binding: short payload")?;
    *at += 16 + len.div_ceil(16) * 16;
    Ok(p)
}

impl Server {
    /// Panics on a configuration [`SessionConfig::check`] refuses; [`Server::try_new`] returns the refusal.
    pub fn new(cfg: SessionConfig, verifier: Box<dyn StatementVerifier>) -> Self {
        Self::try_new(cfg, verifier).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_new(cfg: SessionConfig, verifier: Box<dyn StatementVerifier>) -> Result<Self, String> {
        cfg.check()?;
        Ok(Self {
            cfg,
            verifier,
            hello: None,
            link: None,
            commit: None,
            points: Vec::new(),
            y: Vec::new(),
            streams: BTreeMap::new(),
            proofs: BTreeMap::new(),
            aborted: None,
            verdict: None,
            urandom: File::open("/dev/urandom").expect("/dev/urandom"),
            t0: std::time::Instant::now(),
            handle_s: 0.0,
        })
    }

    /// The proofs received, by (table, rep): kept beside the session record (E1).
    pub fn proofs(&self) -> &BTreeMap<(String, usize), Vec<u8>> {
        &self.proofs
    }

    fn refuse(&mut self, why: String) -> Resp {
        if self.aborted.is_none() {
            self.aborted = Some(why.clone());
        }
        Resp::Refused(why)
    }

    fn coins(&mut self, n: usize) -> Vec<F128> {
        let mut b = vec![0u8; 16 * n];
        self.urandom.read_exact(&mut b).expect("urandom");
        b.chunks(16)
            .map(|c| F128 {
                lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
            })
            .collect()
    }

    /// The root a rep's first round commits: domain, then the table's statement binding.
    fn parse_binding(&self, table: &TableSpec, content: &[u8]) -> Result<Vec<u8>, String> {
        let mut at = 0;
        parse_frame(content, &mut at, OP_DOMAIN)?;
        let (label, digest) = match &table.binding {
            Binding::Mixed { registry_digest, .. } | Binding::Circuit { registry_digest, .. } => (LABEL_MIXED, registry_digest),
            Binding::R1cs { statement_digest } => (LABEL_R1CS, statement_digest),
        };
        if parse_frame(content, &mut at, OP_LABEL)? != label {
            return Err("binding: wrong statement label".into());
        }
        if parse_frame(content, &mut at, OP_BYTES)? != digest {
            return Err("binding: statement digest differs from the configured statement".into());
        }
        if let Binding::Mixed { counts_le, .. } | Binding::Circuit { counts_le, .. } = &table.binding
            && parse_frame(content, &mut at, OP_BYTES)? != counts_le.as_slice()
        {
            return Err("binding: table counts differ from the configured statement".into());
        }
        let cap = parse_frame(content, &mut at, OP_BYTES)?.to_vec();
        if let Binding::Circuit { circuit_digest, .. } = &table.binding {
            if parse_frame(content, &mut at, OP_LABEL)? != LABEL_CIRCUIT {
                return Err("binding: no circuit statement after the root".into());
            }
            if parse_frame(content, &mut at, OP_BYTES)? != circuit_digest {
                return Err("F2: circuit digest differs from the configured statement".into());
            }
            let want = self.commit.as_ref().and_then(|c| c.publics_digest.get(&table.name).cloned().flatten());
            if Some(parse_frame(content, &mut at, OP_BYTES)?) != want.as_ref().map(|d| d.as_slice()) {
                return Err("C4: public words bound in the proof differ from the committed, checked ones".into());
            }
        }
        Ok(cap)
    }

    fn root_of(&self, stream: &str) -> Option<(TableSpec, usize)> {
        let (t, r) = stream.rsplit_once("/rep")?;
        let rep: usize = r.parse().ok()?;
        self.cfg.tables.iter().find(|x| x.name == t).map(|x| (x.clone(), rep))
    }

    pub fn handle(&mut self, req: Req) -> Resp {
        let finish = matches!(req, Req::Finish);
        let t = std::time::Instant::now();
        let r = self.handle_inner(req);
        if !finish {
            self.handle_s += t.elapsed().as_secs_f64();
        }
        r
    }

    fn handle_inner(&mut self, req: Req) -> Resp {
        if let Some(why) = &self.aborted {
            return Resp::Refused(format!("session aborted: {why}"));
        }
        match req {
            Req::Hello(h) => {
                let want = self.cfg.hello();
                if h != want {
                    return self.refuse(format!("R7: session parameters {h} differ from the verifier's {want}"));
                }
                self.hello = Some(h);
                Resp::Ok
            }
            Req::Link(b) => {
                if self.link.is_some() {
                    return self.refuse("link context sent twice".into());
                }
                if let Some(spec) = self.cfg.link.clone() {
                    if self.commit.is_none() {
                        return self.refuse("L1: y sent before root_F, root_B and the link points".into());
                    }
                    let want = 16 * spec.points * self.cfg.tables.len();
                    if b.len() != want {
                        return self.refuse(format!("L1: y is {} bytes, want {want}", b.len()));
                    }
                    self.y = b.chunks(16).map(|c| F128 {
                        lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                        hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
                    }).collect();
                }
                self.link = Some(b);
                Resp::Ok
            }
            Req::Commit { root_f, roots, publics } => {
                let Some(spec) = self.cfg.link.clone() else {
                    return self.refuse("L1: this session has no link exchange".into());
                };
                if self.hello.is_none() {
                    return self.refuse("R7: roots committed before the session parameters".into());
                }
                if self.commit.is_some() {
                    return self.refuse("L1: roots committed twice (the link points are drawn once)".into());
                }
                if root_f.is_empty() {
                    return self.refuse("L1/C2: no root_F: the points must follow both sides' roots".into());
                }
                // A statement with no link points (flock-pure) commits its Σ in root_F's place.
                if spec.points == 0 && root_f != spec.sigma {
                    return self.refuse("statement: Commit names another statement than the verifier's Σ".into());
                }
                let roots: BTreeMap<String, Vec<u8>> = roots.into_iter().collect();
                let publics: BTreeMap<String, Vec<u8>> = publics.into_iter().collect();
                let names: Vec<String> = self.cfg.tables.iter().map(|t| t.name.clone()).collect();
                if roots.keys().cloned().collect::<Vec<_>>() != names || roots.values().any(|r| r.is_empty()) {
                    return self.refuse(format!("L1/L3: root_B must be committed for exactly the tables {names:?}, together"));
                }
                if publics.keys().any(|k| !names.contains(k)) {
                    return self.refuse("public words for a table outside the statement".into());
                }
                let mut pd = BTreeMap::new();
                for n in &names {
                    match self.verifier.statement_publics(n, publics.get(n).map(|v| v.as_slice()).unwrap_or(&[])) {
                        Ok(d) => { pd.insert(n.clone(), d); }
                        Err(e) => return self.refuse(e),
                    }
                }
                self.commit = Some(CommitRec { root_f, roots, publics, publics_digest: pd });
                let pts: Vec<Vec<F128>> = (0..spec.points).map(|_| self.coins(spec.m)).collect();
                self.points = pts;
                Resp::Coins(self.points.concat())
            }
            Req::Open { stream, parent, label, at } => {
                if self.hello.is_none() {
                    return self.refuse("R7: stream opened before the session parameters".into());
                }
                if self.streams.contains_key(&stream) {
                    return self.refuse(format!("stream {stream} opened twice"));
                }
                if parent.is_empty() {
                    let Some((table, rep)) = self.root_of(&stream) else {
                        return self.refuse(format!("unknown root stream {stream}"));
                    };
                    if rep >= self.cfg.reps {
                        return self.refuse(format!("R7: rep {rep} beyond reps = {}", self.cfg.reps));
                    }
                    if label != self.verifier.domain(&table.name, rep) {
                        return self.refuse(format!("stream {stream}: wrong transcript domain"));
                    }
                    if rep > 0 && self.streams.get(&rep_stream(&table.name, rep - 1)).and_then(|s| s.cap.as_ref()).is_none() {
                        return self.refuse(format!("R1: {stream} opened before rep {} committed its root", rep - 1));
                    }
                } else {
                    let Some(p) = self.streams.get(&parent) else {
                        return self.refuse(format!("fork of unknown stream {parent}"));
                    };
                    // F3: a fork follows at least one issued round of its parent, and its rep root's binding round.
                    let root = parent.split('/').take(2).collect::<Vec<_>>().join("/");
                    if at == 0 || self.streams.get(&root).and_then(|r| r.cap.as_ref()).is_none() {
                        return self.refuse(format!("F3: fork {stream} opened at parent position {at}, before {root}'s binding round"));
                    }
                    if p.rounds.len() as u64 != at {
                        return self.refuse(format!("R3: fork {stream} claims parent position {at}, parent has issued {}", p.rounds.len()));
                    }
                    let idx = self.streams.keys().filter(|k| k.starts_with(&format!("{parent}/f"))
                        && !k[parent.len() + 2..].contains('/')).count();
                    if stream != child_stream(&parent, idx as u32) {
                        return self.refuse(format!("R3: fork stream {stream} is not {parent}'s next child"));
                    }
                }
                self.streams.insert(stream, StreamRec { parent, label, at, ..Default::default() });
                Resp::Ok
            }
            Req::Round { stream, content, n } => {
                if (self.cfg.require_link || self.cfg.link.is_some()) && self.link.is_none() {
                    return self.refuse("R5: coin requested before root_F, the link points and y were received".into());
                }
                let Some(rec) = self.streams.get(&stream) else {
                    return self.refuse(format!("round on unopened stream {stream}"));
                };
                if n == 0 || n > 1 << 16 {
                    return self.refuse(format!("bad coin count {n}"));
                }
                if rec.parent.is_empty() && rec.rounds.is_empty() {
                    let (table, rep) = self.root_of(&stream).expect("root checked at open");
                    let cap = match self.parse_binding(&table, &content) {
                        Ok(c) => c,
                        Err(e) => return self.refuse(format!("{stream}: {e}")),
                    };
                    if rep > 0 {
                        let r0 = self.streams[&rep_stream(&table.name, 0)].cap.clone();
                        if r0.as_ref() != Some(&cap) {
                            return self.refuse(format!(
                                "R1: {stream} committed root {} but rep 0 committed {}; no rep-{rep} coin is issued",
                                hex(&sha256(&cap)[..8]),
                                r0.map(|c| hex(&sha256(&c)[..8])).unwrap_or_default()
                            ));
                        }
                    }
                    if self.cfg.link.is_some() {
                        let rb = self.commit.as_ref().and_then(|c| c.roots.get(&table.name));
                        if rb != Some(&cap) {
                            return self.refuse(format!("L1: {stream} binds a root other than the committed root_B"));
                        }
                    }
                    self.streams.get_mut(&stream).unwrap().cap = Some(cap);
                }
                let coins = self.coins(n as usize);
                self.streams.get_mut(&stream).unwrap().rounds.push(RoundRec { sha: sha256(&content), len: content.len(), n, coins: coins.clone() });
                Resp::Coins(coins)
            }
            Req::Proof { table, rep, bytes } => {
                self.proofs.insert((table, rep as usize), bytes);
                Resp::Ok
            }
            Req::Finish => {
                let v = self.finish();
                Resp::Verdict(v.to_string())
            }
        }
    }

    /// Verify every table's every rep against the recorded session (R2 replay, R7 pins, R8 verdict).
    pub fn finish(&mut self) -> Value {
        let mut reps_out = Vec::new();
        let mut ok = self.aborted.is_none();
        let mut why = self.aborted.clone().unwrap_or_default();
        let t = std::time::Instant::now();
        let record = Arc::new(self.streams.clone());
        let n_pts = self.cfg.link.as_ref().map(|l| l.points).unwrap_or(0);
        for (ti, table) in self.cfg.tables.clone().into_iter().enumerate() {
            let publics = self.commit.as_ref().and_then(|c| c.publics.get(&table.name).cloned()).unwrap_or_default();
            let y: Vec<F128> = self.y.get(ti * n_pts..(ti + 1) * n_pts).map(|s| s.to_vec()).unwrap_or_default();
            for rep in 0..self.cfg.reps {
                let root = rep_stream(&table.name, rep);
                let res: Result<(), String> = (|| {
                    let bytes = self.proofs.get(&(table.name.clone(), rep)).ok_or(format!("R7: no proof for {root} (lone rep)"))?;
                    let rec = record.get(&root).ok_or(format!("no live session stream for {root}"))?;
                    let mut ch = ReplayChallenger::new(record.clone(), &root, &self.verifier.domain(&table.name, rep));
                    // A replay failure is the primary reason: the verifier's own error on a proof whose messages
                    // were not the committed ones is a consequence of it.
                    if self.cfg.link.is_some() && (self.points.len() != n_pts || y.len() != n_pts) {
                        return Err(format!("{root}: no complete link context in the session"));
                    }
                    let ctx = LinkCtx { points: &self.points, y: &y, publics: &publics };
                    let cap = self.verifier.verify(&table.name, rep, bytes, &mut ch, &ctx).map_err(|e| match ch.first_failure() {
                        Some(r) => format!("{r} (then the verifier: {e})"),
                        None => e,
                    })?;
                    ch.finish()?;
                    if rec.cap.as_ref() != Some(&cap) {
                        return Err(format!("{root}: proof root differs from the root committed in the session"));
                    }
                    Ok(())
                })();
                if let Err(e) = &res {
                    if ok {
                        why = e.clone();
                    }
                    ok = false;
                }
                reps_out.push(json!({"stream": root, "ok": res.is_ok(), "why": res.err().unwrap_or_default()}));
            }
        }
        let rounds: usize = self.streams.values().map(|s| s.rounds.len()).sum();
        let empty: usize = self.streams.values().map(|s| s.rounds.iter().filter(|r| r.len <= 16).count()).sum();
        let v = json!({
            "accepted": ok, "why": why, "reps": reps_out, "streams": self.streams.len(), "rounds": rounds,
            "rounds_without_prover_message": empty,
            "verify_s": t.elapsed().as_secs_f64(), "session_s": self.t0.elapsed().as_secs_f64(), "handle_s": self.handle_s,
        });
        self.verdict = Some(v.clone());
        v
    }

    /// The session record (R8): what the verifier saw and issued, and its verdict.
    pub fn record_json(&self) -> Value {
        let streams: Vec<Value> = self
            .streams
            .iter()
            .map(|(name, s)| {
                json!({
                    "stream": name, "parent": s.parent, "label": String::from_utf8_lossy(&s.label), "at": s.at,
                    "root_sha256": s.cap.as_ref().map(|c| hex(&sha256(c))),
                    "rounds": s.rounds.iter().enumerate().map(|(k, r)| json!({
                        "k": k, "n": r.n, "msg_len": r.len, "msg_sha256": hex(&r.sha),
                        "coins": r.coins.iter().map(f128_hex).collect::<Vec<_>>(),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        let link = self.commit.as_ref().map(|c| json!({
            "mode": "exchange",
            "sigma": self.cfg.link.as_ref().map(|l| hex(&l.sigma)),
            "root_f_sha256": hex(&sha256(&c.root_f)),
            "root_b_sha256": c.roots.iter().map(|(t, r)| (t.clone(), json!(hex(&sha256(r))))).collect::<serde_json::Map<_, _>>(),
            "publics_sha256": c.publics.iter().map(|(t, p)| (t.clone(), json!(hex(&sha256(p))))).collect::<serde_json::Map<_, _>>(),
            "points": self.points.iter().map(|p| p.iter().map(f128_hex).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "y": self.y.iter().map(f128_hex).collect::<Vec<_>>(),
        }));
        // link_sha256 covers the whole exchange in exchange mode: root_F, every root_B, the points and y.
        let link_sha = match (&self.commit, &self.link) {
            (Some(c), Some(l)) => {
                let mut b = c.root_f.clone();
                for r in c.roots.values() { b.extend_from_slice(r); }
                for p in self.points.iter().flatten() { push_f128(&mut b, *p); }
                b.extend_from_slice(l);
                Some(hex(&sha256(&b)))
            }
            (None, Some(l)) if self.cfg.link.is_none() => Some(hex(&sha256(l))),
            _ => None,
        };
        json!({
            "kind": "flock-live-session/v1",
            "hello": self.hello, "link_sha256": link_sha, "link": link,
            "link_mode": if self.cfg.link.is_some() { "exchange" } else { "stub" },
            "config": {"profile": self.cfg.profile, "reps": self.cfg.reps, "flavor": self.cfg.flavor,
                       "require_link": self.cfg.require_link,
                       "tables": self.cfg.tables.iter().map(|t| json!({"name": t.name, "binding": format!("{:?}", t.binding)})).collect::<Vec<_>>()},
            "proof_sha256": self.proofs.iter().map(|((t, r), b)| json!({"stream": rep_stream(t, *r), "sha256": hex(&sha256(b)), "bytes": b.len()})).collect::<Vec<_>>(),
            "aborted": self.aborted, "verdict": self.verdict, "streams": streams,
        })
    }

    pub fn aborted(&self) -> Option<&str> {
        self.aborted.as_deref()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Verifier side: replay
// ---------------------------------------------------------------------------------------------------------------

struct ReplayState {
    record: Arc<BTreeMap<String, StreamRec>>,
    used: BTreeMap<String, usize>,
    failures: Vec<String>,
}

/// The verifier's challenger over a recorded session. See the crate docs.
pub struct ReplayChallenger {
    stream: String,
    st: Arc<Mutex<ReplayState>>,
    pending: Vec<u8>,
    forks: AtomicU32,
    rounds: u64,
}

impl ReplayChallenger {
    pub fn new(record: Arc<BTreeMap<String, StreamRec>>, root: &str, domain: &[u8]) -> Self {
        let st = Arc::new(Mutex::new(ReplayState { record, used: BTreeMap::new(), failures: Vec::new() }));
        Self::stream(st, root.to_string(), frame_domain(domain))
    }

    fn stream(st: Arc<Mutex<ReplayState>>, stream: String, pending: Vec<u8>) -> Self {
        st.lock().unwrap().used.entry(stream.clone()).or_insert(0);
        Self { stream, st, pending, forks: AtomicU32::new(0), rounds: 0 }
    }

    fn fail(&self, why: String) {
        self.st.lock().unwrap().failures.push(why);
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let content = std::mem::take(&mut self.pending);
        let k = self.rounds as usize;
        self.rounds += 1;
        let mut st = self.st.lock().unwrap();
        *st.used.get_mut(&self.stream).unwrap() = k + 1;
        let rec = st.record.get(&self.stream).and_then(|s| s.rounds.get(k)).cloned();
        match rec {
            None => {
                st.failures.push(format!("R2: {} round {k}: the verifier issued no such coin", self.stream));
                vec![F128::default(); n]
            }
            Some(r) => {
                if r.n as usize != n {
                    st.failures.push(format!("R2: {} round {k}: {n} coins requested, {} issued", self.stream, r.n));
                } else if r.sha != sha256(&content) {
                    st.failures.push(format!(
                        "R2: {} round {k}: the proof's messages differ from those committed before the coin",
                        self.stream
                    ));
                }
                let mut c = r.coins;
                c.resize(n, F128::default());
                c
            }
        }
    }

    pub fn first_failure(&self) -> Option<String> {
        self.st.lock().unwrap().failures.first().cloned()
    }

    /// After the Flock verifier returns: no replay failure, and every recorded round of this rep's streams consumed.
    pub fn finish(&self) -> Result<(), String> {
        let st = self.st.lock().unwrap();
        if let Some(f) = st.failures.first() {
            return Err(f.clone());
        }
        let prefix = format!("{}/", self.stream);
        for (name, rec) in st.record.iter().filter(|(n, _)| **n == self.stream || n.starts_with(&prefix)) {
            let used = st.used.get(name).copied().unwrap_or(0);
            if used != rec.rounds.len() {
                return Err(format!("R2: {name}: {} rounds recorded, the proof consumed {used}", rec.rounds.len()));
            }
        }
        Ok(())
    }
}

impl Challenger for ReplayChallenger {
    framed_challenger!();

    fn grind_pow(&mut self, _bits: u32) -> u64 {
        unreachable!("the verifier's challenger never grinds")
    }

    /// R4: the nonce is a message like any other (bound by the round digest); the coin does not depend on it.
    fn verify_pow(&mut self, nonce: u64, _bits: u32) -> bool {
        self.observe_bytes(&nonce.to_le_bytes());
        true
    }

    fn fork(&mut self, label: &'static [u8]) -> Self {
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }

    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        let idx = self.forks.fetch_add(1, Ordering::SeqCst);
        let name = child_stream(&self.stream, idx);
        {
            let st = self.st.lock().unwrap();
            match st.record.get(&name) {
                None => {
                    drop(st);
                    self.fail(format!("R3: fork {name}: no live child stream in the session"));
                }
                Some(r) if r.parent != self.stream || r.label != label || r.at != self.rounds => {
                    drop(st);
                    self.fail(format!("R3: fork {name}: opened at a different parent position or label"));
                }
                _ => {}
            }
        }
        let mut c = ReplayChallenger::stream(self.st.clone(), name, frame_domain(label));
        c.observe_f128_slice(&seed);
        c
    }
}
