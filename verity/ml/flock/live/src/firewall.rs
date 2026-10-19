//! The firewall's send check on the outer `--zk` prover (PROTOCOL.md §10.1): every live round the device frames is held to
//! the statement's skeleton in the algebraic phase, and byte for byte to an opening the firewall computes itself (the shadow,
//! in lock-step on the same coins) in the opening phase, before the masking challenger sends any of it. A failed check is a
//! `FIREWALL-REFUSED: <what>` reason, and nothing of the refused round leaves: the rep goes on releasing the statement's
//! schedule in the device's place ([`Checker::scheduled`], the shadow's rounds) and the session stops at its `Finish`. So does
//! a worker that fails ([`run_worker`]: an error, a panic, silence past the deadline): a rep's end never depends on its worker.
//! These checks can only fail a round more: the pass decision of record is the rep's Lean frame's ([`LeanFrame::check`]), on
//! each round they pass, against the statement's scheduled round in its place (`pass_vectors.json` holds the two to each other).

use std::sync::mpsc::{Receiver, Sender, channel};

use flock_field::F128;
use flock_transcript::challenger::Challenger;

use crate::{KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, push_header, push_padded};

pub fn refused(what: impl std::fmt::Display) -> String {
    format!("FIREWALL-REFUSED: {what}")
}

/// One framed item of a round, as the transcript absorbs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Domain(Vec<u8>),
    Label(Vec<u8>),
    Bytes(Vec<u8>),
    /// an observe: its kind and the values (the skeleton's: each value's index among the exposed values)
    Obs { kind: u8, vals: Vec<F128> },
    Squeeze { kind: u8, n: usize },
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = |b: &[u8]| if b.iter().all(|c| c.is_ascii_graphic()) { format!("{:?}", String::from_utf8_lossy(b)) } else { format!("{} bytes", b.len()) };
        match self {
            Item::Domain(b) => write!(f, "domain {}", text(b)),
            Item::Label(b) => write!(f, "label {}", text(b)),
            Item::Bytes(b) => write!(f, "bytes ({})", b.len()),
            Item::Obs { kind, vals } => write!(f, "observe {} of {}", if *kind == KIND_SCALAR { "scalar" } else { "slice" }, vals.len()),
            Item::Squeeze { kind, n } => write!(f, "squeeze {} of {n}", if *kind == KIND_SCALAR { "scalar" } else { "slice" }),
        }
    }
}

/// A round's items (transcript-v2 frames: 16-byte header `[op][kind][0;6][len u64 LE]`, payload zero-padded to 16).
pub fn parse(content: &[u8]) -> Result<Vec<Item>, String> {
    let body = |at: usize, w: usize| content.get(at..at + w).ok_or_else(|| refused("a frame overruns its round"));
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < content.len() {
        let h = body(at, 16)?;
        let (op, kind, len) = (h[0], h[1], u64::from_le_bytes(h[8..].try_into().unwrap()) as usize);
        if h[2..8].iter().any(|b| *b != 0) {
            return Err(refused("a frame header's reserved bytes are not zero"));
        }
        at += 16;
        let padded = len.div_ceil(16) * 16;
        let payload = |at: usize| -> Result<Vec<u8>, String> {
            let p = body(at, padded)?;
            if p[len..].iter().any(|b| *b != 0) {
                return Err(refused("a frame's padding is not zero"));
            }
            Ok(p[..len].to_vec())
        };
        match op {
            OP_DOMAIN | OP_LABEL | OP_BYTES if kind == 0 => {
                let p = payload(at)?;
                out.push(match op { OP_DOMAIN => Item::Domain(p), OP_LABEL => Item::Label(p), _ => Item::Bytes(p) });
                at += padded;
            }
            OP_OBSERVE if kind == KIND_SCALAR || kind == KIND_SLICE => {
                let vals = body(at, 16 * len)?.chunks_exact(16)
                    .map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect();
                out.push(Item::Obs { kind, vals });
                at += 16 * len;
            }
            OP_SQUEEZE if kind == KIND_SCALAR || kind == KIND_SLICE => out.push(Item::Squeeze { kind, n: len }),
            _ => return Err(refused(format!("a frame of op {op} and kind {kind}"))),
        }
    }
    Ok(out)
}

/// The index each skeleton value stands for: [`tagged`]'s values carry it in their low word.
const TAG: u64 = 0x6761_7465_7761_7921;

/// `n` stand-in exposed values, value `i` carrying its index `i`, for [`Recorder`] to read back.
pub fn tagged(n: usize) -> Vec<F128> {
    (0..n as u64).map(|i| F128::new(i, TAG)).collect()
}

/// A challenger that records every call a replay makes, as the device's live challenger frames it (observe kinds, and nonce 0
/// absorbed as bytes), answering distinct coins that are neither 0 nor 1 (the replay divides by `1 + r`).
#[derive(Default)]
pub struct Recorder {
    pub items: Vec<Item>,
    coins: u64,
}

impl Recorder {
    fn coin(&mut self) -> F128 {
        self.coins += 1;
        F128::new(0x5eed_0000 + self.coins, 0x1234_5678_9abc_def0)
    }
}

impl Challenger for Recorder {
    fn observe_label(&mut self, l: &[u8]) {
        self.items.push(Item::Label(l.to_vec()));
    }
    fn observe_f128(&mut self, v: F128) {
        self.items.push(Item::Obs { kind: KIND_SCALAR, vals: vec![v] });
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        self.items.push(Item::Obs { kind: KIND_SLICE, vals: vs.to_vec() });
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        self.items.push(Item::Bytes(b.to_vec()));
    }
    fn sample_f128(&mut self) -> F128 {
        self.items.push(Item::Squeeze { kind: KIND_SCALAR, n: 1 });
        self.coin()
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.items.push(Item::Squeeze { kind: KIND_SLICE, n });
        (0..n).map(|_| self.coin()).collect()
    }
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        self.observe_bytes(&0u64.to_le_bytes());
        0
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the algebraic phase does not fork")
    }
}

/// The algebraic phase's items as the device frames them, from the statement's start: the domain, `bind_statement`'s label,
/// digest and cap, then what `replay` (the verifier's replay of the masked zerocheck and lincheck, run on [`tagged`] values
/// into a [`Recorder`]) absorbs and draws, without the pads' label and root that the masking challenger inserts itself.
pub fn skeleton(domain: &[u8], label_r1cs: &[u8], digest: &[u8], cap: &[u8], label_zc: &[u8], label_pads: &[u8],
                replay: impl FnOnce(&mut Recorder) -> Result<(), String>) -> Result<Vec<Item>, String> {
    let mut r = Recorder::default();
    replay(&mut r)?;
    let at = r.items.iter().position(|x| *x == Item::Label(label_zc.to_vec())).ok_or("the replay absorbs no zerocheck label")?;
    match &r.items[at + 1..at + 3] {
        [Item::Label(l), Item::Bytes(_)] if l == label_pads => {}
        _ => return Err("the replay's pads label and root do not follow the zerocheck label".into()),
    }
    r.items.drain(at + 1..at + 3);
    let mut out = vec![Item::Domain(domain.to_vec()), Item::Label(label_r1cs.to_vec()), Item::Bytes(digest.to_vec()), Item::Bytes(cap.to_vec())];
    out.extend(r.items);
    Ok(out)
}

/// The algebraic phase's check: each round's items against the skeleton, one by one. Labels, bytes and the domain must equal
/// the skeleton's; an observe must have its kind and length (its values are masked downstream and are logged here, at the
/// exposed index the skeleton gives); a squeeze its kind and coin count.
pub struct Checker {
    items: Vec<Item>,
    at: usize,
    /// where the last round began
    start: usize,
    round: usize,
    pub values: Vec<Option<F128>>,
}

impl Checker {
    pub fn new(items: Vec<Item>, n_values: usize) -> Self {
        Checker { items, at: 0, start: 0, round: 0, values: vec![None; n_values] }
    }

    /// Whether every item of the algebraic phase has passed (the next round is the opening's).
    pub fn done(&self) -> bool {
        self.at == self.items.len()
    }

    /// A round, refused unless it is the skeleton's next one; a refused round leaves the checker where it was.
    pub fn round(&mut self, content: &[u8], n: usize) -> Result<(), String> {
        let (at, round) = (self.at, self.round);
        let res = self.round_items(content, n);
        if res.is_err() {
            (self.at, self.round) = (at, round);
        }
        self.start = at;
        res
    }

    /// Undoes the last round, which passed here and was refused downstream: [`Checker::scheduled`] gives it again.
    pub fn rewind(&mut self) {
        self.at = self.start;
        self.round -= 1;
    }

    /// The skeleton's next round as an honest device frames it, every observed value zero, which it logs as the values the
    /// firewall passed: what a failed rep releases in the device's place. None once the algebraic phase has ended.
    pub fn scheduled(&mut self) -> Option<(Vec<u8>, usize)> {
        if self.done() {
            return None;
        }
        (self.start, self.round) = (self.at, self.round + 1);
        let mut out = Vec::new();
        while let Some(it) = self.items.get(self.at) {
            self.at += 1;
            match it {
                Item::Domain(b) | Item::Label(b) | Item::Bytes(b) => {
                    push_header(&mut out, match it { Item::Domain(_) => OP_DOMAIN, Item::Label(_) => OP_LABEL, _ => OP_BYTES }, 0, b.len() as u64);
                    push_padded(&mut out, b);
                }
                Item::Obs { kind, vals } => {
                    push_header(&mut out, OP_OBSERVE, *kind, vals.len() as u64);
                    out.resize(out.len() + 16 * vals.len(), 0);
                    let nv = self.values.len();
                    for i in vals.iter().filter(|i| i.hi == TAG && (i.lo as usize) < nv) {
                        self.values[i.lo as usize] = Some(F128::ZERO);
                    }
                }
                Item::Squeeze { kind, n } => {
                    push_header(&mut out, OP_SQUEEZE, *kind, *n as u64);
                    if *n > 0 {
                        return Some((out, *n));
                    }
                }
            }
        }
        Some((out, 0))
    }

    fn round_items(&mut self, content: &[u8], n: usize) -> Result<(), String> {
        let got = parse(content)?;
        let r = self.round;
        self.round += 1;
        for (k, g) in got.iter().enumerate() {
            let want = self.items.get(self.at).ok_or_else(|| refused(format!("algebraic phase, round {r}: {g} past the phase's end")))?;
            let at = |what: String| refused(format!("algebraic phase, round {r}, frame {k}: {what}"));
            match (g, want) {
                (Item::Obs { kind, vals }, Item::Obs { kind: wk, vals: idx }) if kind == wk && vals.len() == idx.len() => {
                    for (v, i) in vals.iter().zip(idx) {
                        if i.hi != TAG || i.lo as usize >= self.values.len() {
                            return Err(at("the skeleton observes a value that is not an exposed one".into()));
                        }
                        self.values[i.lo as usize] = Some(*v);
                    }
                }
                (Item::Obs { .. }, _) => return Err(at(format!("{g} where the statement has {want}"))),
                (Item::Squeeze { .. }, _) | (Item::Label(_), _) | (Item::Bytes(_), _) | (Item::Domain(_), _) if g != want => {
                    return Err(at(format!("{g} where the statement has {want}")));
                }
                _ => {}
            }
            self.at += 1;
        }
        match got.last() {
            Some(Item::Squeeze { n: k, .. }) if *k == n && n > 0 => Ok(()),
            _ => Err(refused(format!("algebraic phase, round {r}: the round does not end with its squeeze of {n}"))),
        }
    }
}

/// The firewall's end of the shadow opening's lock-step: each of the shadow's rounds, to compare with the device's, and the coins
/// back. The shadow ([`ShadowEnd`] in a [`Framer`]) runs on its own thread.
pub struct ShadowLink {
    rx: Receiver<(Vec<u8>, usize)>,
    tx: Option<Sender<Vec<F128>>>,
    pub round: usize,
    /// the shadow's round a device round was refused against, which leaves in its place
    pub last: Option<(Vec<u8>, usize)>,
}

pub struct ShadowEnd {
    tx: Sender<(Vec<u8>, usize)>,
    rx: Receiver<Vec<F128>>,
}

pub fn shadow_pair() -> (ShadowLink, ShadowEnd) {
    let (tx_r, rx_r) = channel();
    let (tx_c, rx_c) = channel();
    (ShadowLink { rx: rx_r, tx: Some(tx_c), round: 0, last: None }, ShadowEnd { tx: tx_r, rx: rx_c })
}

impl ShadowEnd {
    /// The shadow's round: hand it to the firewall and wait for its coins (the shadow stops when the firewall has gone).
    pub fn round(&mut self, content: &[u8], n: usize) -> Vec<F128> {
        self.tx.send((content.to_vec(), n)).ok().and_then(|_| self.rx.recv().ok())
            .unwrap_or_else(|| panic!("the firewall stopped the shadow opening"))
    }
}

impl ShadowLink {
    /// The shadow's next round, or None once its opening has ended (it then has no firewall to wait on).
    pub fn next(&mut self) -> Option<(Vec<u8>, usize)> {
        let x = self.rx.recv().ok()?;
        self.round += 1;
        Some(x)
    }

    /// The device's opening round against the shadow's next one: refused unless the two are byte for byte the same, the
    /// shadow's round then kept in `last`. Returns the shadow's round, the scheduled one the Lean frame's pass decision reads.
    pub fn check(&mut self, content: &[u8], n: usize) -> Result<Vec<u8>, String> {
        let r = self.round;
        self.round += 1;
        let (want, k) = self.rx.recv().map_err(|_| refused(format!("opening round {r}: the firewall's opening ended before the device's")))?;
        #[cfg(feature = "sha512")]
        crate::firewall_transcript::with(|t| t.shadow_round(&want, k));
        if want == content && k == n {
            return Ok(want);
        }
        let why = Self::differ(r, content, n, &want, k);
        self.last = Some((want, k));
        Err(why)
    }

    /// Why the device's opening round `r` is not the shadow's.
    fn differ(r: usize, content: &[u8], n: usize, want: &[u8], k: usize) -> String {
        let (g, w) = match (parse(content), parse(want)) {
            (Ok(g), Ok(w)) => (g, w),
            (Err(e), _) => return e,
            (_, Err(e)) => return format!("the shadow's round: {e}"),
        };
        let show = |x: Option<&Item>| x.map_or("nothing".into(), |x| x.to_string());
        match (0..g.len().max(w.len())).find(|&i| g.get(i) != w.get(i)) {
            Some(i) if show(g.get(i)) == show(w.get(i)) => {
                refused(format!("opening round {r}, frame {i}: the device's {} differs from the firewall's in its payload", show(g.get(i))))
            }
            Some(i) => refused(format!("opening round {r}, frame {i}: the device's {} is not the firewall's {}", show(g.get(i)), show(w.get(i)))),
            None => refused(format!("opening round {r}: the device squeezes {n} coins, the firewall {k}")),
        }
    }

    pub fn answer(&mut self, coins: &[F128]) {
        if let Some(t) = &self.tx {
            let _ = t.send(coins.to_vec());
        }
    }

    /// Stop the shadow: its next round finds no firewall.
    pub fn close(&mut self) {
        self.tx = None;
    }
}

/// What a firewalled rep's worker (the device prover, on a thread of its own: [`run_worker`]) asks the firewall, which serves it
/// on the session's thread.
pub enum Ask<T> {
    /// a round at its squeeze, and where its coins go
    Round(Vec<u8>, usize, Sender<Vec<F128>>),
    /// the level-0 root at the session's `Commit`, and where the extra claims' points go
    Hook(Vec<u8>, Sender<Vec<Vec<F128>>>),
    /// the deadline passed: a silent worker's control sends it as it falls silent, the injected clock's tick, so no test reads
    /// the wall clock
    Tick,
    Done(Result<T, String>),
}

/// The worker's end of [`run_worker`]: each call waits for the firewall's answer. Once the firewall has given the worker up
/// (it failed, or fell silent past the deadline), every call panics and unwinds the worker: nothing it does reaches the stream.
pub struct WorkerEnd<T>(Sender<Ask<T>>);

impl<T> WorkerEnd<T> {
    pub fn round(&self, content: &[u8], n: usize) -> Vec<F128> {
        let (tx, rx) = channel();
        let _ = self.0.send(Ask::Round(content.to_vec(), n, tx));
        rx.recv().unwrap_or_else(|_| panic!("the firewall gave this worker up"))
    }

    pub fn hook(&self, root: &[u8]) -> Vec<Vec<F128>> {
        let (tx, rx) = channel();
        let _ = self.0.send(Ask::Hook(root.to_vec(), tx));
        rx.recv().unwrap_or_else(|_| panic!("the firewall gave this worker up"))
    }

    pub fn tick(&self) {
        let _ = self.0.send(Ask::Tick);
    }
}

/// The wall clock's wait for a worker's next ask: `secs` an ask, `None` once they pass.
pub fn deadline<T>(secs: f64) -> impl FnMut(&Receiver<Ask<T>>) -> Option<Ask<T>> {
    let d = std::time::Duration::from_secs_f64(secs);
    move |rx| rx.recv_timeout(d).ok()
}

/// Runs a firewalled rep's worker on a thread of its own and serves its rounds and its hook here, on the session's thread,
/// until it returns. Every way the worker can fail is `Err`, which the firewall answers by releasing the rest of the schedule
/// itself: an error it returns (a nonzero return code, output that does not decode), a panic, or silence past the deadline
/// (`wait` gives `None`, or the worker's control ticks). A worker given up may still run; whatever it asks later finds no
/// firewall. A stop of the firewall's own (a coin refusal, the verifier's) panics in `round` or `hook`, on this thread.
pub fn run_worker<T: Send + 'static>(work: impl FnOnce(WorkerEnd<T>) -> Result<T, String> + Send + 'static,
                                     round: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, hook: &mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>,
                                     wait: &mut dyn FnMut(&Receiver<Ask<T>>) -> Option<Ask<T>>) -> Result<T, String> {
    let (tx, rx) = channel();
    let end = WorkerEnd(tx.clone());
    // the prover's sequential part runs here, as it did on the main thread: a stack as large as the verifier's threads'
    std::thread::Builder::new().name("flock-worker".into()).stack_size(64 << 20).spawn(move || {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(end))).unwrap_or_else(|e| {
            Err(format!("panicked: {}", e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()))
        });
        let _ = tx.send(Ask::Done(r));
    }).map_err(|e| format!("its thread did not start: {e}"))?;
    loop {
        match wait(&rx) {
            Some(Ask::Round(c, n, back)) => {
                let _ = back.send(round(&c, n));
            }
            Some(Ask::Hook(root, back)) => {
                let _ = back.send(hook(&root));
            }
            Some(Ask::Done(r)) => return r,
            Some(Ask::Tick) | None => return Err("silent past the firewall's deadline".into()),
        }
    }
}

/// Frames every call as the device prover's challenger does (`FsChallenger` under its live hook) and hands each round to
/// `rounds` at its squeeze: M1's CPU prover takes the device path's host side through it (`FC_ZK_FRAMED`), and the firewall's
/// shadow opening frames its rounds through it for the comparison.
pub struct Framer<'a> {
    pub pending: Vec<u8>,
    pub rounds: &'a mut (dyn FnMut(&[u8], usize) -> Vec<F128> + Send),
}

impl Framer<'_> {
    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let c = std::mem::take(&mut self.pending);
        (self.rounds)(&c, n)
    }
}

impl Challenger for Framer<'_> {
    fn observe_label(&mut self, l: &[u8]) {
        push_header(&mut self.pending, OP_LABEL, 0, l.len() as u64);
        push_padded(&mut self.pending, l);
    }
    fn observe_f128(&mut self, v: F128) {
        push_header(&mut self.pending, OP_OBSERVE, KIND_SCALAR, 1);
        self.pending.extend_from_slice(&v.lo.to_le_bytes());
        self.pending.extend_from_slice(&v.hi.to_le_bytes());
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        push_header(&mut self.pending, OP_OBSERVE, KIND_SLICE, vs.len() as u64);
        for v in vs {
            self.pending.extend_from_slice(&v.lo.to_le_bytes());
            self.pending.extend_from_slice(&v.hi.to_le_bytes());
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
    /// the device's live grinding: nonce 0, absorbed like any message
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        self.observe_bytes(&0u64.to_le_bytes());
        0
    }
    fn verify_pow(&mut self, _: u64, _: u32) -> bool {
        unreachable!("the prover's challenger never verifies")
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the single-table prove path does not fork")
    }
}

/// The outer firewall's release (PROTOCOL.md §10.1): one rep's `flock-firewall frame` process (`Flock.Firewall.Frame` in the
/// Lean verifier), which draws the rep's prover randomness from the OS and commits it, computes each round that leaves from
/// the device's unmasked round the firewall passed, and decides the rep's refusals from the coins the stream answered.
#[cfg(feature = "sha512")]
pub struct LeanFrame {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

/// What a Lean frame drew at its start (`Flock.Firewall.Frame.Draws`): the rep's pads, mask, paddings and `τ`'s salt, the
/// pads tree's leaf salts, and the root it computed over them.
#[cfg(feature = "sha512")]
pub struct Drawn {
    pub pads: crate::zk_veil::Pads,
    pub salts: Vec<u8>,
    pub root: [u8; 64],
}

/// The statement's mask slot as the frame's rank check reads it (`flock-circuit`'s `mask_words`): `n` words from word `w0`
/// of each of `blocks` blocks.
#[derive(Clone, Copy, Debug)]
pub struct MaskSlot {
    pub w0: usize,
    pub n: usize,
    pub blocks: usize,
}

#[cfg(feature = "sha512")]
fn f128_wire(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat()).collect()
}

/// The `flock-firewall` the firewall releases through: `FC_FIREWALL_LEAN`, else the Lean verifier's build in this checkout.
#[cfg(feature = "sha512")]
pub fn lean_frame_exe() -> std::path::PathBuf {
    std::env::var_os("FC_FIREWALL_LEAN").map(Into::into)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../core/service/flock/.lake/build/bin/flock-firewall"))
}

/// Makes the Lean server's `flock-firewall tap` (`Flock.Firewall.Frame.tapCmd`) the OS source of every draw of this process
/// that is not a rep's own (`zk_hooks::set_os_source`): the mask slot's words, level 0's hiding, every tree's salts but the
/// pads', the coin tree's nonce. Set it before the first draw. A tap that fails later panics, as an OS generator that fails
/// does.
#[cfg(feature = "sha512")]
pub fn lean_tap(exe: &std::path::Path) -> Result<(), String> {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};
    const MAX: usize = 1 << 24;
    let mut child = Command::new(exe).arg("tap").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
        .map_err(|e| refused(format!("the Lean tap {} does not start ({e}): `lake build flock-firewall` in verity/core/service/flock, or FC_FIREWALL_LEAN", exe.display())))?;
    let (mut stdin, mut stdout) = (child.stdin.take().expect("piped stdin"), child.stdout.take().expect("piped stdout"));
    let mut read = move |out: &mut [u8]| -> std::io::Result<()> {
        for p in out.chunks_mut(MAX) {
            writeln!(stdin, "{}", p.len())?;
            stdin.flush()?;
            stdout.read_exact(p)?;
        }
        Ok(())
    };
    read(&mut [0u8; 32]).map_err(|e| refused(format!("the Lean tap {} does not answer ({e})", exe.display())))?;
    crate::zk_hooks::set_os_source(Box::new(move |out: &mut [u8]| {
        read(out).unwrap_or_else(|e| panic!("the Lean tap failed ({e}): refusing to prove without prover randomness"))
    })).map_err(refused)
}

#[cfg(feature = "sha512")]
impl LeanFrame {
    /// Starts a rep on layout `lay` and the statement's mask slot: the frame draws the rep's randomness and commits it, and
    /// returns its draws, which the rep proves with.
    pub fn start(exe: &std::path::Path, lay: &crate::zk_veil::Layout, mask: MaskSlot) -> Result<(Self, Drawn), String> {
        use crate::zk_veil::{Code, K_PAD, TAU_SALT_BYTES};
        use std::io::{Read, Write};
        use std::process::{Command, Stdio};
        let mut child = Command::new(exe).arg("frame").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
            .map_err(|e| refused(format!("the Lean frame {} does not start ({e}): `lake build flock-firewall` in verity/core/service/flock, or FC_FIREWALL_LEAN", exe.display())))?;
        let (stdin, stdout) = (child.stdin.take().expect("piped stdin"), std::io::BufReader::new(child.stdout.take().expect("piped stdout")));
        let mut f = LeanFrame { child, stdin, stdout };
        let start = serde_json::json!({"layout": {"n_mlv": lay.n_mlv, "n_lc": lay.n_lc, "n_ood0": lay.n_ood0, "k0": lay.k0},
                                       "mask": {"w0": mask.w0, "n": mask.n, "blocks": mask.blocks}});
        writeln!(f.stdin, "{start}").and_then(|_| f.stdin.flush()).map_err(|e| refused(format!("the Lean frame's start: {e}")))?;
        let len = f.reply()?.ok_or_else(|| refused("the Lean frame's start answered without its draws"))?;
        let (s, leaves) = (lay.len(), Code::new(lay.len() + K_PAD).n);
        let tau = 32 * s + 32 * K_PAD;
        let want = tau + TAU_SALT_BYTES + crate::zk_hooks::SALT_BYTES * leaves + 64;
        if len != want {
            return Err(refused(format!("the Lean frame drew {len} bytes where the layout takes {want}")));
        }
        let mut b = vec![0u8; len];
        f.stdout.read_exact(&mut b).map_err(|e| refused(format!("the Lean frame's draws: {e}")))?;
        let words = |o: usize, n: usize| -> Vec<F128> {
            (0..n).map(|i| F128::new(u64::from_le_bytes(b[o + 16 * i..o + 16 * i + 8].try_into().unwrap()),
                                     u64::from_le_bytes(b[o + 16 * i + 8..o + 16 * i + 16].try_into().unwrap()))).collect()
        };
        let pads = crate::zk_veil::Pads { h: words(0, s), mu: words(16 * s, s), pad_h: words(32 * s, K_PAD), pad_mu: words(32 * s + 16 * K_PAD, K_PAD),
                                          tau_salt: b[tau..tau + TAU_SALT_BYTES].to_vec() };
        let drawn = Drawn { pads, salts: b[tau + TAU_SALT_BYTES..len - 64].to_vec(), root: b[len - 64..].try_into().unwrap() };
        Ok((f, drawn))
    }

    fn line(&mut self) -> Result<String, String> {
        use std::io::BufRead;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| refused(format!("the Lean frame's answer: {e}")))?;
        Ok(line.trim_end().to_string())
    }

    fn reply(&mut self) -> Result<Option<usize>, String> {
        let line = self.line()?;
        Self::parse(&line)
    }

    fn parse(line: &str) -> Result<Option<usize>, String> {
        if line == "ok" {
            return Ok(None);
        }
        if let Some(n) = line.strip_prefix("ok ") {
            return n.parse().map(Some).map_err(|_| refused(format!("the Lean frame's answer {line:?}")));
        }
        Err(refused(format!("the Lean frame: {}", line.strip_prefix("err ").unwrap_or(if line.is_empty() { "no answer" } else { line }))))
    }

    /// The table's earlier rep's algebraic-phase coins, before this rep's first round: the rank check covers its claim
    /// points too.
    pub fn prior(&mut self, coins: &[F128]) -> Result<(), String> {
        use std::io::Write;
        write!(self.stdin, "prior {}\n", coins.len()).and_then(|_| self.stdin.write_all(&f128_wire(coins))).and_then(|_| self.stdin.flush())
            .map_err(|e| refused(format!("the Lean frame's prior coins: {e}")))?;
        self.reply().map(|_| ())
    }

    /// The round that leaves for the round `content` (which squeezes `n` coins), or the Lean frame's refusal, after which the
    /// rep is stopped. `fail`: `content` is the statement's scheduled round in place of a device round that never passed the
    /// firewall's own checks (a worker that failed, or a round Rust's checks fail), and the rep stops before it; without it,
    /// `content` is one of the inner proof's rounds, the firewall's own. A device round comes with the scheduled round in its
    /// place ([`LeanFrame::check`]). A round the frame's refusals stop (`stop WHY`: the rank check) panics with its reason,
    /// before anything of it leaves.
    pub fn round(&mut self, content: &[u8], n: usize, fail: bool) -> Result<Vec<u8>, String> {
        use std::io::{Read, Write};
        write!(self.stdin, "round {n} {}{}\n", content.len(), if fail { " fail" } else { "" })
            .and_then(|_| self.stdin.write_all(content)).and_then(|_| self.stdin.flush())
            .map_err(|e| refused(format!("the Lean frame's round: {e}")))?;
        let line = self.line()?;
        if let Some(why) = line.strip_prefix("stop ") {
            panic!("{why}");
        }
        let len = Self::parse(&line)?.ok_or_else(|| refused("the Lean frame answered a round without its length"))?;
        let mut out = vec![0u8; len];
        self.stdout.read_exact(&mut out).map_err(|e| refused(format!("the Lean frame's round: {e}")))?;
        Ok(out)
    }

    /// The pass decision (`Flock.Firewall.Frame.check`) on the device round `content`, which Rust's checks passed, against the
    /// statement's scheduled round `sched` in its place (which squeezes `n` coins): the round that leaves, and why the frame
    /// failed the device round, if it did, the rep then stopped and the scheduled round leaving in its place. The frame's
    /// refusal of the scheduled round itself is an error, and a round its refusals stop panics with its reason, as in
    /// [`LeanFrame::round`].
    pub fn check(&mut self, content: &[u8], sched: &[u8], n: usize) -> Result<(Vec<u8>, Option<String>), String> {
        use std::io::{Read, Write};
        write!(self.stdin, "round {n} {} {}\n", content.len(), sched.len())
            .and_then(|_| self.stdin.write_all(content)).and_then(|_| self.stdin.write_all(sched)).and_then(|_| self.stdin.flush())
            .map_err(|e| refused(format!("the Lean frame's round: {e}")))?;
        let line = self.line()?;
        if let Some(why) = line.strip_prefix("stop ") {
            panic!("{why}");
        }
        let (len, why) = match line.strip_prefix("ok ").map(|r| r.split_once(' ').unwrap_or((r, ""))) {
            Some((len, "pass")) => (len, None),
            Some((len, v)) if v.starts_with("fail ") => (len, Some(v["fail ".len()..].to_string())),
            _ => {
                Self::parse(&line)?;
                return Err(refused(format!("the Lean frame's answer {line:?} to a device round")));
            }
        };
        let len = len.parse().map_err(|_| refused(format!("the Lean frame's answer {line:?}")))?;
        let mut out = vec![0u8; len];
        self.stdout.read_exact(&mut out).map_err(|e| refused(format!("the Lean frame's round: {e}")))?;
        Ok((out, why))
    }

    /// The coins the stream answered the round last released: `Ok(Some(why))` when the frame's refusals stop the rep on them
    /// (a zerocheck outer coin `r_j = 1`, level 0's `ρ_0, ρ_1` dependent over F128, the inner proof's `β = 0`).
    pub fn coins(&mut self, coins: &[F128]) -> Result<Option<String>, String> {
        use std::io::Write;
        write!(self.stdin, "coins {}\n", coins.len()).and_then(|_| self.stdin.write_all(&f128_wire(coins))).and_then(|_| self.stdin.flush())
            .map_err(|e| refused(format!("the Lean frame's coins: {e}")))?;
        let line = self.line()?;
        if let Some(why) = line.strip_prefix("stop ") {
            return Ok(Some(why.to_string()));
        }
        Self::parse(&line).map(|_| None)
    }

    /// Whether the rep ended with every masked phase closed, every pad of the layout used and the inner proof's rounds
    /// released, and if so whether it is stopped.
    pub fn finish(&mut self) -> Result<bool, String> {
        use std::io::{BufRead, Write};
        writeln!(self.stdin, "finish").and_then(|_| self.stdin.flush()).map_err(|e| refused(format!("the Lean frame's finish: {e}")))?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| refused(format!("the Lean frame's finish: {e}")))?;
        match line.trim_end() {
            "ok" => Ok(false),
            "stop" => Ok(true),
            l => Err(refused(format!("the Lean frame: {}", l.strip_prefix("err ").unwrap_or(if l.is_empty() { "no answer" } else { l })))),
        }
    }
}

#[cfg(feature = "sha512")]
impl Drop for LeanFrame {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The masking challenger's inner challenger under the firewall. While a round's `release` is set (the Lean frame's round for
/// the device round the firewall passed), it frames what the masking challenger hands it, as the device's challenger frames,
/// and at the round's squeeze holds that (the speed reference) to the release byte for byte, then hands the release, and
/// only the release, to the stream. With no release set it passes every call through: unfirewalled reps. A firewalled rep's
/// inner proof, the firewall's own, leaves through the Lean frame too (`flock-circuit`'s `zk_finish`).
pub struct Gate<C: Challenger> {
    pub inner: C,
    pub release: Option<Vec<u8>>,
    pending: Vec<u8>,
}

impl<C: Challenger> Gate<C> {
    pub fn new(inner: C) -> Self {
        Gate { inner, release: None, pending: Vec::new() }
    }

    fn held(&self) -> bool {
        self.release.is_some()
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 && kind == KIND_SLICE {
            return Vec::new();
        }
        let release = self.release.take().expect("a held round");
        let mine = std::mem::take(&mut self.pending);
        if mine != release {
            let at = mine.iter().zip(&release).position(|(a, b)| a != b).unwrap_or(mine.len().min(release.len()));
            panic!("{}", refused(format!("the masking challenger's round ({} bytes) and the Lean frame's ({} bytes) differ from byte {at}",
                                         mine.len(), release.len())));
        }
        crate::replay_round(&mut self.inner, &release, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"))
    }
}

impl<C: Challenger> Challenger for Gate<C> {
    fn observe_label(&mut self, l: &[u8]) {
        if !self.held() {
            return self.inner.observe_label(l);
        }
        push_header(&mut self.pending, OP_LABEL, 0, l.len() as u64);
        push_padded(&mut self.pending, l);
    }
    fn observe_f128(&mut self, v: F128) {
        if !self.held() {
            return self.inner.observe_f128(v);
        }
        push_header(&mut self.pending, OP_OBSERVE, KIND_SCALAR, 1);
        self.pending.extend_from_slice(&v.lo.to_le_bytes());
        self.pending.extend_from_slice(&v.hi.to_le_bytes());
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        if !self.held() {
            return self.inner.observe_f128_slice(vs);
        }
        push_header(&mut self.pending, OP_OBSERVE, KIND_SLICE, vs.len() as u64);
        for v in vs {
            self.pending.extend_from_slice(&v.lo.to_le_bytes());
            self.pending.extend_from_slice(&v.hi.to_le_bytes());
        }
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        if !self.held() {
            return self.inner.observe_bytes(b);
        }
        push_header(&mut self.pending, OP_BYTES, 0, b.len() as u64);
        push_padded(&mut self.pending, b);
    }
    fn sample_f128(&mut self) -> F128 {
        if !self.held() {
            return self.inner.sample_f128();
        }
        self.squeeze(KIND_SCALAR, 1)[0]
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        if !self.held() {
            return self.inner.sample_f128_vec(n);
        }
        self.squeeze(KIND_SLICE, n)
    }
    fn grind_pow(&mut self, bits: u32) -> u64 {
        assert!(!self.held(), "a held round grinds: the device frames its nonce as a message");
        self.inner.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.inner.verify_pow(nonce, bits)
    }
    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        assert!(!self.held(), "a fork inside a held round");
        Gate::new(self.inner.fork_from_seed(seed, label))
    }
}

/// This process's CPU seconds so far (user and system, every thread), from `/proc/self/stat` in clock ticks of 1/100 s (Linux's
/// `USER_HZ`); 0 where there is no procfs.
pub fn cpu_seconds() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else { return 0.0 };
    let f: Vec<&str> = s.rsplit_once(')').map_or("", |x| x.1).split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn framed(f: impl FnOnce(&mut Framer<'_>)) -> Vec<(Vec<u8>, usize)> {
        let mut out = Vec::new();
        let mut rounds = |c: &[u8], n: usize| { out.push((c.to_vec(), n)); vec![F128::new(7, 7); n] };
        let mut fr = Framer { pending: crate::frame_domain(b"dom"), rounds: &mut rounds };
        f(&mut fr);
        out
    }

    fn phase(ch: &mut impl Challenger, vals: &[F128]) {
        ch.observe_label(b"flock-r1cs-v0");
        ch.observe_bytes(&[1; 64]);
        ch.observe_bytes(&[2; 64]);
        ch.observe_label(b"zc");
        ch.sample_f128_vec(3);
        ch.observe_f128_slice(&vals[..2]);
        ch.observe_f128(vals[2]);
        ch.sample_f128();
    }

    fn checker() -> Checker {
        let mut r = Recorder::default();
        phase(&mut r, &tagged(3));
        let mut items = vec![Item::Domain(b"dom".to_vec())];
        items.extend(r.items);
        Checker::new(items, 3)
    }

    #[test]
    fn checker_logs_values_and_refuses_another_skeleton() {
        let v = [F128::new(10, 0), F128::new(11, 0), F128::new(12, 0)];
        let rounds = framed(|f| phase(f, &v));
        let mut c = checker();
        for (r, n) in &rounds {
            c.round(r, *n).unwrap();
        }
        assert!(c.done());
        assert_eq!(c.values, v.map(Some).to_vec());
        let bad = framed(|f| {
            f.observe_label(b"flock-r1cs-v1");
            f.sample_f128();
        });
        let e = checker().round(&bad[0].0, bad[0].1).unwrap_err();
        assert!(e.starts_with("FIREWALL-REFUSED: algebraic phase, round 0, frame 1: label"), "{e}");
        let short = framed(|f| {
            f.observe_label(b"flock-r1cs-v0");
            f.observe_bytes(&[1; 64]);
            f.observe_bytes(&[2; 64]);
            f.observe_label(b"zc");
            f.sample_f128_vec(2);
        });
        let e = checker().round(&short[0].0, short[0].1).unwrap_err();
        assert!(e.contains("squeeze slice of 2 where the statement has squeeze slice of 3"), "{e}");
    }

    #[test]
    fn a_failed_rep_goes_on_with_the_skeletons_rounds_as_an_honest_device_frames_them_with_zero_values() {
        let v = [F128::new(10, 0), F128::new(11, 0), F128::new(12, 0)];
        let honest = framed(|f| phase(f, &v));
        let zero = framed(|f| phase(f, &[F128::ZERO; 3]));
        assert_ne!(honest[1], zero[1]);
        let mut c = checker();
        let mut bad = honest[0].0.clone();
        bad[48] ^= 1; // the first label's payload, after the domain frame's 32 bytes and the label's header
        assert!(c.round(&bad, honest[0].1).is_err());
        assert_eq!(c.scheduled(), Some(zero[0].clone()), "a refused round leaves the checker at its start");
        assert_eq!(c.scheduled(), Some(zero[1].clone()));
        assert_eq!((c.scheduled(), c.done()), (None, true));
        assert_eq!(c.values, vec![Some(F128::ZERO); 3], "the values passed are the scheduled ones");
        let mut c = checker();
        c.round(&honest[0].0, honest[0].1).unwrap();
        c.round(&honest[1].0, honest[1].1).unwrap();
        assert_eq!(c.values, v.map(Some).to_vec());
        c.rewind();
        assert_eq!(c.scheduled(), Some(zero[1].clone()), "a round refused downstream is scheduled again");
        assert_eq!(c.values, vec![Some(F128::ZERO); 3]);
    }

    /// A round as the masking challenger hands it on: two observes and a zero-length squeeze inside, then its coin.
    fn masked_round(ch: &mut impl Challenger, v: u64) -> F128 {
        ch.observe_label(b"zc");
        ch.observe_f128_slice(&[F128::new(v, 1), F128::new(v + 1, 2)]);
        ch.sample_f128_vec(0);
        ch.observe_bytes(&[9; 20]);
        ch.observe_f128(F128::new(v + 2, 3));
        ch.sample_f128()
    }

    #[test]
    fn gate_passes_through_without_a_release_and_sends_only_the_release_with_one() {
        let mut direct = Recorder::default();
        masked_round(&mut direct, 10);
        let mut g = Gate::new(Recorder::default());
        masked_round(&mut g, 10);
        assert_eq!(g.inner.items, direct.items, "no release: every call passes through");
        let release = framed(|f| { f.pending.clear(); masked_round(f, 10); }).remove(0).0;
        let mut g = Gate::new(Recorder::default());
        g.release = Some(release);
        let coin = masked_round(&mut g, 10);
        assert_eq!((g.inner.items, coin), (direct.items, F128::new(0x5eed_0001, 0x1234_5678_9abc_def0)), "the release reaches the stream");
        assert!(g.release.is_none(), "a release is used once");
    }

    #[test]
    fn gate_refuses_a_masking_round_the_release_differs_from() {
        let release = framed(|f| { f.pending.clear(); masked_round(f, 10); }).remove(0).0;
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut g = Gate::new(Recorder::default());
            g.release = Some(release.clone());
            masked_round(&mut g, 11);
            g
        })).map(|_| ()).unwrap_err();
        let e = e.downcast_ref::<String>().expect("a refusal message");
        assert!(e.starts_with("FIREWALL-REFUSED: the masking challenger's round") && e.ends_with("differ from byte 48"), "{e}");
    }

    #[test]
    fn shadow_link_names_the_first_differing_frame() {
        let (mut link, mut end) = shadow_pair();
        let t = std::thread::spawn(move || {
            let mut rounds = |c: &[u8], n: usize| end.round(c, n);
            let mut fr = Framer { pending: Vec::new(), rounds: &mut rounds };
            fr.observe_bytes(&[3; 64]);
            fr.sample_f128();
            fr.observe_f128(F128::new(5, 0));
            fr.sample_f128();
        });
        let dev = framed(|f| {
            f.pending.clear();
            f.observe_bytes(&[3; 64]);
            f.sample_f128();
            f.observe_f128(F128::new(6, 0));
            f.sample_f128();
        });
        let shadow = framed(|f| {
            f.pending.clear();
            f.observe_bytes(&[3; 64]);
            f.sample_f128();
            f.observe_f128(F128::new(5, 0));
            f.sample_f128();
        });
        assert_eq!(link.check(&dev[0].0, dev[0].1).unwrap(), shadow[0].0, "a round that passes is the shadow's, the scheduled one");
        assert_eq!(link.last, None);
        link.answer(&[F128::ONE]);
        let e = link.check(&dev[1].0, dev[1].1).unwrap_err();
        assert_eq!(e, "FIREWALL-REFUSED: opening round 1, frame 0: the device's observe scalar of 1 differs from the firewall's in its payload");
        assert_eq!(link.last, Some(shadow[1].clone()), "the shadow's round leaves in the refused one's place");
        link.answer(&[F128::ONE]);
        assert_eq!(link.next(), None, "the shadow's opening has ended");
        assert!(t.join().is_ok());
        let (mut link, mut end) = shadow_pair();
        let t = std::thread::spawn(move || {
            let mut rounds = |c: &[u8], n: usize| end.round(c, n);
            let mut fr = Framer { pending: Vec::new(), rounds: &mut rounds };
            fr.sample_f128();
        });
        assert!(link.next().is_some());
        link.close();
        assert!(t.join().is_err(), "the shadow stops when the firewall closes");
    }

    /// The firewall's side of [`run_worker`] for a test: every ask served in order into `log`, round `k`'s coins `[k]`.
    fn serve<T: Send + 'static>(work: impl FnOnce(WorkerEnd<T>) -> Result<T, String> + Send + 'static,
                                wait: &mut dyn FnMut(&Receiver<Ask<T>>) -> Option<Ask<T>>) -> (Result<T, String>, Vec<String>) {
        let log = std::cell::RefCell::new(Vec::new());
        let r = {
            let mut round = |c: &[u8], n: usize| {
                log.borrow_mut().push(format!("round {} {n}", c[0]));
                vec![F128::new(c[0] as u64, 0); n]
            };
            let mut hook = |root: &[u8]| {
                log.borrow_mut().push(format!("hook {}", root[0]));
                vec![vec![F128::ONE]]
            };
            run_worker(work, &mut round, &mut hook, wait)
        };
        (r, log.into_inner())
    }

    fn served<T>(rx: &Receiver<Ask<T>>) -> Option<Ask<T>> {
        rx.recv().ok()
    }

    #[test]
    fn a_worker_is_served_on_the_firewalls_thread_until_it_returns() {
        let (r, log) = serve(|end| {
            assert_eq!(end.hook(&[9]), vec![vec![F128::ONE]]);
            let coins: Vec<F128> = (0..3u8).flat_map(|k| end.round(&[k], 2)).collect();
            Ok(coins)
        }, &mut served);
        assert_eq!(r.unwrap(), [0, 0, 1, 1, 2, 2].map(|k| F128::new(k, 0)).to_vec());
        assert_eq!(log, ["hook 9", "round 0 2", "round 1 2", "round 2 2"]);
    }

    /// Each way a worker fails is the firewall's `Err` after the rounds it asked, never a panic on the firewall's thread: an
    /// error it returns, a panic, silence (the injected clock: `wait` says the deadline passed at the third ask), and its control's
    /// tick. A worker given up gets nothing more: its next round panics on its own thread.
    #[test]
    fn every_worker_failure_is_an_error_after_the_rounds_it_asked() {
        let (r, log) = serve(|end| {
            end.round(&[0], 1);
            Err::<(), _>("CUDA circuit prover returned error 3".into())
        }, &mut served);
        assert_eq!((r.unwrap_err().as_str(), log), ("CUDA circuit prover returned error 3", vec!["round 0 1".to_string()]));
        let (r, log) = serve(|end: WorkerEnd<()>| {
            end.round(&[0], 1);
            end.round(&[1], 1);
            panic!("the device's output does not decode")
        }, &mut served);
        assert_eq!((r.unwrap_err().as_str(), log.len()), ("panicked: the device's output does not decode", 2));
        let (hold, held) = channel::<()>();
        let (late_tx, late) = channel();
        let mut asks = 0;
        let mut clock = |rx: &Receiver<Ask<()>>| {
            asks += 1;
            if asks == 3 { None } else { rx.recv().ok() }
        };
        let (r, log) = serve(move |end| {
            end.hook(&[4]);
            end.round(&[0], 1);
            let _ = held.recv();
            let _ = late_tx.send(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| end.round(&[1], 1))).is_err());
            Ok(())
        }, &mut clock);
        assert_eq!((r.unwrap_err().as_str(), log), ("silent past the firewall's deadline", vec!["hook 4".to_string(), "round 0 1".to_string()]));
        drop(hold);
        assert_eq!(late.recv(), Ok(true), "a worker given up gets no coins");
        let (r, log) = serve(|end| {
            end.round(&[0], 1);
            end.tick();
            Ok(())
        }, &mut served);
        assert_eq!((r.unwrap_err().as_str(), log.len()), ("silent past the firewall's deadline", 1));
    }

    /// `pass_vectors.json`, which `test_firewall_frame.py` holds the Lean frame's pass decision (`Flock.Firewall.Frame.check`)
    /// to: Rust's checks pass exactly the device rounds the frame passes, [`Checker`] in the algebraic phase against the
    /// skeleton the scheduled rounds stand for (each observed value an exposed one), [`ShadowLink::check`] after it against
    /// the scheduled round.
    #[test]
    fn rusts_checks_pass_the_device_rounds_the_lean_frame_passes() {
        let v: serde_json::Value = serde_json::from_str(include_str!("pass_vectors.json")).unwrap();
        let bytes = |s: &serde_json::Value| -> Vec<u8> {
            let s = s.as_str().unwrap();
            (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
        };
        let rep: Vec<(Vec<u8>, usize)> = v["rep"].as_array().unwrap().iter().zip(v["coins"].as_array().unwrap())
            .map(|(r, n)| (bytes(r), n.as_u64().unwrap() as usize)).collect();
        let alg = v["algebraic"].as_u64().unwrap() as usize;
        let mut items = Vec::new();
        let mut nv = 0u64;
        for (r, _) in &rep[..alg] {
            for it in parse(r).unwrap() {
                items.push(match it {
                    Item::Obs { kind, vals } => Item::Obs { kind, vals: vals.iter().map(|_| { nv += 1; F128::new(nv - 1, TAG) }).collect() },
                    x => x,
                });
            }
        }
        for c in v["cases"].as_array().unwrap() {
            let (at, device) = (c["at"].as_u64().unwrap() as usize, bytes(&c["device"]));
            let got = if at < alg {
                let mut ch = Checker::new(items.clone(), nv as usize);
                for (r, n) in &rep[..at] {
                    ch.round(r, *n).unwrap();
                }
                ch.round(&device, rep[at].1).is_ok()
            } else {
                let (mut link, end) = shadow_pair();
                end.tx.send(rep[at].clone()).unwrap();
                link.check(&device, rep[at].1).is_ok()
            };
            assert_eq!(got, c["pass"].as_bool().unwrap(), "{}", c["case"]);
        }
    }
}
