//! The ZK server's Rust side on the outer `--zk` prover (PROTOCOL.md §10.1). The Lean session loop (`Flock.Firewall.Session`)
//! owns the auditor's connection and each rep's frame: it places, checks and masks every round that leaves, and a rep it
//! stops releases the statement's rounds in the device's place to the session's `Finish`. Here the device runs as a worker
//! ([`run_worker`]) whose every failure (an error, a panic, silence past the deadline) fails its rep without choosing where
//! the session ends, a rep's frame is the loop's ([`LoopRep`]), and the masking challenger hands the loop what it sends
//! ([`Gate`]). A failed check here is a `FIREWALL-REFUSED: <what>` reason kept for the session's `Finish`.

use std::sync::mpsc::{Receiver, Sender, channel};

use flock_field::F128;
use flock_transcript::challenger::Challenger;

use crate::{KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, push_header, push_padded};

pub fn refused(what: impl std::fmt::Display) -> String {
    format!("FIREWALL-REFUSED: {what}")
}

/// What a firewalled rep's worker (the device prover, on a thread of its own: [`run_worker`]) asks the ZK server, which serves it
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

/// The worker's end of [`run_worker`]: each call waits for the ZK server's answer. Once the ZK server has given the worker up
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
/// until it returns. Every way the worker can fail is `Err`, which the ZK server answers by releasing the rest of the schedule
/// itself: an error it returns (a nonzero return code, output that does not decode), a panic, or silence past the deadline
/// (`wait` gives `None`, or the worker's control ticks). A worker given up may still run; whatever it asks later finds no
/// ZK server. A stop of the ZK server's own (a coin refusal, the verifier's) panics in `round` or `hook`, on this thread.
pub fn run_worker<T: Send + 'static>(work: impl FnOnce(WorkerEnd<T>) -> Result<T, String> + Send + 'static,
                                     round: &mut dyn FnMut(&[u8], usize) -> Vec<F128>, hook: &mut dyn FnMut(&[u8]) -> Vec<Vec<F128>>,
                                     wait: &mut dyn FnMut(&Receiver<Ask<T>>) -> Option<Ask<T>>) -> Result<T, String> {
    let (tx, rx) = channel();
    let end = WorkerEnd(tx.clone());
    // the prover's sequential part runs here, as it did on the main thread: a stack as large as the verifier's threads'
    std::thread::Builder::new().name("flock-worker".into()).stack_size(crate::THREAD_STACK).spawn(move || {
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
/// `rounds` at its squeeze: M1's CPU prover takes the device path's host side through it (`FC_ZK_FRAMED`), as does the
/// firewalled CPU device.
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

/// What the Lean session loop names a rep's next round ([`LoopRep::place`]): the algebraic phase's or the opening's
/// scheduled round, squeezing that many coins, the rep's salts for levels ≥ 1 first (its algebraic phase ended and its
/// opening waits for them: [`LeanFinish::salts`]), or none (the inner proof's rounds, which this relays from the Lean
/// finish and the loop's frame holds to its own, `Frame.innerSched`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Alg(usize),
    Open(usize),
    Salts,
    Own,
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

/// The name the Lean session loop knows a session by (`Level0.nameOf`): the hex of the loop's id for it, the four words
/// the loop answers the session's `Hello` with in place of the auditor's answer (`Session.idFrame`), as the wire carries them.
#[cfg(feature = "sha512")]
pub fn level0_session(id: &[F128]) -> String {
    crate::hex(&f128_wire(id))
}

/// The `flock-firewall` the ZK server releases through: `FC_FIREWALL_LEAN`, else the Lean verifier's build in this checkout.
#[cfg(feature = "sha512")]
pub fn lean_frame_exe() -> std::path::PathBuf {
    std::env::var_os("FC_FIREWALL_LEAN").map(Into::into)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../..").join("verity/protocols/flock/verifier/.lake/build/bin/flock-firewall"))
}

/// The `core-firewall` the ZK server keeps its record with (its release side and its value service): `FC_CORE_FIREWALL`,
/// else the build in this checkout's verity/protocols/recursion/verifier/firewall.
#[cfg(feature = "sha512")]
pub fn core_firewall_exe() -> std::path::PathBuf {
    std::env::var_os("FC_CORE_FIREWALL").map(Into::into)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../..").join("verity/protocols/recursion/verifier/firewall/.lake/build/bin/core-firewall"))
}

/// Makes the Lean server's `flock-firewall tap` (`Flock.Firewall.Frame.tapCmd`) the OS source of every sample of this process
/// that is not a rep's own (`zk_hooks::set_os_source`): level 0's hiding, every tree's salts but the pads'. The mask slot's
/// words are the Lean session loop's ([`lean_level0`]), and so is the coin tree's nonce, its `Hello`'s. Set it before the first sample. A tap that fails later panics, as an OS
/// generator that fails does.
#[cfg(feature = "sha512")]
pub fn lean_tap(exe: &std::path::Path) -> Result<(), String> {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};
    const MAX: usize = 1 << 24;
    let mut child = Command::new(exe).arg("tap").stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
        .map_err(|e| refused(format!("the Lean tap {} does not start ({e}): `lake build flock-firewall` in verity/protocols/flock/verifier, or FC_FIREWALL_LEAN", exe.display())))?;
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

/// One rep's frame in the Lean session loop (`Flock.Firewall.Frames`), which samples the rep's prover randomness from the OS
/// at the statement's start, commits it, and places, checks and masks each of the rep's rounds as the loop relays them
/// ([`LoopRep::start`]). The samples never leave the loop: the rep gets their commitment's root alone, and the loop's finish
/// proves with them. Here the rep only asks the loop where it stands ([`LoopRep::place`], [`LoopRep::vals`]).
#[cfg(feature = "sha512")]
pub struct LoopRep {
    /// the loop's frames address, the session's name ([`level0_session`]), the table and the rep
    addr: String,
    name: String,
    table: usize,
    rep: usize,
}

#[cfg(feature = "sha512")]
impl LoopRep {
    /// Starts table `table`'s rep `rep` of the session the Lean session loop names `name`, at its frames address `addr`
    /// ([`LeanSession::frames`], `rep NAME TABLE REP`), and takes the root of the pads tree over its samples
    /// (`Frame.padsRoot`), the samples themselves never.
    pub fn start(addr: &str, name: &str, table: usize, rep: usize) -> Result<(Self, [u8; 64]), String> {
        use std::io::{BufRead, Read, Write};
        let io = |e: std::io::Error| refused(format!("the Lean session loop's frames at {addr}: {e}"));
        let mut tcp = std::net::TcpStream::connect(addr).map_err(io)?;
        let _ = tcp.set_nodelay(true);
        writeln!(tcp, "rep {name} {table} {rep}").and_then(|_| tcp.flush()).map_err(io)?;
        let mut rd = std::io::BufReader::new(tcp.try_clone().map_err(io)?);
        let mut line = String::new();
        rd.read_line(&mut line).map_err(io)?;
        let line = line.trim_end();
        if line != "ok 64" {
            return Err(refused(format!("the Lean frame: {}", line.strip_prefix("err ").unwrap_or(if line.is_empty() { "no answer" } else { line }))));
        }
        let mut root = [0u8; 64];
        rd.read_exact(&mut root).map_err(|e| refused(format!("the Lean frame's pads root: {e}")))?;
        Ok((LoopRep { addr: addr.to_string(), name: name.to_string(), table, rep }, root))
    }

    /// One request `what` about the rep to the loop's frames port (`Flock.Firewall.Frames.ask`): its answer's line, and the
    /// bytes after an `ok N`.
    fn ask(&self, what: &str) -> Result<(String, Vec<u8>), String> {
        use std::io::{BufRead, Read, Write};
        let LoopRep { addr, name, table, rep } = self;
        let io = |e: std::io::Error| refused(format!("the Lean session loop's {what} of table {table} rep {rep}: {e}"));
        let mut tcp = std::net::TcpStream::connect(addr.as_str()).map_err(io)?;
        let _ = tcp.set_nodelay(true);
        writeln!(tcp, "{what} {name} {table} {rep}").and_then(|_| tcp.flush()).map_err(io)?;
        let mut rd = std::io::BufReader::new(tcp);
        let mut line = String::new();
        rd.read_line(&mut line).map_err(io)?;
        let line = line.trim_end().to_string();
        let mut body = Vec::new();
        if let Some(n) = line.strip_prefix("ok ") {
            body = vec![0u8; n.parse().map_err(|_| refused(format!("the Lean session loop's answer {line:?} to a {what}")))?];
            rd.read_exact(&mut body).map_err(io)?;
        }
        Ok((line, body))
    }

    /// The rep's next round as the loop names it (`place NAME I R`).
    pub fn place(&self) -> Result<Place, String> {
        let (line, _) = self.ask("place")?;
        let n = |s: &str| s.parse::<usize>().map_err(|_| refused(format!("the Lean session loop's place {line:?}")));
        match line.split_once(' ') {
            Some(("alg", k)) => Ok(Place::Alg(n(k)?)),
            Some(("open", k)) => Ok(Place::Open(n(k)?)),
            _ if line == "salts" => Ok(Place::Salts),
            _ if line == "own" => Ok(Place::Own),
            _ => Err(refused(format!("the Lean session loop: {}", line.strip_prefix("err ").unwrap_or(&line)))),
        }
    }

    /// The algebraic phase's values the rep's frame took (`vals NAME I R`): a passed device round's, zero in a round it
    /// failed, in the order the transcript absorbs them.
    pub fn vals(&self) -> Result<Vec<F128>, String> {
        let (line, b) = self.ask("vals")?;
        if !line.starts_with("ok ") || b.len() % 16 != 0 {
            return Err(refused(format!("the Lean session loop: {}", line.strip_prefix("err ").unwrap_or(&line))));
        }
        Ok(b.chunks(16).map(|w| F128::new(u64::from_le_bytes(w[..8].try_into().unwrap()), u64::from_le_bytes(w[8..].try_into().unwrap()))).collect())
    }
}

/// A table's request bytes to `flock-firewall finish`'s `table` (`Flock.Firewall.Link.parse`): `z`, the extra lanes,
/// every lane's padding, the level-0 salts.
#[cfg(feature = "sha512")]
pub fn link_body(z: &[F128], extra: &[Vec<F128>], pads: &[Vec<F128>], salts: &[u8]) -> Vec<u8> {
    let mut body = f128_wire(z);
    for v in extra.iter().chain(pads) {
        body.extend(f128_wire(v));
    }
    body.extend_from_slice(salts);
    body
}

/// A rep's request bytes to `flock-firewall finish` (`Flock.Firewall.Finish.parseRep`): the passed values, the extra claims'
/// points, and every round's coins. The rep's samples are its Lean frame's (or, with no session loop, the finish's own) and
/// levels ≥ 1's salts the finish's ([`LeanFinish::salts`]): neither comes from here.
#[cfg(feature = "sha512")]
pub fn rep_body(vals: &[F128], extras: &[Vec<F128>], rounds: &[Vec<F128>]) -> Vec<u8> {
    let mut b = f128_wire(vals);
    b.extend((extras.len() as u64).to_le_bytes());
    for x in extras {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(f128_wire(x));
    }
    b.extend((rounds.len() as u64).to_le_bytes());
    for c in rounds {
        b.extend((c.len() as u64).to_le_bytes());
        b.extend(f128_wire(c));
    }
    b
}

/// The ZK server's proof files of record on a ZK server session: the Lean session loop's finish (`Flock.Firewall.Finish`
/// in the Lean verifier) on the session's statement, which reads each table's level 0 from the loop's own
/// ([`LeanFinish::level0`]), then per rep, once its algebraic phase ended, is asked for its salts for levels ≥ 1, which it
/// samples itself, and at its end given its request, which it proves with the pads the rep's Lean frame drew. It ends at
/// its first refusal, so every later request is refused too.
#[cfg(feature = "sha512")]
pub struct LeanFinish {
    stdin: std::net::TcpStream,
    stdout: std::io::BufReader<std::net::TcpStream>,
    addr: String,
    draw: String,
}

#[cfg(feature = "sha512")]
impl LeanFinish {
    /// The Lean session loop's finish, on its frames port `addr` ([`LeanSession::frames`]), at the session's sample `draw` (the
    /// JSON `Register` answered, `null` for none): the loop sets the statement up as its `Hello` does.
    pub fn connect(addr: &str, draw: &str) -> Result<Self, String> {
        use std::io::{BufRead, Write};
        let fail = |e: String| refused(format!("the Lean session loop's finish at {addr}: {e}"));
        let tcp = std::net::TcpStream::connect(addr).map_err(|e| fail(e.to_string()))?;
        let _ = tcp.set_nodelay(true);
        let reader = tcp.try_clone().map_err(|e| fail(e.to_string()))?;
        let mut f = LeanFinish { stdin: tcp, stdout: std::io::BufReader::new(reader), addr: addr.to_string(), draw: draw.to_string() };
        write!(f.stdin, "finish {}\n{draw}", draw.len()).and_then(|_| f.stdin.flush()).map_err(|e| fail(e.to_string()))?;
        let mut line = String::new();
        f.stdout.read_line(&mut line).map_err(|e| fail(e.to_string()))?;
        match line.trim_end() {
            "ok 0" => Ok(f),
            l => Err(fail(l.strip_prefix("err ").unwrap_or(if l.is_empty() { "no answer" } else { l }).to_string())),
        }
    }

    /// Another finish on the same session loop and sample, beside this one: the loop serves as many as the prover opens,
    /// each holding the requests it serves to what the loop's frames hold.
    pub fn another(&self) -> Result<Self, String> {
        Self::connect(&self.addr, &self.draw)
    }

    fn ask(&mut self, head: &str, body: &[u8]) -> Result<Vec<u8>, String> {
        use std::io::{BufRead, Read, Write};
        writeln!(self.stdin, "{head} {}", body.len()).and_then(|_| self.stdin.write_all(body)).and_then(|_| self.stdin.flush())
            .map_err(|e| refused(format!("the Lean finish's {head}: {e}")))?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|e| refused(format!("the Lean finish's answer to {head}: {e}")))?;
        let line = line.trim_end();
        let n: usize = match line.strip_prefix("ok ").map(str::parse) {
            Some(Ok(n)) => n,
            _ => return Err(refused(format!("the Lean finish: {}", line.strip_prefix("err ").unwrap_or(if line.is_empty() { "no answer" } else { line })))),
        };
        let mut out = vec![0u8; n];
        self.stdout.read_exact(&mut out).map_err(|e| refused(format!("the Lean finish's answer to {head}: {e}")))?;
        Ok(out)
    }

    /// Session table `i`'s level 0, the loop's of root `root`: its level-0 root, which the finish computes from it.
    pub fn level0(&mut self, i: usize, root: &[u8]) -> Result<Vec<u8>, String> {
        self.ask(&format!("level0 {i}"), root)
    }

    /// Rep `r` of session table `i`'s salts for levels ≥ 1, which the finish samples from the OS once, for the rep's proof
    /// file: one byte string a level, levels `1` to `R` in order, each 192 bytes a leaf (`Finish.saltsBytes`). The loop's
    /// finish samples them only once the rep's frame took its algebraic phase's last coins (`Finish.Own.algebraEnded`).
    pub fn salts(&mut self, i: usize, r: usize) -> Result<Vec<Vec<u8>>, String> {
        let b = self.ask(&format!("salts {i} {r}"), &[])?;
        let bad = || refused(format!("the Lean finish's salts of table {i} rep {r}: {} bytes that do not parse", b.len()));
        let u64_at = |o: usize| b.get(o..o + 8).map(|x| u64::from_le_bytes(x.try_into().unwrap()) as usize);
        let n = u64_at(0).ok_or_else(bad)?;
        let (mut o, mut out) = (8usize, Vec::with_capacity(n.min(64)));
        for _ in 0..n {
            let len = u64_at(o).ok_or_else(bad)?;
            out.push(b.get(o + 8..o + 8 + len).ok_or_else(bad)?.to_vec());
            o += 8 + len;
        }
        if o != b.len() {
            return Err(bad());
        }
        Ok(out)
    }

    /// Rep `r` of session table `i`, on its request (`rep_body`): the rep's proof file.
    pub fn rep(&mut self, i: usize, r: usize, body: &[u8]) -> Result<Vec<u8>, String> {
        self.ask(&format!("rep {i} {r}"), body)
    }

    /// Rep `r` of session table `i`'s inner proof, live (`Finish.innerLive`), on its request with the coins of its PIOP and
    /// opening (`rep_body`): its constraints go to `cons` (`Finish.consBytes`), then each of its rounds to `round`, which relays
    /// it onto the rep's stream and answers it with the coins the stream answered it, or with None to stop the rep (which ends
    /// the finish). The finish hands each round to the session loop's frame before it comes here, and the frame releases its
    /// own round (`Frame.innerSched`), stopping the rep where the relayed one differs, and at a round the finish did not hand
    /// it, whatever is relayed; the finish refuses a request whose `final_c` is not the one its passed values fix, and coins
    /// that are not the ones the frame took (`Finish.coinsWhy`).
    /// Returns the names of the constraints that fail on the rep's pads (`Finish.namesBytes`); where there are none, its proof
    /// file is then [`Self::rep`]'s, on every round's coins.
    pub fn inner(&mut self, i: usize, r: usize, body: &[u8], round: &mut dyn FnMut(&[u8], usize) -> Option<Vec<F128>>,
                 cons: &mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<Vec<String>, String> {
        let b = self.live(&format!("inner {i} {r}"), body, round, cons)?;
        parse_names(&b).ok_or_else(|| refused(format!("the Lean finish's failing constraints of table {i} rep {r}: {} bytes that do not parse", b.len())))
    }

    /// A request whose answer comes after the rounds the finish sends live: `round N LEN` and its bytes, answered `coins N`
    /// and their words (or `stop`, which the finish refuses), and `cons LEN` and its bytes, answered nothing.
    fn live(&mut self, head: &str, body: &[u8], round: &mut dyn FnMut(&[u8], usize) -> Option<Vec<F128>>,
            cons: &mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<Vec<u8>, String> {
        use std::io::{BufRead, Read, Write};
        let io = |e: std::io::Error| refused(format!("the Lean finish's {head}: {e}"));
        writeln!(self.stdin, "{head} {}", body.len()).and_then(|_| self.stdin.write_all(body)).and_then(|_| self.stdin.flush()).map_err(io)?;
        loop {
            let mut line = String::new();
            self.stdout.read_line(&mut line).map_err(io)?;
            let words: Vec<&str> = line.split_whitespace().collect();
            let num = |s: &str| s.parse::<usize>().map_err(|_| refused(format!("the Lean finish's {head}: `{}`", line.trim_end())));
            match words.as_slice() {
                ["round", n, len] => {
                    let (n, len) = (num(*n)?, num(*len)?);
                    let mut c = vec![0u8; len];
                    self.stdout.read_exact(&mut c).map_err(io)?;
                    match round(&c, n) {
                        Some(coins) if coins.len() == n => {
                            writeln!(self.stdin, "coins {n}").and_then(|_| self.stdin.write_all(&f128_wire(&coins))).map_err(io)?;
                        }
                        _ => writeln!(self.stdin, "stop").map_err(io)?,
                    }
                    self.stdin.flush().map_err(io)?;
                }
                ["cons", len] => {
                    let mut b = vec![0u8; num(*len)?];
                    self.stdout.read_exact(&mut b).map_err(io)?;
                    cons(&b)?;
                }
                ["ok", n] => {
                    let mut out = vec![0u8; num(*n)?];
                    self.stdout.read_exact(&mut out).map_err(io)?;
                    return Ok(out);
                }
                _ => {
                    let l = line.trim_end();
                    return Err(refused(format!("the Lean finish: {}", l.strip_prefix("err ").unwrap_or(if l.is_empty() { "no answer" } else { l }))));
                }
            }
        }
    }
}

#[cfg(feature = "sha512")]
fn take<'a>(b: &'a [u8], o: &mut usize, n: usize) -> Option<&'a [u8]> {
    let x = b.get(*o..o.checked_add(n)?)?;
    *o += n;
    Some(x)
}

#[cfg(feature = "sha512")]
fn take_len(b: &[u8], o: &mut usize) -> Option<usize> {
    take(b, o, 8).map(|x| u64::from_le_bytes(x.try_into().unwrap()) as usize)
}

/// A name as the Lean finish sends it: its byte count, then its bytes.
#[cfg(feature = "sha512")]
fn take_name(b: &[u8], o: &mut usize) -> Option<String> {
    let k = take_len(b, o)?;
    String::from_utf8(take(b, o, k)?.to_vec()).ok()
}

/// The names of a rep's constraints that fail on its pads, as the Lean finish answers its inner proof (`Finish.namesBytes`).
#[cfg(feature = "sha512")]
pub fn parse_names(b: &[u8]) -> Option<Vec<String>> {
    let mut o = 0usize;
    let n = take_len(b, &mut o)?;
    let out = (0..n).map(|_| take_name(b, &mut o)).collect::<Option<Vec<_>>>()?;
    (o == b.len()).then_some(out)
}

/// The constraints the Lean finish sends before a rep's inner proof (`Finish.consBytes`): each its name and form.
#[cfg(feature = "sha512")]
pub fn parse_cons(b: &[u8]) -> Result<Vec<(String, crate::zk_veil::Lin)>, String> {
    fn word(x: &[u8]) -> F128 {
        F128::new(u64::from_le_bytes(x[..8].try_into().unwrap()), u64::from_le_bytes(x[8..16].try_into().unwrap()))
    }
    let parsed = (|| -> Option<Vec<(String, crate::zk_veil::Lin)>> {
        let mut o = 0usize;
        let n = take_len(b, &mut o)?;
        let mut out = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            let name = take_name(b, &mut o)?;
            let c = word(take(b, &mut o, 16)?);
            let m = take_len(b, &mut o)?;
            let a = take(b, &mut o, m.checked_mul(16)?)?.chunks(16).map(word).collect();
            out.push((name, crate::zk_veil::Lin { c, a }));
        }
        (o == b.len()).then_some(out)
    })();
    parsed.ok_or_else(|| refused(format!("the Lean finish's constraints: {} bytes that do not parse", b.len())))
}

/// The rows the Lean finish adds to the replay's (`Finish.repCons`): the ring switch's, then level 0's lane phase.
pub const RING_SWITCH: [&str; 2] = ["ring switch ab", "ring switch c"];
pub const LANE_PHASE: [&str; 2] = ["ligerito lane phase (c0)", "ligerito lane phase (c1)"];

/// The Lean finish's constraints (`parse_cons`) held to the statement's relations, name for name and in order: the replay's
/// (`replay`), the ring switch's and, under Ligerito, level 0's lane phase. The inner proof's batching coins go one to a
/// constraint and the recorder's relations go by these names, so any other sequence is refused.
#[cfg(feature = "sha512")]
pub fn relations_held(cons: &[(String, crate::zk_veil::Lin)], replay: &[&str], ligerito: bool) -> Result<(), String> {
    let lane: &[&str] = if ligerito { &LANE_PHASE } else { &[] };
    let want: Vec<&str> = replay.iter().chain(&RING_SWITCH).chain(lane).copied().collect();
    let got: Vec<&str> = cons.iter().map(|c| c.0.as_str()).collect();
    if got == want {
        Ok(())
    } else {
        Err(refused(format!("the Lean finish's constraints {got:?} are not the statement's relations {want:?}")))
    }
}

/// A table's level 0 as the Lean session loop sampled and committed it (`Flock.Firewall.Level0`): the mask slot's words it
/// wrote into the witness, the four extra lanes, every lane's padding, the level-0 salts (192 bytes a leaf) and the root, the
/// cap concatenated.
#[cfg(feature = "sha512")]
pub struct LeanLevel0 {
    pub mask: Vec<F128>,
    pub extra: Vec<Vec<F128>>,
    pub pads: Vec<Vec<F128>>,
    pub salts: Vec<u8>,
    pub root: Vec<u8>,
}

/// One table's part of a level-0 request to the Lean session loop ([`lean_level0`]): its statement's `m`, the host's packed
/// witness `z`, and the statement's sizes the answer must have: `mask_words`, `extra_words`, `lanes` (the extra ones
/// included), `t_pad` and `leaves`.
#[cfg(feature = "sha512")]
#[derive(Clone, Copy)]
pub struct Level0Ask<'a> {
    pub m: usize,
    pub z: &'a [F128],
    pub mask_words: usize,
    pub extra_words: usize,
    pub lanes: usize,
    pub t_pad: usize,
    pub leaves: usize,
}

/// The level-0 hiding of record on a session's firewalled tables under the Lean session loop: the loop, on its frames port
/// `addr`, takes every table's packed witness in one request (`level0 SESSION M LEN …`, a pair a table in table order,
/// followed by the witnesses), and only then samples each table's mask slot's words, level 0's hiding and salts from the OS,
/// writes the words into the witness's mask slot, commits level 0 on them and keeps them for the session `session` names
/// ([`level0_session`]) until that session ends. It answers each table in order, `ok N R` and the samples then the root, its
/// last `R` bytes, or the request `err WHY`, nothing sampled. So no table's witness can depend on another table's samples. The
/// loop takes one request a session, with every table's witness, and refuses any other.
#[cfg(feature = "sha512")]
pub fn lean_level0(addr: &str, session: &str, asks: &[Level0Ask]) -> Result<Vec<LeanLevel0>, String> {
    use std::io::{BufRead, Read, Write};
    let fail = |e: String| refused(format!("the Lean loop's level 0 at {addr}: {e}"));
    let tcp = std::net::TcpStream::connect(addr).map_err(|e| fail(e.to_string()))?;
    let _ = tcp.set_nodelay(true);
    let mut out = std::io::BufReader::new(tcp.try_clone().map_err(|e| fail(e.to_string()))?);
    let bodies: Vec<Vec<u8>> = asks.iter().map(|a| f128_wire(a.z)).collect();
    let sizes: String = asks.iter().zip(&bodies).map(|(a, b)| format!(" {} {}", a.m, b.len())).collect();
    let mut w = &tcp;
    write!(w, "level0 {session}{sizes}\n").map_err(|e| fail(e.to_string()))?;
    for b in &bodies {
        w.write_all(b).map_err(|e| fail(e.to_string()))?;
    }
    w.flush().map_err(|e| fail(e.to_string()))?;
    use flock_core::pcs::ligerito::zk::EXTRA_LANES;
    let mut got = Vec::with_capacity(asks.len());
    for (j, a) in asks.iter().enumerate() {
        let mut line = String::new();
        out.read_line(&mut line).map_err(|e| fail(e.to_string()))?;
        let l = line.trim_end();
        let (n, r) = match (l.strip_prefix("ok ").and_then(|v| v.split_once(' ')), l.strip_prefix("err ")) {
            (Some((n, r)), _) => match (n.parse::<usize>(), r.parse::<usize>()) {
                (Ok(n), Ok(r)) => (n, r),
                _ => return Err(fail(format!("table {j}'s answer {l:?}"))),
            },
            (_, Some(why)) => return Err(fail(why.to_string())),
            _ => return Err(fail(format!("table {j}'s answer {l:?}"))),
        };
        let o_extra = 16 * a.mask_words;
        let o_pads = o_extra + 16 * EXTRA_LANES * a.extra_words;
        let o_salts = o_pads + 16 * a.lanes * a.t_pad;
        let o_root = o_salts + crate::zk_hooks::SALT_BYTES * a.leaves;
        if n != o_root + r || r == 0 {
            return Err(fail(format!("table {j}: {n} bytes with a {r}-byte root where the draws take {o_root}")));
        }
        let mut b = vec![0u8; n];
        out.read_exact(&mut b).map_err(|e| fail(e.to_string()))?;
        let words = |o: usize, k: usize| -> Vec<F128> {
            (0..k).map(|i| F128::new(u64::from_le_bytes(b[o + 16 * i..o + 16 * i + 8].try_into().unwrap()),
                                     u64::from_le_bytes(b[o + 16 * i + 8..o + 16 * i + 16].try_into().unwrap()))).collect()
        };
        got.push(LeanLevel0 { mask: words(0, a.mask_words),
                              extra: (0..EXTRA_LANES).map(|e| words(o_extra + 16 * a.extra_words * e, a.extra_words)).collect(),
                              pads: (0..a.lanes).map(|l| words(o_pads + 16 * a.t_pad * l, a.t_pad)).collect(),
                              salts: b[o_salts..o_root].to_vec(), root: b[o_root..].to_vec() });
    }
    Ok(got)
}

/// The ZK server's session loop of record: `flock-firewall session` (`Flock.Firewall.Session` in the Lean verifier) owns the
/// auditor's connection. The prover connects to [`LeanSession::addr`] as it would to the auditor, and each connection is one
/// session, which the Lean loop relays on its own connection to `verifier`: the `Hello` that leaves is its own (at the
/// statement `args` name, `flock-verify verify --zk`'s flags, and the auditor's sample), `Hello`'s answer and every coin are
/// checked there, and only the coins, without their openings, come back. Each rep's Lean frame is one of the loop's
/// ([`LeanSession::frames`]), and a rep's round leaves only as its frame's release. It runs until this handle drops.
#[cfg(feature = "sha512")]
pub struct LeanSession {
    child: std::process::Child,
    _stdin: std::process::ChildStdin,
    addr: String,
    frames: String,
}

#[cfg(feature = "sha512")]
impl LeanSession {
    pub fn start(exe: &std::path::Path, verifier: &str, args: &[String]) -> Result<Self, String> {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        let mut child = Command::new(exe).arg("session").arg("--verifier").arg(verifier).args(args)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
            .map_err(|e| refused(format!("the Lean session {} does not start ({e}): `lake build flock-firewall` in verity/protocols/flock/verifier, or FC_FIREWALL_LEAN; \
                                          with --firewall-state, `lake build core-firewall` in verity/protocols/recursion/verifier/firewall, or FC_CORE_FIREWALL", exe.display())))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let mut stdout = std::io::BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut line = String::new();
        stdout.read_line(&mut line).map_err(|e| refused(format!("the Lean session's start: {e}")))?;
        let mut frames = String::new();
        let _ = stdout.read_line(&mut frames);
        let (Some(addr), Some(frames)) = (line.trim_end().strip_prefix("LISTENING ").map(str::to_string),
                                          frames.trim_end().strip_prefix("FRAMES ").map(str::to_string)) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(refused(format!("the Lean session did not start: {:?}", line.trim_end())));
        };
        Ok(LeanSession { child, _stdin: stdin, addr, frames })
    }

    /// Where the prover connects for a session.
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Where a rep's Lean frame starts ([`LoopRep::start`]).
    pub fn frames(&self) -> &str {
        &self.frames
    }
}

#[cfg(feature = "sha512")]
impl Drop for LeanSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The ZK server's value service of record: `core-firewall values --state DIR --salt-key FILE` (`Flock.Firewall.Values`
/// in the Lean verifier, on the ZK server's record in `DIR`), which holds the server's salt key and derives every value
/// commitment's salt from it (live PROTOCOL.md §10.2). The serving stack commits through it ([`lean_commit`]) and
/// registers one session of each epoch ([`lean_register`]), each registration numbered on the record, and the prover takes
/// its witness's salts from it ([`lean_salts`]). It runs until this handle drops.
#[cfg(feature = "sha512")]
pub struct LeanValues {
    child: std::process::Child,
    _stdin: std::process::ChildStdin,
    addr: String,
}

#[cfg(feature = "sha512")]
impl LeanValues {
    /// The service `exe` (`core-firewall`) on the ZK server's state directory `state` and the key file `key`.
    pub fn start(exe: &std::path::Path, state: &std::path::Path, key: &std::path::Path) -> Result<Self, String> {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        let mut child = Command::new(exe).arg("values").arg("--state").arg(state).arg("--salt-key").arg(key)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()
            .map_err(|e| refused(format!("the Lean value service {} does not start ({e}): `lake build core-firewall` in verity/protocols/recursion/verifier/firewall, or FC_CORE_FIREWALL", exe.display())))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let mut line = String::new();
        let _ = std::io::BufReader::new(child.stdout.take().expect("piped stdout")).read_line(&mut line);
        let Some(addr) = line.trim_end().strip_prefix("VALUES ").map(str::to_string) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(refused(format!("the Lean value service did not start: {:?}", line.trim_end())));
        };
        Ok(LeanValues { child, _stdin: stdin, addr })
    }

    /// Where a commit or a salts request connects.
    pub fn addr(&self) -> &str {
        &self.addr
    }
}

#[cfg(feature = "sha512")]
impl Drop for LeanValues {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One request to the value service at `addr`, its line `line` and its body `body`, sent whole before the client shuts its
/// side down (the service reads a request up to there): the whole answer.
#[cfg(feature = "sha512")]
pub fn values_request(addr: &str, line: &str, body: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::{Read, Write};
    let tcp = std::net::TcpStream::connect(addr).map_err(|e| e.to_string())?;
    let send = || -> std::io::Result<()> {
        let mut w = &tcp;
        w.write_all(format!("{line}\n").as_bytes())?;
        w.write_all(body)?;
        w.flush()?;
        tcp.shutdown(std::net::Shutdown::Write)
    };
    send().map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    (&tcp).read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

/// The answer line at byte `*o` of `b`, past `ok `, its words; the service's `err WHY` as the error.
#[cfg(feature = "sha512")]
fn answer_line(b: &[u8], o: &mut usize) -> Result<Vec<String>, String> {
    let nl = b[*o..].iter().position(|&x| x == b'\n').ok_or("an answer with no line")?;
    let l = String::from_utf8_lossy(&b[*o..*o + nl]).to_string();
    *o += nl + 1;
    match (l.strip_prefix("ok "), l.strip_prefix("err ")) {
        (Some(ws), _) => Ok(ws.split(' ').map(String::from).collect()),
        (_, Some(why)) => Err(why.to_string()),
        _ => Err(format!("an answer line {l:?}")),
    }
}

/// The row schemas in the order the value service indexes them (`Flock.Registered.SCHEMAS`).
#[cfg(feature = "sha512")]
pub const VALUE_SCHEMAS: [&str; 3] = [crate::circuit::ROW_SCHEMA, crate::circuit::ROW_SCHEMA_V2, crate::circuit::ROW_SCHEMA_SEG];

/// A table a commit request carries: its frame-v3-sha512 domain id, its row schema and its rows' unsalted 64-byte digests in
/// leaf order.
#[cfg(feature = "sha512")]
pub struct CommitTable {
    pub domain: Vec<u8>,
    pub schema: &'static str,
    pub digests: Vec<[u8; 64]>,
}

/// A table as the value service committed it: its root, its tag and its leaves' `b || c` in leaf order.
#[cfg(feature = "sha512")]
pub struct Committed {
    pub root: Vec<u8>,
    pub tag: Vec<u8>,
    pub bcs: Vec<[u8; crate::circuit::BC_BYTES]>,
}

/// Commits `tables` through the value service at `addr` (`commit J D₀ N₀ S₀ …`, the body every table's digests and
/// nothing else): the session the service drew, and each table as it committed it.
#[cfg(feature = "sha512")]
pub fn lean_commit(addr: &str, tables: &[CommitTable]) -> Result<(Vec<u8>, Vec<Committed>), String> {
    use crate::circuit::BC_BYTES;
    let fail = |e: String| refused(format!("the Lean value service's commit at {addr}: {e}"));
    let mut line = format!("commit {}", tables.len());
    for t in tables {
        if !VALUE_SCHEMAS.contains(&t.schema) {
            return Err(fail(format!("a table of row schema {}, none of the service's", t.schema)));
        }
        line += &format!(" {} {} {}", crate::hex(&t.domain), t.digests.len(), t.schema);
    }
    let body: Vec<u8> = tables.iter().flat_map(|t| t.digests.iter().flatten().copied()).collect();
    let b = values_request(addr, &line, &body).map_err(fail)?;
    let mut o = 0;
    let hex = |w: &[String], k: usize| w.get(k).ok_or("a short answer line".to_string()).and_then(|x| crate::circuit::hexb(x));
    let s = hex(&answer_line(&b, &mut o).map_err(fail)?, 0).map_err(fail)?;
    let mut out = Vec::with_capacity(tables.len());
    for (j, t) in tables.iter().enumerate() {
        let w = answer_line(&b, &mut o).map_err(fail)?;
        let n: usize = w.get(2).and_then(|x| x.parse().ok()).ok_or_else(|| fail(format!("table {j}'s answer {w:?}")))?;
        if n != t.digests.len() || b.len() < o + BC_BYTES * n {
            return Err(fail(format!("table {j}: {n} leaves answered of its {}, {} bytes left", t.digests.len(), b.len() - o)));
        }
        let bcs = b[o..o + BC_BYTES * n].chunks(BC_BYTES).map(|x| x.try_into().unwrap()).collect();
        o += BC_BYTES * n;
        out.push(Committed { root: hex(&w, 0).map_err(fail)?, tag: hex(&w, 1).map_err(fail)?, bcs });
    }
    if o != b.len() {
        return Err(fail(format!("{} bytes past its tables", b.len() - o)));
    }
    Ok((s, out))
}

/// A table ref (`Flock.Firewall.Salts.TableRef`): table `j` of the commit session `s`, its domain, row schema and leaf
/// count, and the root and tag the service answered.
#[cfg(feature = "sha512")]
pub fn table_ref(s: &[u8], j: usize, t: &CommitTable, c: &Committed) -> Vec<u8> {
    let schema = VALUE_SCHEMAS.iter().position(|&x| x == t.schema).expect("a committed table's schema") as u8;
    let parts: [&[u8]; 7] = [s, &(j as u32).to_be_bytes(), &t.domain, &(t.digests.len() as u64).to_be_bytes(), &[schema], &c.root, &c.tag];
    parts.concat()
}

/// A refs file (`Flock.Firewall.Salts.parseRefs`): `be32 T ‖ be32 R`, the table refs, then each row ref `(t, i)`, leaf `i`
/// of table ref `t`.
#[cfg(feature = "sha512")]
pub fn refs_file(tables: &[Vec<u8>], rows: &[(usize, usize)]) -> Vec<u8> {
    let mut b = [(tables.len() as u32).to_be_bytes(), (rows.len() as u32).to_be_bytes()].concat();
    b.extend(tables.iter().flatten());
    for &(t, i) in rows {
        b.extend((t as u32).to_be_bytes());
        b.extend((i as u64).to_be_bytes());
    }
    b
}

/// The number of row refs a refs file names, from its counts and its length.
#[cfg(feature = "sha512")]
pub fn refs_rows(refs: &[u8]) -> Result<usize, String> {
    const TABLE_REF: usize = 64 + 4 + 64 + 8 + 1 + 64 + 64;
    let count = |o: usize| refs.get(o..o + 4).map(|x| u32::from_be_bytes(x.try_into().unwrap()) as usize);
    let (Some(t), Some(r)) = (count(0), count(4)) else { return Err(format!("a refs file of {} bytes, shorter than its counts", refs.len())) };
    if refs.len() != 8 + TABLE_REF * t + 12 * r {
        return Err(format!("a refs file of {} bytes, where {t} table refs and {r} row refs take {}", refs.len(), 8 + TABLE_REF * t + 12 * r));
    }
    Ok(r)
}

/// Registers the commit session `s` through the value service at `addr` (`register S`): the registration's number on the
/// ZK server's record, the number of sessions the service's epoch opened (1: the service opens one an epoch, refusing a
/// commit while its session is not yet registered), and its record (`Flock.Firewall.Salts.recordOf`:
/// `be64 n ‖ be64 m ‖ be32 T ‖` the session's table refs `‖` its record tag), which the service appended to the ZK server's
/// record whole, where the session loop admits it (`core-firewall session --salt-key`).
#[cfg(feature = "sha512")]
pub fn lean_register(addr: &str, s: &[u8]) -> Result<(u64, u64, Vec<u8>), String> {
    let fail = |e: String| refused(format!("the Lean value service's registration at {addr}: {e}"));
    let b = values_request(addr, &format!("register {}", crate::hex(s)), &[]).map_err(fail)?;
    let mut o = 0;
    let w = answer_line(&b, &mut o).map_err(fail)?;
    let (Some(m), Some(len)) = (w.first().and_then(|x| x.parse::<u64>().ok()), w.get(1).and_then(|x| x.parse::<usize>().ok())) else {
        return Err(fail(format!("an answer {w:?}")));
    };
    let rec = &b[o..];
    if rec.len() != len || len < 20 || rec[8..16] != m.to_be_bytes() {
        return Err(fail(format!("a record of {} bytes, where the answer names {len} and {m} sessions", rec.len())));
    }
    Ok((u64::from_be_bytes(rec[..8].try_into().unwrap()), m, rec.to_vec()))
}

/// The salts of the rows the refs file `refs` names, in its row refs' order, from the value service at `addr`, which checks
/// every table ref's tag.
#[cfg(feature = "sha512")]
pub fn lean_salts(addr: &str, refs: &[u8]) -> Result<Vec<u8>, String> {
    let fail = |e: String| refused(format!("the Lean value service's salts at {addr}: {e}"));
    let b = values_request(addr, "salts", refs).map_err(fail)?;
    let mut o = 0;
    let w = answer_line(&b, &mut o).map_err(fail)?;
    let len: usize = w.first().and_then(|x| x.parse().ok()).ok_or_else(|| fail(format!("an answer {w:?}")))?;
    let want = crate::circuit::SALT_BYTES * refs_rows(refs).map_err(fail)?;
    if len != want || b.len() != o + len {
        return Err(fail(format!("{len} bytes of salts in an answer of {}, where the refs' rows take {want}", b.len() - o)));
    }
    Ok(b[o..].to_vec())
}

/// The masking challenger's inner challenger under the ZK server. While a round's `release` is set (the device's unmasked
/// round, which the Lean session loop's frame of the rep checks and masks, or a firewalled rep's inner proof's round, the
/// session's Lean finish's, which the frame holds to its own: `flock-circuit`'s `zk_finish_fw`), what the masking
/// challenger hands it is dropped, and at the round's squeeze the release, and only the release, goes to the stream. With
/// no release set it passes every call through: unserved reps.
pub struct Gate<C: Challenger> {
    pub inner: C,
    pub release: Option<Vec<u8>>,
}

impl<C: Challenger> Gate<C> {
    pub fn new(inner: C) -> Self {
        Gate { inner, release: None }
    }

    fn held(&self) -> bool {
        self.release.is_some()
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        if n == 0 && kind == KIND_SLICE {
            return Vec::new();
        }
        let release = self.release.take().expect("a held round");
        crate::replay_round(&mut self.inner, &release, n).unwrap_or_else(|e| panic!("LIVE-ERROR replay: {e}"))
    }
}

impl<C: Challenger> Challenger for Gate<C> {
    fn observe_label(&mut self, l: &[u8]) {
        if !self.held() {
            self.inner.observe_label(l);
        }
    }
    fn observe_f128(&mut self, v: F128) {
        if !self.held() {
            self.inner.observe_f128(v);
        }
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        if !self.held() {
            self.inner.observe_f128_slice(vs);
        }
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        if !self.held() {
            self.inner.observe_bytes(b);
        }
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

    /// Every call a challenger takes, in order, answering distinct coins.
    #[derive(Default)]
    struct Recorder {
        items: Vec<String>,
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
            self.items.push(format!("label {l:?}"));
        }
        fn observe_f128(&mut self, v: F128) {
            self.items.push(format!("scalar {} {}", v.lo, v.hi));
        }
        fn observe_f128_slice(&mut self, vs: &[F128]) {
            self.items.push(format!("slice {:?}", vs.iter().map(|v| (v.lo, v.hi)).collect::<Vec<_>>()));
        }
        fn observe_bytes(&mut self, b: &[u8]) {
            self.items.push(format!("bytes {b:?}"));
        }
        fn sample_f128(&mut self) -> F128 {
            self.items.push("squeeze 1".into());
            self.coin()
        }
        fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
            self.items.push(format!("squeeze {n}"));
            (0..n).map(|_| self.coin()).collect()
        }
        fn grind_pow(&mut self, _bits: u32) -> u64 {
            unreachable!("no test grinds")
        }
        fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
            unreachable!("no test forks")
        }
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
        assert_eq!((&g.inner.items, coin), (&direct.items, F128::new(0x5eed_0001, 0x1234_5678_9abc_def0)), "the release reaches the stream");
        assert!(g.release.is_none(), "a release is used once");
        let release = framed(|f| { f.pending.clear(); masked_round(f, 10); }).remove(0).0;
        let mut g = Gate::new(Recorder::default());
        g.release = Some(release);
        masked_round(&mut g, 11);
        assert_eq!(g.inner.items, direct.items, "what the masking challenger handed over is dropped: only the release leaves");
    }

    /// The ZK server's side of [`run_worker`] for a test: every ask served in order into `log`, round `k`'s coins `[k]`.
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

    /// Each way a worker fails is the ZK server's `Err` after the rounds it asked, never a panic on the ZK server's thread: an
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

    #[cfg(feature = "sha512")]
    #[test]
    fn lean_constraints_are_held_to_the_relations_name_for_name() {
        let lin = || crate::zk_veil::Lin { c: F128::ZERO, a: vec![F128::ONE] };
        let named = |ns: &[&str]| ns.iter().map(|n| (n.to_string(), lin())).collect::<Vec<_>>();
        let replay = ["zerocheck final a·b", "zerocheck c_eval", "lincheck"];
        let honest = [&replay[..], &RING_SWITCH, &LANE_PHASE].concat();
        assert_eq!(relations_held(&named(&honest), &replay, true), Ok(()));
        assert_eq!(relations_held(&named(&honest[..5]), &replay, false), Ok(()));
        let renamed = [&replay[..2], &["lincheck sum"], &RING_SWITCH, &LANE_PHASE].concat();
        let swapped = [&replay[..], &["ring switch c", "ring switch ab"], &LANE_PHASE].concat();
        for bad in [renamed, swapped, honest[..6].to_vec(), [&honest[..], &["extra"]].concat()] {
            let e = relations_held(&named(&bad), &replay, true).unwrap_err();
            assert!(e.starts_with("FIREWALL-REFUSED: the Lean finish's constraints"), "{e}");
        }
    }

    /// A table ref and a refs file at the offsets `Flock.Firewall.Salts.TableRef.parse` and `parseRefs` read them.
    #[cfg(feature = "sha512")]
    #[test]
    fn a_refs_file_is_laid_out_as_the_value_service_reads_it() {
        let t = CommitTable { domain: vec![1; 64], schema: crate::circuit::ROW_SCHEMA_V2, digests: vec![[0; 64]; 3] };
        let c = Committed { root: vec![2; 64], tag: vec![3; 64], bcs: vec![] };
        let r = table_ref(&[4; 64], 5, &t, &c);
        assert_eq!(r.len(), 269);
        assert_eq!((&r[..64], &r[64..68], &r[68..132], &r[132..140], r[140], &r[141..205], &r[205..]),
                   (&[4; 64][..], &[0, 0, 0, 5][..], &[1; 64][..], &3u64.to_be_bytes()[..], 1, &[2; 64][..], &[3; 64][..]));
        let f = refs_file(&[r.clone()], &[(0, 2), (0, 0)]);
        assert_eq!((&f[..8], &f[8..8 + 269]), (&[0, 0, 0, 1, 0, 0, 0, 2][..], &r[..]));
        assert_eq!(&f[8 + 269..], &[[0u8, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]].concat()[..]);
        assert_eq!(refs_rows(&f), Ok(2));
        assert!(refs_rows(&f[..f.len() - 1]).is_err() && refs_rows(&f[..7]).is_err());
    }

    #[cfg(feature = "sha512")]
    #[test]
    fn the_failing_names_parse_as_the_lean_finish_sends_them() {
        let names = |ns: &[&[u8]]| -> Vec<u8> {
            let mut b = (ns.len() as u64).to_le_bytes().to_vec();
            for n in ns {
                b.extend((n.len() as u64).to_le_bytes());
                b.extend_from_slice(n);
            }
            b
        };
        assert_eq!(parse_names(&names(&[])), Some(vec![]));
        assert_eq!(parse_names(&names(&[b"zerocheck c_eval", b"ring switch c"])),
                   Some(vec!["zerocheck c_eval".to_string(), "ring switch c".to_string()]));
        let two = names(&[b"lincheck", b"ring switch ab"]);
        for bad in [[&two[..], &[0]].concat(), two[..two.len() - 1].to_vec(), two[..4].to_vec(), names(&[&[0xff]])] {
            assert_eq!(parse_names(&bad), None, "{bad:?}");
        }
    }
}
