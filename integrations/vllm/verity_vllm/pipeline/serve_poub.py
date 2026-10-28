"""serve-openai's remote bandwidth session (``--poub-verifier HOST:PORT``): the Commit's session
(`protocols.poub.remote.CommitSession`, as `pipeline.poub_session` builds it for a declared Commit) on the
server's engine.

~~~text
preflight  refused before the model loads (`refusal`): an address that isn't HOST:PORT, a plan other than serve, a
           store read in place (--poub-session-lib) without kv.cu (--poub-kv-lib), prefix caching (a hit brings a
           page back with its KV), and a store read in place under PoUW H1 (--protocols), whose windows hash the weights
           where the engine holds them, which that store holds encoded while its session lasts
build      before the engine is built, the session's engine arguments (synchronous scheduling, no prefix caching; the KV
           pool the engine's own, outside the store) and the store binder on vLLM (`CommitSession.engine_args`,
           `install`); after it, `CommitSession.check`, which registers --poub-filler-gib of zeros beside the weights.
           The engine then runs at LAG 0, so the bench row the server commits under declares ``async_scheduling``
           false, as a declared Commit's row does under its session (`CommitSession.target`)
begin      once the committer has read every weight a seal reads, its weights tree at its build (`serve_seal.committer`)
           and the registered tensors' bytes (`serve_commit.CommitWindows.read_extras`), and before the first request:
           the session registers the store, which a store read in place holds encoded from then until the session
           closes (`in_place`), and waits for the server's first engine step
serving    the engine loop's steps through `Paced`: while the session lasts each is a pass's (`Lockstep.step`), so the
           engine steps as the session asks and the memory map is of the server's steps from its first; once the
           session has ended its record is written (`end`) and the later steps are unaudited. A server that idles
           while the session still asks runs no step, so the verifier's window lapses and the session is rejected
stop       the session ended if it hasn't (rejected if it still asked for a step); summary.json's ``poub_session`` is
           its verdict and the server's engine steps, audited and not (its whole record is ``poub_session.json``
           beside the records), and a rejected session fails the server's exit
~~~
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.pipeline import poub_session as RS


@dataclass(frozen=True, kw_only=True)
class PoubOptions:
    poub_verifier: str | None = option("--poub-verifier", default=None, metavar="HOST:PORT",
                                         help="the remote bandwidth session's verifier: the store binder on the engine from its "
                                              "build, and the store proved to it in lockstep with the server's engine steps "
                                              "from its first, a pass a step (serve_poub); a rejected session fails the exit")
    poub_plan: str = option("--poub-plan", default="serve", help="the session's mode: serve")
    poub_filler_gib: float = option("--poub-filler-gib", type=float, default=0.0,
                                      help="GiB of zeros on the engine's GPU registered after the build as the session's filler")
    poub_weight_blocks: int | None = option("--poub-weight-blocks", type=int, default=None,
                                              help="register weight tensors in order up to this many 4 KiB blocks (default: all)")
    poub_kv_lib: str | None = option("--poub-kv-lib", default=None,
                                       help="kv.cu built at the codec's field and mask: the store's encode on the GPU "
                                            "(default: the CPU reference)")
    poub_session_lib: str | None = option("--poub-session-lib", default=None,
                                            help="session.cu's build: the session reads the store where the engine holds it, "
                                                 "encoded in place by --poub-kv-lib (in_place)")
    poub_step_ms: float = option("--poub-step-ms", type=float, default=0.0,
                                   help="the engine step a paced session's pass waits on, in ms, added to the calibration's "
                                        "pass times")


def refusal(opts: Any) -> str | None:
    """Why serve-openai can't run ``opts``' session (module docstring's preflight), or None."""
    r = opts.remote
    if not r.poub_verifier:
        return None
    try:
        RS.commit_session(r, None, None)
    except ValueError as e:
        return f"--poub-verifier: {e}"
    if opts.prefix_caching:
        return "--poub-verifier refuses --prefix-caching: a prefix hit brings a page back with its KV"
    if r.poub_session_lib and opts.protocols:
        return ("--poub-session-lib holds the weights encoded where the engine holds them while its session lasts, and "
                "--protocols' H1 windows hash them there: serve both with the store's host mirror (no --poub-session-lib)")
    return None


class Paced:
    """The server's engine as `EngineLoop` steps it: each step the session's while it lasts (`Lockstep.step`), and the
    session ended on the engine loop as soon as it has (`ServeSession.end`); the rest is the engine's."""

    def __init__(self, engine: Any, session: ServeSession) -> None:
        self._engine, self._session = engine, session

    def step(self) -> Any:
        s = self._session
        out = s.cs.lockstep.step(self._engine.step)
        if s.cs.lockstep.ended and s.record is None:
            s.end()
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


class ServeSession:
    """The server's `CommitSession` from ``opts.remote`` (module docstring), its record under ``out``."""

    def __init__(self, opts: Any, out: Path) -> None:
        self.cs = RS.commit_session(opts.remote, out, None)
        self.record: dict[str, Any] | None = None

    def engine_args(self, kw: dict[str, Any]) -> dict[str, Any]:
        """The engine's arguments under the session, the store binder installed on vLLM: just before the build."""
        kw = self.cs.engine_args(kw)
        self.cs.install()
        return kw

    def built(self, llm: Any) -> None:
        from verity_vllm.engine.engine_profile import get_model_runner

        self.cs.check(llm.vllm, get_model_runner(llm.vllm))

    def begin(self, windows: Any) -> None:
        """The session registered and waiting for the first step, once ``windows`` (a `CommitWindows`, or None) has read
        the registered tensors' bytes its seals carry."""
        if windows is not None:
            windows.read_extras()
        self.cs.begin()

    def paced(self, engine: Any) -> Paced:
        return Paced(engine, self)

    def end(self) -> dict[str, Any]:
        """The session's record, ended on the first call (`CommitSession.end`), its ``run`` the server's engine steps so
        far (``poub_session.json``'s: those at the session's end)."""
        lock = self.cs.lockstep
        if self.record is None and lock is None:
            self.record = {"verifier": self.cs.verifier, "plan": self.cs.plan, "accepted": False,
                           "error": "the server stopped before its session began", "run": None}
        if self.record is None:
            rec = self.cs.end()
            self.cs.log(f"[serve] remote session over the server's first {lock.audited} engine steps, a pass each: "
                        f"{'ACCEPTED' if rec['accepted'] else 'REJECTED'}" + (f" ({rec['error']})" if rec["error"] else ""))
            mm = rec["memory_map"]
            self.record = {k: rec.get(k) for k in ("verifier", "plan", "accepted", "error", "encode")} | {
                "agrees": mm.get("agrees") if mm else None, "path": str(self.cs.out / "poub_session.json")}
        if lock is not None:
            self.record["run"] = {"steps": lock.audited + lock.unaudited, "audited": lock.audited, "unaudited": lock.unaudited}
        return self.record

    def describe(self) -> dict[str, Any]:
        cs = self.cs
        return {"verifier": cs.verifier, "plan": cs.plan, "async_scheduling": False, "weight_blocks": cs.weight_blocks,
                "filler_gib": cs.filler_gib, "step_ms": cs.step_ms, "in_place": cs.backend_of is not None,
                "kv_lib": cs.kv_lib, "store": cs.binder.stats() if cs.binder is not None else None}


def session_of(opts: Any, out: Path) -> ServeSession | None:
    return ServeSession(opts, out) if opts.remote.poub_verifier else None


def accepted(session: ServeSession | None) -> bool:
    """Whether the session, ended at the stop if it hadn't, accepted (True without one)."""
    return session is None or bool(session.end()["accepted"])
