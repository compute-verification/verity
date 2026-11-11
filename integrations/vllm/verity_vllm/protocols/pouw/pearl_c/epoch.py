"""Pearl-C on the device under the PoUW adapter (`verity_vllm.protocols.pouw`): one epoch an install, made by the install
or ahead of it (`Epochs`, which the adapter makes over its own parts and this module never imports).

An epoch is made in one order: the weights committed under a fresh hm96 key (`Epochs.commit_weights`), then the salt
coin opened after their root and the device run made under that salt (`Epochs.make`: the settings' in-process ``coins``,
or a live salt-only session their ``gate`` opens on the epoch's root; `pearl_c.PearlCDeviceRun`, B̃ formed and its first
pass begun).  `Epochs.install` then hooks the model's linears and its runner's steps onto the run.  `Epochs.prepare`
makes the epoch with the model untouched, for a later install that is handed it (`Prepared`, the pouw setting
``prepared``): that install commits no weight and opens no coin, it only hooks the model.  `Epochs.keep_rows` writes the
served model's weight rows once, which every install's retained passes link (the pouw setting ``kept_rows``).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root
from verity.core.protocols.pouw.protocol import Matmul, Shape

from verity_vllm.protocols.interface import Option, Target
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.h1 import settings as PHS
from verity_vllm.protocols.pouw.h1 import window as PHW
from verity_vllm.protocols.pouw.pearl_c import passes as PP


@dataclass
class Prepared:
    """An epoch made ahead of its install (`Epochs.prepare`), the model untouched: the weights' commitment under the
    epoch's own hm96 key (``committed``, `Epochs.commit_weights`' answer), the device run made over it once its salt
    coin was open (``run``), the live session that coin came from (``session``, None for in-process coins), the layers'
    weights it was made over (``weights``, in order) and what each part took (``seconds``).  The install that is handed
    it takes it (`take`) and from then on closes it as it closes its own; one no install took is closed by `close`."""

    entry: str
    weights: tuple[Any, ...]
    committed: dict[str, Any]
    run: Any
    session: Any = None
    seconds: dict[str, float] = field(default_factory=dict)
    taken: bool = False

    def of(self, entry: str, layers: list[Any]) -> Prepared:
        """This epoch, as an install of `entry` over `layers` may take it: made for that entry over those layers' own
        weights, and taken by no install; ValueError otherwise."""
        if self.entry != entry or self.taken or len(self.weights) != len(layers) or any(
                a is not m.weight for a, m in zip(self.weights, layers)):
            raise ValueError(f"{entry}: the setting 'prepared' is not an epoch made ahead for this install: it is "
                             + ("taken or closed" if self.taken else f"{self.entry}'s" if self.entry != entry
                                else "over other weights than the model's linears'"))
        return self

    def take(self, entry: str, layers: list[Any]) -> tuple[Any, Any]:
        """The run and the session, for the one install that takes this epoch (`of`)."""
        self.of(entry, layers).taken = True
        return self.run, self.session

    def close(self) -> None:
        """An epoch no install took: its run finished and its session closed (`pouw_h1_window.finish`).  Nothing once an
        install has taken it, or it is closed."""
        if not self.taken:
            self.taken = True
            PHW.finish(self.run, self.session)


def settings_of(raw: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Prepared | None]:
    """The pouw settings ``kept_rows`` (`Epochs.keep_rows`' answer) and ``prepared`` (`Epochs.prepare`'s) of `raw`;
    ValueError for another kind of value."""
    kept, prepared = raw.get("kept_rows"), raw.get("prepared")
    if kept is not None and not isinstance(kept, Mapping):
        raise ValueError("pouw setting 'kept_rows' is the served model's weight rows files, as `keep_rows` returns them")
    if prepared is not None and not isinstance(prepared, Prepared):
        raise ValueError("pouw setting 'prepared' is an epoch made ahead of its install, as `prepare` returns it")
    return kept, prepared


def _weights(layers: list[Any]) -> dict[int, Any]:
    return {w: m.weight for w, m in enumerate(layers)}


@dataclass(frozen=True)
class Epochs:
    """Pearl-C's epochs over the PoUW adapter's own parts: its ``name``, its ``settings`` of a raw mapping, the
    ``layers`` it executes of a target (the scheme, the linears in order and their shapes; its ValueError for a layer
    the scheme refuses), the ``executor`` it hooks a layer onto, its hm96 keys by entry as it holds them at each call
    (``salt_keys``) and its registration ``claim`` of a scheme."""

    name: str
    settings: Callable[[Mapping[str, Any]], Any]
    layers: Callable[[str, Target], tuple[Any, list[Any], dict[int, tuple[int, int]]]]
    executor: Callable[[Any, int], Any]
    salt_keys: Callable[[], dict[str, bytes]]
    claim: Callable[[Any], dict[str, Any]]

    def commit_weights(self, scheme: str, entry: str, settings: Any, layers: list[Any],
                       shapes: Mapping[int, tuple[int, int]]) -> dict[str, Any]:
        """The adapter's `commit_weights` under Pearl-C: the epoch's fresh hm96 key (the adapter's key of `entry`) and
        each weight's root under it, on the device.  For an install handed an epoch made ahead, what that epoch
        committed."""
        if settings.prepared is not None:
            return dict(settings.prepared.of(entry, layers).committed)
        key = self.salt_keys()[entry] = PPD.new_salt_key(scheme)
        roots = PPD.weight_roots(PPD.Kernel.load(settings.kernel, settings.cubin), _weights(layers), scheme, salt_key=key)
        return {"weight_roots": {str(w): r.hex() for w, r in roots.items()},
                "shapes": {str(w): list(v) for w, v in shapes.items()}}

    def keep_rows(self, scheme: str, target: Target, directory: Path) -> dict[str, Any]:
        """With the model untouched: every linear layer's weight rows written once under `directory`
        (`pearl_c.keep_rows`, each file named by the device's digests of its rows), the setting ``kept_rows`` of the
        model's later installs.  A layer the scheme refuses is the adapter's ValueError, as at install."""
        settings = self.settings(target.context.settings_of(self.name))
        _, layers, _ = self.layers(scheme, target)
        return PPD.keep_rows(PPD.Kernel.load(settings.kernel, settings.cubin), _weights(layers), scheme, Path(directory))

    def make(self, entry: str, settings: Any, s: Any, layers: list[Any], shapes: Mapping[int, tuple[int, int]],
             committed: Mapping[str, Any]) -> tuple[Any, Any, dict[str, float]]:
        """The epoch over weights committed as `committed`: its salt coin opened after their root (the settings'
        ``coins``, or the live session ``challenger`` names, which ``gate`` opens on the epoch's root) and its device
        run made under that salt.  Returns the run, the session (None for in-process coins; closed here when the run
        can't be made) and the seconds the coin and the run took."""
        roots = {int(w): bytes.fromhex(r) for w, r in committed["weight_roots"].items()}
        session, k, t0 = None, None, time.perf_counter()
        if settings.challenger is not None:
            session, commitment, opening, _, carried, k = PHW.salt_session(entry, settings, weights_root(roots), True)
        else:
            commitment, opening = settings.coins.commitment, settings.coins.open("salt", weights_root(roots))
            declared = Workload(tuple(Matmul(w, Shape(16, cols, n), w) for w, (n, cols) in shapes.items()))
            Epoch.start(declared, roots, commitment, opening)    # refuses any other coin, as every install's start does
        t1 = time.perf_counter()
        with PHW.closed_on_error(session):
            run = PPD.PearlCDeviceRun(s, entry, PPD.Kernel.load(settings.kernel, settings.cubin), _weights(layers), roots,
                                      commitment, opening, schedule=settings.schedule, salt_key=self.salt_keys().get(entry),
                                      pass_index=k, rows=settings.kept_rows)
            run.commits["registration"] = self.claim(s)
            if session is not None:
                run.commits.update(coin_source=PH.CHALLENGER, session=carried)
        return run, session, {"salt": round(t1 - t0, 4), "run": round(time.perf_counter() - t1, 4)}

    def prepare(self, scheme: str, target: Target) -> Prepared:
        """An epoch made ahead of its install, the model untouched, in the install's own order: the weights committed
        under a fresh hm96 key (`commit_weights`), then the salt coin opened after their root and the device run made
        under that salt (`make`).  The settings are the install's, less its transcript dir and runner, which only its
        hooks read.  A refusal is the install's (ValueError), with no session left open."""
        entry = f"{self.name}:{scheme}"
        settings = self.settings(target.context.settings_of(self.name))
        if scheme not in PPD.SCHEMES or settings.prepared is not None:
            raise ValueError(f"{entry}: an epoch is made ahead on Pearl-C's device path ({', '.join(PPD.SCHEMES)}), from "
                             "settings that hold none yet")
        PHS.coins_given(scheme, entry, settings)
        s, layers, shapes = self.layers(scheme, target)
        t0 = time.perf_counter()
        committed = self.commit_weights(scheme, entry, settings, layers, shapes)
        t1 = time.perf_counter()
        run, session, seconds = self.make(entry, settings, s, layers, shapes, committed)
        return Prepared(entry, tuple(m.weight for m in layers), committed, run, session,
                        {"weights": round(t1 - t0, 4), **seconds})

    def install(self, scheme: str, entry: str, settings: Any, target: Target) -> Option:
        """The adapter's `install` under Pearl-C: the epoch handed it (the setting ``prepared``), else one made now over
        the weights the composition committed, and the model's linears and its runner's steps hooked onto its run."""
        s, layers, shapes = self.layers(scheme, target)
        if settings.prepared is not None:
            run, session = settings.prepared.take(entry, layers)
        else:
            committed = (target.weight_commitments.get(entry)
                         or self.commit_weights(scheme, entry, settings, layers, shapes))
            run, session, _ = self.make(entry, settings, s, layers, shapes, committed)
        with PHW.closed_on_error(session):
            h = target.hooks(entry)
            for w, layer in enumerate(layers):
                h.wrap(layer.quant_method, "apply", lambda orig, w=w: self.executor(run, w))
            PP.hook_steps(h, run, PPD.CALL_CAPACITY, settings.transcript_dir, settings.runner)
        return Option(name=entry, hooks=h, commits=run.commits, details=run.details, end_pass=run.end_pass,
                      release=run.finish if session is None else lambda: PHW.finish(run, session))
