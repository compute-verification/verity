"""The Commit's token record (`commit/tokens_p<pair>.json`) and the link that accepts a shorter executed prefix on it.

A request that stops at EOS before its cap has an executed prefix shorter than its cap (`executed_prefix.executed_prefix_of_record`), and
the committer's own count of it is never sufficient by itself.  A full row links it to the Match account of the same attempt
(`executed_prefix.link_to_match_account`); a config run has no Match stage, so the Commit writes its own account beside the commitment:
the tokens the driver served (the engine's request outputs, recorded apart from the committed store) and, for each served token, the
identity of the committed sampled id it was served from.  `link_to_commit_account` accepts the prefix only when every served token opens
from the run root, verified, to the committed sampled id at its identity.

PROTOCOL SUMMARY (Commit stage: the execution extent check's account of record when there is no Match stage)
  Purpose: bind each request's served output, and so its executed prefix, to the commitment itself instead of to a Match record.
  Inputs: the executed prefix of record, the driver's served tokens and finish reasons, the Program's declared stop rule (EOS ids, LAG)
    and the engine's async_scheduling read-back (stop_rule_of_record), the binding map and the committer's openings (`OpenedReader`),
    the rows-past-the-stop check.
  Outputs: `commit/tokens_p<pair>.json` (`CommitTokens`, `commit_tokens_record`); `link_to_commit_account` {ok, required, problems,
    tokens_committed, served_by_request, executed_by_request, lag_attested, record}, recorded as binding.coverage.commit_account_link;
    `account_link_error`, the one decision over the Match link and this one; `request_entries`, the one reader of either record's
    requests.
  Invariants: every point of COMMIT_LINK_RULE holds per request or the link names the failure; the record's tokens are the engine's
    served output, never read back from the store; each token is opened with its Merkle path and verified against the run root;
    `executed_prefix_len` in the record is a cross-check, recomputed and never trusted; with a Match record beside the Commit both links
    must hold when the row committed its sampled ids, and a row that committed none writes no token record (its Match link alone
    decides, the token-record link recorded as not applied); a link that places no shorter prefix is recorded, not decisive.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verity_vllm.check import executed_prefix as EP
from verity_vllm.check.replay.index import SAMPLER_OP
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit.executed_prefix import ACTIVE_REFERENCE_ONLY
from verity_vllm.commit.opened import OpenedReader, UnverifiedUnlinked
from verity.ml.definitions.numerics.rows import SAMPLED_MEMBER
from verity_vllm.program.sampling_event import circuit_token_id

#: the committed family a served token is opened from
FAMILY = (SAMPLER_OP, SAMPLED_MEMBER)


class CommitTokens:
    """The Commit's token record, `commit/tokens_p<pair>.json`: the tokens the driver served, written beside the commitment, each tied to
    the committed sampled id it was served from.  The per-request names are the Match record's (capture/tokens.json), read through
    `request_entries`: the Match record's names that differ are `ALIASES`, and its requests carry no `index` (their order is the
    workload's).  Every field name the writer, the link and the tests use is one of these, so a rename is one edit."""
    SCHEMA = "verity-vllm/commit-tokens/v1"
    FILE = "tokens_p{pair}.json"
    SCHEMA_KEY, PROGRAM_DIGEST, PAIR, RUN_ROOT, TOKENS_SOURCE = "schema", "program_digest", "pair", "run_root", "tokens_source"
    STOP_RULE, EOS_IDS, LAG, LAG_VALUE, ASYNC, SOURCE = "stop_rule", "eos_token_ids", "lag", "value", "async_scheduling", "source"
    REQUESTS, REQUEST_ID, INDEX, CAP = "requests", "request_id", "index", "cap"
    GENERATED, SERVED_LEN, FINISH, EXECUTED_LEN, IDENTITIES = ("generated_token_ids", "served_output_len", "finish_reason",
                                                               "executed_prefix_len", "token_identities")
    #: the Match record's per-request name -> this record's
    ALIASES = {"max_tokens": CAP}
    IDENTITY = ("step", "op_path", "output_member", "invocation")
    SERVED_OUTPUT = "engine served output (RequestOutput.outputs[0].token_ids)"
    ENGINE_READ_BACK = "engine read-back"
    STOP, LENGTH = "stop", "length"


def commit_tokens_record(epr: dict[str, Any], *, order: dict[str, str], served_tokens: dict[str, list[int]] | None,
                         finish_reasons: dict[str, Any] | None, identities: dict[str, list[dict[str, Any] | None]], stop: dict[str, Any],
                         program_digest: str, pair: int, run_root: str) -> dict[str, Any]:
    """One pair's token record.  `served_tokens` / `finish_reasons`: {r<i>: ...} as the driver served them (the engine's request outputs,
    never the committed store); `identities`: {r<i>: [the committed sampled-id identity of request-relative step s, or None]}; `stop`:
    stop_rule_of_record (`eos`, `declared_lag`, `engine_lag`, `async_scheduling`).  `executed_prefix_len` is the Commit's own n_r, a
    cross-check that `link_to_commit_account` recomputes and never trusts."""
    C = CommitTokens
    wid = {v: k for k, v in order.items()}
    reqs = []
    for rid, rec in sorted((epr.get("requests") or {}).items(), key=lambda kv: int(kv[0][1:])):
        toks = None if served_tokens is None or served_tokens.get(rid) is None else [int(t) for t in served_tokens[rid]]
        reqs.append({C.REQUEST_ID: wid.get(rid, rid), C.INDEX: int(rid[1:]), C.CAP: int(rec["cap"]), C.GENERATED: toks,
                     C.SERVED_LEN: None if toks is None else len(toks), C.FINISH: (finish_reasons or {}).get(rid),
                     C.EXECUTED_LEN: rec.get("executed_prefix_len"), C.IDENTITIES: list(identities.get(rid) or [])})
    eos = stop.get("eos")
    return {C.SCHEMA_KEY: C.SCHEMA, C.PROGRAM_DIGEST: program_digest, C.PAIR: int(pair), C.RUN_ROOT: run_root, C.TOKENS_SOURCE: C.SERVED_OUTPUT,
            C.STOP_RULE: {C.EOS_IDS: None if eos is None else sorted(int(x) for x in eos), C.LAG: stop.get("declared_lag")},
            C.LAG: {C.LAG_VALUE: stop.get("engine_lag"), C.ASYNC: stop.get("async_scheduling"), C.SOURCE: C.ENGINE_READ_BACK},
            C.REQUESTS: reqs}


def request_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The per-request entries of a token record or a Match record (capture/tokens.json) under `CommitTokens`' names: an aliased name
    (`CommitTokens.ALIASES`) reads as this record's, unless the entry carries this record's name too, and an entry with no `index` takes its
    position in the record's request order.  A position that is not the request's is refused downstream by its request_id."""
    C = CommitTokens
    out = []
    for i, e in enumerate(record.get(C.REQUESTS) or []):
        e = dict(e)
        for alias, name in C.ALIASES.items():
            if alias in e:
                e.setdefault(name, e.pop(alias))
        e.setdefault(C.INDEX, i)
        out.append(e)
    return out


COMMIT_LINK_RULE = ("the executed prefix rests on the Commit's token record: (1) every served token is the circuit's 32-bit id zero-extended "
                    "and equals the committed sampled id opened from the run root at its identity and verified; (2) the driver served the same list; (3) the stop predicate holds under "
                    "the declared EOS ids, with a finish_reason that agrees; (4) n_r recomputed = min(cap, a_r + LAG) on an EOS stop, the cap "
                    "on a length stop; (5) LAG declared == the record's == the engine's read-back; (6) nothing committed past the stop")

#: per-request token problems named in full before the rest are counted
TOKEN_PROBLEMS_SHOWN = 3


def _commit_link_request(rid: str, rec: dict[str, Any], e: dict[str, Any], *, opened: Any, workload_id: str, arrive: int,
                         served: list[int] | None, finish: Any, eos_set: set[int] | None, lag: int | None,
                         family: tuple[str, str]) -> dict[str, Any]:
    """Points (1)-(4) of `link_to_commit_account` for one request's entry `e` of the record; `rec` = its executed_prefix_of_record entry."""
    C = CommitTokens
    probs: list[str] = []
    cap = int(rec["cap"])
    if e.get(C.REQUEST_ID) != workload_id:
        probs.append(f"{rid}: record request_id {e.get(C.REQUEST_ID)!r} != the workload's {workload_id!r}")
    if e.get(C.CAP) != cap:
        probs.append(f"{rid}: record cap {e.get(C.CAP)!r} != the declared cap {cap}")
    gen = e.get(C.GENERATED)
    if not isinstance(gen, list):
        return {"problems": probs + [f"{rid}: no generated_token_ids in the token record"], "committed": False, "served": None, "executed": None}
    gen = [int(t) for t in gen]
    a = len(gen)
    if e.get(C.SERVED_LEN) != a:
        probs.append(f"{rid}: record served_output_len {e.get(C.SERVED_LEN)!r} != its {a} generated_token_ids")
    same = served is not None and [int(t) for t in served] == gen
    if not same:
        first = None if served is None else next((i for i, (x, y) in enumerate(zip(served, gen)) if int(x) != y), min(len(served), a))
        probs.append(f"{rid}: (2) the driver served {None if served is None else len(served)} token(s), the record {a} (first differing position "
                     f"{first}): not one serving transcript")
    if finish is not None and e.get(C.FINISH) != finish:
        probs.append(f"{rid}: record finish_reason {e.get(C.FINISH)!r} != the driver's {finish!r}")
    ids = list(e.get(C.IDENTITIES) or [])
    if len(ids) != a:
        probs.append(f"{rid}: (1) {len(ids)} token identities for {a} served tokens (one per served token)")
    token_probs: list[str] = []
    got: list[int | None] = []
    for s in range(a):
        ident, step = (ids[s] if s < len(ids) else None), arrive + s
        try:
            circuit_token_id(gen[s])
        except ValueError as err:
            token_probs.append(f"{rid}: (1) served token {s}: {err}")
            got.append(None)
            continue
        if not isinstance(ident, dict) or any(ident.get(k) is None for k in C.IDENTITY):
            token_probs.append(f"{rid}: (1) served token {s} has no committed identity in the token record")
            got.append(None)
            continue
        if (ident["op_path"], ident["output_member"]) != tuple(family) or int(ident["step"]) != step:
            token_probs.append(f"{rid}: (1) served token {s}'s identity (step {ident['step']}, {ident['op_path']}/{ident['output_member']}) is not "
                               f"the sampled id of request step {s} (engine step {step}, {family[0]}/{family[1]})")
            got.append(None)
            continue
        v, why = opened(rid, ident)
        got.append(None if v is None else int(v))
        if v is None:
            token_probs.append(f"{rid}: (1) served token {s} not opened from the run root at engine step {step}: {why}")
        elif int(v) != gen[s]:
            token_probs.append(f"{rid}: (1) served token {s} is {gen[s]} but the committed sampled id at engine step {step} is {int(v)}")
    probs += token_probs[:TOKEN_PROBLEMS_SHOWN]
    if len(token_probs) > TOKEN_PROBLEMS_SHOWN:
        probs.append(f"{rid}: (1) {len(token_probs) - TOKEN_PROBLEMS_SHOWN} more served token(s) not bound to the commitment")
    bound = not token_probs and len(ids) == a
    toks = got if bound else gen
    fin = e.get(C.FINISH)
    first_eos = next((k for k, t in enumerate(toks) if t is not None and eos_set is not None and int(t) in eos_set), None)
    last_eos = a > 0 and first_eos == a - 1
    if a == 0:
        probs.append(f"{rid}: (3) served no token at all: not a stop of the Program's rule")
    elif a > cap:
        probs.append(f"{rid}: (3) served {a} tokens over the declared cap {cap}")
    elif first_eos is not None and first_eos < a - 1:
        probs.append(f"{rid}: (3) served past an EOS: token {first_eos} is the declared EOS id {toks[first_eos]} and {a - 1 - first_eos} more were "
                     "served after it (the Program's rule stops at the first EOS)")
    elif fin == C.STOP and not last_eos:
        probs.append(f"{rid}: (3) finish_reason {C.STOP!r} but the last served token {toks[-1]} is not a declared EOS id "
                     f"{sorted(eos_set)[:8] if eos_set is not None else None}")
    elif fin == C.LENGTH and a != cap:
        probs.append(f"{rid}: (3) finish_reason {C.LENGTH!r} but served {a} of its cap {cap}")
    elif fin not in (C.STOP, C.LENGTH):
        probs.append(f"{rid}: (3) finish_reason {fin!r} is neither {C.STOP!r} nor {C.LENGTH!r}: not a stop of the Program's rule")
    n = cap if a == cap else (min(cap, a + int(lag)) if lag is not None and last_eos and fin == C.STOP else None)
    if 0 < a < cap and lag is None:
        probs.append(f"{rid}: (4) stopped before its cap at {a} but the LAG is not attested: the executed prefix cannot be placed")
    if n is None or e.get(C.EXECUTED_LEN) != n or rec.get("executed_prefix_len") != n:
        probs.append(f"{rid}: (4) executed prefix recomputed {n} != the record's executed_prefix_len {e.get(C.EXECUTED_LEN)!r} / the Commit's "
                     f"{rec.get('executed_prefix_len')!r} (served {a}, cap {cap}, LAG {lag})")
    return {"problems": probs, "committed": bound and same, "served": a, "executed": n}


def link_to_commit_account(epr: dict[str, Any], record: dict[str, Any], *, opened: Any, order: dict[str, str], arrivals: dict[str, int],
                           served_tokens: dict[str, list[int]] | None, finish_reasons: dict[str, Any] | None, stop: dict[str, Any],
                           program_digest: str, run_root: str, pair: int, family: tuple[str, str], past: dict[str, Any] | None,
                           path: str | None = None, sha256: str | None = None) -> dict[str, Any]:
    """The executed prefix of record ACCEPTED on the Commit's own token record (`CommitTokens`, `commit_tokens_record`) in place of a Match
    account: every point of COMMIT_LINK_RULE, per request, each failure a problem by name.  `opened(rid, identity) -> (value | None, why)`
    reads the committed sampled id at an identity, opened from the run root and verified; `served_tokens` / `finish_reasons`: the driver's,
    {r<i>: ...}; `stop`: stop_rule_of_record; `family`: the sampled-id (op_path, member); `past`: rows_past_the_stop.  The record must be
    this attempt's (`program_digest`, `run_root`, `pair`) and its tokens the engine's served output.  `required`: some prefix was reduced
    from its cap (an EOS stop) -- only then does the link decide (`account_link_error`).  Returns {"ok", "required", "problems",
    "tokens_committed", "served_by_request", "executed_by_request", "lag_attested", "record", ...}."""
    C = CommitTokens
    problems: list[str] = []
    reduced = sorted(r for r, v in (epr.get("requests") or {}).items() if v.get("eos_stop"))
    eos_set = None if stop.get("eos") is None else {int(x) for x in stop["eos"]}
    decl = stop.get("declared_lag")
    for key, want in ((C.SCHEMA_KEY, C.SCHEMA), (C.PROGRAM_DIGEST, program_digest), (C.RUN_ROOT, run_root), (C.PAIR, pair)):
        if record.get(key) != want:
            problems.append(f"record {key} {record.get(key)!r} != {want!r}: not this attempt's token record")
    if record.get(C.TOKENS_SOURCE) != C.SERVED_OUTPUT:
        problems.append(f"record tokens_source {record.get(C.TOKENS_SOURCE)!r}: the served tokens must be the {C.SERVED_OUTPUT}, recorded apart from "
                        "the committed store (tokens read back from the store would make the check circular)")
    srule, lg = record.get(C.STOP_RULE) or {}, record.get(C.LAG) or {}
    rec_eos = srule.get(C.EOS_IDS)
    if (None if rec_eos is None else sorted(int(x) for x in rec_eos)) != (None if eos_set is None else sorted(eos_set)):
        problems.append(f"record stop_rule.eos_token_ids {rec_eos} != the Program's declared {None if eos_set is None else sorted(eos_set)}")
    legs = {"stop_rule.lag": srule.get(C.LAG), "lag.value": lg.get(C.LAG_VALUE), "the engine's read-back": stop.get("engine_lag")}
    lag_attested = (decl is not None and all(v is not None and int(v) == int(decl) for v in legs.values())
                    and lg.get(C.ASYNC) == stop.get("async_scheduling") and lg.get(C.SOURCE) == C.ENGINE_READ_BACK)
    if not lag_attested:
        problems.append(f"(5) lag not attested: the Program declares LAG {decl!r}; " + ", ".join(f"{k} {v!r}" for k, v in legs.items())
                        + f"; record async_scheduling {lg.get(C.ASYNC)!r} vs the engine's {stop.get('async_scheduling')!r}, source {lg.get(C.SOURCE)!r}")
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for e in request_entries(record):
        by_rid.setdefault(f"r{e.get(C.INDEX)}", []).append(e)
    wid = {v: k for k, v in order.items()}
    committed: dict[str, bool] = {}
    served_n: dict[str, int | None] = {}
    executed_n: dict[str, int | None] = {}
    for rid, rec in sorted((epr.get("requests") or {}).items(), key=lambda kv: int(kv[0][1:])):
        es = by_rid.get(rid) or []
        if len(es) != 1:
            problems.append(f"{rid}: {len(es)} entries in the token record (one per request)")
            committed[rid], served_n[rid], executed_n[rid] = False, None, None
            continue
        one = _commit_link_request(rid, rec, es[0], opened=opened, workload_id=wid.get(rid, rid), arrive=int(arrivals.get(rid, 0)),
                                   served=None if served_tokens is None else served_tokens.get(rid), finish=(finish_reasons or {}).get(rid),
                                   eos_set=eos_set, lag=(int(decl) if lag_attested else None), family=family)
        problems += one["problems"]
        committed[rid], served_n[rid], executed_n[rid] = one["committed"], one["served"], one["executed"]
    if not isinstance(past, dict) or past.get("ok") is not True or int(past.get("entries_n") or 0):
        problems.append(f"(6) {(past or {}).get('error') or 'rows_past_the_stop not run'}")
    return {"rule": COMMIT_LINK_RULE, "required": bool(reduced), "reduced_requests": reduced,
            "record": {"path": path, "sha256": sha256, "schema": record.get(C.SCHEMA_KEY), "requests_n": len(record.get(C.REQUESTS) or [])},
            "tokens_committed": committed, "served_by_request": served_n, "executed_by_request": executed_n, "lag_attested": lag_attested,
            "problems": problems, "ok": not problems, "classification": ACTIVE_REFERENCE_ONLY}


def account_link_error(match_link: dict[str, Any], commit_link: dict[str, Any] | None) -> str | None:
    """The one decision over the executed prefix's two account links.  With a Match record beside the Commit (`match_dir`) the Match link
    must hold, and so must the commit link when it places a prefix; with none, the commit link is the account.  A commit link that places
    nothing (every request ran its cap, or it was not applied) is recorded, not decisive.  With no commit link at all the Match link
    decides alone."""
    errs = []
    if (match_link.get("match_dir") or commit_link is None) and not match_link.get("ok"):
        errs.append("executed prefix not linked to the Match account: " + "; ".join(match_link.get("problems", [])[:3]))
    if commit_link is not None and commit_link.get("required") and not commit_link.get("ok"):
        errs.append("executed prefix not bound to the commitment by the Commit's token record: " + "; ".join(commit_link.get("problems", [])[:3]))
    return " | ".join(errs) or None


def sampled_ids_committed(bmap: dict[str, Any] | None) -> bool:
    """Whether the binding map binds a committed sampled id (`FAMILY`) at all: a row that committed none has no token to open."""
    return any(((e.get("identity") or {}).get("op_path"), (e.get("identity") or {}).get("output_member")) == FAMILY
               for e in (bmap or {}).get("entries") or [])


def sampled_identities(store: Any, rid: str, arrive: int, n: int) -> list[dict[str, Any] | None]:
    """The committed identity of request `rid`'s sampled id at request-relative steps 0..n-1 (engine step arrive + s) in the binding
    map `store` indexes, or None where nothing is bound there."""
    out: list[dict[str, Any] | None] = []
    for s in range(n):
        es, canon, inv, _bound = store.entries(rid, arrive + s, SAMPLER_OP, SAMPLED_MEMBER)
        out.append(dict(zip(CommitTokens.IDENTITY, (arrive + s, canon, SAMPLED_MEMBER, inv))) if es else None)
    return out


def opener(store: Any) -> Callable[[str, dict[str, Any]], tuple[int | None, str]]:
    """`opened(rid, identity) -> (value | None, why)`: the committed sampled id at an identity of the record, read through `store` (a
    `CommittedStore` over an `OpenedReader`: the leaves are opened and verified against the run root before a byte is returned; a read
    that does not verify is None by name, as is a committed word that is not a 32-bit id zero-extended).  The identity's invocation must
    be the one its op_path resolves to."""
    def opened(rid: str, ident: dict[str, Any]) -> tuple[int | None, str]:
        _canon, inv = store.resolve(str(ident["op_path"]))
        if int(ident["invocation"]) != inv:
            return None, f"invocation {ident['invocation']} is not the one {ident['op_path']} resolves to ({inv})"
        w, why = store.words(rid, int(ident["step"]), str(ident["op_path"]), str(ident["output_member"]), 0, 1)
        if w is None:
            return None, why
        try:
            return circuit_token_id(np.asarray(w).astype(np.int64)[0]), "opened"
        except ValueError as e:
            return None, f"the committed sampled id: {e}"
    return opened


def commit_link(com: Any, rc: Any, bmap: dict[str, Any], epr: dict[str, Any], *, out_dir: str | None, pair: int, order: dict[str, str],
                arrivals: dict[str, int], stop: dict[str, Any], served: dict[str, list[int]] | None, finish: dict[str, Any] | None,
                past: dict[str, Any], program_digest: str) -> dict[str, Any]:
    """Writes this pair's token record into `out_dir` and links the executed prefix to it, opening every served token from the
    committer `com` against its run commitment `rc` through the binding map."""
    store = UnverifiedUnlinked(CommittedStore(bmap, OpenedReader(com, rc)))
    ids = {rid: sampled_identities(store, rid, int(arrivals.get(rid, 0)), len((served or {}).get(rid) or [])) for rid in epr.get("requests") or {}}
    run_root = bytes(rc.run_root).hex()
    record = commit_tokens_record(epr, order=order, served_tokens=served, finish_reasons=finish, identities=ids, stop=stop,
                                  program_digest=program_digest, pair=pair, run_root=run_root)
    raw = (json.dumps(record, indent=1) + "\n").encode()
    path = None
    if out_dir:
        path = Path(out_dir) / CommitTokens.FILE.format(pair=pair)
        path.write_bytes(raw)
    return link_to_commit_account(epr, json.loads(raw), opened=opener(store), order=order, arrivals=arrivals, served_tokens=served,
                                  finish_reasons=finish, stop=stop, program_digest=program_digest, run_root=run_root, pair=pair, family=FAMILY,
                                  past=past, path=None if path is None else str(path), sha256=hashlib.sha256(raw).hexdigest())


def account_links(com: Any, rc: Any, bmap: dict[str, Any], epr: dict[str, Any], *, out_dir: str | None, pair: int, order: dict[str, str],
                  arrivals: dict[str, int], match_dir: str | None, stop: dict[str, Any], served: dict[str, list[int]] | None,
                  finish: dict[str, Any] | None, past: dict[str, Any], program_digest: str) -> dict[str, Any]:
    """Both account links of the executed prefix at the binding map, and the decision: {"match_account_link", "commit_account_link",
    "error"} (`error` None = accepted).  A token record that cannot be built is a commit link with that problem, never a silent fallback.
    A row with a Match record beside it that committed no sampled id writes no token record: its commit link is `not_applied`, by name,
    and places nothing, so the Match link alone decides.  With no Match record the token record is built either way: it is the account."""
    match = EP.link_to_match_account(epr, EP.load_match_account(match_dir), order=order, declared_lag=stop.get("declared_lag"), served_tokens=served)
    reduced = sorted(r for r, v in (epr.get("requests") or {}).items() if v.get("eos_stop"))
    if match_dir and not sampled_ids_committed(bmap):
        commit = {"rule": COMMIT_LINK_RULE, "required": False, "reduced_requests": reduced, "record": None, "lag_attested": None, "problems": [],
                  "ok": None, "not_applied": f"no committed {FAMILY[0]}/{FAMILY[1]} in the binding map: the Match link alone decides",
                  "classification": ACTIVE_REFERENCE_ONLY}
    else:
        try:
            commit = commit_link(com, rc, bmap, epr, out_dir=out_dir, pair=pair, order=order, arrivals=arrivals, stop=stop, served=served,
                                 finish=finish, past=past, program_digest=program_digest)
        except Exception as e:  # noqa: BLE001
            commit = {"rule": COMMIT_LINK_RULE, "required": bool(reduced), "reduced_requests": reduced, "record": None, "lag_attested": False,
                      "problems": [f"token record not built: {e!r}"[:300]], "ok": False, "classification": ACTIVE_REFERENCE_ONLY}
    error = account_link_error(match, commit)
    for name, ln in (("Match account", match), ("Commit's token record", commit)):
        if ln.get("not_applied"):
            print(f"[delta] execution_extent: executed prefix <-> {name} (pair {pair}): not applied ({ln['not_applied']})", flush=True)
        elif ln["required"] or ln["problems"]:
            print(f"[delta] execution_extent: executed prefix <-> {name} (pair {pair}): {'LINKED' if ln['ok'] else 'NOT LINKED'}"
                  + (f"; PROBLEMS {ln['problems'][:3]}" if ln["problems"] else ""), flush=True)
    return {"match_account_link": match, "commit_account_link": commit, "error": error}
