"""`research notes claim | approach | approaches`: the approach registry (the notes' kb/onboarding.md is the how-to).

  research notes claim C/SLUG --lane L [--title T --hypothesis H --type T [--attacks C/SLUG]] [--cite TOKEN]... [--reopen WHY]
  research notes approach C/SLUG STATUS --by L [--reason R] [--cite TOKEN]... [--superseded-by C/SLUG] [--owner L] [--at UTC]
                          [--title T --hypothesis H --type T [--attacks C/SLUG]]      # a new name: the record of work already done
  research notes approach C/SLUG --by L [--cite TOKEN]... [--uncite TOKEN]...         # no status: its evidence alone
  research notes approaches [--campaign C] [--status S[,S]] [--owner L] [--grep WORDS] [--refresh] [--json]
  research notes approaches render [--campaign C] [--refresh] [--root DIR]            # campaigns/<c>/APPROACHES.md in the notes
  research notes approaches check [--refresh] [--root DIR]                            # the lint: exit 1 on any error

An approach is a line of work with a hypothesis, an owner lane and a status.  It is a record in the evidence store, kind
`approach/v1`, meta `{campaign, slug}`, no payload, so its `art:` id is a function of its name `<campaign>/<slug>`: the same name
is the same record everywhere.  Every fact about it is a label on that record (store README §8), in the `approach` vocabulary group:

    title, hypothesis   one line each                     approach_type    scheme | route | attack
    owner               the lane that owns it             approach_status  live | parked | killed | superseded | merged
    reason              one line: why the status changed (required unless live)
    superseded_by       the <campaign>/<slug> that replaces it          attacks   the <campaign>/<slug> an attack targets
    cites               one evidence token per label (EVIDENCE_RE)      approach  on a run or artifact: the approach it belongs to
    uncites             one token the approach no longer cites

The status describes the entry's own line of work: `live` its owner works on it now; `parked` nobody does, it is not refuted, anyone
may claim it; `killed` it cannot work (an attack that found nothing is killed: "target held"); `superseded` replaced; `merged` done
and adopted (landed in main, chosen, or an attack's finding accepted and acted on).

The state is a fold over the labels in ts order, one change per (ts, by): the newest value of each key wins, `cites` accumulate
until an `uncites` takes one back, and a change that makes a live approach live under another lane without a `reason` is ignored
(the loser of a claim race; `check` warns about it).  Every command refuses (exit 2, nothing written) what the fold would reject: a
missing field, a non-live status without a reason, a merged one without evidence, a `superseded_by` / `attacks` that names no
approach, an evidence token of no known form, a field over `MAX_FIELD` characters or with a line break.  Claims and changes go through to the remote at once (a
claim nobody can see is not a claim), and `claim` re-reads the remote after writing, exiting 3 when another claim landed first.
Without a usable remote, `claim` writes a pending claim file into the lane's own folder (`lanes/<lane>/<stamp>-claim-….md`, exit
4) and the steward's rule records it at its next pass, handing a refusal back to the lane.

The public view is one generated file per campaign in the notes, `campaigns/<c>/APPROACHES.md`: `render` writes it (the steward's
`[approaches]` rule, every `every_min`), and the campaign's `BRIEF.md` front matter `registry: full | titles` says what it shows:
`titles` (the default) only the title, status, owner, dates and evidence; `full` also the hypothesis and the reason.  The notes
repo is public; the store is not.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .store import labels_sync as LS
from .store import vocab
from .store.local import LocalStore, manifest_key
from .store.model import ArtifactManifest, Label, utc_now

KIND = "approach/v1"
STATUSES = vocab.APPROACH_STATUSES
TYPES = vocab.APPROACH_TYPES
LIVE, PARKED, KILLED, SUPERSEDED, MERGED = STATUSES
ATTACK = "attack"
MAX_FIELD = 240
REGISTRY_FILE = "APPROACHES.md"
BRIEF = "BRIEF.md"
REGISTRY_MODES = ("titles", "full")
NAME_RE = re.compile(r"^(?P<campaign>[a-z0-9][a-z0-9-]{0,39})/(?P<slug>[a-z0-9][a-z0-9-]{0,63})$")
LANE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
EVIDENCE_RE = re.compile(
    r"^(?:r\d{8}-\d{6}-[0-9a-f]{4}"                         # a run (attempt) id
    r"|(?:art|drv):[0-9a-f]{8,64}"                          # a store object, full or abbreviated
    r"|(?:note|notes-asset):\S+"                            # a notes id or a notes asset path
    r"|approach:[a-z0-9][a-z0-9-]{0,39}/[a-z0-9][a-z0-9-]{0,63}"
    r"|[A-Za-z0-9_.-]+#\d+"                                 # a pull request or issue: <repo>#<n>
    r"|[A-Za-z0-9_.-]+@[0-9a-f]{7,40}"                      # a commit: <repo>@<sha>
    r"|https://\S+)$")
AT_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?Z$")
#: Seconds a claim waits before re-reading the remote: a rival claim stamped in the same second (ties break by lane name) or
#: earlier is visible by then unless its upload took longer, so both claimants read the same winner.
CLAIM_SETTLE_S = 2.0


class Refused(Exception):
    """A write the registry's rules reject; nothing was written."""


class NoRemote(Exception):
    """No usable remote: a claim nobody else can see is not a claim (the CLI falls back to a pending claim file)."""


@dataclass
class Approach:
    name: str
    art: str
    title: str | None = None
    hypothesis: str | None = None
    type: str | None = None
    attacks: str | None = None
    owner: str | None = None
    status: str | None = None
    reason: str | None = None
    superseded_by: str | None = None
    cites: list[str] = field(default_factory=list)
    claimed: str | None = None
    updated: str | None = None
    history: list[dict] = field(default_factory=list)
    ignored: list[dict] = field(default_factory=list)

    @property
    def campaign(self) -> str:
        return self.name.split("/", 1)[0]


# ---- names and records -----------------------------------------------------------------------------------------------------------

def parse_name(name: str) -> tuple[str, str]:
    m = NAME_RE.match(name or "")
    if not m:
        raise Refused(f"{name!r} is not an approach name: <campaign>/<slug>, lowercase letters, digits and '-'")
    return m["campaign"], m["slug"]


def manifest(name: str) -> ArtifactManifest:
    campaign, slug = parse_name(name)
    return ArtifactManifest(kind=KIND, meta={"campaign": campaign, "slug": slug})


def record_id(name: str) -> str:
    """The approach's `art:` id: a function of its name alone."""
    return manifest(name).id


def exists(store: LocalStore, name: str) -> bool:
    """Whether the record is here or on the remote (one HEAD; no refresh)."""
    art = record_id(name)
    if store.index.artifact(art) is not None:
        return True
    r = store.remote
    return bool(r is not None and r.head(manifest_key(art.split(":", 1)[1])))


# ---- the fold --------------------------------------------------------------------------------------------------------------------

def _changes(labels: Iterable[Label]) -> list[tuple[str, str, dict[str, list]]]:
    """(ts, by, {key: [values]}) per change, in ts order (a tie by lane name); one command's labels share ts and by."""
    out: dict[tuple[str, str], dict[str, list]] = {}
    for lab in sorted(labels, key=lambda x: (x.ts, x.by)):
        out.setdefault((lab.ts, lab.by), {}).setdefault(lab.key, []).append(lab.value)
    return [(ts, by, kv) for (ts, by), kv in sorted(out.items())]


def handover(owner: str | None, new_owner: str, by: str, reason: str | None) -> bool:
    """Whether a live -> live change stands: the same owner, or a hand-over with a reason written by someone other than the new
    owner (the owner itself, or a coordinator).  A lane that makes itself the owner of someone else's live approach lost a race."""
    return new_owner == owner or (bool(reason) and by != new_owner)


def fold(name: str, labels: Iterable[Label]) -> Approach:
    a = Approach(name=name, art=record_id(name))
    for ts, by, kv in _changes(labels):
        one = {k: v[-1] for k, v in kv.items()}
        new_owner = one.get(vocab.OWNER, a.owner if a.owner is not None else by)
        if a.status == LIVE and one.get(vocab.APPROACH_STATUS) == LIVE and not handover(a.owner, new_owner, by, one.get(vocab.REASON)):
            a.ignored.append({"ts": ts, "by": by, "owner": new_owner})
            continue
        for key, attr in ((vocab.TITLE, "title"), (vocab.HYPOTHESIS, "hypothesis"), (vocab.APPROACH_TYPE, "type"),
                          (vocab.ATTACKS, "attacks"), (vocab.OWNER, "owner"), (vocab.SUPERSEDED_BY, "superseded_by")):
            if key in one:
                setattr(a, attr, one[key])
        a.cites += [c for c in kv.get(vocab.CITES, []) if c not in a.cites]
        a.cites = [c for c in a.cites if c not in kv.get(vocab.UNCITES, ())]
        if vocab.APPROACH_STATUS in one:
            a.status, a.reason, a.updated = one[vocab.APPROACH_STATUS], one.get(vocab.REASON), ts
            if a.owner is None:
                a.owner = by
            a.claimed = a.claimed or ts
            a.history.append({"ts": ts, "status": a.status, "by": by, "owner": a.owner, "reason": a.reason})
        elif vocab.REASON in one:
            a.reason = one[vocab.REASON]
    return a


def problems(a: Approach, known=lambda name: True) -> list[str]:
    """Why the folded state is not a valid registry entry, one line each (empty when it is)."""
    out = []
    for attr in ("title", "hypothesis", "type", "owner", "status"):
        if not getattr(a, attr):
            out.append(f"no {attr}")
    for key, attr in ((vocab.TITLE, "title"), (vocab.HYPOTHESIS, "hypothesis"), (vocab.REASON, "reason")):
        v = getattr(a, attr)
        if isinstance(v, str) and (len(v) > MAX_FIELD or "\n" in v):
            out.append(f"{key} must be one line of at most {MAX_FIELD} characters (details go in a note; cite it)")
    if a.status and a.status not in STATUSES:
        out.append(f"status {a.status!r} is not one of {', '.join(STATUSES)}")
    if a.type and a.type not in TYPES:
        out.append(f"type {a.type!r} is not one of {', '.join(TYPES)}")
    if a.status and a.status != LIVE and not a.reason:
        out.append(f"{a.status} needs a reason")
    if a.owner and not LANE_RE.match(a.owner):
        out.append(f"owner {a.owner!r} is not a lane name")
    if a.status == SUPERSEDED and not a.superseded_by:
        out.append("superseded needs superseded_by")
    if a.status == MERGED and not a.cites:
        out.append("merged needs evidence: cite the PR, commit, note or artifact that shows it")
    if a.type == ATTACK and not a.attacks:
        out.append("an attack needs attacks (its target)")
    if a.attacks and a.type != ATTACK:
        out.append("attacks is for an approach of type attack")
    for key, ref in ((vocab.SUPERSEDED_BY, a.superseded_by), (vocab.ATTACKS, a.attacks)):
        if ref and (not NAME_RE.match(ref) or ref == a.name or not known(ref)):
            out.append(f"{key} {ref!r} names no other approach")
    for c in a.cites:
        if str(c).startswith("store:"):
            out.append(f"evidence {c!r} is a private agent store's path: copy it into the notes, then --cite note:ID --uncite {c}")
        elif not EVIDENCE_RE.match(str(c)):
            out.append(f"evidence {c!r} is of no known form")
    return out


# ---- reading ---------------------------------------------------------------------------------------------------------------------

def current_labels(store: LocalStore, art: str) -> list[Label]:
    """Every assertion about the record on either side; remote-only ones land in this store too (the claim reads the remote)."""
    if store.remote is None:
        return store.labels(art)
    out = []
    for row in LS.compare(store, art):
        if row["label"] is None or row.get("error"):
            continue
        lab = Label.from_doc(row["label"])
        if row["where"] == "remote":
            LS.write_label_file(store, lab)
        out.append(lab)
    return out


def load(store: LocalStore) -> list[Approach]:
    """Every approach this store's catalog knows (after `refresh`, every one on the remote), folded, sorted by name."""
    out = []
    for row in store.index.select_artifacts(kind=KIND):
        meta = (row.get("manifest") or {}).get("meta") or {}
        name = f"{meta.get('campaign')}/{meta.get('slug')}"
        if NAME_RE.match(name) and record_id(name) == row["id"]:
            out.append(fold(name, store.labels(row["id"])))
    return sorted(out, key=lambda a: a.name)


def _words(*texts: str | None) -> set[str]:
    return {w for t in texts if t for w in re.findall(r"[a-z0-9]{3,}", t.lower().replace("-", " "))}


def similar(approaches: list[Approach], title: str, hypothesis: str, n: int = 5) -> list[Approach]:
    mine = _words(title, hypothesis)
    scored = [(len(mine & _words(a.name.split("/")[1], a.title, a.hypothesis)), a) for a in approaches]
    return [a for s, a in sorted(scored, key=lambda t: -t[0]) if s >= 2][:n]


# ---- writing ---------------------------------------------------------------------------------------------------------------------

def _ts(at: str | None, labels: list[Label]) -> str:
    """The change's ts: strictly after the approach's newest label, so the fold orders it after everything its writer saw."""
    newest = max((x.ts for x in labels), default="")
    if at is None:
        ts = utc_now()
        if ts <= newest:
            ts = (dt.datetime.strptime(newest, "%Y-%m-%dT%H:%M:%SZ") + dt.timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return ts
    m = AT_RE.match(at)
    if not m:
        raise Refused(f"--at {at!r}: YYYY-MM-DDTHH:MM[:SS]Z")
    ts = f"{m[1]}T{m[2]}:{m[3]}:{m[4] or '00'}Z"
    if ts > utc_now():
        raise Refused(f"--at {at} is in the future")
    if labels and ts <= max(x.ts for x in labels):
        raise Refused(f"--at {at} is not after the approach's newest label ({max(x.ts for x in labels)}): history is append-only")
    return ts


def _write(store: LocalStore, name: str, labels: list[Label], change: dict[str, object], by: str, at: str | None,
           cites: list[str], uncites: list[str] = ()) -> Approach:
    """Validate the folded state with `change` applied, then put the record and the change's labels and push them."""
    if not LANE_RE.match(by or ""):
        raise Refused(f"{by!r} is not a lane name")
    if both := set(cites) & set(uncites):
        raise Refused(f"{min(both)!r} is both cited and uncited")
    cited = fold(name, labels).cites if labels else []
    if missing := [c for c in uncites if c not in cited]:
        raise Refused(f"{name} does not cite {missing[0]!r}")
    ts = _ts(at, labels)
    ref = cites[0] if cites else None
    new = [Label(target=record_id(name), key=k, value=v, by=by, ts=ts, ref=ref) for k, v in change.items() if v is not None]
    new += [Label(target=record_id(name), key=vocab.CITES, value=c, by=by, ts=ts, ref=ref) for c in cites]
    new += [Label(target=record_id(name), key=vocab.UNCITES, value=c, by=by, ts=ts, ref=ref) for c in uncites]
    for lab in new:
        if why := vocab.check(lab.key, lab.value):
            raise Refused(why)
    after = fold(name, labels + new)
    if bad := problems(after, known=lambda n: n != name and NAME_RE.match(n) is not None and exists(store, n)):
        raise Refused(f"{name}: " + "; ".join(bad))
    m = manifest(name)
    art = store.put_manifest(m)
    for lab in new:
        store.add_label(lab.target, lab.key, lab.value, lab.by, ref=lab.ref, ts=lab.ts)
    if store.remote is not None and store.write_through:
        store.push(art)
        if failed := LS.push_targets(store, [art])["failed"]:
            raise RuntimeError(f"written here but NOT on the remote ({failed[0]}): `research data push --pending` uploads it")
    return after


def pending_claim(root: Path, name: str, lane: str, fields: dict[str, object], why: str) -> Path:
    """The folder fallback: without a usable remote, the claim goes into the lane's own folder as `<stamp>-claim-….md`, which
    the steward's rule records in the store at its next pass (a pending claim is not a claim until then)."""
    parse_name(name)
    if not LANE_RE.match(lane or ""):
        raise Refused(f"{lane!r} is not a lane name")
    for k, v in fields.items():
        if isinstance(v, str) and (len(v) > MAX_FIELD or "\n" in v):
            raise Refused(f"{k} must be one line of at most {MAX_FIELD} characters")
    bad = [c for c in fields.get("cites") or [] if not EVIDENCE_RE.match(str(c))]
    if bad:
        raise Refused(f"evidence {bad[0]!r} is of no known form")
    now = dt.datetime.now(dt.timezone.utc)
    d = root / "lanes" / lane
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{now:%Y%m%dT%H%M%SZ}-claim-{name.replace('/', '--')}.md"
    front = {"kind": "claim", "approach": name, "lane": lane, "created": f"{now:%Y-%m-%dT%H:%M:%SZ}",
             **{k: (" ".join(v) if isinstance(v, list) else v) for k, v in fields.items() if v}}
    p.write_text("---\n" + "".join(f"{k}: {v}\n" for k, v in front.items()) + "---\n\n"
                 f"Pending claim of `{name}` by `{lane}`: `research notes claim` found no usable remote ({why}). The steward records "
                 f"it at its next pass; until `research notes approaches` shows it, it is not a claim.\n")
    return p


def read_pending(p: Path) -> dict[str, str]:
    text = p.read_text(errors="replace")
    end = text.find("\n---\n", 4)
    if not text.startswith("---\n") or end < 0:
        return {}
    return {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in text[4:end].splitlines()) if k.strip()}


def import_pending(store: LocalStore, root: Path, now: dt.datetime, done: list[str]) -> list[str]:
    """Record every pending claim file not yet in `done` (the steward's memory); one line each, and a handoff to a refused lane."""
    lines = []
    for p in sorted((root / "lanes").glob("*/*Z-claim-*--*.md")):
        rel = str(p.relative_to(root))
        if rel in done:
            continue
        done.append(rel)
        f = read_pending(p)
        try:
            rc, line = claim(store, f.get("approach", ""), f.get("lane", ""), title=f.get("title"), hypothesis=f.get("hypothesis"),
                             type_=f.get("type"), attacks=f.get("attacks"), cites=(f.get("cites") or "").split(), reopen=f.get("reopen"))
        except Refused as e:
            rc, line = 2, f"refused: {e}"
        lines.append(f"{now:%H:%MZ} {'CLAIM' if rc == 0 else 'CLAIM-REFUSED'} {rel}: {line}")
        if rc != 0 and LANE_RE.match(f.get("lane", "")):
            h = root / "lanes" / f["lane"] / f"{now:%Y%m%dT%H%MZ}-handoff-steward-claim.md"
            h.write_text(f"---\nlane: {f['lane']}\nkind: handoff\nfrom: steward\ncreated: {now:%Y-%m-%dT%H:%MZ}\n---\n\n"
                         f"# your pending claim of {f.get('approach')} was not recorded: {line}\n\nFrom `{rel}`.\n")
    return lines


def claim(store: LocalStore, name: str, lane: str, *, title: str | None = None, hypothesis: str | None = None,
          type_: str | None = None, attacks: str | None = None, cites: list[str] = (), reopen: str | None = None) -> tuple[int, str]:
    """(exit code, one line).  0 claimed (or already yours), 2 refused, 3 lost a race to another claim."""
    if store.remote is None or not store.write_through:
        raise NoRemote("no usable remote with write-through (" + (getattr(store, "remote_error", None) or "write-through off") + ")")
    art = record_id(name)
    labels = current_labels(store, art)
    cur = fold(name, labels) if labels else None
    if cur and cur.status == LIVE and cur.owner != lane:
        raise Refused(f"{name} is live under {cur.owner} since {cur.updated}: write to lanes/{cur.owner}/ (or the coordinator) first")
    if cur and cur.status == LIVE:
        return 0, f"{name} is already yours ({lane}), live since {cur.updated}"
    if cur and cur.status in (KILLED, SUPERSEDED, MERGED) and not reopen:
        why = f" (superseded by {cur.superseded_by})" if cur.superseded_by else ""
        raise Refused(f"{name} is {cur.status}{why}: {cur.reason}.  Reopen it with --reopen WHY (what differs this time)")
    if not cur and not (title and hypothesis and type_):
        raise Refused(f"{name} is new: give --title, --hypothesis and --type ({' | '.join(TYPES)})")
    change = {vocab.APPROACH_STATUS: LIVE, vocab.OWNER: lane, vocab.REASON: reopen, vocab.TITLE: title,
              vocab.HYPOTHESIS: hypothesis, vocab.APPROACH_TYPE: type_, vocab.ATTACKS: attacks}
    _write(store, name, labels, change, lane, None, list(cites))
    time.sleep(CLAIM_SETTLE_S)
    won = fold(name, current_labels(store, art))
    if won.owner != lane or won.status != LIVE:
        return 3, f"lost {name} to {won.owner} (claimed {won.updated}): your claim stays in the history, ignored; talk to lanes/{won.owner}/"
    return 0, f"claimed {name} for {lane} (live); cite approach:{name} in your checkpoint"


def set_status(store: LocalStore, name: str, status: str | None, by: str, *, reason: str | None = None, owner: str | None = None,
               superseded_by: str | None = None, cites: list[str] = (), at: str | None = None, title: str | None = None,
               hypothesis: str | None = None, type_: str | None = None, attacks: str | None = None,
               uncites: list[str] = ()) -> Approach:
    """A change of status, or with `status` None a change of evidence alone (`cites`, `uncites`), which leaves the status, its
    reason and the history as they are."""
    if status is not None and status not in STATUSES:
        raise Refused(f"status {status!r} is not one of {', '.join(STATUSES)}")
    art = record_id(name)
    labels = current_labels(store, art)
    cur = fold(name, labels) if labels else None
    if status is None:
        if cur is None:
            raise Refused(f"{name} is not recorded: give its status")
        if any((reason, owner, superseded_by, title, hypothesis, type_, attacks)):
            raise Refused("without a status a change carries only --cite and --uncite")
        if not (cites or uncites):
            raise Refused("nothing to change: give a status, --cite or --uncite")
        return _write(store, name, labels, {}, by, at, list(cites), list(uncites))
    if cur is None and not (title and hypothesis and type_):
        raise Refused(f"{name} is new: give --title, --hypothesis and --type to record it (or `research notes claim` it)")
    if cur and cur.status == LIVE and status == LIVE and not handover(cur.owner, owner or cur.owner, by, reason):
        raise Refused(f"{name} is live under {cur.owner}: its owner or a coordinator hands it over (--owner NEW --reason WHY)")
    change = {vocab.APPROACH_STATUS: status, vocab.REASON: reason, vocab.OWNER: owner or (None if cur else by),
              vocab.SUPERSEDED_BY: superseded_by, vocab.TITLE: title, vocab.HYPOTHESIS: hypothesis, vocab.APPROACH_TYPE: type_,
              vocab.ATTACKS: attacks}
    return _write(store, name, labels, change, by, at, list(cites), list(uncites))


# ---- the lint --------------------------------------------------------------------------------------------------------------------

def brief_mode(root: Path, campaign: str) -> tuple[str | None, str | None]:
    """(registry mode, error) from campaigns/<c>/BRIEF.md's front matter; (None, why) when there is no BRIEF."""
    p = root / "campaigns" / campaign / BRIEF
    if not p.is_file():
        return None, f"campaigns/{campaign}/{BRIEF} is missing: a campaign with approaches needs its brief"
    text = p.read_text(errors="replace")
    front = text[4:text.find("\n---", 4)] if text.startswith("---\n") and text.find("\n---", 4) > 0 else ""
    m = re.search(r"^registry:\s*(\S+)\s*$", front, re.M)
    mode = m.group(1) if m else REGISTRY_MODES[0]
    if mode not in REGISTRY_MODES:
        return None, f"campaigns/{campaign}/{BRIEF}: registry: {mode!r} is not one of {', '.join(REGISTRY_MODES)}"
    return mode, None


def orphaned(root: Path, a: Approach) -> str | None:
    """Why a live approach has no live owner (its lane, or the lane that took it over, is done), or None."""
    from . import notes  # noqa: PLC0415

    lane = notes.latest(a.owner, notes.successors(root))
    if not (root / "lanes" / lane).is_dir():
        return f"owner lane {lane} has no folder lanes/{lane}/"
    rep = notes.report_for(root, lane)
    if rep is None:                                         # a coordinator's folder: handoffs, no report
        return None
    st = notes.lane_status(rep, notes._utcnow(), 30)  # noqa: SLF001
    return f"owner lane {lane} is {st['state']}" if st["state"] in notes.DONE else None


def check(store: LocalStore, root: Path | None) -> tuple[list[str], list[str]]:
    """(errors, warnings) over every approach record the catalog knows, and each campaign's brief when `root` is given."""
    errors, warnings = [], []
    rows = store.index.select_artifacts(kind=KIND)
    names = set()
    for row in rows:
        meta = (row.get("manifest") or {}).get("meta") or {}
        name = f"{meta.get('campaign')}/{meta.get('slug')}"
        if set(meta) != {"campaign", "slug"} or not NAME_RE.match(name) or record_id(name) != row["id"]:
            errors.append(f"{row['id']}: an {KIND} record's meta must be exactly {{campaign, slug}} of a valid name")
        else:
            names.add(name)
    approaches = load(store)
    for a in approaches:
        errors += [f"{a.name}: {p}" for p in problems(a, known=lambda n: n in names and n != a.name)]
        warnings += [f"{a.name}: ignored a live claim by {x['by']} at {x['ts']} (live under {a.owner})" for x in a.ignored]
        if root is not None and a.status == LIVE and a.owner and (why := orphaned(root, a)):
            warnings.append(f"{a.name}: ORPHAN, live but {why}: park it or hand it over")
    if root is not None:
        for c in sorted({a.campaign for a in approaches}):
            if (err := brief_mode(root, c)[1]) is not None:
                errors.append(err)
    return errors, warnings


# ---- the public view -------------------------------------------------------------------------------------------------------------

def _day(ts: str | None) -> str:
    return (ts or "?")[:10]


def render_campaign(campaign: str, approaches: list[Approach], mode: str) -> str:
    rows = [a for a in approaches if a.campaign == campaign]
    counts = " · ".join(f"{s} {n}" for s in STATUSES if (n := sum(a.status == s for a in rows)))
    asof = max((a.updated or "" for a in rows), default="")
    lines = [f"<!-- generated by `research notes approaches render` from the evidence store; do not edit (claim / approach change it) -->",
             f"# Approaches: {campaign}", "",
             f"As of {asof or 'never'}; registry: {mode} (`campaigns/{campaign}/{BRIEF}`); {counts or 'none yet'}.",
             "Live view: `research notes approaches --campaign " + campaign + " --refresh`.  Before starting: `research notes claim`.", ""]
    for s in STATUSES:
        group = [a for a in rows if a.status == s]
        if not group:
            continue
        lines += [f"## {s}", ""]
        for a in group:
            lines.append(f"- **{a.name}** ({a.type}) · owner `{a.owner}` · since {_day(a.claimed)}, updated {_day(a.updated)} · {a.title}")
            if a.attacks:
                lines.append(f"  - attacks `{a.attacks}`")
            if mode == "full" and a.hypothesis:
                lines.append(f"  - hypothesis: {a.hypothesis}")
            if mode == "full" and a.reason and a.status != LIVE:
                lines.append(f"  - why: {a.reason}")
            if a.superseded_by:
                lines.append(f"  - superseded by `{a.superseded_by}`")
            if a.cites:
                lines.append("  - evidence: " + " ".join(f"`{c}`" for c in a.cites))
        lines.append("")
    return "\n".join(lines)


def render(store: LocalStore, root: Path, campaign: str | None = None) -> list[Path]:
    """Write campaigns/<c>/APPROACHES.md for every campaign with approaches (or one); returns the files whose content changed."""
    approaches = load(store)
    changed = []
    for c in sorted({a.campaign for a in approaches if campaign in (None, a.campaign)}):
        mode = brief_mode(root, c)[0] or REGISTRY_MODES[0]
        p = root / "campaigns" / c / REGISTRY_FILE
        text = render_campaign(c, approaches, mode)
        if not p.is_file() or p.read_text(errors="replace") != text:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            changed.append(p)
    return changed


def steward_lines(root: Path, now: dt.datetime, state: dict, config: Path | None = None) -> list[str]:
    """The steward's `[approaches]` rule (steward.toml: `store`, `every_min` = 10): refresh that store from the remote, then render
    every campaign's APPROACHES.md for the watch's `--sync` to commit.  A line only when a file changed or the rule failed."""
    import tomllib  # noqa: PLC0415

    path = config or root / "steward.toml"
    try:
        conf = tomllib.loads(path.read_text()).get("approaches") if path.is_file() else None
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [f"{now:%H:%MZ} APPROACHES-FAILED {path.name}: {e}"]
    if not isinstance(conf, dict):
        return []
    last = state.get("approaches_at")
    if last is not None and now.timestamp() - last < float(conf.get("every_min", 10)) * 60:
        return []
    state["approaches_at"] = now.timestamp()
    store = LocalStore(Path(str(conf["store"])).expanduser() if conf.get("store") else None)
    lines = []
    if store.remote is not None:
        store.refresh()
        lines += import_pending(store, root, now, state.setdefault("approach_claims", []))
    changed = render(store, root)
    return lines + [f"{now:%H:%MZ} RENDERED {p.relative_to(root)}" for p in changed]


# ---- CLI -------------------------------------------------------------------------------------------------------------------------

def _line(a: Approach) -> str:
    tail = f"  -- {a.reason}" if a.reason and a.status != LIVE else ""
    return f"{a.name:<34} {a.type or '?':<6} {a.status or '?':<10} {a.owner or '?':<22} {_day(a.updated)}  {a.title or ''}{tail}"


def _store(ns) -> LocalStore:
    st = LocalStore(ns.store)
    if getattr(ns, "refresh", False):
        if st.remote is None:
            print(f"research notes approaches: no usable remote ({st.remote_error or 'none configured'}): the local catalog only",
                  file=sys.stderr)
        else:
            st.refresh()
    return st


def main(argv: list[str], default_root: Path) -> int:
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"research notes {cmd}", description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", default=None, help="the evidence store root (default $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--root", type=Path, default=None, help="the notes root (default $RESEARCH_NOTES or ~/.research/notes)")
    if cmd == "approaches":
        ap.add_argument("action", nargs="?", choices=("list", "render", "check"), default="list")
        ap.add_argument("--campaign")
        ap.add_argument("--status", help="one or more of " + ",".join(STATUSES))
        ap.add_argument("--owner")
        ap.add_argument("--grep", nargs="+", default=[], metavar="WORD", help="every word in the name, title, hypothesis or reason")
        ap.add_argument("--refresh", action="store_true", help="catch the store up from the remote first (warm: seconds)")
        ap.add_argument("--json", action="store_true")
    else:
        ap.add_argument("name", metavar="CAMPAIGN/SLUG")
        if cmd == "approach":
            ap.add_argument("status", nargs="?", choices=STATUSES, help="omit it to change the evidence alone")
            ap.add_argument("--by", required=True, metavar="LANE")
            ap.add_argument("--reason", help="one line: why (required unless live)")
            ap.add_argument("--owner", metavar="LANE", help="hand the approach to this lane")
            ap.add_argument("--superseded-by", metavar="CAMPAIGN/SLUG")
            ap.add_argument("--at", metavar="UTC", help="when it happened (the record of past work): YYYY-MM-DDTHH:MMZ")
        else:
            ap.add_argument("--lane", required=True)
            ap.add_argument("--reopen", metavar="WHY", help="take over a killed, superseded or merged approach: what differs this time")
        ap.add_argument("--title")
        ap.add_argument("--hypothesis")
        ap.add_argument("--type", choices=TYPES)
        ap.add_argument("--attacks", metavar="CAMPAIGN/SLUG", help="an attack's target")
        ap.add_argument("--cite", action="append", default=[], metavar="TOKEN", help="evidence: a run id, art:, note:, <repo>#<n>, ...")
        if cmd == "approach":
            ap.add_argument("--uncite", action="append", default=[], metavar="TOKEN", help="evidence the approach no longer cites")
    ns = ap.parse_args(rest)
    root = ns.root or default_root
    try:
        if cmd == "claim":
            try:
                rc, line = claim(LocalStore(ns.store), ns.name, ns.lane, title=ns.title, hypothesis=ns.hypothesis, type_=ns.type,
                                 attacks=ns.attacks, cites=ns.cite, reopen=ns.reopen)
            except NoRemote as e:
                p = pending_claim(root, ns.name, ns.lane, {"type": ns.type, "title": ns.title, "hypothesis": ns.hypothesis,
                                                           "attacks": ns.attacks, "reopen": ns.reopen, "cites": ns.cite}, str(e))
                print(f"PENDING: {e}; wrote {p} for the steward to record at its next pass (sync your lane folder). It is not a "
                      f"claim until `research notes approaches --refresh` shows it", file=sys.stderr)
                return 4
            print(line, file=sys.stdout if rc == 0 else sys.stderr)
            if rc == 0 and ns.title:
                st = LocalStore(ns.store)
                for a in similar([x for x in load(st) if x.name != ns.name], ns.title, ns.hypothesis or ""):
                    print(f"  similar: {_line(a)}", file=sys.stderr)
            return rc
        if cmd == "approach":
            a = set_status(LocalStore(ns.store), ns.name, ns.status, ns.by, reason=ns.reason, owner=ns.owner,
                           superseded_by=ns.superseded_by, cites=ns.cite, at=ns.at, title=ns.title, hypothesis=ns.hypothesis,
                           type_=ns.type, attacks=ns.attacks, uncites=ns.uncite)
            print(f"{a.name}: {a.status} (owner {a.owner})" + (f": {a.reason}" if a.reason else "")
                  + ("" if ns.status else f"; cites {' '.join(a.cites) or 'nothing'}"))
            return 0
    except Refused as e:
        print(f"research notes {cmd}: refused: {e}.  Nothing written", file=sys.stderr)
        return 2
    st = _store(ns)
    if ns.action == "check":
        errors, warnings = check(st, root if root.is_dir() else None)
        for w in warnings:
            print(f"WARN  {w}")
        for e in errors:
            print(f"ERROR {e}")
        print(f"approaches check: {len(load(st))} approaches, {len(errors)} errors, {len(warnings)} warnings", file=sys.stderr)
        return 1 if errors else 0
    if ns.action == "render":
        for p in render(st, root, ns.campaign):
            print(f"rendered {p}")
        return 0
    want = set((ns.status or "").split(",")) - {""}
    rows = [a for a in load(st)
            if (ns.campaign in (None, a.campaign)) and (not want or a.status in want) and (ns.owner in (None, a.owner))
            and all(w.lower() in " ".join(filter(None, (a.name, a.title, a.hypothesis, a.reason))).lower() for w in ns.grep)]
    if ns.json:
        print(json.dumps([asdict(a) for a in rows], indent=1))
    else:
        for a in rows:
            print(_line(a))
        print(f"{len(rows)} approaches" + ("" if ns.refresh else " (this store's catalog; --refresh catches up from the remote)"),
              file=sys.stderr)
    return 0
