# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "trafilatura", "warcio", "py3langid", "beautifulsoup4", "lxml", "mwparserfromhell"]
# ///
"""Build the evaluation corpus of the routed-mixture probes: the Notion page's section 4 (CPU only).

    uv run corpus.py [--work DIR] fetch STRATUM     raw documents to DIR/raw/STRATUM.jsonl (resumable: known ids are skipped)
    uv run corpus.py [--work DIR] mixed             the mixed-domain pairs to DIR/raw/mixed.jsonl
    uv run corpus.py [--work DIR] assemble --out TREE
                                                    hygiene, equal bytes per stratum, the splits and MANIFEST.sha512 in TREE

Every document was first published on or after ``DATE`` (each source's own date field; a source without one is not used).
``assemble`` keeps a document only if it has at least ``MIN_BYTES`` (one window), cuts it to ``CAP_BYTES``, drops exact
and near duplicates inside the corpus (13-word shingles), and drops a document when ``PROBES`` evenly spaced 13-word
spans hit OLMo 2's pretraining mix (which holds DCLM and StarCoder's Stack subset) in the infini-gram index at least
``HIT_LIMIT`` times. FineWeb and the Stack v2 have no public n-gram index, so the check covers them only through that
mix's overlap. TREE holds ``corpus.jsonl`` (the broad set, equal bytes per stratum), ``calibration.jsonl`` (B1's
codebook split, disjoint documents), ``mixed.jsonl`` (windows that join two strata, reported apart: stratum
``mixed:A+B``), ``b2-docs.txt`` (B2's 2M-token subset of the broad documents), ``build.json`` (sources, counts, every
dropped document and why) and ``MANIFEST.sha512``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import html as htmllib
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import threading
import time
import unicodedata
from pathlib import Path

import requests

DATE = dt.date(2026, 1, 1)
UNTIL = dt.date(2026, 10, 19)
MIN_BYTES = 4608
CAP_BYTES = 16384
WINDOW_BYTES = 8192
STRATA = ["news", "arxiv", "code", "wikipedia", "qa", "fiction", "gov-legal", "edu"]
#: bytes per stratum of the broad set (about 4,900 windows of 8 KiB over eight strata) and of the calibration split
#: (about 500 windows over eight strata)
TARGET_BYTES = 5_000_000
CALIBRATION_BYTES = 520_000
B2_TOKENS = 2_000_000
BYTES_PER_TOKEN = 4.0
SHINGLE = 13
NEAR_DUP = 0.2
PROBES = 8
HIT_LIMIT = 2
INDEXES = ["v4_olmo-mix-1124_llama"]
UA = "verity-research-corpus/0.1 (compute-verification; evaluation corpus, not training)"
CC_CRAWL = "CC-MAIN-2026-39"
CC = "https://data.commoncrawl.org"
WORK = Path(os.environ.get("CORPUS_WORK", "/tmp/routed-corpus"))

_session = threading.local()


def http() -> requests.Session:
    s = getattr(_session, "s", None)
    if s is None:
        s = _session.s = requests.Session()
        s.headers["User-Agent"] = UA
    return s


def fresh_session() -> None:
    """A worker process's initializer: a forked child must not reuse the parent's TLS connections."""
    _session.s = None


def get(url: str, *, tries: int = 4, pause: float = 2.0, **kw) -> requests.Response | None:
    for i in range(tries):
        try:
            r = http().get(url, timeout=kw.pop("timeout", 60), **kw)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code in (200, 206):
            return r
        if r is not None and r.status_code in (403, 404, 410, 451):
            return None
        if r is not None and r.status_code == 429:
            wait = r.headers.get("Retry-After", "")
            time.sleep(min(300, int(wait)) if wait.isdigit() else 60)
            continue
        time.sleep(pause * (2 ** i))
    return None


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def cap(text: str, limit: int = CAP_BYTES) -> str:
    data = text.encode()
    if len(data) <= limit:
        return text
    i = limit
    while i > 0 and data[i] & 0xC0 == 0x80:
        i -= 1
    return data[:i].decode()


def in_range(day: dt.date | None) -> bool:
    return day is not None and DATE <= day <= UNTIL


def parse_day(s) -> dt.date | None:
    if not s:
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(s))
    if not m:
        return None
    try:
        return dt.date(int(m[1]), int(m[2]), int(m[3]))
    except ValueError:
        return None


def english(text: str) -> bool:
    import py3langid
    lang, _ = py3langid.classify(text[:4000])
    return lang == "en"


class Sink:
    """DIR/raw/STRATUM.jsonl, appended under a lock; ids already there are skipped."""

    def __init__(self, stratum: str, work: Path):
        self.path = work / "raw" / f"{stratum}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stratum, self.lock = stratum, threading.Lock()
        self.ids, self.bytes, self.keys, self.source_bytes = set(), 0, {}, {}
        if self.path.exists():
            for line in self.path.open():
                d = json.loads(line)
                self.ids.add(d["id"])
                n = len(cap(d["text"]).encode())
                self.bytes += n
                self.source_bytes[d["source"]] = self.source_bytes.get(d["source"], 0) + n
                k = d.get("group")
                if k:
                    self.keys[k] = self.keys.get(k, 0) + 1

    def full(self, target: int) -> bool:
        return self.bytes >= target

    def group_count(self, key: str) -> int:
        return self.keys.get(key, 0)

    def add(self, doc_id: str, text: str, *, url: str, date: dt.date, source: str, group: str | None = None,
            **extra) -> bool:
        text = cap(clean(text), CAP_BYTES * 2)
        if len(text.encode()) < MIN_BYTES:
            return False
        with self.lock:
            if doc_id in self.ids:
                return False
            self.ids.add(doc_id)
            n = len(cap(text).encode())
            self.bytes += n
            self.source_bytes[source] = self.source_bytes.get(source, 0) + n
            if group:
                self.keys[group] = self.keys.get(group, 0) + 1
            rec = {"id": doc_id, "stratum": self.stratum, "source": source, "url": url, "date": date.isoformat(),
                   "group": group, **extra, "text": text}
            with self.path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True


# ---------------------------------------------------------------------------------------------------------------- news

def warc_records(url: str, limit_bytes: int | None = None):
    """(target URI, HTTP headers, HTML bytes) of each response record of a WARC streamed from ``url``."""
    from warcio.archiveiterator import ArchiveIterator
    r = http().get(url, stream=True, timeout=120)
    r.raise_for_status()
    for rec in ArchiveIterator(r.raw):
        if rec.rec_type != "response":
            continue
        ctype = rec.http_headers.get_header("Content-Type") if rec.http_headers else ""
        if not ctype or "html" not in ctype:
            continue
        yield rec.rec_headers.get_header("WARC-Target-URI"), rec.content_stream().read()


def extract(html_bytes: bytes, url: str) -> tuple[str, dt.date | None] | None:
    import trafilatura
    try:
        doc = trafilatura.bare_extraction(html_bytes, url=url, with_metadata=True, include_comments=False,
                                          include_tables=False, favor_precision=True, date_extraction_params={
                                              "original_date": True, "extensive_search": False})
    except Exception:
        return None
    if doc is None:
        return None
    get_ = doc.get if isinstance(doc, dict) else lambda k: getattr(doc, k, None)
    text = get_("text")
    if not text:
        return None
    title = get_("title") or ""
    return (f"{title}\n\n{text}" if title else text), parse_day(get_("date"))


def _news_file(path: str, per_file: int = 60, per_domain: int = 2) -> list[dict]:
    """In a worker process (trafilatura and lxml are not thread-safe): up to ``per_file`` English articles dated in
    range from one CC-NEWS WARC, at most ``per_domain`` per host."""
    out, hosts = [], {}
    try:
        for uri, body in warc_records(f"{CC}/{path}"):
            host = re.sub(r"^www\.", "", requests.utils.urlparse(uri).netloc.lower())
            if hosts.get(host, 0) >= per_domain:
                continue
            got = extract(body, uri)
            if not got:
                continue
            text, day = got
            if not in_range(day) or len(text.encode()) < MIN_BYTES or not english(text):
                continue
            hosts[host] = hosts.get(host, 0) + 1
            out.append({"uri": uri, "host": host, "text": text, "day": day})
            if len(out) >= per_file:
                break
    except Exception as e:
        print(f"news: {path}: {type(e).__name__}: {e}", file=sys.stderr)
    return out


def fetch_news(sink: Sink, target: int, rng: random.Random) -> None:
    paths = []
    for month in range(1, 10):
        r = get(f"{CC}/crawl-data/CC-NEWS/2026/{month:02d}/warc.paths.gz")
        if r:
            ps = gzip.decompress(r.content).decode().split()
            paths += rng.sample(ps, min(4, len(ps)))
    rng.shuffle(paths)
    with cf.ProcessPoolExecutor(4, initializer=fresh_session) as ex:
        for docs in ex.map(_news_file, paths):
            for d in docs:
                if sink.full(target):
                    break
                if sink.group_count(d["host"]) < 3:
                    sink.add("news:" + hashlib.sha1(d["uri"].encode()).hexdigest()[:16], d["text"], url=d["uri"],
                             date=d["day"], source="CC-NEWS", group=d["host"])
            print(f"news: {sink.bytes} bytes", file=sys.stderr)
            if sink.full(target):
                ex.shutdown(cancel_futures=True)
                break


# --------------------------------------------------------------------------------------------------------------- arxiv

ARXIV_GROUPS = {"math": ["math"], "cs": ["cs"], "physics": ["astro-ph", "cond-mat", "hep-ph", "hep-th", "quant-ph",
                                                            "physics", "gr-qc"], "bio": ["q-bio"]}


def arxiv_ids(archive: str, month: int) -> list[str]:
    r = get(f"https://arxiv.org/list/{archive}/2026-{month:02d}?skip=0&show=2000", timeout=120)
    if not r:
        return []
    return sorted(set(re.findall(r"/abs/(26%02d\.\d{4,5})" % month, r.text)))


def arxiv_source(aid: str) -> str | None:
    r = get(f"https://export.arxiv.org/e-print/{aid}", timeout=120)
    if not r:
        return None
    data = r.content
    files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".tex") and m.size < 2_000_000:
                    files[m.name] = tf.extractfile(m).read().decode("utf-8", "replace")
    except tarfile.TarError:
        try:
            raw = gzip.decompress(data)
        except OSError:
            raw = data
        if raw[:4] == b"%PDF":
            return None
        files["main.tex"] = raw.decode("utf-8", "replace")
    mains = [t for t in files.values() if "\\documentclass" in t and "\\begin{document}" in t]
    if not mains:
        return None
    main = max(mains, key=len)
    body = main[main.index("\\begin{document}"):]
    if len(body.encode()) < MIN_BYTES:
        rest = sorted((t for t in files.values() if t is not main), key=len, reverse=True)
        body += "\n" + "\n".join(rest[:3])
    body = "\n".join(line for line in body.split("\n") if not line.lstrip().startswith("%"))
    return body


def fetch_arxiv(sink: Sink, target: int, rng: random.Random) -> None:
    per_group = target // len(ARXIV_GROUPS)
    for group, archives in ARXIV_GROUPS.items():
        pool = []
        for month in range(1, 10):
            for a in archives:
                ids = arxiv_ids(a, month)
                pool += rng.sample(ids, min(len(ids), 40 // len(archives) + 2))
                time.sleep(1)
        rng.shuffle(pool)
        got = sum(1 for k in sink.ids if k.startswith(f"arxiv:{group}:")) * 12000
        for aid in pool:
            if got >= per_group:
                break
            if f"arxiv:{group}:{aid}" in sink.ids or any(k.endswith(f":{aid}") for k in sink.ids):
                continue
            body = arxiv_source(aid)
            time.sleep(3)
            if body and sink.add(f"arxiv:{group}:{aid}", body, url=f"https://arxiv.org/abs/{aid}",
                                 date=dt.date(2026, int(aid[2:4]), 1), source="arXiv e-print", group=group):
                got += len(cap(clean(body)).encode())
        print(f"arxiv {group}: {sink.bytes} bytes", file=sys.stderr)


# ---------------------------------------------------------------------------------------------------------------- code

CODE_LANGS = {"Python": (".py",), "TypeScript": (".ts", ".tsx"), "JavaScript": (".js", ".mjs"), "Go": (".go",),
              "Rust": (".rs",), "C++": (".cpp", ".cc", ".hpp", ".h"), "Java": (".java",), "C": (".c", ".h"),
              "C#": (".cs",), "Kotlin": (".kt",), "Swift": (".swift",), "Ruby": (".rb",), "PHP": (".php",),
              "Scala": (".scala",), "Julia": (".jl",), "Haskell": (".hs",), "Lua": (".lua",), "Zig": (".zig",)}
PERMISSIVE = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "0bsd"]
VENDORED = re.compile(r"(^|/)(node_modules|vendor|third_party|third-party|external|deps|dist|build|out|target|"
                      r"\.venv|venv|site-packages|generated|gen|migrations|test_data|testdata|fixtures|examples?)/", re.I)


def gh(path: str) -> dict | list | None:
    for i in range(4):
        p = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github+json", path], capture_output=True,
                           text=True)
        if p.returncode == 0:
            return json.loads(p.stdout)
        if "rate limit" in p.stderr.lower() or "secondary" in p.stderr.lower():
            time.sleep(30 * (i + 1))
            continue
        return None
    return None


def code_files(repo: dict, exts: tuple[str, ...], rng: random.Random) -> list[tuple[str, str]]:
    full, branch = repo["full_name"], repo["default_branch"]
    tree = gh(f"repos/{full}/git/trees/{branch}?recursive=1")
    if not tree or "tree" not in tree:
        return []
    cands = [t["path"] for t in tree["tree"] if t["type"] == "blob" and t["path"].endswith(exts)
             and 5000 <= t.get("size", 0) <= 80000 and not VENDORED.search(t["path"])
             and not re.search(r"(\.min\.|\.d\.ts$|_pb2|\.pb\.|generated|bundle)", t["path"], re.I)]
    rng.shuffle(cands)
    out = []
    for path in cands[:6]:
        r = get(f"https://raw.githubusercontent.com/{full}/{branch}/{requests.utils.quote(path)}", tries=2)
        if not r:
            continue
        text = r.content.decode("utf-8", "replace")
        lines = text.split("\n")
        if max(map(len, lines)) > 400 or sum(map(len, lines)) / max(1, len(lines)) > 100 or "\ufffd" in text:
            continue
        out.append((path, text))
        if len(out) >= 2:
            break
    return out


def fetch_code(sink: Sink, target: int, rng: random.Random) -> None:
    queries = []
    for lang in CODE_LANGS:
        for lic in rng.sample(PERMISSIVE[:4], 2):
            month = rng.randrange(1, 10)
            queries.append((lang, lic, f"2026-{month:02d}-01..2026-{month:02d}-28"))
    rng.shuffle(queries)
    repos = []
    for lang, lic, created in queries:
        q = requests.utils.quote(f"created:{created} license:{lic} language:\"{lang}\" stars:>=15 fork:false "
                                 "archived:false")
        res = gh(f"search/repositories?q={q}&sort=stars&order=desc&per_page=30")
        time.sleep(2.5)
        for item in (res or {}).get("items", []):
            if parse_day(item.get("created_at")) and in_range(parse_day(item["created_at"])) and not item.get("fork"):
                repos.append((lang, item))
    rng.shuffle(repos)
    seen = set()
    for lang, repo in repos:
        if sink.full(target):
            break
        if repo["full_name"] in seen or sink.group_count(repo["full_name"]):
            continue
        seen.add(repo["full_name"])
        for path, text in code_files(repo, CODE_LANGS[lang], rng):
            lic = (repo.get("license") or {}).get("spdx_id")
            sink.add(f"code:{repo['full_name']}:{path}", text, url=f"https://github.com/{repo['full_name']}/blob/"
                     f"{repo['default_branch']}/{path}", date=parse_day(repo["created_at"]), source="GitHub",
                     group=repo["full_name"], license=lic, language=lang)
        print(f"code: {sink.bytes} bytes ({len(seen)} repos)", file=sys.stderr)


# ----------------------------------------------------------------------------------------------------------- wikipedia

WIKI_SHARE = {"en": 0.5, "de": 0.1, "fr": 0.1, "es": 0.1, "ru": 0.1, "ja": 0.1}


def wiki_api(lang: str, **params) -> dict | None:
    params.update(format="json", formatversion=2)
    r = get(f"https://{lang}.wikipedia.org/w/api.php", params=params)
    return r.json() if r else None


#: link prefixes that aren't article text: files, categories and interlanguage links in the six languages
WIKI_NONTEXT = {"file", "image", "media", "category", "datei", "bild", "kategorie", "fichier", "catégorie", "archivo",
                "imagen", "categoría", "файл", "изображение", "категория", "ファイル", "画像", "カテゴリ"}


def wiki_plain(wikitext: str) -> str:
    """Plain text of an article's wikitext: templates, references and non-text links removed, headings kept as
    ``== Title ==`` lines (as the TextExtracts API gives them)."""
    import mwparserfromhell as mw

    code = mw.parse(wikitext)
    for tag in code.filter_tags(recursive=False, matches=lambda t: str(t.tag).lower() in ("ref", "references", "gallery")):
        try:
            code.remove(tag)
        except ValueError:
            pass
    for link in code.filter_wikilinks(recursive=True):
        prefix = str(link.title).lstrip(":").split(":", 1)
        if len(prefix) == 2 and (prefix[0].strip().lower() in WIKI_NONTEXT
                                 or re.fullmatch(r"[a-z]{2,3}(-[a-z]+)?", prefix[0].strip())):
            try:
                code.remove(link)
            except ValueError:
                pass
    for h in code.filter_headings(recursive=False):
        code.replace(h, f"\n\u27e6{h.level}\u27e7{h.title.strip_code().strip()}\u27e6/\u27e7\n")
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"[^\S\n]{2,}", " ", text)
    return re.sub("\u27e6(\\d)\u27e7(.*?)\u27e6/\u27e7", lambda m: f"{'=' * int(m[1])} {m[2]} {'=' * int(m[1])}", text)


def _wiki_batch(lang: str, start: dt.datetime, known: set) -> list[tuple[str, dt.date, str]]:
    """(title, creation day, plain text) of every article of at least 9,000 wikitext bytes among the 500 created in
    ``lang`` from ``start`` on, but those whose ids are ``known``. Wikitext comes 50 pages a request (TextExtracts
    gives one full page a request, and Wikimedia rate-limits anonymous clients)."""
    res = wiki_api(lang, action="query", list="logevents", letype="create", lenamespace=0, ledir="newer",
                   lestart=start.strftime("%Y-%m-%dT%H:%M:%SZ"), lelimit=500)
    created = {e["title"]: parse_day(e["timestamp"]) for e in (res or {}).get("query", {}).get("logevents", [])
               if "title" in e and in_range(parse_day(e.get("timestamp")))}
    titles = sorted(created)
    out = []
    for i in range(0, len(titles), 50):
        q = wiki_api(lang, action="query", prop="revisions", rvprop="content|size", rvslots="main",
                     titles="|".join(titles[i:i + 50]))
        for p in (q or {}).get("query", {}).get("pages", []):
            rev = (p.get("revisions") or [{}])[0]
            if (p.get("missing") or rev.get("size", 0) < 9000 or p["title"] not in created
                    or f"wikipedia:{lang}:{p['title']}" in known):
                continue
            wikitext = rev.get("slots", {}).get("main", {}).get("content", "")
            if wikitext.lstrip().startswith("#"):
                continue
            text = wiki_plain(wikitext)
            if text and "may refer to" not in text[:500]:
                out.append((p["title"], created[p["title"]], text))
    return out


def fetch_wikipedia(sink: Sink, target: int, rng: random.Random) -> None:
    span = (UNTIL - DATE).days * 86400
    rng = random.Random(f"{rng.random()}:{len(sink.ids)}")  # a resumed fetch draws new batches
    for lang, share in WIKI_SHARE.items():
        want = int(target * share)
        got = sum(len(cap(json.loads(line)["text"]).encode()) for line in sink.path.open()
                  if line.startswith('{"id": "wikipedia:%s:' % lang)) if sink.path.exists() else 0
        starts = [dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=rng.randrange(span))
                  for _ in range(400)]
        # Wikimedia answers 429 to an anonymous client that runs much faster than this
        with cf.ThreadPoolExecutor(2) as ex:
            futs = [ex.submit(_wiki_batch, lang, s, set(sink.ids)) for s in starts]
            for fut in cf.as_completed(futs):
                for title, first, text in fut.result():
                    if got < want and sink.add(f"wikipedia:{lang}:{title}", f"{title}\n\n{text}",
                                               url=f"https://{lang}.wikipedia.org/wiki/{requests.utils.quote(title)}",
                                               date=first, source="Wikipedia", group=lang, lang=lang):
                        got += len(cap(clean(text)).encode())
                print(f"wikipedia {lang}: {got} bytes", file=sys.stderr)
                if got >= want:
                    for f in futs:
                        f.cancel()
                    break


# ------------------------------------------------------------------------------------------------------------------ qa

SE_SITES = ["stackoverflow", "math", "physics", "superuser", "serverfault", "askubuntu", "unix", "english", "cooking",
            "diy", "scifi", "history", "money", "travel", "academia", "workplace", "stats", "cs", "electronics",
            "gaming", "rpg", "biology", "chemistry", "law", "philosophy", "worldbuilding", "softwareengineering",
            "codereview", "tex", "gis", "dsp", "bicycles", "gardening", "music", "movies", "politics", "aviation",
            "linguistics", "astronomy", "mathoverflow.net", "earthscience", "economics", "parenting", "pets"]


def html_text(fragment: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(fragment, "lxml")
    for pre in soup.find_all("pre"):
        pre.replace_with("\n" + pre.get_text() + "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote", "div", "tr"]):
        blk.insert_after("\n")
    return htmllib.unescape(soup.get_text())


def fetch_qa(sink: Sink, target: int, rng: random.Random) -> None:
    flt = get("https://api.stackexchange.com/2.3/filters/create", params={
        "include": "question.body;question.answers;answer.body;answer.score;question.title;question.link;"
                   "question.creation_date;question.answer_count", "unsafe": "false"})
    fid = flt.json()["items"][0]["filter"]
    per_site = target // len(SE_SITES) + 1
    frm = int(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    for site in SE_SITES:
        got, page = 0, 1
        while got < per_site and page <= 3:
            r = get("https://api.stackexchange.com/2.3/questions", params={
                "site": site, "fromdate": frm, "sort": "votes", "order": "desc", "pagesize": 100, "page": page,
                "filter": fid})
            if not r:
                break
            d = r.json()
            for q in d.get("items", []):
                answers = sorted(q.get("answers", []), key=lambda a: -a.get("score", 0))
                if not answers:
                    continue
                text = q["title"] + "\n\n" + html_text(q["body"]) + "".join(
                    "\n\nAnswer:\n\n" + html_text(a["body"]) for a in answers[:5])
                day = dt.datetime.fromtimestamp(q["creation_date"], dt.timezone.utc).date()
                if in_range(day) and sink.add(f"qa:{site}:{q['question_id']}", text, url=q["link"], date=day,
                                              source="Stack Exchange", group=site, license="CC BY-SA 4.0"):
                    got += min(len(text.encode()), CAP_BYTES)
                if got >= per_site:
                    break
            if d.get("backoff"):
                time.sleep(d["backoff"])
            if not d.get("has_more"):
                break
            page += 1
            time.sleep(1)
        print(f"qa {site}: {got} bytes, quota {d.get('quota_remaining')}", file=sys.stderr)


# --------------------------------------------------------------------------------------- Common Crawl (fiction, edu)

def cluster_index(work: Path) -> Path:
    p = work / "cc" / f"cluster-{CC_CRAWL[-7:]}.idx"
    if not p.exists() or p.stat().st_size < 50_000_000:
        p.parent.mkdir(parents=True, exist_ok=True)
        r = http().get(f"{CC}/cc-index/collections/{CC_CRAWL}/indexes/cluster.idx", timeout=900)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def cdx_blocks(idx: Path, prefixes: list[str]) -> dict[str, list[tuple[str, int, int]]]:
    """For each SURT prefix, the (cdx file, offset, length) blocks that may hold its captures."""
    lines = idx.read_text().split("\n")
    keys = [ln.split(" ", 1)[0] for ln in lines if ln]
    import bisect
    out = {}
    for pre in prefixes:
        i = max(0, bisect.bisect_left(keys, pre) - 1)
        blocks = []
        while i < len(keys) and (keys[i] < pre or keys[i].startswith(pre)):
            f = lines[i].split("\t")
            blocks.append((f[1], int(f[2]), int(f[3])))
            i += 1
        out[pre] = blocks
    return out


def cdx_lines(block: tuple[str, int, int]) -> list[dict]:
    name, off, length = block
    r = get(f"{CC}/cc-index/collections/{CC_CRAWL}/indexes/{name}", headers={"Range": f"bytes={off}-{off + length - 1}"},
            timeout=120)
    if r is None:
        return []
    out = []
    for ln in gzip.decompress(r.content).decode("utf-8", "replace").split("\n"):
        if not ln:
            continue
        surt, _, rest = ln.split(" ", 2)
        try:
            d = json.loads(rest)
        except json.JSONDecodeError:
            continue
        d["surt"] = surt
        out.append(d)
    return out


def warc_html(d: dict) -> bytes | None:
    off, length = int(d["offset"]), int(d["length"])
    r = get(f"{CC}/{d['filename']}", headers={"Range": f"bytes={off}-{off + length - 1}"}, timeout=120, tries=3)
    if r is None:
        return None
    from warcio.archiveiterator import ArchiveIterator
    for rec in ArchiveIterator(io.BytesIO(r.content)):
        if rec.rec_type == "response":
            return rec.content_stream().read()
    return None


def fetch_cc(sink: Sink, target: int, rng: random.Random, work: Path, prefixes: dict[str, str], parse, group_of,
             per_group: int, sample_blocks: int) -> None:
    idx = cluster_index(work)
    blocks = cdx_blocks(idx, list(prefixes))
    cands = []
    for pre, bl in blocks.items():
        pat = re.compile(prefixes[pre])
        chosen = rng.sample(bl, min(sample_blocks, len(bl)))
        with cf.ThreadPoolExecutor(8) as ex:
            for lines in ex.map(cdx_lines, chosen):
                cands += [d for d in lines if d["surt"].startswith(pre) and d.get("status") == "200"
                          and "html" in d.get("mime", "") and pat.search(d["url"])]
        print(f"cc {pre}: {len(bl)} blocks, {len(cands)} candidates so far", file=sys.stderr)
    rng.shuffle(cands)
    tried = 0
    with cf.ProcessPoolExecutor(6, initializer=fresh_session) as ex:
        while cands and not sink.full(target):
            batch, cands = cands[:96], cands[96:]
            batch = [d for d in batch if sink.group_count(group_of(d)) < per_group]
            tried += len(batch)
            for d, got in zip(batch, ex.map(_english_page, [parse] * len(batch), batch)):
                g = group_of(d)
                if got and sink.group_count(g) < per_group and not sink.full(target):
                    sink.add(f"{sink.stratum}:" + hashlib.sha1(d["url"].encode()).hexdigest()[:16], got[0],
                             url=d["url"], date=got[1], source=f"Common Crawl {CC_CRAWL}", group=g)
            print(f"{sink.stratum}: {tried} tried, {len(cands)} left, {sink.bytes} bytes", file=sys.stderr)


def _english_page(parse, d: dict):
    """In a worker process: ``parse(d)`` if it gives an English page."""
    try:
        got = parse(d)
    except Exception:
        return None
    return got if got and english(got[0]) else None


SUBSTACK_FICTION = 284


def _substack_posts(pub: dict) -> list[tuple[str, str, dt.date, str]]:
    """(url, title, day, text) of up to three free posts of at least 1,000 words a Fiction publication made in range."""
    base = f"https://{pub['custom_domain']}" if pub.get("custom_domain") else f"https://{pub['subdomain']}.substack.com"
    r = get(f"{base}/api/v1/archive", params={"sort": "new", "limit": 50}, tries=2)
    posts = [p for p in (r.json() if r else []) if p.get("audience") == "everyone" and p.get("type") == "newsletter"
             and (p.get("wordcount") or 0) >= 1000 and in_range(parse_day(p.get("post_date")))]
    out = []
    for p in posts[:3]:
        r = get(f"{base}/api/v1/posts/{p['slug']}", tries=2)
        body = (r.json() if r else {}).get("body_html")
        if body:
            title = " ".join(x for x in (p.get("title"), p.get("subtitle")) if x)
            out.append((p.get("canonical_url") or f"{base}/p/{p['slug']}", title, parse_day(p["post_date"]),
                        html_text(body)))
    return out


def fetch_fiction(sink: Sink, target: int, rng: random.Random, work: Path) -> None:
    """Free posts from Substack's Fiction category: the fiction sites (Royal Road, AO3, FanFiction.net) keep Common
    Crawl's bot out, and what the 2026 crawl holds of Wattpad and the like is older than DATE."""
    pubs = []
    for page in range(40):
        r = get(f"https://substack.com/api/v1/category/public/{SUBSTACK_FICTION}/all", params={"page": page})
        d = r.json() if r else {}
        pubs += d.get("publications", [])
        if not d.get("more"):
            break
    rng.shuffle(pubs)
    print(f"fiction: {len(pubs)} publications", file=sys.stderr)
    with cf.ThreadPoolExecutor(8) as ex:
        for pub, posts in zip(pubs, ex.map(_substack_posts, pubs)):
            for url, title, day, text in posts:
                if not sink.full(target) and english(text):
                    sink.add("fiction:" + hashlib.sha1(url.encode()).hexdigest()[:16], f"{title}\n\n{text}", url=url,
                             date=day, source="Substack (Fiction)", group=pub.get("subdomain"))
            if sink.full(target):
                break
    print(f"fiction: {sink.bytes} bytes", file=sys.stderr)


#: academic domains, and only URLs that carry the year (news, blog and course pages): the extracted date still decides
EDU_PREFIXES = {"edu,": r"/2026[/-]", "uk,ac,": r"/2026[/-]", "au,edu,": r"/2026[/-]"}


def _edu_page(d: dict):
    body = warc_html(d)
    if not body:
        return None
    got = extract(body, d["url"])
    return got if got and in_range(got[1]) else None


def fetch_edu(sink: Sink, target: int, rng: random.Random, work: Path) -> None:
    def host(d):
        return ".".join(requests.utils.urlparse(d["url"]).netloc.lower().split(".")[-3:])
    fetch_cc(sink, target, rng, work, EDU_PREFIXES, _edu_page, host, 4, 3000)


# ----------------------------------------------------------------------------------------------------------- gov-legal

def pdf_text(url: str) -> str | None:
    r = get(url, timeout=120, tries=2)
    if not r or r.content[:4] != b"%PDF":
        return None
    p = subprocess.run(["pdftotext", "-layout", "-", "-"], input=r.content, capture_output=True)
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def fetch_gov_legal(sink: Sink, target: int, rng: random.Random) -> None:
    fr_target, court_target, bill_target, uk_target = (int(target * s) for s in (0.35, 0.3, 0.15, 0.2))
    got = sink.source_bytes.get("Federal Register", 0)
    for month in rng.sample(range(1, 10), 9):
        if got >= fr_target:
            break
        r = get("https://www.federalregister.gov/api/v1/documents.json", params={
            "per_page": 100, "conditions[publication_date][gte]": f"2026-{month:02d}-01",
            "conditions[publication_date][lte]": f"2026-{month:02d}-28", "order": "relevance",
            "fields[]": ["raw_text_url", "title", "type", "publication_date", "document_number", "html_url"]})
        docs = (r.json() if r else {}).get("results", [])
        rng.shuffle(docs)
        for doc in docs[:60]:
            if got >= fr_target or not doc.get("raw_text_url"):
                continue
            t = get(doc["raw_text_url"])
            if not t:
                continue
            text = html_text(t.text) if "<" in t.text[:200] else t.text
            if sink.add(f"gov-legal:fr:{doc['document_number']}", f"{doc['title']}\n\n{text}", url=doc["html_url"],
                        date=parse_day(doc["publication_date"]), source="Federal Register", group="federal-register"):
                got += min(len(text.encode()), CAP_BYTES)
        print(f"gov-legal FR: {got} bytes", file=sys.stderr)
    got = sink.source_bytes.get("CourtListener", 0)
    # published federal appellate opinions: most state appellate rulings are a paragraph, under MIN_BYTES
    url = ("https://www.courtlistener.com/api/rest/v4/search/?type=o&filed_after=01/01/2026&stat_Published=on&q=*"
           "&court=" + "%20".join([f"ca{i}" for i in range(1, 12)] + ["cadc", "cafc"]))
    pages = 0
    while url and got < court_target and pages < 60:
        r = get(url)
        if not r:
            break
        d = r.json()
        pages += 1
        res = d.get("results", [])
        rng.shuffle(res)
        for op in res[:10]:
            if got >= court_target or not in_range(parse_day(op.get("dateFiled"))):
                continue
            paths = [o["local_path"] for o in op.get("opinions", []) if o.get("local_path", "").endswith(".pdf")]
            text = pdf_text("https://storage.courtlistener.com/" + paths[0]) if paths else None
            if text and sink.add(f"gov-legal:court:{op['cluster_id']}", f"{op['caseName']} ({op['court']})\n\n{text}",
                                 url="https://www.courtlistener.com" + op["absolute_url"],
                                 date=parse_day(op["dateFiled"]), source="CourtListener", group=op["court_id"]):
                got += min(len(text.encode()), CAP_BYTES)
            time.sleep(1)
        url = d.get("next")
        for _ in range(rng.randrange(0, 4)):
            r = get(url) if url else None
            if not r:
                break
            url = r.json().get("next")
        print(f"gov-legal court: {got} bytes", file=sys.stderr)
    got = sink.source_bytes.get("GovInfo BILLS", 0)
    r = get(f"https://api.govinfo.gov/published/{DATE}/{UNTIL}", params={
        "collection": "BILLS", "pageSize": 1000, "offsetMark": "*", "api_key": "DEMO_KEY"})
    bills = [p for p in (r.json() if r else {}).get("packages", []) if in_range(parse_day(p.get("dateIssued")))]
    rng.shuffle(bills)
    for b in bills:
        if got >= bill_target:
            break
        pid = b["packageId"]
        page = get(f"https://www.govinfo.gov/content/pkg/{pid}/html/{pid}.htm")
        if not page:
            continue
        text = html_text(page.text)
        if sink.add(f"gov-legal:bill:{pid}", text, url=f"https://www.govinfo.gov/app/details/{pid}",
                    date=parse_day(b["dateIssued"]), source="GovInfo BILLS", group="us-bills"):
            got += min(len(text.encode()), CAP_BYTES)
        time.sleep(0.5)
    print(f"gov-legal bills: {got} bytes", file=sys.stderr)
    got = sink.source_bytes.get("legislation.gov.uk", 0)
    for kind in ["uksi", "ukpga", "ssi", "wsi"]:
        r = get(f"https://www.legislation.gov.uk/{kind}/2026/data.feed")
        if not r:
            continue
        links = sorted(set(re.findall(rf"https?://www\.legislation\.gov\.uk/{kind}/2026/\d+", r.text)))
        rng.shuffle(links)
        for link in links[:40]:
            if got >= uk_target * (["uksi", "ukpga", "ssi", "wsi"].index(kind) + 1) / 4:
                break
            page = get(link + "/made/data.html") or get(link + "/data.html")
            if not page:
                continue
            text = html_text(page.text)
            n = link.rsplit("/", 1)[1]
            if sink.add(f"gov-legal:uk:{kind}:{n}", text, url=link, date=DATE, source="legislation.gov.uk",
                        group=kind, license="OGL v3.0"):
                got += min(len(text.encode()), CAP_BYTES)
            time.sleep(1)
        print(f"gov-legal UK {kind}: {got} bytes", file=sys.stderr)


# --------------------------------------------------------------------------------------------------------------- mixed

def fetch_mixed(work: Path, rng: random.Random, pairs: int = 220) -> None:
    """Windows that join two strata: a 2026 arXiv paper's opening followed by code from the GitHub repository its
    abstract links (created in 2026), and a 2026 Hacker News story's article followed by its discussion."""
    sink = Sink("mixed", work)
    half = WINDOW_BYTES // 2 - 64
    have = {"arxiv+code": 0, "news+forum": 0}
    for k in sink.ids:
        have["arxiv+code" if ":arxiv+code:" in k else "news+forum"] += 1
    # news + forum (Hacker News)
    start = int(dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    end = int(dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc).timestamp())
    tries = 0
    while have["news+forum"] < pairs and tries < 60:
        tries += 1
        t0 = rng.randrange(start, end - 86400 * 7)
        r = get("https://hn.algolia.com/api/v1/search_by_date", params={
            "tags": "story", "numericFilters": f"created_at_i>{t0},created_at_i<{t0 + 86400 * 3},num_comments>40",
            "hitsPerPage": 50})
        for hit in (r.json() if r else {}).get("hits", [])[:12]:
            if have["news+forum"] >= pairs or not hit.get("url"):
                continue
            key = f"mixed:news+forum:{hit['objectID']}"
            if key in sink.ids:
                continue
            page = get(hit["url"], tries=1, timeout=30)
            got = extract(page.content, hit["url"]) if page else None
            if not got or len(got[0].encode()) < half or not english(got[0]):
                continue
            if not in_range(got[1]):
                continue
            item = get(f"https://hn.algolia.com/api/v1/items/{hit['objectID']}")
            if not item:
                continue
            comments = []

            def walk(node, depth=0):
                for c in node.get("children", []):
                    if c.get("text"):
                        comments.append(html_text(c["text"]).strip())
                    if depth < 1:
                        walk(c, depth + 1)
            walk(item.json())
            forum = "\n\n".join(comments)
            if len(forum.encode()) < half:
                continue
            text = cap(clean(got[0]), half).rstrip() + "\n\nDiscussion:\n\n" + cap(clean(forum), half)
            if sink.add(key, text, url=hit["url"], date=got[1], source="Hacker News", group="news+forum",
                        forum=f"https://news.ycombinator.com/item?id={hit['objectID']}"):
                have["news+forum"] += 1
        print(f"mixed news+forum: {have['news+forum']}", file=sys.stderr)
    # arxiv + code
    tries = 0
    while have["arxiv+code"] < pairs and tries < 40:
        tries += 1
        month = rng.randrange(1, 10)
        # the phrase matches about 1,600 abstracts a month (abs:github alone about 130)
        r = get("https://export.arxiv.org/api/query", params={
            "search_query": f'abs:"github.com" AND submittedDate:[2026{month:02d}010000 TO 2026{month:02d}282359]',
            "start": rng.randrange(0, 1400), "max_results": 100})
        time.sleep(3)
        if not r:
            continue
        for entry in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
            if have["arxiv+code"] >= pairs:
                break
            aid = re.search(r"<id>https?://arxiv.org/abs/(26\d{2}\.\d{4,5})", entry)
            repo = re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", entry)
            if not aid or not repo:
                continue
            full = repo[1].rstrip(".")
            key = f"mixed:arxiv+code:{aid[1]}"
            if key in sink.ids:
                continue
            meta = gh(f"repos/{full}")
            if not meta or not in_range(parse_day(meta.get("created_at"))):
                continue
            lic = ((meta.get("license") or {}).get("spdx_id") or "").lower()
            if lic not in PERMISSIVE:
                continue
            lang = meta.get("language")
            if lang not in CODE_LANGS:
                continue
            files = code_files(meta, CODE_LANGS[lang], rng)
            body = arxiv_source(aid[1])
            time.sleep(3)
            if not files or not body:
                continue
            text = cap(clean(body), half).rstrip() + "\n\n" + f"// {files[0][0]}\n" + cap(clean(files[0][1]), half)
            if sink.add(key, text, url=f"https://arxiv.org/abs/{aid[1]}", date=dt.date(2026, int(aid[1][2:4]), 1),
                        source="arXiv e-print + GitHub", group="arxiv+code", code=f"https://github.com/{full}",
                        license=lic):
                have["arxiv+code"] += 1
        print(f"mixed arxiv+code: {have['arxiv+code']}", file=sys.stderr)


# ------------------------------------------------------------------------------------------------------------ assemble

def words(text: str) -> list[str]:
    return text.split()


def shingles(text: str) -> set[int]:
    w = words(text)
    return {hash(" ".join(w[i:i + SHINGLE])) for i in range(0, max(0, len(w) - SHINGLE + 1))}


#: what the API's firewall refuses (403) in a query: quotes, SQL operators and comments
UNSAFE = re.compile(r"['\"\u2018\u2019\u201c\u201d`=;<>#]|--|/\*|\*/")


def probes(text: str, n: int = PROBES) -> list[str]:
    """Up to ``n`` 13-word spans, one from each of ``n`` equal stretches of the text: the first in its stretch that
    has nothing the firewall refuses and no word the query language reads as an operator (code has fewer)."""
    w = words(text)
    if len(w) < SHINGLE * 2:
        return []
    step = (len(w) - SHINGLE) / n
    out = []
    for i in range(n):
        s, end = int(i * step), int((i + 1) * step) if i < n - 1 else len(w) - SHINGLE
        while s <= end and (UNSAFE.search(" ".join(w[s:s + SHINGLE])) or {"AND", "OR"} & set(w[s:s + SHINGLE])):
            s += 1
        if s <= end:
            out.append(" ".join(w[s:s + SHINGLE]))
    return out


#: the API refuses (403) bursts of concurrent queries but serves a few a second
_ig_lock, _ig_last, IG_INTERVAL = threading.Lock(), [0.0], 0.3


def infinigram(query: str, index: str) -> int | None:
    refused = 0
    for i in range(6):
        with _ig_lock:
            time.sleep(max(0.0, _ig_last[0] + IG_INTERVAL - time.monotonic()))
            _ig_last[0] = time.monotonic()
        try:
            r = http().post("https://api.infini-gram.io/", json={"index": index, "query_type": "count",
                                                                   "query": query}, timeout=60)
            d = r.json()
            if "count" in d:
                return int(d["count"])
            refused += r.status_code == 403
            if refused == 2:
                return None
        except (requests.RequestException, ValueError):
            pass
        time.sleep(min(20, 3 * 2 ** i))
    return None


def contamination(text: str) -> dict:
    """How many probes occur in any index. Probes go four to a query (an OR, the most a clause takes), and only a
    group that occurs is asked one probe at a time."""
    qs = probes(text)
    found, failed = set(), 0
    for ix in INDEXES:
        for g in range(0, len(qs), 4):
            group = [j for j in range(g, min(g + 4, len(qs))) if j not in found]
            if not group:
                continue
            c = infinigram(" OR ".join(qs[j] for j in group), ix)
            if c is None:
                failed += len(group)
                continue
            for j in group if c > 0 else []:
                cj = infinigram(qs[j], ix) if len(group) > 1 else c
                if cj is None:
                    failed += 1
                elif cj > 0:
                    found.add(j)
    return {"probes": len(qs), "hits": len(found), "failed": failed}


def assemble(work: Path, out: Path, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cache_path = work / "contamination.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    dropped: list[dict] = []
    kept: dict[str, list[dict]] = {}
    for stratum in [*STRATA, "mixed"]:
        p = work / "raw" / f"{stratum}.jsonl"
        docs = [json.loads(line) for line in p.open()] if p.exists() else []
        random.Random(f"{seed}:{stratum}").shuffle(docs)
        seen_hash, seen_sh = set(), set()
        ok = []
        for d in docs:
            d["text"] = cap(clean(d["text"]), WINDOW_BYTES if stratum == "mixed" else CAP_BYTES)
            if len(d["text"].encode()) < MIN_BYTES:
                dropped.append({"id": d["id"], "why": "short"})
                continue
            h = hashlib.sha512(re.sub(r"\s+", " ", d["text"]).encode()).hexdigest()
            if h in seen_hash:
                dropped.append({"id": d["id"], "why": "exact duplicate"})
                continue
            sh = shingles(d["text"])
            if sh and len(sh & seen_sh) / len(sh) > NEAR_DUP:
                dropped.append({"id": d["id"], "why": f"near duplicate ({len(sh & seen_sh) / len(sh):.2f})"})
                continue
            seen_hash.add(h)
            seen_sh |= sh
            ok.append(d)
        kept[stratum] = ok
    # documents are checked in the order the splits take them, until a stratum's clean bytes cover them with a margin;
    # the rest are spare, neither checked nor used
    need = int((TARGET_BYTES + CALIBRATION_BYTES) * 1.1)
    unchecked = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for stratum, ds in kept.items():
            clean_docs, have, i = [], 0, 0
            while i < len(ds) and (stratum == "mixed" or have < need):
                batch, i = ds[i:i + 64], i + 64
                todo = [d for d in batch if d["id"] not in cache or cache[d["id"]]["failed"]]
                for d, c in zip(todo, ex.map(lambda d: contamination(d["text"]), todo)):
                    cache[d["id"]] = c
                cache_path.write_text(json.dumps(cache))
                for d in batch:
                    c = cache[d["id"]]
                    if c["failed"] > c["probes"] // 2:
                        dropped.append({"id": d["id"], "why": "infini-gram unreachable"})
                    elif c["hits"] >= HIT_LIMIT:
                        dropped.append({"id": d["id"],
                                        "why": f"pretraining overlap ({c['hits']} of {c['probes']} spans)"})
                    else:
                        d["overlap"] = c
                        clean_docs.append(d)
                        have += len(d["text"].encode())
            unchecked += len(ds) - min(i, len(ds))
            print(f"infini-gram {stratum}: {len(clean_docs)} clean of {min(i, len(ds))} checked, {have} bytes",
                  file=sys.stderr)
            kept[stratum] = clean_docs
    avail = {s: sum(len(d["text"].encode()) for d in kept[s]) for s in STRATA}
    per = min(TARGET_BYTES, min(avail[s] for s in STRATA if avail[s]) - CALIBRATION_BYTES)
    broad, calib, spare = [], [], []
    for s in STRATA:
        b = c = 0
        for d in kept[s]:
            n = len(d["text"].encode())
            if c + n <= CALIBRATION_BYTES:
                calib.append(d)
                c += n
            elif b + n <= per + n // 2 and b < per:
                broad.append(d)
                b += n
            else:
                spare.append(d["id"])
    b2, t = [], 0
    for d in sorted(broad, key=lambda d: hashlib.sha512(f"{seed}:{d['id']}".encode()).hexdigest()):
        if t >= B2_TOKENS * BYTES_PER_TOKEN:
            break
        b2.append(d["id"])
        t += len(d["text"].encode())
    for d in kept["mixed"]:
        d["stratum"] = "mixed:" + d["group"]

    def write(name: str, docs: list[dict]) -> None:
        with (out / name).open("w") as f:
            for d in sorted(docs, key=lambda d: d["id"]):
                f.write(json.dumps({k: d[k] for k in ("id", "stratum", "text", "source", "url", "date")}
                                   | {k: d[k] for k in ("license", "lang", "language", "code", "forum", "overlap")
                                      if d.get(k)}, ensure_ascii=False) + "\n")
    write("corpus.jsonl", broad)
    write("calibration.jsonl", calib)
    write("mixed.jsonl", kept["mixed"])
    (out / "b2-docs.txt").write_text("\n".join(sorted(b2)) + "\n")
    stats = {s: {"docs": sum(d["stratum"] == s for d in broad), "bytes": sum(len(d["text"].encode()) for d in broad
                                                                             if d["stratum"] == s),
                 "calibration_docs": sum(d["stratum"] == s for d in calib), "available_bytes": avail[s]}
             for s in STRATA}
    build = {"date_from": DATE.isoformat(), "date_until": UNTIL.isoformat(), "min_bytes": MIN_BYTES,
             "cap_bytes": CAP_BYTES, "shingle_words": SHINGLE, "near_duplicate": NEAR_DUP, "probes": PROBES,
             "hit_limit": HIT_LIMIT, "indexes": INDEXES, "seed": seed, "common_crawl": CC_CRAWL,
             "per_stratum_bytes": per, "strata": stats, "mixed_docs": len(kept["mixed"]), "b2_docs": len(b2),
             "b2_bytes": t, "dropped": dropped, "spare": len(spare) + unchecked}
    (out / "build.json").write_text(json.dumps(build, indent=1, ensure_ascii=False) + "\n")
    (out / "corpus.py").write_text(Path(__file__).read_text())
    lines = []
    for p in sorted(out.iterdir()):
        if p.name != "MANIFEST.sha512":
            lines.append(f"{hashlib.sha512(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "MANIFEST.sha512").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in build.items() if k != "dropped"}, indent=1))


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--seed", type=int, default=20261019)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("stratum", choices=STRATA)
    f.add_argument("--target", type=int, default=int((TARGET_BYTES + CALIBRATION_BYTES) * 1.5))
    sub.add_parser("mixed")
    a = sub.add_parser("assemble")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rng = random.Random(f"{args.seed}:{args.cmd}:{getattr(args, 'stratum', '')}")
    if args.cmd == "fetch":
        sink = Sink(args.stratum, args.work)
        fn = {"news": fetch_news, "arxiv": fetch_arxiv, "code": fetch_code, "wikipedia": fetch_wikipedia,
              "qa": fetch_qa, "gov-legal": fetch_gov_legal}.get(args.stratum)
        if fn:
            fn(sink, args.target, rng)
        else:
            {"fiction": fetch_fiction, "edu": fetch_edu}[args.stratum](sink, args.target, rng, args.work)
        print(f"{args.stratum}: {sink.bytes} bytes in {sink.path}")
    elif args.cmd == "mixed":
        fetch_mixed(args.work, rng)
    else:
        assemble(args.work, args.out, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
