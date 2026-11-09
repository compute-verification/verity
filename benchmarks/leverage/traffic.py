"""Honest traffic for calibrating the range clamps' boxes, and a held-out set for checking them.

Each hand-written set mixes prose, code, structured data, mathematics, several languages, chat transcripts in ChatML
and documents separated by <|endoftext|>, so that attention sinks and massive activations at delimiters and special
tokens show up in both. The held-out set also holds the prompts the fault experiments decode from
(`prompts.PROMPTS[:8]`), and the calibration set the other prompts, so no box is fitted to a text it is checked on; and
texts whose openings move or stretch the sink and that no calibration text resembles (`HELDOUT_OPENINGS`: a run of one
token, digits a model's sink can't sit on).

`mixed(split, n)` draws more traffic from three public datasets pinned by revision (`SOURCES`): chats in ChatML from
OpenAssistant's conversation trees (23 languages), documents from a 10,000-document sample of the Pile (web, code,
law, papers, mail, mathematics, subtitles), and passages from Belebele in 20 scripts. Each source is in a fixed
order, its languages or subsets taken in turn; "heldout" is the first 40% / 40% / 20% of n from each, and
"calibration" the items after the largest held-out set (`HELDOUT_MAX`), so the two never share a text. Every such text
is cut to `CHARS` characters; `long_texts` takes held-out Pile documents long enough to reach far past the positions
calibration sees.
"""
import gzip
import json
import urllib.request

from lm import cache_dir
from prompts import PROMPTS

SOURCES = {
    "oasst1": ("OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b", "2023-04-12_oasst_ready.trees.jsonl.gz"),
    "pile": ("NeelNanda/pile-10k", "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
             "data/train-00000-of-00001-4746b8785c874cc7.parquet"),
    "belebele": ("facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", "data/{}.jsonl"),
}
BELEBELE_LANGS = ("jpn_Jpan", "kor_Hang", "rus_Cyrl", "ukr_Cyrl", "arb_Arab", "heb_Hebr", "hin_Deva", "ben_Beng",
                  "tha_Thai", "ell_Grek", "zho_Hans", "zho_Hant", "vie_Latn", "tur_Latn", "amh_Ethi", "kat_Geor",
                  "hye_Armn", "tam_Taml", "khm_Khmr", "pes_Arab")
SHARES = (("oasst1", 0.4), ("pile", 0.4), ("belebele", 0.2))
HELDOUT_MAX = 400
CHARS = 6000


def fetch_dataset(name: str, file: str):
    repo, rev, _ = SOURCES[name]
    p = cache_dir() / "datasets" / repo / rev / file
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{file}", timeout=600) as r:
            tmp.write_bytes(r.read())
        tmp.rename(p)
    return p


def round_robin(groups: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Each group's items in order, taking the groups in turn (largest first), as (group, item)."""
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))
    out, i = [], 0
    while any(i < len(groups[k]) for k in keys):
        out.extend((k, groups[k][i]) for k in keys if i < len(groups[k]))
        i += 1
    return out


def chat_texts() -> list[tuple[str, str]]:
    """Each conversation tree's prompt and its best-ranked replies, up to two exchanges, in ChatML."""
    by_lang: dict[str, list[str]] = {}
    with gzip.open(fetch_dataset("oasst1", SOURCES["oasst1"][2]), "rt") as f:
        for line in f:
            m = json.loads(line)["prompt"]
            lang, turns = m["lang"], []
            while m is not None and len(turns) < 4:
                turns.append(("user" if m["role"] == "prompter" else "assistant", m["text"]))
                replies = sorted(m.get("replies") or [], key=lambda r: (r.get("rank") is None, r.get("rank") or 0))
                m = replies[0] if replies else None
            text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + "".join(
                f"<|im_start|>{role}\n{t}<|im_end|>\n" for role, t in turns)
            by_lang.setdefault(lang, []).append(text[:CHARS])
    return round_robin(by_lang)


def pile_texts(chars: int = CHARS) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq
    t = pq.read_table(fetch_dataset("pile", SOURCES["pile"][2])).to_pylist()
    by_set: dict[str, list[str]] = {}
    for row in t:
        meta = row["meta"] if isinstance(row["meta"], dict) else json.loads(row["meta"])
        by_set.setdefault(meta["pile_set_name"], []).append(row["text"][:chars])
    return round_robin(by_set)


def belebele_texts() -> list[tuple[str, str]]:
    """Each language's distinct passages in order, three to a text."""
    by_lang = {}
    for lang in BELEBELE_LANGS:
        seen: list[str] = []
        for line in fetch_dataset("belebele", SOURCES["belebele"][2].format(lang)).read_text().splitlines():
            p = json.loads(line)["flores_passage"]
            if p not in seen:
                seen.append(p)
        by_lang[lang] = ["\n\n".join(seen[i:i + 3]) for i in range(0, len(seen) - 2, 3)]
    return round_robin(by_lang)


def mixed(split: str, n: int) -> list[tuple[str, str]]:
    """n texts of `split` ("heldout" or "calibration") as (source:group, text)."""
    if split == "heldout" and n > HELDOUT_MAX:
        raise ValueError(f"at most {HELDOUT_MAX} held-out texts")
    out = []
    for name, share in SHARES:
        items = {"oasst1": chat_texts, "pile": pile_texts, "belebele": belebele_texts}[name]()
        k = round(n * share)
        start = 0 if split == "heldout" else round(HELDOUT_MAX * share)
        out.extend((f"{name}:{g}", t) for g, t in items[start:start + k])
    return out


def long_texts(n: int, chars: int) -> list[tuple[str, str]]:
    """The first n held-out Pile documents of at least `chars` characters, cut to that: long-context traffic, from the
    held-out range `mixed` draws from, so none is in calibration."""
    docs = pile_texts(chars)[:round(HELDOUT_MAX * dict(SHARES)["pile"])]
    return [(f"long:{g}", t) for g, t in docs if len(t) >= chars][:n]

CALIBRATION_TEXTS = (
    "WASHINGTON (Reuters) - The Senate on Thursday passed a sweeping infrastructure bill that would spend $550 billion "
    "on roads, bridges, broadband and water systems over the next five years, sending the measure to the House, where "
    "its fate remains uncertain. The vote was 69 to 30, with 19 Republicans joining all 50 Democrats in support. "
    "\"This is a historic investment,\" the majority leader said on the Senate floor. Critics argued that the bill "
    "would add to the deficit and that its funding mechanisms, including unspent pandemic relief money, were "
    "unreliable. The Congressional Budget Office estimated that the legislation would increase deficits by $256 "
    "billion over a decade.\n\nIn the House, progressive members have said they will not vote for the bill unless it "
    "is paired with a larger package of social spending.",
    "import heapq\nfrom dataclasses import dataclass, field\nfrom typing import Iterable\n\n\n@dataclass(order=True)\n"
    "class Task:\n    priority: int\n    name: str = field(compare=False)\n    deps: tuple[str, ...] = field(default=(), "
    "compare=False)\n\n\nclass Scheduler:\n    \"\"\"Runs tasks in priority order once their dependencies are done.\"\"\"\n\n"
    "    def __init__(self, tasks: Iterable[Task]):\n        self.pending = {t.name: t for t in tasks}\n        "
    "self.done: set[str] = set()\n        self.heap: list[Task] = []\n\n    def ready(self) -> list[Task]:\n        "
    "return [t for t in self.pending.values() if all(d in self.done for d in t.deps)]\n\n    def run(self) -> list[str]:\n"
    "        order = []\n        while self.pending:\n            for t in self.ready():\n                "
    "heapq.heappush(self.heap, t)\n                del self.pending[t.name]\n            if not self.heap:\n"
    "                raise RuntimeError(f\"cycle among {sorted(self.pending)}\")\n            t = heapq.heappop(self.heap)\n"
    "            self.done.add(t.name)\n            order.append(t.name)\n        return order\n",
    "{\n  \"id\": \"chatcmpl-8f3a2b\",\n  \"object\": \"chat.completion\",\n  \"created\": 1715000000,\n  \"model\": "
    "\"example-1\",\n  \"choices\": [\n    {\n      \"index\": 0,\n      \"message\": {\"role\": \"assistant\", \"content\": "
    "\"The answer is 42.\"},\n      \"finish_reason\": \"stop\"\n    }\n  ],\n  \"usage\": {\"prompt_tokens\": 17, "
    "\"completion_tokens\": 6, \"total_tokens\": 23},\n  \"tags\": [\"a\", \"b\", null, true, false, 3.14159, -2e-7]\n}\n",
    "La Révolution française est une période de bouleversements politiques et sociaux en France, qui commence en 1789 "
    "et se termine en 1799. Elle met fin à l'Ancien Régime en remplaçant la monarchie absolue par une monarchie "
    "constitutionnelle, puis par la Première République. Les idées des Lumières, la crise financière de l'État et "
    "les mauvaises récoltes ont contribué à son déclenchement. La Déclaration des droits de l'homme et du citoyen, "
    "adoptée le 26 août 1789, proclame que « les hommes naissent et demeurent libres et égaux en droits ».",
    "人工智能是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。自从人工智能诞生以来，理论和技术日益成熟，"
    "应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。",
    "Let $f:\\mathbb{R}^n \\to \\mathbb{R}$ be convex and $L$-smooth, i.e. $\\|\\nabla f(x) - \\nabla f(y)\\|_2 \\le "
    "L\\|x - y\\|_2$. Gradient descent with step $\\eta = 1/L$ satisfies\n\\[\n  f(x_k) - f(x^\\star) \\le "
    "\\frac{L\\|x_0 - x^\\star\\|_2^2}{2k}.\n\\]\n\\begin{proof}\nBy smoothness, $f(x_{k+1}) \\le f(x_k) - \\frac{1}{2L}"
    "\\|\\nabla f(x_k)\\|^2$. By convexity, $f(x_k) - f^\\star \\le \\langle \\nabla f(x_k), x_k - x^\\star \\rangle$. "
    "Combining the two and telescoping over $k$ gives the claim. $\\square$\n\\end{proof}\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nCan you explain what a hash "
    "function is, in two sentences?<|im_end|>\n<|im_start|>assistant\nA hash function maps data of any size to a "
    "fixed-size value, called a digest, in a way that is fast to compute. A cryptographic hash function also makes it "
    "infeasible to find two inputs with the same digest or to recover an input from its digest.<|im_end|>\n"
    "<|im_start|>user\nWhat is SHA-512's digest size?<|im_end|>\n<|im_start|>assistant\n",
    "# Release notes\n\n## 2.4.0\n\n### Added\n\n- A `--dry-run` flag for `deploy`, which prints the plan and exits.\n"
    "- Support for Python 3.13.\n\n### Fixed\n\n- `sync` no longer follows symlinks out of the tree (#1432).\n"
    "- A race in the cache's eviction under heavy load.\n\n| command | p50 (ms) | p99 (ms) |\n|---|---:|---:|\n"
    "| `build` | 412 | 1,830 |\n| `test` | 2,210 | 9,004 |\n| `deploy` | 87 | 240 |\n\n> **Note:** the old "
    "`--force` flag is deprecated and will be removed in 3.0.\n",
    "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\ntypedef struct node {\n    int key;\n    struct "
    "node *left, *right;\n} node;\n\nstatic node *insert(node *root, int key) {\n    if (!root) {\n        node *n = "
    "calloc(1, sizeof *n);\n        n->key = key;\n        return n;\n    }\n    if (key < root->key)\n        "
    "root->left = insert(root->left, key);\n    else if (key > root->key)\n        root->right = insert(root->right, "
    "key);\n    return root;\n}\n\nint main(int argc, char **argv) {\n    node *root = NULL;\n    for (int i = 1; i < "
    "argc; i++)\n        root = insert(root, atoi(argv[i]));\n    printf(\"%d\\n\", root ? root->key : -1);\n    "
    "return 0;\n}\n",
    "Die Energiewende bezeichnet den Übergang von der nicht-nachhaltigen Nutzung fossiler Energieträger sowie der "
    "Kernenergie zu einer nachhaltigen Energieversorgung mittels erneuerbarer Energien. Ziel ist es, die "
    "Treibhausgasemissionen deutlich zu senken. Neben dem Ausbau von Wind- und Solarenergie gehören dazu auch die "
    "Steigerung der Energieeffizienz, der Ausbau der Stromnetze und die Entwicklung von Speichertechnologien.",
    "date,station,temp_c,precip_mm,wind_kmh\n2024-01-01,ZRH,-1.4,0.0,12\n2024-01-02,ZRH,0.8,2.3,18\n2024-01-03,ZRH,"
    "3.1,11.7,31\n2024-01-04,GVA,4.6,0.4,9\n2024-01-05,GVA,2.2,0.0,7\n2024-01-06,BSL,-3.9,0.0,22\n\nSee "
    "https://example.org/data/v2/stations?id=ZRH&from=2024-01-01 for the full series, or call +41 44 123 45 67. "
    "Checksum: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n",
    "The fog comes\non little cat feet.\n\nIt sits looking\nover harbor and city\non silent haunches\nand then moves on."
    "<|endoftext|>Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do "
    "shake the darling buds of May,\nAnd summer's lease hath all too short a date.<|endoftext|>Roses are red,\n"
    "violets are blue,\nsugar is sweet,\nand so are you.",
)

HELDOUT_TEXTS = (
    "The rain had not stopped for three days. Marguerite stood at the window of the small apartment above the bakery, "
    "watching the water run in sheets down the glass, and tried to remember the last time she had spoken to her "
    "brother. It had been at their mother's funeral, she thought, or perhaps a little after, when he had called to "
    "ask about the house. He had sounded tired then, and distant, as if he were calling from much farther away than "
    "the other side of the city. She picked up the phone, put it down again, and went to make tea instead.",
    "export async function fetchJson<T>(url: string, opts: RequestInit = {}): Promise<T> {\n  const res = await "
    "fetch(url, { ...opts, headers: { Accept: 'application/json', ...(opts.headers ?? {}) } });\n  if (!res.ok) {\n"
    "    throw new Error(`HTTP ${res.status} for ${url}`);\n  }\n  return (await res.json()) as T;\n}\n\nconst cache = "
    "new Map<string, unknown>();\nexport const memo = <A extends unknown[], R>(f: (...a: A) => R) => (...a: A): R => "
    "{\n  const k = JSON.stringify(a);\n  if (!cache.has(k)) cache.set(k, f(...a));\n  return cache.get(k) as R;\n};\n",
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n  <title>Status</title>\n  <style>\n"
    "    body { font-family: system-ui, sans-serif; margin: 2rem; }\n    .ok { color: #1a7f37; } .bad { color: #cf222e; "
    "}\n  </style>\n</head>\n<body>\n  <h1>Service status</h1>\n  <ul>\n    <li class=\"ok\">API &mdash; operational</li>\n"
    "    <li class=\"bad\">Search &mdash; degraded</li>\n  </ul>\n  <script>document.title += ' (' + new "
    "Date().toISOString() + ')';</script>\n</body>\n</html>\n",
    "El cambio climático es uno de los mayores desafíos de nuestro tiempo. Las temperaturas medias del planeta han "
    "aumentado aproximadamente 1,1 grados desde la era preindustrial, y los científicos advierten que superar el "
    "umbral de 1,5 grados tendría consecuencias graves: sequías más intensas, olas de calor, pérdida de "
    "biodiversidad y el aumento del nivel del mar. ¿Qué podemos hacer? Reducir las emisiones, proteger los bosques y "
    "adaptar nuestras ciudades.",
    "富士山は、静岡県と山梨県にまたがる活火山で、標高は3776メートルです。日本で最も高い山であり、その美しい姿は古くから"
    "多くの芸術作品の題材となってきました。2013年には「富士山－信仰の対象と芸術の源泉」として世界文化遺産に登録されました。"
    "毎年夏の登山シーズンには、多くの人々が山頂を目指します。",
    "Москва — столица России, крупнейший по численности населения город страны и один из крупнейших городов мира. "
    "Город расположен на реке Москве в центре Восточно-Европейской равнины. Москва является важным политическим, "
    "экономическим, культурным и научным центром. Здесь находятся Кремль, Красная площадь и множество музеев.",
    "<|im_start|>system\nYou are a concise assistant that answers in one line.<|im_end|>\n<|im_start|>user\nTranslate "
    "'good morning' into Italian.<|im_end|>\n<|im_start|>assistant\nBuongiorno.<|im_end|>\n<|im_start|>user\nAnd into "
    "Greek?<|im_end|>\n<|im_start|>assistant\nΚαλημέρα.<|im_end|>\n<|im_start|>user\nWhich of the two is older?"
    "<|im_end|>\n<|im_start|>assistant\n",
    "-- monthly active users by country\nSELECT c.name AS country, COUNT(DISTINCT e.user_id) AS mau\nFROM events e\n"
    "JOIN users u ON u.id = e.user_id\nJOIN countries c ON c.code = u.country_code\nWHERE e.ts >= now() - interval "
    "'30 days'\nGROUP BY c.name\nHAVING COUNT(DISTINCT e.user_id) > 100\nORDER BY mau DESC\nLIMIT 20;\n\n$ psql -h "
    "db.internal -U report -c '\\copy (select * from mau) to mau.csv csv header'\n$ wc -l mau.csv\n21 mau.csv\n",
    "1. Definitions. In this Agreement, \"Confidential Information\" means any information disclosed by either party "
    "to the other, whether orally or in writing, that is designated as confidential or that reasonably should be "
    "understood to be confidential given the nature of the information and the circumstances of disclosure.\n2. "
    "Obligations. The Receiving Party shall (a) not use Confidential Information for any purpose outside the scope "
    "of this Agreement, and (b) limit access to those of its employees who need such access.",
    "[server]\nhost = \"0.0.0.0\"\nport = 8443\nworkers = 16\n\n[tls]\ncert = \"/etc/ssl/certs/server.pem\"\nkey = "
    "\"/etc/ssl/private/server.key\"\nmin_version = \"1.3\"\n\n[limits]\nmax_body_bytes = 10_485_760\nrate = { burst = "
    "200, per_second = 50 }\n\n# yaml for the sidecar\n---\nsidecar:\n  image: registry.example.com/proxy:1.27.3\n  "
    "resources:\n    limits: {cpu: \"500m\", memory: 256Mi}\n",
    "서울은 대한민국의 수도이자 최대 도시로, 한강을 중심으로 발전해 왔다. 조선 시대부터 600년 넘게 수도의 역할을 해 왔으며, "
    "현재는 정치, 경제, 문화의 중심지이다. 경복궁, 남산서울타워, 북촌 한옥마을 등 많은 관광 명소가 있다.",
    "Abstract. We study the problem of estimating the mean of a distribution from samples, some of which have been "
    "corrupted by an adversary. We show that a simple filtering algorithm achieves the optimal error rate up to "
    "constant factors, and runs in nearly linear time.<|endoftext|>Ingredients: 2 cups flour, 1 tsp baking soda, 1/2 "
    "tsp salt, 1 cup butter, 3/4 cup sugar, 2 eggs, 2 cups chocolate chips. Preheat the oven to 375°F. Mix the dry "
    "ingredients. Cream the butter and sugar, beat in the eggs, then stir in the flour mixture and the chips. Bake "
    "for 9 to 11 minutes.<|endoftext|>\"Are you coming tonight?\" asked Sam.\n\"I don't know,\" said Priya. \"It depends "
    "on whether the train is running.\"\n\"It's always running.\"\n\"Not on Sundays.\"\n\"Today is Saturday.\"\n\"Then I'm "
    "coming.\"<|endoftext|>The quarterly report shows revenue of $4.2 million, up 12% from the same quarter last year, "
    "while operating costs rose 7%. The board approved a dividend of $0.15 per share, payable on March 31. Headcount "
    "grew from 87 to 104, mostly in engineering and support. Management expects growth to slow in the second half as "
    "the new pricing takes effect and two large contracts come up for renewal.",
)

HELDOUT_OPENINGS = (
    "00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............\n00000010: 0300 3e00 0100 0000 1011 0000 0000 "
    "0000  ..>.............\n00000020: 4000 0000 0000 0000 c8a1 0400 0000 0000  @...............\n00000030: 0000 0000 "
    "4000 3800 0d00 4000 1f00 1e00  ....@.8...@.....\n",
    "2026-03-14 09:21:07 INFO worker[12]: started job 8812 (queue=default)\n2026-03-14 09:21:08 WARN worker[12]: "
    "retrying fetch of s3://bucket/part-00017 after timeout (attempt 2/5)\n2026-03-14 09:21:11 INFO worker[12]: fetched "
    "41.2 MiB in 2.8 s\n2026-03-14 09:21:15 ERROR worker[7]: job 8809 failed: disk quota exceeded\n",
    "3141592653589793238462643383279 is about as far as most people get, but the record for reciting the digits of pi "
    "from memory is over seventy thousand. Memorizers usually turn the digits into images and walk through a familiar "
    "building, placing one image in each room.",
    "哈哈哈哈哈哈哈哈哈哈，太好笑了！你昨天发的那个视频我看了三遍。最后那只猫从桌子上跳下来的时候，我差点把咖啡喷到屏幕上。"
    "你是在哪里找到的？能不能再发几个类似的给我？",
)

CALIBRATION = PROMPTS[8:] + CALIBRATION_TEXTS
HELDOUT = PROMPTS[:8] + HELDOUT_TEXTS + HELDOUT_OPENINGS

RUN_UNITS = {"calibration": (" and", " of", " to", " a", " in", " is", " that"),
             "heldout": (" the", "<|endoftext|>", ".", ",", "哈哈哈哈", "0", " I", " de")}
RUN_LENGTHS = {"calibration": (512,), "heldout": (8, 16, 24, 32, 64, 128, 512)}


def run_texts(split: str, specials: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Texts with a run of one token after an ordinary opening, then an ordinary text of `split`: a long enough run
    forms the sink again, and the text after it puts sinks on ordinary tokens (`lm.long_runs`). Calibration's runs are
    long, of other ordinary tokens than the held-out ones, and of every special token in `specials`, whose runs grow
    sink-sized values before the threshold, as the position-0 sweep covers every token there. The held-out runs also
    run short of the threshold, and `<|endoftext|>`'s among them sit in other texts than calibration's."""
    tails = CALIBRATION_TEXTS if split == "calibration" else HELDOUT_TEXTS
    units = RUN_UNITS[split] + (tuple(specials) if split == "calibration" else ())
    lengths = RUN_LENGTHS[split]
    return [(f"run:{u.strip()}x{n}", "The log continues:" + u * n + " " + tails[(i * len(lengths) + j) % len(tails)])
            for i, u in enumerate(units) for j, n in enumerate(lengths)]
