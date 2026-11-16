"""kind_at(path, text) -> a function offset -> kind, for the sweep's `kind` column: prose, printed, python, shell, rust, cuda,
lean, flag, env, path, key, stored.  Python by tokenize (comment, docstring: prose; a string: printed, key when it is a dict key
or a subscript; a name: python); Markdown: prose (a code span holding an identifier keeps prose); TOML: a key or table name
is key, a long value prose; shell by line (a comment prose, an option flag, UPPER= env, a quoted string printed); C family
by comment and string; Lean: lean; JSON: key or stored."""
import io, re, tokenize

def _py(text):
    spans = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except Exception:
        return None
    starts = [0]
    for ln in text.split("\n"):
        starts.append(starts[-1] + len(ln) + 1)
    def off(p): return starts[p[0] - 1] + p[1]
    prev = None
    for i, t in enumerate(toks):
        s, e = off(t.start), off(t.end)
        if t.type == tokenize.COMMENT:
            spans.append((s, e, "prose"))
        elif t.type == tokenize.STRING:
            body = t.string.lstrip("rbfuRBFU")
            if body.startswith(('"""', "'''")):
                spans.append((s, e, "prose"))
            else:
                nxt = next((x for x in toks[i + 1:] if x.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT)), None)
                key = (nxt is not None and nxt.string in (":", "]") and prev is not None and prev.string in ("{", ",", "[", "(")) or \
                      (prev is not None and prev.string == "[" and nxt is not None and nxt.string == "]")
                spans.append((s, e, "key" if key else "printed"))
        elif t.type == tokenize.NAME:
            spans.append((s, e, "python"))
        if t.type not in (tokenize.NL, tokenize.COMMENT):
            prev = t
    return spans

def kind_at(path, text):
    ext = path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    if ext == "lean":
        return lambda o: "lean"
    if ext in ("md", "txt", "rst"):
        return lambda o: "prose"
    if ext in ("py", "pyi"):
        spans = _py(text)
        if spans is None:
            return lambda o: "python"
        import bisect
        starts = [s for s, _, _ in spans]
        def f(o):
            i = bisect.bisect_right(starts, o) - 1
            while i >= 0 and spans[i][1] <= o:
                break
            return spans[i][2] if i >= 0 and spans[i][0] <= o < spans[i][1] else "python"
        return f
    lines = text.split("\n")
    cum = [0]
    for ln in lines:
        cum.append(cum[-1] + len(ln) + 1)
    def line_of(o):
        import bisect
        return bisect.bisect_right(cum, o) - 1
    if ext in ("toml",):
        def f(o):
            ln = lines[line_of(o)]; col = o - cum[line_of(o)]
            if ln.lstrip().startswith("#") or "#" in ln[:col] and not re.search(r'["\'].*#', ln[:col]):
                return "prose"
            if ln.lstrip().startswith("["):
                return "key"
            if "=" in ln and col < ln.index("="):
                return "key"
            val = ln[ln.index("=") + 1:] if "=" in ln else ln
            return "prose" if (" " in val.strip().strip('"') and len(val.strip()) > 40) else "key"
        return f
    if ext in ("sh", "bash", "service", "timer", "conf", "cfg", "ini", "yaml", "yml"):
        def f(o):
            ln = lines[line_of(o)]; col = o - cum[line_of(o)]
            if ln.lstrip().startswith("#") or re.search(r"(^|\s)#", ln[:col]):
                return "prose"
            tok = re.search(r"[\w.-]*$", ln[:col]).group(0) if col else ""
            if ln[max(0, col - 2 - len(tok)):col].endswith("--") or ln[:col].rstrip().endswith("--"):
                return "flag"
            word = re.search(r"[\w-]+", ln[col:])
            m = re.search(r"[A-Z][A-Z0-9_]*(?==|\}|:-|:=)", ln[max(0, col - 40):col + 40])
            if m and m.group(0) and re.search(r"[A-Z]", ln[col:col + 1] or "a") or re.match(r"^\s*(export\s+)?[A-Z][A-Z0-9_]*=", ln):
                return "env"
            if ln[:col].count('"') % 2 == 1 or ln[:col].count("'") % 2 == 1:
                return "printed"
            return "shell"
        return f
    if ext in ("rs", "c", "h", "cc", "cpp", "hpp", "cu", "cuh", "ts", "js", "go"):
        lang = {"rs": "rust", "cu": "cuda", "cuh": "cuda"}.get(ext, "cuda" if ext in ("c", "h", "cc", "cpp", "hpp") else "rust")
        spans = [(mo.start(), mo.end(), "prose") for mo in re.finditer(r"//[^\n]*|/\*.*?\*/", text, re.S)] + \
                [(mo.start(), mo.end(), "printed") for mo in re.finditer(r'"(?:[^"\\\n]|\\.)*"', text)]
        spans.sort()
        def f(o):
            for s, e, k in spans:
                if s <= o < e:
                    return k
            return lang
        return f
    if ext in ("json", "jsonl"):
        def f(o):
            ln = lines[line_of(o)]; col = o - cum[line_of(o)]
            m = re.search(r'"\s*:', ln[col:])
            return "key" if m and ln[col:col + m.start()].count('"') == 0 else "stored"
        return f
    return lambda o: "prose"
