"""Drop from terms_map.toml's [words] the rows whose key (case folded) draw_words_map.toml renames: one map per word."""
import re, sys, tomllib
repo, mine = sys.argv[1], sys.argv[2]
d = tomllib.load(open(mine, "rb"))
keys = {k.casefold() for k in d.get("words", {})} | {k.casefold() for t in d.get("words_in", []) for k in t["words"]}
p = f"{repo}/tools/move/terms_map.toml"
lines = open(p).read().split("\n")
out, sect, dropped = [], None, []
for ln in lines:
    m = re.match(r"^\[\[?([^\]]+)\]\]?\s*$", ln)
    if m:
        sect = m.group(1)
    k = re.match(r'^"((?:[^"\\]|\\.)*)"\s*=', ln)
    if sect in ("words", "words_in.words") and k and k.group(1).casefold() in keys:
        dropped.append(k.group(1)); continue
    out.append(ln)
open(p, "w").write("\n".join(out))
print(len(dropped), "terms_map rows dropped")
s = open(p).read()
block = ('# The SHA-512 row engine\'s salts and seed tiles are derived from a key or a seed, not sampled\n[[words_in]]\n'
         'files = ["benchmarks/torch_gpu/sha512_rows/**"]\nwords = { "draws" = "derives", "drawn" = "generated" }\n\n')
assert s.count(block) == 1, "sha512_rows table not found"
open(p, "w").write(s.replace(block, ""))
print("sha512_rows table dropped")
