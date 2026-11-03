# pyenv.sh, sourced by the ladder's scripts: P, the node's python, with an overlay (on PYTHONPATH) of the modules the
# ladder needs and it lacks, installed without dependencies as prep.sh does, so the closure is listed whole. The overlay
# is a cache outside the run dir (custody would upload it), keyed by the python and the list, and moved into place whole.
P=${LADDER_PYTHON:-/workspace/jobs/venv312/bin/python}
[ -x "$P" ] || { echo "no python at $P; set LADDER_PYTHON to one with torch" >&2; exit 2; }
need=$("$P" - <<'EOF'
import importlib.util
mods = [("tiktoken", "tiktoken==0.14.0"), ("regex", "regex"), ("safetensors", "safetensors"),
        ("huggingface_hub", "huggingface_hub<1"), ("requests", "requests"), ("urllib3", "urllib3"), ("idna", "idna"),
        ("certifi", "certifi"), ("charset_normalizer", "charset-normalizer"), ("filelock", "filelock"),
        ("fsspec", "fsspec"), ("packaging", "packaging"), ("yaml", "pyyaml"), ("tqdm", "tqdm"),
        ("typing_extensions", "typing-extensions")]
print(" ".join(p for m, p in mods if importlib.util.find_spec(m) is None))
EOF
)
echo "pyenv: $P, missing: ${need:-none}"
if [ -n "$need" ]; then
  T="${TMPDIR:-/tmp}/ladder-pylib-$(printf '%s %s' "$P" "$need" | sha256sum | cut -c1-16)"
  if [ ! -d "$T" ]; then
    t=$(mktemp -d "$T.XXXXXX")
    # shellcheck disable=SC2086
    uv pip install -q --python "$P" --target "$t" --no-deps $need
    mv -T "$t" "$T" 2>/dev/null || rm -rf "$t"
  fi
  export PYTHONPATH="$T${PYTHONPATH:+:$PYTHONPATH}"
fi
