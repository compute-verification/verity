for d in /workspace/research/src/*/; do
  l="$d.venv"
  if [ -L "$l" ] && [ "$(readlink "$l")" = /agent/repos/verity/.venv ]; then echo "removing $l"; rm "$l"; fi
done
echo done
