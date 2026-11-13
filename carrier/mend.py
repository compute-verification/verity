"""One commit on top of a carried head that replaces a string in one file, made with git plumbing (no checkout).

    python3 mend.py PARENT PATH OLD NEW COUNT MESSAGE_FILE

It refuses unless PATH at PARENT holds OLD exactly COUNT times, and prints the new commit's id. The commit's author and
committer are the checkout's own configured identity (no GIT_AUTHOR_* or GIT_COMMITTER_* is read from here).
"""
import os
import subprocess
import sys
import tempfile


def git(*a, env=None, data=None):
    p = subprocess.run(["git", *a], capture_output=True, input=data, env=env)
    if p.returncode:
        sys.exit(f"git {' '.join(a)}: {p.stderr.decode()[-400:]}")
    return p.stdout


def main():
    parent, path, old, new, count, message = sys.argv[1:7]
    for k in list(os.environ):
        if k.startswith(("GIT_AUTHOR_", "GIT_COMMITTER_")):
            sys.exit(f"{k} is set: the commit is the checkout's own identity's")
    text = git("show", f"{parent}:{path}").decode("utf-8")
    if text.count(old) != int(count):
        sys.exit(f"{path} at {parent[:12]} holds the old text {text.count(old)} time(s), not {count}: nothing made")
    mode = git("ls-tree", parent, "--", path).decode().split()[0]
    blob = git("hash-object", "-w", "--stdin", data=text.replace(old, new).encode("utf-8")).decode().strip()
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(d, "index")}
        git("read-tree", parent, env=env)
        git("update-index", "--cacheinfo", f"{mode},{blob},{path}", env=env)
        tree = git("write-tree", env=env).decode().strip()
    commit = git("commit-tree", tree, "-p", parent, "-F", message).decode().strip()
    changed = git("diff", "--stat", parent, commit).decode().strip().splitlines()
    print(commit)
    print("\n".join(changed), file=sys.stderr)


main()
