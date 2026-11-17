"""The jobs' image builder (pods/nebius/image/build.sh), which a person runs on a node: it parses, and its dry run prints the
content tag a pod names the image by and stops before docker or k3s is touched."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from research.pods import registry

IMAGE = Path(registry.__file__).resolve().parent / "nebius" / "image"


def test_build_parses():
    subprocess.run(["bash", "-n", str(IMAGE / "build.sh")], check=True)


def test_build_dry_run_prints_the_content_tag_and_builds_nothing(tmp_path, monkeypatch):
    calls = tmp_path / "sudo.calls"
    (tmp_path / "sudo").write_text(f'#!/bin/sh\necho "$*" >> {calls}\n[ "$2" = docker ] && echo /var/lib/docker\nexit 0\n')
    (tmp_path / "sudo").chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
    monkeypatch.setenv("MIN_FREE_GB", "0")
    monkeypatch.setenv("EST_GB", "0")
    for name in ("IMAGE_NAME", "DOCKERFILE"):
        monkeypatch.delenv(name, raising=False)
    r = subprocess.run(["bash", str(IMAGE / "build.sh"), "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tag = "verity-base:" + hashlib.sha256((IMAGE / "Dockerfile").read_bytes()).hexdigest()[:16]
    assert r.stdout.splitlines()[0] == f"tag: {tag}"
    ran = calls.read_text().splitlines()
    assert ran == ["-n k3s ctr -n k8s.io images ls -q", "-n docker info -f {{.DockerRootDir}}"]
