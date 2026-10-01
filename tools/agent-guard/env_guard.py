#!/usr/bin/env python3
"""`env` / `printenv` that never print secret values: installed ahead of /usr/bin on agent VMs (tools/agent-guard/install.sh).

A secret is any variable Cursor lists in CLOUD_AGENT_ALL_SECRET_NAMES, any name matching KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL,
and any multi-line value (a multi-line value defeats `env | cut -d= -f1`, which is how the Nebius key leaked four times).
`env CMD ...` and `env -i ...` run CMD unchanged through /usr/bin/env; only printing the environment is redacted.
"""
import os
import re
import sys

REAL = {"env": "/usr/bin/env", "printenv": "/usr/bin/printenv"}
PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE", re.I)


def secret(name: str, value: str) -> bool:
    listed = {n for n in re.split(r"[,\s]+", os.environ.get("CLOUD_AGENT_ALL_SECRET_NAMES", "")) if n}
    return name in listed or bool(PATTERN.search(name)) or "\n" in value


def show(name: str, value: str) -> str:
    return f"{name}=[REDACTED]" if secret(name, value) else f"{name}={value}"


def main() -> int:
    tool = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    if tool == "printenv":
        names = [a for a in args if not a.startswith("-")]
        if not names:
            print("\n".join(show(k, v) for k, v in sorted(os.environ.items())))
            return 0
        rc = 0
        for n in names:
            if n not in os.environ:
                rc = 1
            elif secret(n, os.environ[n]):
                print("[REDACTED]")
            else:
                print(os.environ[n])
        return rc
    prints_env = all(a.startswith("-") or ("=" in a and not a.startswith("=")) for a in args) and "-i" not in args
    if prints_env:
        env = dict(os.environ)
        for a in args:
            if "=" in a and not a.startswith("-"):
                k, v = a.split("=", 1)
                env[k] = v
        print("\n".join(show(k, v) for k, v in sorted(env.items())))
        return 0
    os.execv(REAL["env"], [REAL["env"], *args])


if __name__ == "__main__":
    sys.exit(main())
