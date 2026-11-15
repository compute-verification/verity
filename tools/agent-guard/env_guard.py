#!/usr/bin/env python3
"""`env` / `printenv` that never print secret values: installed ahead of /usr/bin on agent VMs (tools/agent-guard/install.sh).

A secret is any variable Cursor lists in CLOUD_AGENT_ALL_SECRET_NAMES, any name matching KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL,
and any multi-line value (a multi-line value defeats `env | cut -d= -f1`, which is how the Nebius key leaked four times).
`env CMD ...` and `env -i ...` run CMD unchanged through /usr/bin/env; only printing the environment is redacted. Whatever
`env` would print without running a command (`env -u NAME`, `env -v`, an option this file doesn't know) prints redacted, and
`-0` / `--null` keep their NUL-separated form, so `env -0 | cut -z -d= -f1` lists names.
"""
import os
import re
import sys

REAL = {"env": "/usr/bin/env", "printenv": "/usr/bin/printenv"}
PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE", re.I)
NULL = ("-0", "--null")


def secret(name: str, value: str) -> bool:
    listed = {n for n in re.split(r"[,\s]+", os.environ.get("CLOUD_AGENT_ALL_SECRET_NAMES", "")) if n}
    return name in listed or bool(PATTERN.search(name)) or "\n" in value


def show(name: str, value: str) -> str:
    return f"{name}=[REDACTED]" if secret(name, value) else f"{name}={value}"


def emit(lines: list[str], null: bool) -> None:
    sep = "\0" if null else "\n"
    sys.stdout.write("".join(line + sep for line in lines))


def parse_env(args: list[str]) -> tuple[bool, bool, list[str], dict[str, str], list[str]]:
    """`env`'s argv as (null, ignore, unset names, NAME=VALUE assignments, the command and its arguments, if any)."""
    null = ignore = False
    unset: list[str] = []
    assigns: dict[str, str] = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a in NULL:
            null = True
        elif a in ("-i", "--ignore-environment", "-"):
            ignore = True
        elif a in ("-u", "--unset"):
            i += 1
            unset += args[i:i + 1]
        elif a.startswith("--unset="):
            unset.append(a.split("=", 1)[1])
        elif a.startswith("-u") and not a.startswith("--"):
            unset.append(a[2:])
        elif a == "--":
            i += 1
            break
        elif a.startswith("-"):
            pass                     # unknown to this file: without a command it still prints redacted
        elif "=" in a and not a.startswith("="):
            k, v = a.split("=", 1)
            assigns[k] = v
        else:
            break
        i += 1
    return null, ignore, unset, assigns, args[i:]


def main() -> int:
    tool = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    if tool == "printenv":
        null = any(a in NULL for a in args)
        names = [a for a in args if not a.startswith("-")]
        if not names:
            emit([show(k, v) for k, v in sorted(os.environ.items())], null)
            return 0
        rc, out = 0, []
        for n in names:
            if n not in os.environ:
                rc = 1
            elif secret(n, os.environ[n]):
                out.append("[REDACTED]")
            else:
                out.append(os.environ[n])
        emit(out, null)
        return rc
    null, ignore, unset, assigns, command = parse_env(args)
    if command:
        os.execv(REAL["env"], [REAL["env"], *args])
    env = {} if ignore else dict(os.environ)
    for n in unset:
        env.pop(n, None)
    env.update(assigns)
    emit([show(k, v) for k, v in sorted(env.items())], null)
    return 0


if __name__ == "__main__":
    sys.exit(main())
