"""How many times a program enters one of its functions: a breakpoint at the function's first instruction, held by ptrace
(x86-64 Linux), with no debugger on the machine.

  python3 calls.py BINARY SYMBOL STDIN STDOUT [ARG...]

Runs ``BINARY ARG...`` with the files STDIN and STDOUT as its standard input and output and prints one line of JSON: the
entries into SYMBOL (a function's name as `nm` prints it), the program's exit code (minus its signal, if one ended
it), the address the breakpoint was put at, and the threads traced. Every thread the program makes is traced. While a
thread steps over the breakpoint it is lifted, so an entry another thread makes in that instant is not counted: the
count is exact for a function one thread at a time enters, which is what `replays.sh` counts.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import struct
import subprocess
import sys

libc = ctypes.CDLL(None, use_errno=True)
libc.ptrace.restype = ctypes.c_long
libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
TRACEME, PEEKTEXT, POKETEXT, CONT, SINGLESTEP, GETREGS, SETREGS = 0, 1, 4, 7, 9, 12, 13
SETOPTIONS, O_TRACECLONE, O_EXITKILL, WALL = 0x4200, 0x8, 0x100000, 0x40000000
MASK = (1 << 64) - 1


class Regs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "r15", "r14", "r13", "r12", "rbp", "rbx", "r11", "r10", "r9", "r8", "rax", "rcx", "rdx", "rsi", "rdi", "orig_rax",
        "rip", "cs", "eflags", "rsp", "ss", "fs_base", "gs_base", "ds", "es", "fs", "gs")]


def ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> int:
    ctypes.set_errno(0)
    r = libc.ptrace(request, pid, ctypes.c_void_p(addr), ctypes.c_void_p(data))
    if r == -1 and ctypes.get_errno():
        raise OSError(ctypes.get_errno(), f"ptrace({request}): {os.strerror(ctypes.get_errno())}")
    return r & MASK


def value_of(binary: str, symbol: str) -> int:
    for line in subprocess.run(["nm", binary], capture_output=True, text=True, check=True).stdout.splitlines():
        f = line.split()
        if len(f) == 3 and f[2] == symbol and f[1] in "Tt":
            return int(f[0], 16)
    raise SystemExit(f"{binary} has no function {symbol}")


def entries(binary: str, symbol: str, stdin: str, stdout: str, args: list[str]) -> dict:
    binary = os.path.realpath(binary)
    value = value_of(binary, symbol)
    with open(binary, "rb") as f:
        pie = struct.unpack_from("<H", f.read(18), 16)[0] == 3      # e_type ET_DYN: loaded at a base of the kernel's choice
    pid = os.fork()
    if pid == 0:
        os.dup2(os.open(stdin, os.O_RDONLY), 0)
        os.dup2(os.open(stdout, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), 1)
        libc.ptrace(TRACEME, 0, None, None)
        os.execv(binary, [binary, *args])
    os.waitpid(pid, 0)                                               # stopped at its exec
    base = 0
    if pie:
        with open(f"/proc/{pid}/maps") as maps:
            base = min(int(l.split("-")[0], 16) for l in maps if l.rstrip().endswith(binary))
    addr = base + value
    word = ptrace(PEEKTEXT, pid, addr)
    trap = (word & ~0xFF) | 0xCC
    ptrace(POKETEXT, pid, addr, trap)
    ptrace(SETOPTIONS, pid, 0, O_TRACECLONE | O_EXITKILL)
    ptrace(CONT, pid)
    count, threads = 0, {pid}
    while True:
        tid, status = os.waitpid(-1, WALL)
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            if tid == pid:
                rc = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -os.WTERMSIG(status)
                return {"entries": count, "rc": rc, "address": hex(addr), "pie": pie, "threads": len(threads)}
            continue
        sig = os.WSTOPSIG(status)
        if sig == signal.SIGTRAP and status >> 16:                   # the stop a clone makes in the thread that made it
            ptrace(CONT, tid)
        elif sig == signal.SIGSTOP and tid not in threads:           # a new thread's first stop
            threads.add(tid)
            ptrace(CONT, tid)
        elif sig == signal.SIGTRAP:
            regs = Regs()
            ptrace(GETREGS, tid, 0, ctypes.addressof(regs))
            if regs.rip - 1 == addr:
                count += 1
                ptrace(POKETEXT, tid, addr, word)
                regs.rip = addr
                ptrace(SETREGS, tid, 0, ctypes.addressof(regs))
                ptrace(SINGLESTEP, tid)
                os.waitpid(tid, WALL)
                ptrace(POKETEXT, tid, addr, trap)
            ptrace(CONT, tid)
        else:
            ptrace(CONT, tid, 0, sig)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise SystemExit("usage: calls.py BINARY SYMBOL STDIN STDOUT [ARG...]")
    print(json.dumps(entries(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])))
