"""Scratch for MEM-310's sizing, never merged: writes Lean probes of what the kernel pays for one ECPP step.

    python3 gen.py CERT OUT.lean STEP[,STEP...] [--variants kernel,decide,rfl] [--fermat]

CERT is the PARI/GP certificate `cm-cert-p16448-c21065` (art:8bd64630…, internal/efficient-crypto/p2-prime/). Each
STEP is an index into it; the probe states, for that step's N, curve a4, point (x, y), cofactor s and q = (N+1-t)/s,
that the plain-`Nat` Jacobian check accepts: R = [q-1]([s]P) has Z coprime to N and R = -[s]P. `--fermat` adds
3^(N-1) mod N = 1 by a square-and-multiply loop, the cost of one N-1 style exponentiation of the same size.
"""
import ast
import sys

sys.set_int_max_str_digits(0)

HEADER = '''/-! Scratch probe for MEM-310 (never merged): the kernel's cost of one ECPP step's check, on plain `Nat`. -/
set_option maxRecDepth 200000
set_option Elab.async false

namespace PrimeCertProbe

def sub (N a b : Nat) : Nat := (a + N - b) % N

/-- Doubling in Jacobian coordinates on y^2 = x^3 + a x + b, every value reduced mod N (dbl-2007-bl, a general). -/
def dbl (N a : Nat) : Nat × Nat × Nat → Nat × Nat × Nat
  | (X, Y, Z) =>
    let YY := Y * Y % N
    let S := 4 * (X * YY % N) % N
    let ZZ := Z * Z % N
    let M := (3 * (X * X % N) + a * (ZZ * ZZ % N)) % N
    let X3 := sub N (M * M % N) (2 * S % N)
    let Y3 := sub N (M * sub N S X3 % N) (8 * (YY * YY % N) % N)
    (X3, Y3, 2 * (Y * Z % N) % N)

/-- Addition in Jacobian coordinates (add-1998-cmo-2); Z3 = Z1 Z2 H, so a unit Z3 means neither input was O and
H was a unit. -/
def add (N : Nat) : Nat × Nat × Nat → Nat × Nat × Nat → Nat × Nat × Nat
  | (X1, Y1, Z1), (X2, Y2, Z2) =>
    let Z1Z1 := Z1 * Z1 % N
    let Z2Z2 := Z2 * Z2 % N
    let U1 := X1 * Z2Z2 % N
    let U2 := X2 * Z1Z1 % N
    let S1 := Y1 * (Z2 * Z2Z2 % N) % N
    let S2 := Y2 * (Z1 * Z1Z1 % N) % N
    let H := sub N U2 U1
    let R := sub N S2 S1
    let HH := H * H % N
    let HHH := H * HH % N
    let V := U1 * HH % N
    let X3 := sub N (sub N (R * R % N) HHH) (2 * V % N)
    let Y3 := sub N (R * sub N V X3 % N) (S1 * HHH % N)
    (X3, Y3, Z1 * Z2 % N * H % N)

/-- Left-to-right double-and-add over bits `i - 1` down to `0` of `k`. -/
def ladder (N a k : Nat) (P : Nat × Nat × Nat) : Nat → Nat × Nat × Nat → Nat × Nat × Nat
  | 0, acc => acc
  | i + 1, acc =>
    let d := dbl N a acc
    ladder N a k P i (bif k.testBit i then add N d P else d)

def smul (N a k : Nat) (P : Nat × Nat × Nat) : Nat × Nat × Nat :=
  ladder N a k P (Nat.log2 k) P

/-- One Goldwasser–Kilian step: Q = [s]P, R = [q-1]Q with Z(R) a unit mod N, and R = -Q. -/
def check (N a x y s q : Nat) : Bool :=
  let Q := smul N a s (x, y, 1)
  let R := smul N a (q - 1) Q
  let zq2 := Q.2.2 * Q.2.2 % N
  let zr2 := R.2.2 * R.2.2 % N
  Nat.gcd R.2.2 N == 1 && R.1 * zq2 % N == Q.1 * zr2 % N &&
    (R.2.1 * (zq2 * Q.2.2 % N) + Q.2.1 * (zr2 * R.2.2 % N)) % N == 0

def powLoop (N b e : Nat) : Nat → Nat → Nat
  | 0, acc => acc
  | i + 1, acc =>
    let a2 := acc * acc % N
    powLoop N b e i (bif e.testBit i then a2 * b % N else a2)

def powMod (N b e : Nat) : Nat := powLoop N b e (Nat.log2 e) b
'''


def mark(name: str, when: str) -> str:
    return f'#eval (do IO.println s!"mark {name} {when} {{← IO.monoMsNow}}" : IO Unit)'


def main() -> None:
    cert_path, out, steps = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    variants = ["kernel"]
    if "--variants" in sys.argv:
        variants = sys.argv[sys.argv.index("--variants") + 1].split(",")
    cert = ast.literal_eval(open(cert_path).read().strip())
    body = [HEADER]
    for i in steps:
        N, t, s, a4, (x, y) = cert[i]
        q = (N + 1 - t) // s
        assert (N + 1 - t) % s == 0
        bits = N.bit_length()
        body.append(f"\n/-- Step {i}: N of {bits} bits, s of {s.bit_length()} bits, q of {q.bit_length()} bits. -/")
        body.append(f"def N{i} : Nat := {N}\ndef a{i} : Nat := {a4 % N}\ndef x{i} : Nat := {x % N}")
        body.append(f"def y{i} : Nat := {y % N}\ndef s{i} : Nat := {s}\ndef q{i} : Nat := {q}")
        for v in variants:
            goal = f"check N{i} a{i} x{i} y{i} s{i} q{i} = true"
            proof = {"kernel": "by decide +kernel", "decide": "by decide", "rfl": "rfl"}[v]
            body.append(f"{mark(f'step{i}_{v}', 'start')}\ntheorem step{i}_{v} : {goal} := {proof}\n{mark(f'step{i}_{v}', 'end')}")
        if "--fermat" in sys.argv:
            body.append(f"{mark(f'fermat{i}', 'start')}\ntheorem fermat{i} : powMod N{i} 3 (N{i} - 1) = 1 := by decide +kernel\n{mark(f'fermat{i}', 'end')}")
    body.append("\nend PrimeCertProbe\n")
    open(out, "w").write("\n".join(body))


if __name__ == "__main__":
    main()
