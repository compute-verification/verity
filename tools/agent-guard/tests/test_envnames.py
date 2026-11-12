"""`envnames.py` prints the names an env file sets and nothing of any other file: no value, and no line of a key file that a
glob over a secrets directory matched beside the env files."""
import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1]
ENVNAMES = GUARD / "envnames.py"

#: dummy values: none is a credential, and each is told apart in the output
VALUES = {"R2_ACCOUNT": "dummy-account-0001", "R2_TOKEN": "dummy-token-0002", "RELAY_URL": "https://relay.invalid/hook?x=1"}
REFUSED = "{}: refused: line {} is not NAME=VALUE, so this is not an env file; nothing printed"


def run(*files, cwd: Path, stdin: bool = False) -> subprocess.CompletedProcess:
    """`envnames.py FILE...` by path, or from stdin as `ssh HOST python3 - FILE... < envnames.py` hands it over."""
    if stdin:
        with open(ENVNAMES, "rb") as script:
            return subprocess.run([sys.executable, "-", *map(str, files)], stdin=script, capture_output=True, cwd=cwd)
    return subprocess.run([sys.executable, "-B", ENVNAMES, *map(str, files)], capture_output=True, cwd=cwd)


def lines(done: subprocess.CompletedProcess) -> list[str]:
    assert done.stderr == b""
    return done.stdout.decode().splitlines()


def shows_nothing_of(done: subprocess.CompletedProcess, *paths: Path) -> bool:
    """No line of these files, and no value of VALUES, is anywhere in what the reader wrote."""
    out = done.stdout + done.stderr
    texts = [line.strip() for p in paths for line in p.read_bytes().splitlines()] + [v.encode() for v in VALUES.values()]
    return not any(t in out for t in texts if len(t) > 3)


def env_file(path: Path) -> Path:
    path.write_text("# the relay's settings\n\n" + "".join(f"{name}={value}\n" for name, value in VALUES.items())
                    + "export WITH_SPACES='two words'\n  # an indented comment\nEMPTY_QUOTES=\"\"\n")
    return path


def key_file(path: Path) -> Path:
    """A throwaway private key, made here and never used: what a `*` over a secrets directory finds beside the env files."""
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "throwaway", "-f", str(path)], check=True, capture_output=True)
    return path


def test_an_env_files_names_are_listed_and_no_value(tmp_path):
    good = env_file(tmp_path / "relay.env")
    done = run("relay.env", cwd=tmp_path)
    assert lines(done) == ["relay.env: R2_ACCOUNT R2_TOKEN RELAY_URL WITH_SPACES EMPTY_QUOTES"] and done.returncode == 0
    assert shows_nothing_of(done) and b"two words" not in done.stdout
    (tmp_path / "empty.env").write_text("# nothing set yet\n\n")
    (tmp_path / "dos.env").write_bytes(b"A=1\r\nB='x y'\r\n")
    done = run(good, "empty.env", "dos.env", cwd=tmp_path)
    assert lines(done) == [f"{good}: R2_ACCOUNT R2_TOKEN RELAY_URL WITH_SPACES EMPTY_QUOTES", "empty.env: (no variables)", "dos.env: A B"]
    assert done.returncode == 0


def test_a_generated_private_key_is_refused_whole_beside_the_env_file_the_same_glob_matched(tmp_path):
    env_file(tmp_path / "relay.env")
    key = key_file(tmp_path / "drive.pem")
    assert key.read_text().startswith("-----BEGIN ") and len(key.read_text().splitlines()) > 3
    done = run(*sorted(p.name for p in tmp_path.iterdir()), cwd=tmp_path)          # what `DIR/*` expands to
    assert lines(done) == [REFUSED.format("drive.pem", 1), REFUSED.format("drive.pem.pub", 1),
                           "relay.env: R2_ACCOUNT R2_TOKEN RELAY_URL WITH_SPACES EMPTY_QUOTES"]
    assert done.returncode == 1 and shows_nothing_of(done, key, tmp_path / "drive.pem.pub")


def test_a_pem_shaped_file_is_refused_at_its_first_line_and_at_a_base64_line_that_reads_as_an_assignment(tmp_path):
    body = ["QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo", "ZHVtbXlkdW1teWR1bW15ZHVtbXk=", "dGhyb3dhd2F5"]
    pem = tmp_path / "service.pem"
    pem.write_text("\n".join(["-----BEGIN " "PRIVATE KEY-----", *body, "-----END " "PRIVATE KEY-----"]) + "\n")
    done = run(pem, cwd=tmp_path)
    assert lines(done) == [REFUSED.format(pem, 1)] and done.returncode == 1 and shows_nothing_of(done, pem)
    # the same body with no armor: line 2 is `NAME=` to a filter that cuts at `=`, and the name it would print is the secret
    bare = tmp_path / "token"
    bare.write_text("# a key as base64\n" + body[1] + "\n")
    done = run(bare, cwd=tmp_path)
    assert lines(done) == [REFUSED.format(bare, 2)] and done.returncode == 1 and shows_nothing_of(done, bare)
    padded = tmp_path / "padded"
    padded.write_text("MFRGGZDFMZTWQ2LK====\n")
    assert lines(run(padded, cwd=tmp_path)) == [REFUSED.format(padded, 1)]


def test_a_mixed_file_is_refused_whole_though_its_first_lines_are_assignments(tmp_path):
    mixed = tmp_path / "mixed.env"
    mixed.write_text("".join(f"{name}={value}\n" for name, value in VALUES.items()) + "dummy-password-on-its-own-line\nLAST=1\n")
    done = run(mixed, cwd=tmp_path)
    assert lines(done) == [REFUSED.format(mixed, 4)] and done.returncode == 1
    assert shows_nothing_of(done, mixed) and b"R2_TOKEN" not in done.stdout


def test_a_multi_line_values_later_lines_are_not_names(tmp_path):
    # the second line of the quoted value reads as NAME=VALUE on its own; its "name" is the value's text
    quoted = tmp_path / "quoted.env"
    quoted.write_text('FIRST=1\nKEY_PEM="dummy-line-one\nZHVtbXlzZWNvbmRsaW5l=tail\ndummy-line-three"\nAFTER=2\n')
    done = run(quoted, cwd=tmp_path)
    assert lines(done) == [REFUSED.format(quoted, 3)] and done.returncode == 1 and shows_nothing_of(done, quoted)
    assert b"ZHVtbXlzZWNvbmRsaW5l" not in done.stdout
    continued = tmp_path / "continued.env"
    continued.write_text("KEY=dummy-part-one\\\nZHVtbXlwYXJ0dHdv=x\n")
    done = run(continued, cwd=tmp_path)
    assert lines(done) == [REFUSED.format(continued, 2)] and done.returncode == 1 and shows_nothing_of(done, continued)
    single = tmp_path / "single.env"
    single.write_text("KEY='dummy-open\nZHVtbXlvcGVu=x'\n")
    assert lines(run(single, cwd=tmp_path)) == [REFUSED.format(single, 2)]
    # quotes that close on their line, an escaped quote and a `#` in a value are one line's value
    closed = tmp_path / "closed.env"
    closed.write_text("A=\"it's fine\"\nB='say \"hi\"'\nC=a\\\"b\nD=x # y\nE=\"a\\\"b\"\n")
    assert lines(run(closed, cwd=tmp_path)) == [f"{closed}: A B C D E"]


def test_a_file_that_cannot_be_read_gets_one_line_and_the_rest_are_still_listed(tmp_path):
    env_file(tmp_path / "relay.env")
    (tmp_path / "dir.env").mkdir()
    binary = tmp_path / "blob.key"
    binary.write_bytes(bytes(range(256)) * 4)
    done = run("missing.env", "dir.env", "blob.key", "relay.env", cwd=tmp_path)
    assert lines(done) == ["missing.env: not a readable file", "dir.env: not a readable file", REFUSED.format("blob.key", 1),
                           "relay.env: R2_ACCOUNT R2_TOKEN RELAY_URL WITH_SPACES EMPTY_QUOTES"]
    assert done.returncode == 1
    none = run(cwd=tmp_path)
    assert none.returncode == 2 and none.stdout == b"" and none.stderr.startswith(b"usage: python3 envnames.py FILE...")


def test_it_runs_from_stdin_as_over_ssh_with_the_same_output(tmp_path):
    env_file(tmp_path / "relay.env")
    key_file(tmp_path / "id_ed25519")
    by_path, piped = run("relay.env", "id_ed25519", cwd=tmp_path), run("relay.env", "id_ed25519", cwd=tmp_path, stdin=True)
    assert lines(piped) == lines(by_path) == ["relay.env: R2_ACCOUNT R2_TOKEN RELAY_URL WITH_SPACES EMPTY_QUOTES", REFUSED.format("id_ed25519", 1)]
    assert piped.returncode == by_path.returncode == 1 and shows_nothing_of(piped, tmp_path / "id_ed25519")
