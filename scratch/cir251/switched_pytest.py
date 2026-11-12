"""Run pytest with the switch applied in memory only, as a switch pull request would write it: part ``a`` rewrites the bodies of
`dsa_topk_reads.cut` and `dsa_topk_mask_cut.plan` to bind the five `_v2` units (every other reference to a `_v1`, a test's
included, still means the `_v1`); part ``b`` drops the DSA norm's three `opt_in` marks.  Nothing in the tree is written.
Usage: switched_pytest.py MAX_GB {a,b,ab,none} PYTEST_ARGS...; exits itself past MAX_GB of RSS."""
import inspect
import os
import sys
import textwrap
import threading
import time

MAX_GB, PART, ARGS = float(sys.argv[1]), sys.argv[2], sys.argv[3:]


def guard():
    while True:
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        if rss > MAX_GB:
            print(f"GUARD: RSS {rss:.2f} GB past {MAX_GB}", flush=True)
            os._exit(9)
        time.sleep(0.2)


threading.Thread(target=guard, daemon=True).start()

import pytest  # noqa: E402

from verity.definitions.registry import dsa_topk_mask_cut as X, dsa_topk_reads as R  # noqa: E402


def rewrite(module, fn, swaps):
    src = textwrap.dedent(inspect.getsource(fn))
    for a, b in swaps:
        assert a in src, (fn.__name__, a)
        src = src.replace(a, b)
    exec(compile(src, f"<switched {module.__name__}.{fn.__name__}>", "exec"), module.__dict__)


if "a" in PART:
    rewrite(X, X.plan, [("bind(DsaTopKDigitPick,", "bind(DsaTopKDigitPickV2,"), ("bind(DsaTopKMaskBlock,", "bind(DsaTopKMaskBlockV2,")])
    rewrite(R, R.cut, [("bind(X.DsaTopKMaskWhole,", "bind(X.DsaTopKMaskWholeV2,"), ("bind(DsaMaskPositions,", "bind(DsaMaskPositionsV2,"),
                       ("bind(DsaPositionsMerge,", "bind(DsaPositionsMergeV2,")])
    R.CUTS["DsaTopKReadRows_v1"] = R.cut
    from verity.circuits.defs import bind
    from verity.commitments.gates.embedding import read

    print("switched: the cut at 131,072 keys binds", sorted({u.base_id for u, _ in R.cut(
        bind(R.DsaTopKReadRows, N=131072, TOPK=2048, BLOCK=R.BLOCK, READ=read(131072, 576)))}), flush=True)
if "b" in PART:
    from verity.definitions import boolean as BB
    from verity.definitions.registry import boolean_dsa_norm as BN

    BB.load()
    for d in (BN.DsaRMSNormSumSqV3, BN.DsaRMSNormApplyV3, BN.DsaRMSNormV3):
        d.opt_in = False
    BB._TABLE.clear()
    BB._VERSIONS.clear()
    print("switched: the norm's opt_in marks dropped", flush=True)
raise SystemExit(pytest.main(["-p", "no:cacheprovider", *ARGS]))
