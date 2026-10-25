"""The top-up-off arms of the bandwidth audit's breakdown (``llama31-8b-fp8-1gpu-bandwidth-ablation``), built with memory
accounting's supported ``BandwidthArm`` (``benchmarks/vllm_encoded/README.md``). Spans time where each step's time goes,
but the top-up runs beside the step, so its cost shows only as the other parts' stretch: these two arms are the spans
arms with the top-up never started, and the top-up's contention is the audit's and plain's time over theirs.

~~~text
bandwidth-plain-spans-no-topup   PLAIN_SPANS with topup=False
bandwidth-audit-spans-no-topup   AUDIT_SPANS with topup=False
~~~

Both are token-controlled by ``bandwidth-plain`` and measure cost only: ``credit`` keeps them uncredited.
"""

from __future__ import annotations

from benchmarks.vllm_encoded.bench_arm import PLAIN, BandwidthArm

PLAIN_SPANS_NO_TOPUP = BandwidthArm("bandwidth-plain-spans-no-topup", encoded=False, control=PLAIN.id, spans=True,
                                    topup=False)
AUDIT_SPANS_NO_TOPUP = BandwidthArm("bandwidth-audit-spans-no-topup", encoded=True, control=PLAIN.id, spans=True,
                                    topup=False)

__all__ = ["AUDIT_SPANS_NO_TOPUP", "PLAIN_SPANS_NO_TOPUP"]
