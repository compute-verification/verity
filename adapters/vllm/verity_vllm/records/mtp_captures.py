"""The schema ids of the property records GLM-5.3's MTP draft-layer captures write, each named once: its capture
(`tests/properties/glm53_mtp_<name>_capture_gpu.py`) seals its record under it, and `properties.record.REGISTRY` names its
property by it.  A capture's record is read only as a property record (`properties.record.PropertyRecord`: its `ok` and its
digest), so it has no typed fields here.
"""
from __future__ import annotations

EH_NORM = "verity-vllm/glm53-mtp-eh-norm-capture/v1"
EH_PROJ = "verity-vllm/glm53-mtp-eh-proj-capture/v1"
INDEXER_LOGITS = "verity-vllm/glm53-mtp-indexer-logits-capture/v1"
