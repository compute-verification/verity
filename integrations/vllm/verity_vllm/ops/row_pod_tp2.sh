#!/usr/bin/env bash
# sweep/row_pod_tp2.sh -- [lane vllm-tp-n] compatibility shim: the TP2 row runner is `tp_stage.sh` at WORLD=2 (one pipeline for every world;
# world-2 identities, flags and evidence layout unchanged).  Callers of record (R19 notes, research Tools) keep working; new callers use
#   WORLD=N bash verity_vllm/ops/tp_stage.sh <row-id> <ROLE> <hf-repo> <revision> [stages]
WORLD=${WORLD:-2} exec bash "$(dirname "${BASH_SOURCE[0]}")/tp_stage.sh" "$@"
