# Runbook step 1: 8x H200 at sm_90 with 132 SMs, the pinned torch and vLLM, GlmMoeDsaForCausalLM registered, the checkpoint as pinned
# (names, sizes, config digest; the shard digests are hashed in warm.sh). Reports the bootstrap's built artifacts without requiring them.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
{
stamp boot
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader
df -h /workspace /workspace/fs | tail -2
pgrep -af 'pod_bootstrap' | grep -v pgrep | head -3
for f in /workspace/cp/fa2/build/matReqG/verity_fa2_matReqG.so /workspace/cp/fa2/build/matReq/verity_fa2_matReq.so \
         /workspace/cp/fa2/build/fa3_matReqG/verity_fa3_matReqG.so /workspace/cp/norm_tap/build/verity_norm_tap.so \
         /workspace/cp/router_tap/build/verity_router_tap.so /workspace/cp/nc_build; do
  [ -e "$f" ] && echo "artifact present: $f" || echo "artifact absent: $f"
done
"$PY" - "$CK" "$CMAP" <<'EOF'
import hashlib, json, os, shutil, subprocess, sys
import importlib.metadata as md
import torch, vllm
ck, cmap = sys.argv[1], sys.argv[2]
fail, rec = [], {}
rec["jit_tools"] = {t: shutil.which(t) for t in ("ninja", "nvcc")}
if not all(rec["jit_tools"].values()):
    fail.append("ninja/nvcc not on PATH (FlashInfer's all-reduce JIT)")
else:
    rec["nvcc"] = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
n = torch.cuda.device_count()
rec["gpus"] = [{"name": torch.cuda.get_device_name(i), "cc": list(torch.cuda.get_device_capability(i)),
                "sms": torch.cuda.get_device_properties(i).multi_processor_count,
                "mem_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)} for i in range(n)]
if n != 8 or any(g["cc"] != [9, 0] or g["sms"] != 132 or "H200" not in g["name"] for g in rec["gpus"]):
    fail.append("devices")
rec.update(torch=torch.__version__, cuda=torch.version.cuda, vllm=vllm.__version__, torchcodec_dist=md.version("torchcodec"),
           cutlass_dsl=md.version("nvidia-cutlass-dsl"))
if vllm.__version__ != "0.28.1rc1.dev472+gd9105ea80":
    fail.append("vllm version")
if torch.__version__ != "2.13.0+cu129":
    fail.append("torch version")
from vllm.model_executor.models.registry import ModelRegistry
rec["registered"] = "GlmMoeDsaForCausalLM" in ModelRegistry.get_supported_archs()
if not rec["registered"]:
    fail.append("GlmMoeDsaForCausalLM not registered")
try:
    import torchcodec
    import vllm.multimodal  # noqa: F401
    rec["torchcodec"] = [torchcodec.__version__, torchcodec.__file__]
except Exception as e:  # noqa: BLE001
    fail.append(f"vllm.multimodal: {e!r}"[:300])
try:
    from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
    rec["ll_bf16_available"] = bool(ll_bf16.is_available())
except Exception as e:  # noqa: BLE001
    rec["ll_bf16_available"] = f"import failed: {e!r}"[:300]
pin = next(c for c in json.load(open("manifests/checkpoints.json"))["checkpoints"] if c.get("role") == "GLM53")
sums = {}
for ln in open(os.path.join(ck, "SHA256SUMS")):
    h, p = ln.split(None, 1)
    sums[p.strip().lstrip("*")] = h
tree = {e["path"]: e["size"] for e in json.load(open(ck + ".tree.json")) if e["type"] == "file"}
shards = pin["safetensors_sha256"]
on_disk = sorted(f for f in os.listdir(ck) if f.endswith(".safetensors"))
rec["shards"] = {"pinned": len(shards), "on_disk": len(on_disk),
                 "sums_differ": sorted(f for f, h in shards.items() if sums.get(f) != h),
                 "size_differs": sorted(f for f in shards if not os.path.isfile(os.path.join(ck, f)) or os.path.getsize(os.path.join(ck, f)) != tree.get(f)),
                 "unpinned_on_disk": sorted(set(on_disk) - set(shards))}
cfg = hashlib.sha256(open(os.path.join(ck, "config.json"), "rb").read()).hexdigest()
rec["config_sha256"] = cfg
want = json.load(open(cmap))["pins"]["checkpoint"]["config_sha256"]
if cfg != pin["config_sha256"] or cfg != want:
    fail.append("config.json digest")
s = rec["shards"]
if s["pinned"] != 141 or s["on_disk"] != 141 or s["sums_differ"] or s["size_differs"] or s["unpinned_on_disk"]:
    fail.append("shards")
rec["fail"] = fail
json.dump(rec, open(os.path.join(os.environ["RESEARCH_RUN_DIR"], "out", "boot.json"), "w"), indent=1)
print(json.dumps(rec, indent=1))
print("BOOT", "OK" if not fail else "FAIL " + "; ".join(fail))
sys.exit(1 if fail else 0)
EOF
} 2>&1 | tee "$O/boot.log"
exit "${PIPESTATUS[0]}"
