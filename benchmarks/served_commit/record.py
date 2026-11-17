"""record.py RESULT: from a served-arm run's RESULT (`verity_bench vllm`'s result.json), the compiled Commit arm over its
stock control at each timed point, paired by rep, into the run's result; the batch-32 mixed point last, the one the
dashboard shows, which also writes its time per output token as `OVERHEAD`'s percent over stock. A failed gate leaves no
timed point, so it writes nothing."""
import json
import sys
from pathlib import Path

from verity_bench import kit

ARM = "served-commit-rows-4k"
METRICS = (("circuits/committed-throughput-over-stock", "tokens_per_s"),
           ("circuits/committed-time-to-first-token-over-stock", "time_to_first_token_s"))
OVERHEAD = "benchmarks/served-overhead-commit"


def main(path: Path) -> None:
    doc = json.loads(path.read_text())
    arms = {a["id"]: a for a in doc["identity"]["arms"]}
    if ARM not in arms or not doc["points"]:
        return
    control = arms[ARM]["control"]
    w = kit.writer(path.parent / "metrics")
    card = w.stamp["card"]
    model = doc["identity"]["preset"]["model"]
    machine = "cpu" if card == "none" else "-".join(card.lower().split()).removeprefix("nvidia-")
    short = f"{model.rsplit('/', 1)[-1].lower()}-{machine}"
    for point in sorted(doc["points"], key=lambda p: p["point"]["headline"]):
        at = point["point"]
        fields = {"model": model, "machine": machine, "point": at["id"], "control": control}

        def reps(metric: str) -> list[list[float]]:
            return [[r["metrics"][metric] for r in point["arms"][a]["reps"]] for a in (ARM, control)]

        for name, metric in METRICS:
            p = kit.paired(*reps(metric))
            ci = p["ci95"] and [x + 1 for x in p["ci95"]]
            m = w.measurement(name, p["ratio"], "times stock", ci, mark="measured", member=f"{short}-{at['id']}",
                              inputs=[path.name], extra={**fields, "n": p["n"]})
            print(json.dumps({k: m[k] for k in ("name", "value", "unit", "member", "model", "machine", "point")}))
        if at["headline"]:
            p = kit.paired(*reps("time_per_output_token_s"))
            ci = p["ci95"] and [x * 100 for x in p["ci95"]]
            m = w.measurement(OVERHEAD, p["gamma"] * 100, "percent", ci, mark="measured", member=short,
                              inputs=[path.name], extra={**fields, "n": p["n"], "ratio": p["ratio"]})
            print(json.dumps({k: m[k] for k in ("name", "value", "unit", "member", "model", "machine", "point")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
