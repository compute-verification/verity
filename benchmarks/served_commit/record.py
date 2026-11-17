"""record.py RESULT: from a served-arm run's RESULT (`verity_bench vllm`'s result.json), the compiled Commit arm over its
stock control at each timed point, paired by rep, into the run's result; the batch-32 mixed point last, the one the
dashboard shows. A failed gate leaves no timed point, so it writes nothing."""
import json
import sys
from pathlib import Path

from verity_bench import kit

ARM = "served-commit-rows-4k"
METRICS = (("circuits/committed-throughput-over-stock", "tokens_per_s"),
           ("circuits/committed-time-to-first-token-over-stock", "time_to_first_token_s"),
           ("circuits/committed-time-per-output-token-over-stock", "time_per_output_token_s"))


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
    for point in sorted(doc["points"], key=lambda p: p["point"]["headline"]):
        at = point["point"]
        fields = {"model": model, "machine": machine, "point": at["id"], "control": control}
        member = f"{model.rsplit('/', 1)[-1].lower()}-{machine}-{at['id']}"
        for name, metric in METRICS:
            reps = [[r["metrics"][metric] for r in point["arms"][a]["reps"]] for a in (ARM, control)]
            p = kit.paired(*reps)
            ci = p["ci95"] and [x + 1 for x in p["ci95"]]
            m = w.measurement(name, p["ratio"], "times stock", ci, mark="measured", member=member,
                              inputs=[path.name], extra={**fields, "n": p["n"]})
            print(json.dumps({k: m[k] for k in ("name", "value", "unit", "member", "model", "machine", "point")}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
