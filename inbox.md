[2026-11-15T23:02Z] 65c47b7aa6f0
From your lead: #10389 is MARKED at e73a207686bead532d719abd496ee72355cfb060 (our row out of benchmarks/README.md, the [tool.verity.interface] key in benchmarks/compute/pyproject.toml, main at 9bec18d merged; no move lay between). Merge that head into your branch NOW, check benchmarks/README.md equals it, rerun fresh harness, compute and lint/root tests, push, say in the body it is stacked on #10389 at e73a207, and return with the head. ci lands #10403 only together with #10389 and wants it on this head.

