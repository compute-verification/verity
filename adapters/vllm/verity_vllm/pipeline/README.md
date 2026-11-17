# verity_vllm.pipeline

The adapter's top layer: the command line, and the stages it takes a workload through. Each stage is a command of
`verity-vllm`, run on its own or in order by `verity-vllm row`, and each writes its records beside the workload for
the next stage and the checks to read.

| Stage | What it does | Command |
| --- | --- | --- |
| Build | Derives the workload's Programs from the pinned vLLM model class, on a CPU. | `verity-vllm build` |
| Match | Serves the workload under the observer and checks the Programs against what vLLM executed. | `verity-vllm match` |
| Commit | Serves the workload with the committer attached, and holds what it committed to what it served. | `verity-vllm commit` |
| Replay | Replays a sample of a Commit's committed values, in the Commit's process or later on a CPU. | `verity-vllm commit-replay` |

The package also holds the committing endpoint, `verity-vllm serve-openai`: an OpenAI-compatible server on the
in-process engine, with every step committed.

## Try it

~~~sh
uv run verity-vllm --help
~~~

This lists every command with a line on what it does, and needs no GPU. Serving and committing need one.
