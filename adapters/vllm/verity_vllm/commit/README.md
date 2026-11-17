# verity_vllm.commit

This is the Commit: it commits a served run's values while vLLM serves. Its GPU committer commits each value it is
handed under `rows-sha512`, the scheme every Commit commits under. The package also holds what an audit needs of the
Commit afterwards: the challenge's samples and the opened values.

Values reach the committer from the engine through the adapter's acquisition, `acquire`, and its
[sources](../acquire/sources/README.md). The package's
docstring lists its modules.
