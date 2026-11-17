# verity_vllm.query

The query decides which of a derived Program's values a Commit must commit. Under the query of record,
`Q_module_body_v1`, those are every value that crosses a module-body boundary in the Program's own dataflow, plus what
an opt-in policy adds, such as the norm scales or the router's softmax. The decision reads the Program's structure and
its [correspondence](../correspondence/README.md), and no table of names.

Its answer is the required-value manifest, bound to what the runtime commits, with the check that a Commit used it.
The package also gives
a derived Program as a call graph (`program_view`), which the benchmarks read. The package's docstring lists its
modules, `required`'s says what is required and where it lives.
