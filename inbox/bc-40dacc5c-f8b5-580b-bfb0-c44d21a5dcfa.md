
## 17:31Z
Post Linear comments only with `research msg linear comment ISSUE "text" --as soundness`, never the Linear MCP connector. The connector posts as Daniel.

## 17:47Z (changes your task)
Daniel corrected the lifecycle at 17:42Z (full wording on Linear TOP-137, 17:50Z):
- The registered compiler B runs at the start of the verification phase, after the trace is committed.
- Its output is an execution block, partitioned into inference blocks, each partitioned into proof blocks.
- The developer proves it ran B correctly on committed advice of at most l_a bits (a public parameter), and that the output explains the committed trace: each proof block's inputs and outputs are committed values.
- The coins come after.
So the restated Declaration is: committed trace, then B(advice) as the execution block with its two-level partition, then a proof that B ran correctly on advice of at most l_a bits, and that the block explains the trace. The circuit and partition are B's output. Add l_a as a public parameter, and add no second description. Tell INFONLY (#6313) when the "explains the trace" link exists, since its tie reads it.
