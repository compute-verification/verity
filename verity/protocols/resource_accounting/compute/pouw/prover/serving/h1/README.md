# H1's serving side

H1 is PoUW's scheme for a served model's int8 linears, and this package is what runs beside the model for it: each
served int8 linear call becomes its own Program, committed while serving and staged for proof when the challenger's
coins sample it. Its receipts and the device halves of its registered rows, call tables and records are the workload's
(the device adapter's), reached through the H1 device the workload hands in (`serving.tensors.h1_device`).

The package's docstring states the scheme, and each module's docstring says what the module holds. An adapter reaches
it through [the serving package](../README.md)'s exports, never directly. There is nothing here to run on its own.
