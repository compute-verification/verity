"""The capture plan's capture classes (`check.capture_plan`): how the serve captures each value it commits."""

#: a module's returned tensor, captured at its forward's output
MODULE_OUTPUT = "module_output"
#: a value two Calls of one module body exchange (`<module>/<fx node>`), captured at the call inside the body
CALL_BOUNDARY = "call_boundary"
#: a shared-scale FP8 GEMM's products per row, captured at the GEMM
SCALE_PRODUCTS = "scale_products"
#: a field of a Program's `run` input (`<op_path>/<member>`), captured where the runner sets it
RUN_VALUE = "run_value"
#: the token select's ids, the runner's return
SAMPLED = "sampled"
#: the prompt ids, captured at the embedding's input
INPUT_IDS = "input_ids"
CLASSES = (MODULE_OUTPUT, CALL_BOUNDARY, SCALE_PRODUCTS, RUN_VALUE, SAMPLED, INPUT_IDS)

__all__ = ["CALL_BOUNDARY", "CLASSES", "INPUT_IDS", "MODULE_OUTPUT", "RUN_VALUE", "SAMPLED", "SCALE_PRODUCTS"]
