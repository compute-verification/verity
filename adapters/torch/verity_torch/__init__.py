"""The PyTorch adapter: `frontend` translates a PyTorch module, graph or op into circuits, and `pouw` is PoUW's serving code
over PyTorch (Pearl-C's run on the device, H1's receipts and device halves), which hands `verity`'s PoUW serving package its
tensors when imported. The vLLM adapter sits on it."""
