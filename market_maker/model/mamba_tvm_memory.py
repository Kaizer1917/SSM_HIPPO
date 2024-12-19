import tvm

def optimize_memory_layout(x, layout="NCHW"):
    """Optimize tensor memory layout.
    
    Args:
        x: tvm.te.Tensor, Input tensor to optimize
        layout: str, Memory layout to use ('NCHW' supported)
    
    Returns:
        tvm.te.Tensor: Optimized tensor with specified memory layout
        
    Raises:
        ValueError: If layout is not supported or input is invalid
    """
    if not isinstance(x, tvm.te.Tensor):
        raise ValueError("Input must be a TVM tensor")
        
    supported_layouts = ["NCHW"]
    if layout not in supported_layouts:
        raise ValueError(f"Unsupported layout: {layout}. Supported layouts: {supported_layouts}")

    if layout == "NCHW":
        return tvm.te.compute(
            x.shape,
            lambda n, c, h, w: x[n, c, h, w],
            name="optimized_layout"
        )
    # Add more layouts as needed
    return x
