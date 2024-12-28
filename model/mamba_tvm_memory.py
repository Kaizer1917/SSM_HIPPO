import tvm
from functools import lru_cache
import numpy as np

def optimize_memory_layout(x, layout="NCHW", device_type="cpu"):
    """Enhanced tensor memory layout optimization"""
    @lru_cache(maxsize=64)  # Increased cache size
    def get_optimized_layout(shape, layout_type, dev_type):
        # Create compute definition
        if layout_type == "NCHW":
            # Add vectorization hint for CPU
            if dev_type == "cpu":
                return tvm.te.compute(
                    shape,
                    lambda n, c, h, w: x[n, c, h, w],
                    name="optimized_layout",
                    attrs={"layout": layout_type, "vectorize": True}
                )
            # Add thread binding for GPU
            elif dev_type == "gpu":
                return tvm.te.compute(
                    shape,
                    lambda n, c, h, w: x[n, c, h, w],
                    name="optimized_layout",
                    attrs={"layout": layout_type, "gpu_thread_bound": True}
                )
    
    # Get optimized layout with hardware hints
    opt_layout = get_optimized_layout(tuple(x.shape), layout, device_type)
    
    # Enhanced memory optimizations
    with tvm.transform.PassContext(opt_level=3):
        # Apply memory optimizations
        opt_layout = tvm.tir.transform.StorageRewrite()(opt_layout)
        opt_layout = tvm.tir.transform.VectorizeLoop()(opt_layout)
        opt_layout = tvm.tir.transform.InjectDoubleBuffer()(opt_layout)
        
        # Add hardware-specific optimizations
        if device_type == "gpu":
            opt_layout = tvm.tir.transform.InjectPTXIntrinsics()(opt_layout)
            opt_layout = tvm.tir.transform.ThreadSync("shared")(opt_layout)
        
    return opt_layout
