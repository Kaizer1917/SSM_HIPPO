import tvm
from functools import lru_cache

def optimize_memory_layout(x, layout="NCHW"):
    """Optimize tensor memory layout with caching"""
    @lru_cache(maxsize=32)
    def get_optimized_layout(shape, layout_type):
        if layout_type == "NCHW":
            return tvm.te.compute(
                shape,
                lambda n, c, h, w: x[n, c, h, w],
                name="optimized_layout",
                attrs={"layout": layout_type}
            )
    
    # Use cached layout if available
    opt_layout = get_optimized_layout(tuple(x.shape), layout)
    
    # Add memory pool for better allocation
    with tvm.transform.PassContext(opt_level=3):
        opt_layout = tvm.tir.transform.StorageRewrite()(opt_layout)
    
    return opt_layout
