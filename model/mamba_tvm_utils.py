import tvm

def get_optimal_target():
    """Get optimal target based on hardware"""
    if tvm.cuda(0).exist:
        return tvm.target.cuda()
    elif tvm.rocm(0).exist:
        return tvm.target.rocm()
    else:
        return tvm.target.Target("llvm")

def optimize_for_hardware(mod, target):
    """Apply hardware-specific optimizations"""
    with tvm.transform.PassContext(opt_level=3):
        if target.kind.name == "cuda":
            # CUDA-specific optimizations
            mod = tvm.tir.transform.InjectPTXIntrinsics()(mod)
            mod = tvm.tir.transform.LowerWarpMemory()(mod)
        elif target.kind.name == "llvm":
            # CPU-specific optimizations
            mod = tvm.tir.transform.VectorizeLoop()(mod)
            mod = tvm.tir.transform.LoopPartition()(mod)
    return mod
