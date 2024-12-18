import torch
import torch.nn as nn
import tvm
from tvm import relax, te
from tvm.script import relax as R
import numpy as np

class MambaBlockTVM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Initialize TVM module
        self.tvm_module = optimize_ssm_hippo()
        
        # Regular PyTorch layers
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        
    def forward(self, x, training_progress=0.0):
        # Convert input to TVM format
        x_tvm = tvm.nd.array(x.detach().numpy())
        
        # Get optimized HiPPO matrices
        A, B = optimize_hippo_transition('legs', self.args.d_state, training_progress, x.device)
        A_tvm = tvm.nd.array(A.detach().numpy())
        B_tvm = tvm.nd.array(B.detach().numpy())
        
        # Run TVM optimized computation
        y_tvm = self.tvm_module["ssm_forward"](x_tvm, A_tvm, B_tvm)
        
        # Convert back to PyTorch
        y = torch.from_numpy(y_tvm.numpy()).to(x.device)
        
        return y
