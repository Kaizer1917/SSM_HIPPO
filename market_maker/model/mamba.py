import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from .hippo import transition, optimize_hippo_transition

class ModelArgs:
    def __init__(self, d_model=128, n_layer=4, seq_len=96, d_state=16, expand=2, dt_rank='auto',
                 d_conv=4, pad_multiple=8, conv_bias=True, bias=False,
                 num_channels=24, patch_len=16, stride=8, forecast_len=96, sigma=0.5, reduction_ratio=8, verbose=False):
        self.d_model = d_model
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.d_state = d_state
        self.v = verbose
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.pad_multiple = pad_multiple
        self.conv_bias = conv_bias
        self.bias = bias
        self.num_channels = num_channels
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_len = forecast_len
        self.sigma = sigma
        self.reduction_ratio = reduction_ratio
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.forecast_len % self.pad_multiple != 0:
            self.forecast_len += (self.pad_multiple - self.forecast_len % self.pad_multiple)

        # Optimized state space parameters
        self.d_state_min = d_state
        self.d_state_max = d_state * 2

        # Enhanced patch embedding parameters
        self.patch_overlap = 0.5  # New parameter for overlapping patches
        self.stride = max(1, int(patch_len * (1 - self.patch_overlap)))

        # Progressive expansion parameters
        self.expand_factor = 1.5
        self.max_expansion = 3

class ChannelMixup(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            B, V, L = x.shape
            perm = torch.randperm(V)
            lambda_ = torch.normal(mean=0, std=self.sigma, size=(V,)).to(x.device)
            x_mixed = x + lambda_.unsqueeze(1) * x[:, perm]
            return x_mixed
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).squeeze(-1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).squeeze(-1))))
        out = self.sigmoid(avg_out + max_out)
        return out.unsqueeze(-1)

class PatchMamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([MambaBlock(args) for _ in range(args.n_layer)])

    def forward(self, x, training_progress=0.0):
        for layer in self.layers:
            x = x + layer(x, training_progress)
        return x

class SSM_HIPPOBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.patch_mamba = PatchMamba(args)
        self.channel_attention = ChannelAttention(args.d_model, args.reduction_ratio)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x = self.patch_mamba(x)
        attn = self.channel_attention(x.permute(0, 2, 1))
        x = x * attn.permute(0, 2, 1)
        return self.norm(x)

class SSM_HIPPO(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        # Calculate number of patches
        self.num_patches = (args.seq_len - args.patch_len) // args.stride + 1
        
        # Input projection with skip connection
        self.input_proj = nn.Sequential(
            nn.Linear(args.num_channels, args.d_model),
            nn.LayerNorm(args.d_model),
            nn.GELU()
        )
        self.input_skip = nn.Linear(args.num_channels, args.d_model)
        
        # Residual patch embedding
        self.patch_embed = nn.ModuleList([
            nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(args.d_model * (1 + (i-1)//2 if i > 0 else 1), 
                             args.d_model * (1 + i//2)),
                    nn.LayerNorm(args.d_model * (1 + i//2)),
                    nn.GELU()
                ),
                'skip': nn.Linear(
                    args.d_model * (1 + (i-1)//2 if i > 0 else 1),
                    args.d_model * (1 + i//2)
                ) if i > 0 else None
            }) for i in range(args.n_layer)
        ])
        
        # Initialize SSM blocks with varying dimensions
        self.ssm_blocks = nn.ModuleList([
            MambaBlock(
                ModelArgs(
                    d_model=args.d_model * (1 + i//2),
                    n_layer=args.n_layer,
                    seq_len=args.seq_len,
                    d_state=args.d_state * (1 + i//2),
                    expand=args.expand,
                    dt_rank=args.dt_rank,
                    d_conv=args.d_conv,
                    pad_multiple=args.pad_multiple,
                    conv_bias=args.conv_bias,
                    bias=args.bias,
                    num_channels=args.num_channels,
                    patch_len=args.patch_len,
                    stride=args.stride,
                    forecast_len=args.forecast_len,
                    sigma=args.sigma,
                    reduction_ratio=args.reduction_ratio,
                    verbose=args.v
                )
            ) for i in range(args.n_layer)
        ])
        
        final_dim = args.d_model * (1 + (args.n_layer-1)//2)
        self.norm_f = RMSNorm(final_dim)
        self.output_proj = nn.Linear(
            final_dim * self.num_patches,
            args.num_channels * args.forecast_len
        )

    def forward(self, x, training_progress=0.0):
        B, C, L = x.shape
        
        # Input projection with skip connection
        x_proj = x.transpose(1, 2)  # [B, L, C]
        x = self.input_proj(x_proj)  # [B, L, d_model]
        x = x + self.input_skip(x_proj)  # Skip connection
        x = x.transpose(1, 2)  # [B, d_model, L]
        
        # Create patches with residual connections
        patches = []
        for i in range(0, L - self.args.patch_len + 1, self.args.stride):
            patch = x[:, :, i:i+self.args.patch_len]  # [B, d_model, patch_len]
            patches.append(patch.mean(dim=2))  # [B, d_model]
        
        x = torch.stack(patches, dim=1)  # [B, num_patches, d_model]
        
        # Progressive feature extraction with skip connections
        for i in range(self.args.n_layer):
            # Residual patch embedding
            x_res = x
            x = self.patch_embed[i]['main'](x)
            if i > 0 and self.patch_embed[i]['skip'] is not None:
                x = x + self.patch_embed[i]['skip'](x_res)
            
            # SSM block with residual
            x = x + self.ssm_blocks[i](x, training_progress)
        
        x = self.norm_f(x)
        x = x.reshape(B, -1)
        x = self.output_proj(x)
        
        return x.reshape(B, C, -1)

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x, training_progress=0.0):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_inner)
        (x, res) = x_and_res.chunk(2, dim=-1)  # shape (b, l, d_inner)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        # Get optimized HiPPO matrices here
        A, B = optimize_hippo_transition(
            'legs',
            self.args.d_state,
            training_progress,
            x.device
        )
        
        y = self.ssm(x, A, B)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x, A, B):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Computes the selective scan using HiPPO transition matrices."""
        N = A.shape[-1]  # State dimension (d_state)
        L = delta.shape[-2]  # Sequence length
        
        # Initialize HiPPO transition
        measure = 'legs'
        A_hippo, B_hippo = transition(measure, N)
        A_hippo = torch.as_tensor(A_hippo, dtype=torch.float, device=u.device)
        B_hippo = torch.as_tensor(B_hippo, dtype=torch.float, device=u.device)[:, 0]  # Take first column and flatten
        
        # Initialize state
        x = torch.zeros((*u.shape[:-2], u.shape[-1], N), device=u.device)  # (batch, d_inner, d_state)
        ys = []
        
        # Scan with HiPPO transitions
        for i in range(L):
            # Update state using discretized state space equation
            x = x + delta[..., i, :].unsqueeze(-1) * (
                torch.matmul(x, A_hippo.T) + u[..., i, :].unsqueeze(-1) * B_hippo
            )
            # Generate output - reshape C to match dimensions
            C_i = C[..., i, :].view(*C.shape[:-2], 1, -1)  # Add dimension to match x
            y = torch.sum(x * C_i, dim=-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=-2)  # Shape (..., L, D)
        y = y + u * D.unsqueeze(-2)  # Add skip connection
        
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


if __name__ == "__main__":
    args = ModelArgs(
        d_model=128,          # Dimension of the model
        n_layer=4,            # Number of C-Mamba blocks
        seq_len=96,           # Length of input sequence (look-back window)
        num_channels=17,      # Number of numerical channels in your data
        patch_len=16,         # Length of each patch
        stride=8,             # Stride for patching
        forecast_len=96,      # Number of future time steps to predict
        d_state=16,           # Dimension of SSM state
        expand=2,             # Expansion factor for inner dimension
        dt_rank='auto',       # Rank for delta projection, 'auto' sets it to d_model/16
        d_conv=4,             # Kernel size for temporal convolution
        pad_multiple=8,       # Padding to ensure sequence length is divisible by this
        conv_bias=True,       # Whether to use bias in convolution
        bias=False,           # Whether to use bias in linear layers
        sigma=0.5,            # Standard deviation for channel mixup
        reduction_ratio=4,     # Reduction ratio for channel attention
        verbose=False
    )
    model = SSM_HIPPO(args)
    print(model)
    # Example input
    x = torch.randn(32, args.num_channels, args.seq_len)
    output = model(x)
    print(output.shape)  # Should be (32, forecast_len)
