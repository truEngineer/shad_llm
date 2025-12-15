import torch
import triton
import triton.language as tl
from torch.library import triton_op


def swiglu_ref(a, b):
    return torch.nn.functional.silu(a) * b


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16),
    ],
    key=['n_elements'], 
)
@triton.jit
def swiglu_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    af = a.to(tl.float32)
    sigmoid_af = 1.0 / (1.0 + tl.exp(-af))
    silu_af = af * sigmoid_af
    silu_a = silu_af.to(a.dtype)
    output = silu_a * b

    tl.store(output_ptr + offsets, output, mask=mask)


@triton_op("llm_scaling_week::swiglu_fwd", mutates_args={})
def swiglu_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_contiguous(), "Tensor a must be contiguous"
    assert b.is_contiguous(), "Tensor b must be contiguous"
    assert a.shape == b.shape, "Tensors a and b must have the same shape"
    assert a.device == b.device, "Tensors must be on the same device"
    assert a.dtype == b.dtype, "Tensors must have the same dtype"
    
    output = torch.empty_like(a)
    n_elements = a.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    swiglu_kernel[grid](
        a, b, output,
        n_elements,
        # BLOCK_SIZE
    )
    
    return output
