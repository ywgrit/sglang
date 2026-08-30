import torch
import triton
import triton.language as tl


@triton.jit
def _relu2_static_fp8_kernel(
    x_ptr,
    scale_ptr,
    output_ptr,
    num_elements,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Match the standalone ReLU2 kernel: NaN and negative values become zero,
    # and the activation is rounded to BF16 before static quantization.
    relu = tl.where(x > 0.0, x, 0.0)
    activated = (relu * relu).to(tl.bfloat16).to(tl.float32)
    scale_inv = 1.0 / tl.load(scale_ptr).to(tl.float32)
    quantized = tl.clamp(activated * scale_inv, fp8_min, fp8_max).to(tl.float8e4nv)
    tl.store(
        output_ptr + offsets,
        quantized.to(tl.uint8, bitcast=True),
        mask=mask,
    )


def relu2_and_static_quant_fp8(
    x: torch.Tensor, input_scale: torch.Tensor
) -> torch.Tensor:
    """Fuse BF16 ReLU2 with static per-tensor E4M3FN quantization."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.dim() >= 2 and x.numel() > 0 and x.is_contiguous()
    assert input_scale.numel() == 1 and input_scale.device == x.device

    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    block_size = 1024
    fp8 = torch.finfo(torch.float8_e4m3fn)
    _relu2_static_fp8_kernel[(triton.cdiv(x.numel(), block_size),)](
        x,
        input_scale,
        output.view(torch.uint8),
        x.numel(),
        fp8_min=fp8.min,
        fp8_max=fp8.max,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output
