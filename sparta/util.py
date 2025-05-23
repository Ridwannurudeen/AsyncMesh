import torch

def dequantize(q: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype):
        """
        Recover float activations from quantized ints and scale factors.
        
        Args:
            q: quantized tensor (a x b)
            scale: scale tensor (a x b)
        Returns:
            acts_hat: dequantized float tensor same shape as q
        """
        acts_fp32 = q.float() * scale
        return acts_fp32.to(target_dtype)
    
def quantize(d: torch.Tensor, num_bits: int):
    """
    Uniformly quantize each head-vector in `d` to `num_bits` bits (symmetric).
    
    Args:
        d: Tensor of shape (a x b)
        num_bits: number of bits for quantization (e.g., 8)
    Returns:
        q: quantized ints of same shape (torch.int8, torch.int16, ...)
        scale: float scale factors of shape (bs, seqlen, num_heads, 1)
    """
    # symmetric quantization in range [-Q, +Q]
    Q = 2**(num_bits-1) - 1
    # compute max absolute value per head-vector
    # shape -> (a x b)
    max_val = d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    # scale factor to map float -> int
    scale = max_val / Q
    # quantize: round to nearest integer in [-Q, +Q]
    q = (d / scale).round().clamp(-Q, Q).to(torch.int8 if num_bits<=8 else torch.int16)
    return q, scale
