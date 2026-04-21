import torch
import torch.distributed as dist

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    from einops import rearrange
    def index_first_axis(*a, **k): raise NotImplementedError("flash_attn not installed")
    def pad_input(*a, **k): raise NotImplementedError("flash_attn not installed")
    def unpad_input(*a, **k): raise NotImplementedError("flash_attn not installed")

try:
    from flash_attn.utils.distributed import all_gather
except ImportError:
    def all_gather(x, *a, **k): return [x]


def gather_and_pad_tensor(tensor, padding_value=0):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)
    if tensor.shape[0] < max_size:
        pad_shape = list(tensor.shape)
        pad_shape[0] = max_size - tensor.shape[0]
        tensor = torch.cat([tensor, torch.full(pad_shape, padding_value, device=tensor.device, dtype=tensor.dtype)], dim=0)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def unpad_and_slice_tensor(tensor, original_length, rank=0):
    if not dist.is_initialized():
        return tensor[:original_length]
    return tensor[rank * original_length : (rank + 1) * original_length]


_ring_attn_group = None

def get_ring_attn_group():
    return _ring_attn_group

def set_ring_attn_group(group):
    global _ring_attn_group
    _ring_attn_group = group
