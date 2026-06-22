"""
RawCacheBlock: stores KV without any RoPE rotation.

Use for visual tokens in block_P — they appear at offset=0 in every view
so no re-rotation is needed. Keys are stored at their absolute M-RoPE
positions and returned unchanged.
"""
import torch
from typing import Optional, Dict, Any


class RawCacheBlock:
    """
    KV cache block that stores keys/values as-is (no de-rotation on write,
    no re-rotation on read). Always placed at offset=0 in every view.

    Interface matches CacheBlock: get_seq_length, get_kv_with_offset,
    append, clear.
    """

    def __init__(self, config=None):
        self.config = config
        self._key_cache: Dict[int, list] = {}
        self._val_cache: Dict[int, list] = {}
        self._seq_len:   Dict[int, int]  = {}

    def append(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if layer_idx not in self._key_cache:
            self._key_cache[layer_idx] = []
            self._val_cache[layer_idx] = []
            self._seq_len[layer_idx]   = 0
        self._key_cache[layer_idx].append(key_states.detach())
        self._val_cache[layer_idx].append(value_states.detach())
        self._seq_len[layer_idx] += key_states.shape[-2]

    def get_kv_with_offset(self, layer_idx: int, offset: int = 0):
        """Return stored KV unchanged; offset is ignored."""
        keys   = torch.cat(self._key_cache[layer_idx], dim=-2)
        values = torch.cat(self._val_cache[layer_idx], dim=-2)
        return keys, values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len.get(layer_idx, 0)

    def __contains__(self, layer_idx) -> bool:
        try:
            return layer_idx in self._seq_len
        except TypeError:
            return False

    def clear(self) -> None:
        self._key_cache.clear()
        self._val_cache.clear()
        self._seq_len.clear()
