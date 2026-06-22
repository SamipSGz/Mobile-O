"""
MRopeCombinedCacheView and MRopeSharedCacheManager.

Extends AsyncReasoning's cache classes to support Qwen2-VL's M-RoPE
(4-D cos/sin of shape (3, batch, seq_len, head_dim/2)).

CombinedCacheView.update() crashes with assert cos.ndim == 3 at line 87.
MRopeCombinedCacheView detects 4D cos/sin and handles it correctly.
MRopeSharedCacheManager converts 2D position_ids -> 3D (temporal, h=0, w=0).
"""
import sys, os
# vendored: shared_cache now lives in ~/Mobile-O (no AsyncReasoning path needed)

import torch
from typing import Dict, Optional, Sequence, Tuple, Any

from shared_cache import SharedCacheManager
from shared_cache.combined_cache import CombinedCacheView, combine_cache_from_structure
from shared_cache.cache_block import using_rotary_cache


class MRopeCombinedCacheView(CombinedCacheView):
    """CombinedCacheView extended to handle Qwen2-VL's 4D M-RoPE cos/sin."""

    def update(
        self,
        key_states:   torch.Tensor,
        value_states: torch.Tensor,
        layer_idx:    int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cache_kwargs.get("cos") if cache_kwargs else None

        # Standard 3D cos (Qwen3, LLaMA, etc.) — use parent
        if cos is None or cos.ndim != 4:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # ── 4D M-RoPE branch: cos/sin shape = (3, batch, seq_len, head_dim/2) ──
        sin = cache_kwargs["sin"]
        num_workers = len(self.cache_structure)
        num_new = key_states.shape[-2]

        write_to = (
            [seq[-1] for seq in self.cache_structure]
            if self.write_to is None
            else self.write_to
        )
        assert key_states.shape[0] == num_workers

        if self.input_mask is None:
            selectors = [slice(0, num_new)] * num_workers
        else:
            mask = self.input_mask.to(dtype=torch.bool, device=key_states.device)
            selectors = [mask[wi] for wi in range(num_workers)]

        if self.position_ids is None:
            cp_global = cache_kwargs.get("cache_position")
            cp_by_worker = [cp_global] * num_workers
        else:
            cp_by_worker = self.position_ids.to(device=key_states.device)

        # Slice 4D cos/sin per worker: batch dim is dim=1, seq dim is dim=2
        mkw_by_worker = [
            dict(
                cache_position=(cp_by_worker[wi][sel]
                                if not isinstance(sel, slice)
                                else cp_by_worker[wi][sel]),
                cos=cos[:, wi : wi + 1, sel, :],
                sin=sin[:, wi : wi + 1, sel, :],
            )
            for wi, sel in enumerate(selectors)
        ]

        with using_rotary_cache(self.rotary_cache):
            for wi, (wt, sel, wkw) in enumerate(
                zip(write_to, selectors, mkw_by_worker)
            ):
                wt.append(
                    key_states=key_states[wi : wi + 1, ..., sel, :],
                    value_states=value_states[wi : wi + 1, ..., sel, :],
                    layer_idx=layer_idx,
                    cache_kwargs=wkw,
                )
        return combine_cache_from_structure(self.cache_structure, layer_idx=layer_idx)


class MRopeSharedCacheManager(SharedCacheManager):
    """
    SharedCacheManager for Qwen2-VL.
    1. Converts 2D position_ids -> 3D M-RoPE (temporal=pos, h=0, w=0).
    2. Wraps returned CombinedCacheView in MRopeCombinedCacheView.
    """

    def get_input_kwargs(
        self,
        input_ids,
        attention_mask=None,
        write_to=None,
        active_worker_indices=None,
    ):
        result = super().get_input_kwargs(
            input_ids, attention_mask, write_to, active_worker_indices
        )

        # 2D -> 3D M-RoPE: temporal=pos, height=0, width=0
        pos2d = result["position_ids"]          # (num_workers, num_tokens)
        result["position_ids"] = torch.stack(
            [pos2d, torch.zeros_like(pos2d), torch.zeros_like(pos2d)], dim=0
        )                                        # (3, num_workers, num_tokens)

        # Replace CombinedCacheView with MRopeCombinedCacheView
        old = result["past_key_values"]
        result["past_key_values"] = MRopeCombinedCacheView(
            cache_structure=old.cache_structure,
            write_to=old.write_to,
            input_mask=old.input_mask,
            position_ids=old.position_ids,       # keep 2D for cache_position_by_worker
            override_length=old.override_length,
            rotary_cache=old.rotary_cache,
        )
        return result
