# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Torch-based HashTable, API-compatible with spconv.pytorch.hash.HashTable.

Sorted arrays + searchsorted (vendor-agnostic GPU ops) instead of a CUDA
open-addressing table; same semantics (map/query/enumerate keys).
"""

import torch


class HashTable:
    def __init__(
        self,
        device: torch.device,
        key_dtype: torch.dtype,
        value_dtype: torch.dtype,
        max_size: int = -1,
    ):
        if key_dtype not in (torch.int32, torch.int64):
            raise AssertionError
        # parity: CUDA tables are fixed-size (require max_size); CPU tables are dynamic (-1).
        if device.type != "cpu" and max_size <= 0:
            raise AssertionError(
                "you must provide max_size for fixed-size cuda hash table, "
                "usually *2 of num of keys"
            )
        self.device = device
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.max_size = max_size
        self.keys_sorted = torch.empty(0, dtype=key_dtype, device=device)
        self.values_sorted = torch.empty(0, dtype=value_dtype, device=device)

    def insert(self, keys: torch.Tensor, values: torch.Tensor | None = None):
        keys = keys.to(self.device).to(self.key_dtype).reshape(-1)
        if values is None:
            values = torch.zeros(
                keys.shape[0], dtype=self.value_dtype, device=self.device
            )
        else:
            values = values.to(self.device).to(self.value_dtype).reshape(-1)
        all_keys = torch.cat([self.keys_sorted, keys])
        all_vals = torch.cat([self.values_sorted, values])
        # last write wins for duplicate keys (matches table overwrite)
        sk, order = torch.sort(all_keys, stable=True)
        sv = all_vals[order]
        uniq, _inverse, counts = torch.unique_consecutive(
            sk, return_inverse=True, return_counts=True
        )
        last_pos = torch.cumsum(counts, 0) - 1
        self.keys_sorted = uniq
        self.values_sorted = sv[last_pos]
        if self.max_size > 0:
            if self.keys_sorted.numel() > self.max_size:
                raise AssertionError("hash table full")

    def query(
        self, keys: torch.Tensor, values: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = keys.to(self.device).to(self.key_dtype).reshape(-1)
        n_stored = self.keys_sorted.numel()
        out_v = torch.zeros(keys.shape[0], dtype=self.value_dtype, device=self.device)
        is_empty = torch.ones(keys.shape[0], dtype=torch.bool, device=self.device)
        if n_stored == 0 or keys.numel() == 0:
            return out_v, is_empty
        pos = torch.searchsorted(self.keys_sorted, keys)
        pos_c = pos.clamp(max=n_stored - 1)
        found = (self.keys_sorted[pos_c] == keys) & (pos < n_stored)
        out_v[found] = self.values_sorted[pos_c[found]]
        is_empty = ~found
        if values is not None:
            values.reshape(-1)[: keys.shape[0]].copy_(out_v)
        return out_v, is_empty

    def insert_exist_keys(self, keys: torch.Tensor, values: torch.Tensor):
        keys = keys.to(self.device).to(self.key_dtype).reshape(-1)
        values = values.to(self.device).to(self.value_dtype).reshape(-1)
        n_stored = self.keys_sorted.numel()
        if n_stored == 0:
            return torch.zeros(keys.shape[0], dtype=torch.bool, device=self.device)
        pos = torch.searchsorted(self.keys_sorted, keys)
        pos_c = pos.clamp(max=n_stored - 1)
        found = (self.keys_sorted[pos_c] == keys) & (pos < n_stored)
        self.values_sorted[pos_c[found]] = values[found]
        return found

    def assign_arange_(self):
        n = self.keys_sorted.numel()
        self.values_sorted = torch.arange(n, dtype=self.value_dtype, device=self.device)
        # parity: count is a shape-[1] device tensor, int32 for int32 keys else int64 (as items()).
        count_dtype = torch.int32 if self.key_dtype == torch.int32 else torch.int64
        return torch.tensor([n], dtype=count_dtype, device=self.device)

    def items(self, max_size: int = -1):
        """Return (keys, values, count) buffers sized max_size (table capacity
        if -1), first `count` entries valid - matches reference signature."""
        if max_size == -1:
            max_size = self.max_size
        n = min(self.keys_sorted.numel(), max_size)
        keys = torch.zeros([max_size], dtype=self.key_dtype, device=self.device)
        values = torch.zeros([max_size], dtype=self.value_dtype, device=self.device)
        keys[:n] = self.keys_sorted[:n]
        values[:n] = self.values_sorted[:n]
        count_dtype = torch.int32 if self.key_dtype == torch.int32 else torch.int64
        count = torch.tensor([n], dtype=count_dtype, device=self.device)
        return keys, values, count
