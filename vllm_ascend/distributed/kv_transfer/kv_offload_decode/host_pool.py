# SPDX-License-Identifier: Apache-2.0
"""Runner-owned DSA Host KV pool backed by a Mooncake shared segment."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)

DSA_HOST_POOL_ALIGNMENT = 2 * 1024 * 1024
DSA_HOST_TE_REGISTRATION_LIMIT_BYTES = 64 * 1024 * 1024 * 1024
DSA_MAIN_K_WIDTH = 512
DSA_MAIN_V_WIDTH = 64


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class DSAHostPoolTopology:
    """Tensor-parallel ownership information for one shared Host pool."""

    tp_rank: int = 0
    tp_size: int = 1
    owner_rank: int = 0
    device_id: int = 0
    dp_rank: int = 0
    tp_group: Any = None

    def __post_init__(self) -> None:
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")
        if self.tp_rank < 0 or self.tp_rank >= self.tp_size:
            raise ValueError(
                f"tp_rank out of range: rank={self.tp_rank}, size={self.tp_size}"
            )
        if self.owner_rank < 0 or self.owner_rank >= self.tp_size:
            raise ValueError(
                "owner_rank out of range: "
                f"owner={self.owner_rank}, size={self.tp_size}"
            )


@dataclass(frozen=True)
class DSAHostKVPoolLayout:
    """Interleaved, per-layer aligned ``[K][V]`` Host-pool layout."""

    layer_names: tuple[str, ...]
    num_blocks: int
    block_size: int
    dtype: torch.dtype = torch.bfloat16
    alignment: int = DSA_HOST_POOL_ALIGNMENT
    k_width: int = DSA_MAIN_K_WIDTH
    v_width: int = DSA_MAIN_V_WIDTH

    def __post_init__(self) -> None:
        if not self.layer_names:
            raise ValueError("DSA Host pool requires at least one layer")
        for name, value in (
            ("num_blocks", self.num_blocks),
            ("block_size", self.block_size),
            ("alignment", self.alignment),
            ("k_width", self.k_width),
            ("v_width", self.v_width),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.alignment % self.dtype.itemsize != 0:
            raise ValueError(
                "alignment must be a multiple of dtype itemsize: "
                f"alignment={self.alignment}, itemsize={self.dtype.itemsize}"
            )

    @property
    def num_layers(self) -> int:
        return len(self.layer_names)

    @property
    def k_shape(self) -> tuple[int, int, int, int]:
        return (self.num_blocks, self.block_size, 1, self.k_width)

    @property
    def v_shape(self) -> tuple[int, int, int, int]:
        return (self.num_blocks, self.block_size, 1, self.v_width)

    @property
    def k_numel(self) -> int:
        return self.num_blocks * self.block_size * self.k_width

    @property
    def v_numel(self) -> int:
        return self.num_blocks * self.block_size * self.v_width

    @property
    def k_stride_numel(self) -> int:
        return (
            _align_up(
                self.k_numel * self.dtype.itemsize,
                self.alignment,
            )
            // self.dtype.itemsize
        )

    @property
    def v_stride_numel(self) -> int:
        return (
            _align_up(
                self.v_numel * self.dtype.itemsize,
                self.alignment,
            )
            // self.dtype.itemsize
        )

    @property
    def layer_stride_numel(self) -> int:
        return self.k_stride_numel + self.v_stride_numel

    @property
    def total_numel(self) -> int:
        return self.num_layers * self.layer_stride_numel

    @property
    def total_nbytes(self) -> int:
        return self.total_numel * self.dtype.itemsize

    @property
    def layer_stride_nbytes(self) -> int:
        return self.layer_stride_numel * self.dtype.itemsize

    @property
    def te_registration_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return ``(offset, size)`` ranges that keep every layer intact.

        HIXL registers these NPU-visible addresses as device memory and
        treats 64 GiB as an exclusive upper bound for every individual region.
        All layers in this pool have the same aligned K/V stride, so the
        maximum number of layers per region can be computed directly.
        """
        max_layers = (
            DSA_HOST_TE_REGISTRATION_LIMIT_BYTES - 1
        ) // self.layer_stride_nbytes
        if max_layers <= 0:
            raise ValueError(
                "One DSA Host KV layer exceeds the TE registration limit: "
                f"layer_bytes={self.layer_stride_nbytes}, "
                f"limit={DSA_HOST_TE_REGISTRATION_LIMIT_BYTES}"
            )

        ranges: list[tuple[int, int]] = []
        for first_layer in range(0, self.num_layers, max_layers):
            layer_count = min(max_layers, self.num_layers - first_layer)
            ranges.append(
                (
                    first_layer * self.layer_stride_nbytes,
                    layer_count * self.layer_stride_nbytes,
                )
            )
        return tuple(ranges)

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": 1,
            "order": "layer_interleaved_kv_aligned",
            "layer_names": self.layer_names,
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "dtype": str(self.dtype),
            "alignment": self.alignment,
            "k_width": self.k_width,
            "v_width": self.v_width,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class DSAHostMemoryRegion:
    """One shared allocation and its local tensor view."""

    tensor: torch.Tensor
    handle: Any = None
    register_location: str | None = None
    release_callback: Callable[[Any], None] | None = None
    _released: bool = field(default=False, init=False)

    def release(self) -> None:
        if self._released:
            return
        if self.release_callback is not None:
            self.release_callback(self.handle)
        self.handle = None
        self._released = True


class DSAHostRegionAllocator(Protocol):

    def __call__(
        self,
        *,
        numel: int,
        dtype: torch.dtype,
        alignment: int,
        name: str,
        topology: DSAHostPoolTopology,
    ) -> DSAHostMemoryRegion: ...


def _align_host_tensor(
    tensor: torch.Tensor,
    alignment: int,
) -> torch.Tensor:
    data_ptr = int(tensor.data_ptr())
    aligned_addr = _align_up(data_ptr, alignment)
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _select_shared_segment_mode() -> tuple[bool, bool]:
    """Choose a shared-segment mode that exposes an NPU-accessible VA."""
    try:
        from mooncake.shared_segment import shared_segment_supported
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake shared_segment support is required for the DSA Host pool"
        ) from exc

    # Temporarily force the PR #3952 mmap + HostRegister path for RoCE.
    # if shared_segment_supported(mmap=False):
    #     return False, False
    if shared_segment_supported(mmap=True, host_register=True):
        return True, True
    raise RuntimeError(
        "Mooncake shared_segment cannot expose an NPU-accessible address"
    )


def allocate_mooncake_host_region(
    *,
    numel: int,
    dtype: torch.dtype,
    alignment: int,
    name: str,
    topology: DSAHostPoolTopology,
) -> DSAHostMemoryRegion:
    """Create or map one Mooncake shared segment for the Decode Host pool."""
    try:
        from mooncake.shared_segment import create_shared_segment
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake shared_segment support is required for the DSA Host pool"
        ) from exc

    mmap, host_register = _select_shared_segment_mode()
    if topology.tp_size > 1 and topology.tp_group is None:
        raise RuntimeError(
            "create_shared_segment requires tp_group when tp_size > 1: "
            f"tp={topology.tp_rank}/{topology.tp_size}"
        )

    extra = (int(alignment) + dtype.itemsize - 1) // dtype.itemsize
    alloc_numel = int(numel) + extra
    segment_name = f"{name}_dp{int(topology.dp_rank)}"
    logger.info(
        "Creating DSA Host shared segment name=%s tp=%s/%s owner=%s "
        "device=%s mmap=%s host_register=%s numel=%s",
        segment_name,
        topology.tp_rank,
        topology.tp_size,
        topology.owner_rank,
        topology.device_id,
        mmap,
        host_register,
        numel,
    )
    segment = create_shared_segment(
        segment_name,
        blocks={
            "pool": {
                "count": 1,
                "shape": (alloc_numel,),
                "dtype": dtype,
            }
        },
        world_size=int(topology.tp_size),
        rank_id=int(topology.tp_rank),
        owner_rank=int(topology.owner_rank),
        device_id=int(topology.device_id),
        tp_group=topology.tp_group,
        mmap=mmap,
        host_register=host_register,
    )
    if host_register and os.getenv("VLLM_ASCEND_SKIP_MIGRATEPAGES") is None:
        os.environ["VLLM_ASCEND_SKIP_MIGRATEPAGES"] = "1"

    raw = segment.tensors("pool")[0]
    aligned = _align_host_tensor(raw, alignment)[: int(numel)]
    device = getattr(aligned, "device", None)
    if device is None or getattr(device, "type", None) == "cpu":
        raise RuntimeError(
            "Mooncake shared segment did not expose an NPU tensor: "
            f"device={device}, mmap={mmap}, host_register={host_register}"
        )
    return DSAHostMemoryRegion(
        tensor=aligned,
        handle=segment,
        register_location=f"npu:{int(topology.device_id)}",
    )


class DSAHostKVPool:
    """One shared Host allocation, sliced into aligned per-layer K/V views.

    Owner rank (Decode TP0) creates the segment; other TP ranks map it.
    """

    def __init__(
        self,
        layout: DSAHostKVPoolLayout,
        region: DSAHostMemoryRegion,
        topology: DSAHostPoolTopology | None = None,
    ) -> None:
        self.layout = layout
        self.region = region
        self.topology = topology or DSAHostPoolTopology()
        self._registered_engine: Any = None
        self._registered_ranges: list[tuple[int, int]] = []
        self._closed = False
        self._validate_region()

        flat = region.tensor.reshape(-1)[: layout.total_numel]
        self.k_caches: list[torch.Tensor] = []
        self.v_caches: list[torch.Tensor] = []
        offset = 0
        for _ in layout.layer_names:
            k_cache = flat.narrow(
                0,
                offset,
                layout.k_numel,
            ).view(layout.k_shape)
            offset += layout.k_stride_numel
            v_cache = flat.narrow(
                0,
                offset,
                layout.v_numel,
            ).view(layout.v_shape)
            offset += layout.v_stride_numel
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)
        self._validate_layer_alignment()

    @classmethod
    def allocate(
        cls,
        layout: DSAHostKVPoolLayout,
        allocator: DSAHostRegionAllocator = allocate_mooncake_host_region,
        topology: DSAHostPoolTopology | None = None,
    ) -> DSAHostKVPool:
        resolved_topology = topology or DSAHostPoolTopology()
        region = allocator(
            numel=layout.total_numel,
            dtype=layout.dtype,
            alignment=layout.alignment,
            name="dsa_main_kv_pool",
            topology=resolved_topology,
        )
        try:
            return cls(layout, region, resolved_topology)
        except Exception:
            region.release()
            raise

    @property
    def data_ptr(self) -> int:
        return int(self.region.tensor.data_ptr())

    @property
    def nbytes(self) -> int:
        return self.layout.total_nbytes

    @property
    def is_registered(self) -> bool:
        return self._registered_engine is not None

    @property
    def is_owner(self) -> bool:
        return self.topology.tp_rank == self.topology.owner_rank

    def _validate_region(self) -> None:
        tensor = self.region.tensor
        if tensor.dtype != self.layout.dtype:
            raise TypeError(
                f"DSA Host dtype mismatch: expected {self.layout.dtype}, "
                f"got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError("DSA Host allocation must be contiguous")
        if tensor.numel() < self.layout.total_numel:
            raise ValueError(
                "DSA Host allocation is too small: "
                f"need={self.layout.total_numel}, got={tensor.numel()}"
            )
        if tensor.data_ptr() % self.layout.alignment != 0:
            raise ValueError(
                "DSA Host allocation is not aligned: "
                f"ptr=0x{tensor.data_ptr():x}, "
                f"alignment={self.layout.alignment}"
            )

    def _validate_layer_alignment(self) -> None:
        for layer_id, (k_cache, v_cache) in enumerate(
            zip(self.k_caches, self.v_caches, strict=True)
        ):
            for name, tensor in (
                ("main_k", k_cache),
                ("rope", v_cache),
            ):
                if tensor.data_ptr() % self.layout.alignment != 0:
                    raise ValueError(
                        "DSA Host layer view is not aligned: "
                        f"layer={layer_id}, {name}=0x{tensor.data_ptr():x}, "
                        f"alignment={self.layout.alignment}"
                    )

    def register(self, engine: Any) -> None:
        """Register the mapped pool with TE in layer-aligned subregions."""
        if self._closed:
            raise RuntimeError("cannot register a closed DSA Host pool")
        if self._registered_engine is engine:
            return
        if self._registered_engine is not None:
            raise RuntimeError(
                "DSA Host pool is already registered with another engine"
            )

        ranges = tuple(
            (self.data_ptr + offset, size)
            for offset, size in self.layout.te_registration_ranges
        )
        logger.info(
            "Registering DSA Host pool with TE: ptr=0x%x bytes=%s "
            "regions=%s limit=%s",
            self.data_ptr,
            self.nbytes,
            len(ranges),
            DSA_HOST_TE_REGISTRATION_LIMIT_BYTES,
        )

        registered: list[tuple[int, int]] = []
        try:
            for index, (ptr, size) in enumerate(ranges):
                self._register_te_range(engine, ptr, size)
                registered.append((ptr, size))
                logger.info(
                    "Registered DSA Host TE region %s/%s: ptr=0x%x bytes=%s",
                    index + 1,
                    len(ranges),
                    ptr,
                    size,
                )
        except Exception:
            rollback_failed: list[tuple[int, int]] = []
            for ptr, size in reversed(registered):
                try:
                    self._unregister_te_range(engine, ptr, size)
                except Exception:
                    rollback_failed.append((ptr, size))
                    logger.exception(
                        "Failed to roll back DSA Host TE region: "
                        "ptr=0x%x bytes=%s",
                        ptr,
                        size,
                    )
            if rollback_failed:
                self._registered_engine = engine
                self._registered_ranges = list(reversed(rollback_failed))
            raise

        self._registered_ranges = registered
        self._registered_engine = engine

    def _register_te_range(self, engine: Any, ptr: int, size: int) -> None:
        location = self.region.register_location
        if location is None:
            result = engine.register_memory(ptr, size)
        else:
            try:
                result = engine.register_memory(
                    ptr,
                    size,
                    location=location,
                )
            except TypeError:
                result = engine.register_memory(ptr, size, location)
        if result not in (0, None):
            raise RuntimeError(
                "Mooncake register_memory failed for DSA Host pool region: "
                f"result={result}, ptr=0x{ptr:x}, size={size}, "
                f"location={location}"
            )

    @staticmethod
    def _unregister_te_range(engine: Any, ptr: int, size: int) -> None:
        unregister_memory = getattr(engine, "unregister_memory", None)
        if unregister_memory is None:
            raise RuntimeError(
                "Mooncake engine must unregister the DSA Host pool before release"
            )
        try:
            result = unregister_memory(ptr)
        except TypeError:
            result = unregister_memory(ptr, size)
        if result not in (0, None):
            raise RuntimeError(
                "Mooncake unregister_memory failed for DSA Host pool region: "
                f"result={result}, ptr=0x{ptr:x}, size={size}"
            )

    def unregister(self) -> None:
        if self._registered_engine is None:
            return
        engine = self._registered_engine
        failed: list[tuple[int, int]] = []
        first_error: Exception | None = None
        for ptr, size in reversed(self._registered_ranges):
            try:
                self._unregister_te_range(engine, ptr, size)
            except Exception as exc:
                failed.append((ptr, size))
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            self._registered_ranges = list(reversed(failed))
            raise RuntimeError(
                "Failed to unregister one or more DSA Host TE regions"
            ) from first_error
        self._registered_ranges.clear()
        self._registered_engine = None

    def close(self) -> None:
        if self._closed:
            return
        self.unregister()
        self.region.release()
        self._closed = True
