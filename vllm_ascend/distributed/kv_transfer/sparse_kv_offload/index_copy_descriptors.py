"""Build graph-stable Mooncake index_copy descriptors on the NPU."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element

_DESCRIPTOR_BLOCK_SIZE = 32


@triton.jit(
    do_not_specialize=[
        "slots_ptr",
        "chunk_sources_ptr",
        "chunk_destinations_ptr",
        "chunk_counts_ptr",
        "rows",
        "host_capacity",
    ]
)
def compact_index_copy_descriptors_kernel(
    slots_ptr,
    chunk_sources_ptr,
    chunk_destinations_ptr,
    chunk_counts_ptr,
    rows,
    host_capacity,
    BLOCK: tl.constexpr,
):
    """Compact valid source/destination pairs within fixed-size chunks."""
    chunk = tl.program_id(0)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    active = offsets < rows
    slots = tl.load(slots_ptr + offsets, mask=active, other=-1).to(tl.int64)
    valid = active & (slots >= 0) & (slots < host_capacity)
    valid_i32 = valid.to(tl.int32)
    chunk_offset = chunk * BLOCK
    count = 0
    for index in range(BLOCK):
        if get_element(valid_i32, (index,)) == 1:
            tl.store(
                chunk_sources_ptr + chunk_offset + count,
                get_element(offsets, (index,)),
            )
            tl.store(
                chunk_destinations_ptr + chunk_offset + count,
                get_element(slots, (index,)),
            )
            count += 1
    tl.store(chunk_counts_ptr + chunk, count)


@triton.jit(
    do_not_specialize=[
        "chunk_sources_ptr",
        "chunk_destinations_ptr",
        "chunk_counts_ptr",
        "sources_ptr",
        "destinations_ptr",
        "count_ptr",
    ]
)
def gather_index_copy_descriptors_kernel(
    chunk_sources_ptr,
    chunk_destinations_ptr,
    chunk_counts_ptr,
    sources_ptr,
    destinations_ptr,
    count_ptr,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Gather compacted chunks in token order and publish the valid count."""
    offsets = tl.arange(0, BLOCK)
    write_position = 0
    for chunk in range(NUM_CHUNKS):
        count = tl.load(chunk_counts_ptr + chunk)
        active = offsets < count
        chunk_offset = chunk * BLOCK
        sources = tl.load(
            chunk_sources_ptr + chunk_offset + offsets,
            mask=active,
            other=0,
        )
        destinations = tl.load(
            chunk_destinations_ptr + chunk_offset + offsets,
            mask=active,
            other=0,
        )
        tl.store(sources_ptr + write_position + offsets, sources, mask=active)
        tl.store(
            destinations_ptr + write_position + offsets,
            destinations,
            mask=active,
        )
        write_position += count
    tl.store(count_ptr, write_position)


@triton.jit(
    do_not_specialize=[
        "sources_ptr",
        "destinations_ptr",
        "count_ptr",
        "capacity",
    ]
)
def pad_index_copy_descriptors_kernel(
    sources_ptr,
    destinations_ptr,
    count_ptr,
    capacity,
    BLOCK: tl.constexpr,
):
    """Pad the fixed graph shape with the last valid pair.

    Repeating one valid copy is idempotent. An all-invalid input preserves the
    legacy callback behavior by filling both index arrays with zero.
    """
    count = tl.load(count_ptr)
    has_valid = count > 0
    fallback_position = tl.maximum(count - 1, 0)
    fallback_source = tl.load(
        sources_ptr + fallback_position,
        mask=has_valid,
        other=0,
    )
    fallback_destination = tl.load(
        destinations_ptr + fallback_position,
        mask=has_valid,
        other=0,
    )
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    padding = (offsets < capacity) & (offsets >= count)
    tl.store(sources_ptr + offsets, fallback_source, mask=padding)
    tl.store(destinations_ptr + offsets, fallback_destination, mask=padding)


def build_index_copy_descriptors(
    slots: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    count: torch.Tensor,
    chunk_sources: torch.Tensor,
    chunk_destinations: torch.Tensor,
    chunk_counts: torch.Tensor,
    host_capacity: int,
) -> None:
    """Build fixed-shape descriptors without a device-to-host round trip."""
    rows = slots.numel()
    capacity = sources.numel()
    tensors = (
        slots,
        sources,
        destinations,
        count,
        chunk_sources,
        chunk_destinations,
        chunk_counts,
    )
    if slots.dtype != torch.int64:
        raise ValueError("Mooncake index_copy slots must use int64")
    if sources.dtype != torch.int64 or destinations.dtype != torch.int64:
        raise ValueError("Mooncake index_copy descriptors must use int64")
    if count.dtype != torch.int32 or chunk_counts.dtype != torch.int32:
        raise ValueError("Mooncake index_copy descriptor counts must use int32")
    if any(
        tensor.device != slots.device or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError(
            "Mooncake index_copy descriptor tensors must be contiguous "
            "on one device"
        )
    if rows <= 0 or capacity <= 0 or rows > capacity or host_capacity <= 0:
        raise ValueError(
            "Mooncake index_copy descriptor dimensions are invalid: "
            f"rows={rows}, capacity={capacity}, "
            f"host_capacity={host_capacity}"
        )
    if destinations.numel() != capacity:
        raise ValueError(
            "Mooncake source/destination descriptor capacities differ"
        )
    if (
        chunk_sources.numel() < capacity
        or chunk_destinations.numel() < capacity
    ):
        raise ValueError("Mooncake compact descriptor workspaces are too small")
    if count.numel() < 1:
        raise ValueError("Mooncake descriptor count buffer is empty")

    num_chunks = triton.cdiv(capacity, _DESCRIPTOR_BLOCK_SIZE)
    if chunk_counts.numel() < num_chunks:
        raise ValueError(
            "Mooncake compact descriptor count workspace is too small"
        )

    compact_index_copy_descriptors_kernel[(num_chunks,)](
        slots,
        chunk_sources,
        chunk_destinations,
        chunk_counts,
        rows,
        host_capacity,
        BLOCK=_DESCRIPTOR_BLOCK_SIZE,
    )
    gather_index_copy_descriptors_kernel[(1,)](
        chunk_sources,
        chunk_destinations,
        chunk_counts,
        sources,
        destinations,
        count,
        BLOCK=_DESCRIPTOR_BLOCK_SIZE,
        NUM_CHUNKS=num_chunks,
    )
    pad_index_copy_descriptors_kernel[
        (triton.cdiv(capacity, _DESCRIPTOR_BLOCK_SIZE),)
    ](
        sources,
        destinations,
        count,
        capacity,
        BLOCK=_DESCRIPTOR_BLOCK_SIZE,
    )


def warmup_index_copy_descriptors(manager) -> None:
    """Compile descriptor kernels before graph capture."""
    device = manager.d2h_src_idx_npu.device
    slots = torch.zeros(1, dtype=torch.int64, device=device)
    build_index_copy_descriptors(
        slots,
        manager.d2h_src_idx_npu,
        manager.d2h_dst_idx_npu,
        manager.d2h_index_count_npu,
        manager.d2h_index_chunk_src_npu,
        manager.d2h_index_chunk_dst_npu,
        manager.d2h_index_chunk_count_npu,
        1,
    )
    torch.npu.synchronize()
