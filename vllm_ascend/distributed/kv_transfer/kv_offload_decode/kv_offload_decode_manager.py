import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu
from memfabric_hybrid import offload
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import KVOffloadDecodeConfig
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.host_backend import (
    SFA_KV_OFFLOAD_BACKEND_MOONCAKE,
    ensure_mooncake_host_is_pd_decode_only,
    kv_transfer_extra_config,
    resolve_sfa_kv_offload_backend,
)
from vllm_ascend.distributed.kv_transfer.kv_offload_decode.host_pool import (
    DSAHostMemoryRegion,
    DSAHostKVPool,
    allocate_mooncake_host_region,
)


# Main BF16 cache: [k_cache, v_cache, k_cache_cpu, v_cache_cpu,
# topk_buffer_k, topk_buffer_v]. Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
# TODO remove KV_OFFLOAD_COLOCATE_DEBUG after PD disaggregate is done:
# the npu k_cache/v_cache entries only exist for colocate debug (prefill
# staging) and are deleted together with the prefill path.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5

FSA_EXTERNAL_PLAN_READY_MARKER = 0x5A45
FSA_PAIRED_SELECTION_COPY_MARKER = 0x5A56
FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT = 16376
FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT = 16
FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT = 8
FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT = (
    (
        FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT
        + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
        - 1
    )
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT = (
    (
        FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT
        + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
        + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
        - 1
    )
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)


_SUBSCRIBED_COMPUTE_STREAMS: set[object] = set()
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
_CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER = 3 * _CPU_CACHE_ALIGNMENT
_VLLM_NULL_BLOCK_COUNT = 1


def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


@dataclass(frozen=True)
class KVOffloadDecodeMemoryBudget:
    npu_limit_blocks: int
    dram_limit_blocks: int
    workload_limit_blocks: int
    final_num_blocks: int
    final_planner_bytes: int
    planned_host_bytes: int
    planned_device_bytes: int
    host_alignment_reserve_bytes: int
    limiting_factor: str


def _split_host_device_kv_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[list[KVCacheSpec], list[KVCacheSpec]]:
    host_specs: list[KVCacheSpec] = []
    device_specs: list[KVCacheSpec] = []
    for spec in kv_cache_spec.values():
        if getattr(spec, "store_on_host", False):
            host_specs.append(spec)
        else:
            device_specs.append(spec)
    if not host_specs:
        raise ValueError("KV offload decode requires at least one host KV cache spec")
    if not device_specs:
        raise ValueError("KV offload decode requires at least one device KV cache spec")
    block_sizes = {spec.block_size for spec in host_specs + device_specs}
    if len(block_sizes) != 1:
        raise ValueError(
            "KV offload decode memory planning requires one shared block size, "
            f"got {sorted(block_sizes)}"
        )
    return host_specs, device_specs


def plan_kv_offload_decode_memory(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    available_device_memory_bytes: int,
    dram_limit_bytes: int,
    keep_device_kv_cache: bool,
) -> KVOffloadDecodeMemoryBudget:
    """Bound KV offload decode blocks by NPU, DRAM, and active demand."""
    host_specs, device_specs = _split_host_device_kv_specs(kv_cache_spec)
    host_page_size_bytes = sum(spec.page_size_bytes for spec in host_specs)
    device_page_size_bytes = sum(spec.page_size_bytes for spec in device_specs)
    total_page_size_bytes = host_page_size_bytes + device_page_size_bytes

    host_alignment_reserve_bytes = (
        len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    )
    usable_dram_bytes = max(dram_limit_bytes - host_alignment_reserve_bytes, 0)
    dram_limit_blocks = usable_dram_bytes // host_page_size_bytes

    npu_page_size_bytes = (
        total_page_size_bytes if keep_device_kv_cache else device_page_size_bytes
    )
    npu_limit_blocks = max(available_device_memory_bytes, 0) // npu_page_size_bytes

    max_blocks_per_request = max(
        cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
        for spec in host_specs + device_specs
    )
    workload_limit_blocks = (
        max_blocks_per_request * vllm_config.scheduler_config.max_num_seqs
        + _VLLM_NULL_BLOCK_COUNT
    )

    limits = {
        "npu": npu_limit_blocks,
        "dram": dram_limit_blocks,
        "workload": workload_limit_blocks,
    }
    limiting_factor = min(limits, key=limits.get)
    final_num_blocks = limits[limiting_factor]
    host_page_tokens = min((int(spec.block_size) for spec in host_specs), default=0)
    host_pages_for_max_len = max(
        (
            cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )
            for spec in host_specs
        ),
        default=0,
    )
    logger.info(
        "KVPOOL layout=UNIFIED_HOST_MAIN host_page_tokens=%s required_blocks=%s "
        "npu=%s dram=%s workload=%s final_blocks=%s limiting=%s",
        host_page_tokens,
        host_pages_for_max_len,
        npu_limit_blocks,
        dram_limit_blocks,
        workload_limit_blocks,
        final_num_blocks,
        limiting_factor,
    )
    if host_pages_for_max_len > final_num_blocks:
        raise ValueError(
            "KV offload Host pool cannot cover one max_model_len request: "
            f"need={host_pages_for_max_len} Host pages, have={final_num_blocks} "
            f"blocks ({limiting_factor}-limited)."
        )
    final_planner_bytes = final_num_blocks * total_page_size_bytes
    planned_host_bytes = final_num_blocks * host_page_size_bytes
    planned_device_bytes = final_num_blocks * npu_page_size_bytes
    return KVOffloadDecodeMemoryBudget(
        npu_limit_blocks=npu_limit_blocks,
        dram_limit_blocks=dram_limit_blocks,
        workload_limit_blocks=workload_limit_blocks,
        final_num_blocks=final_num_blocks,
        final_planner_bytes=final_planner_bytes,
        planned_host_bytes=planned_host_bytes,
        planned_device_bytes=planned_device_bytes,
        host_alignment_reserve_bytes=host_alignment_reserve_bytes,
        limiting_factor=limiting_factor,
    )


def get_kv_offload_decode_cpu_pool_size_bytes(
    kv_cache_config: KVCacheConfig,
) -> int:
    """Return a safe upper bound for aligned host KV allocations."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            layer_specs.update(group.kv_cache_spec.kv_cache_specs)
        else:
            layer_specs.update(
                (layer_name, group.kv_cache_spec)
                for layer_name in group.layer_names
            )
    host_specs = [
        spec
        for spec in layer_specs.values()
        if getattr(spec, "store_on_host", False)
    ]
    if not host_specs:
        raise ValueError("KV offload decode requires host-resident KV cache specs")
    raw_host_bytes = kv_cache_config.num_blocks * sum(
        spec.page_size_bytes for spec in host_specs
    )
    alignment_reserve_bytes = (
        len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    )
    return raw_host_bytes + alignment_reserve_bytes


class KVOffloadDecodeManager:
    """
    A manager responsible to the offload KV cache.
    It enlarge the availble memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """
    _CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024

    @staticmethod
    def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        data_ptr = tensor.data_ptr()
        aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
        offset = (aligned_addr - data_ptr) // tensor.element_size()
        return tensor[int(offset):]

    @classmethod
    def _empty_aligned_cpu_tensor(
        cls,
        shape: list[int],
        dtype: torch.dtype,
        alignment: int = _CPU_CACHE_ALIGNMENT,
    ) -> torch.Tensor:
        num_elements = int(np.prod(shape))
        extra_elements = cdiv(alignment, torch.empty((), dtype=dtype).element_size())
        tensor = offload.empty([num_elements + extra_elements], dtype=dtype, pin_memory=True)
        return cls._align_memory(tensor, alignment)[:num_elements].view(shape)

    @staticmethod
    def empty_aligned_int8_cpu_tensors(
        sizes: list[int],
        alignment: int = _CPU_CACHE_ALIGNMENT,
    ) -> list[torch.Tensor]:
        chunk_nums = [cdiv(size, alignment) for size in sizes]
        total_chunk_num = 1 + sum(chunk_nums)
        raw_tensor = offload.empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
        base_addr = raw_tensor.data_ptr()
        if base_addr % alignment:
            base_addr = (base_addr // alignment + 1) * alignment
        base_offset = base_addr - raw_tensor.data_ptr()
        allocate_tensors = []
        for size, chunk_num in zip(sizes, chunk_nums):
            allocate_tensors.append(raw_tensor[base_offset:base_offset + size])
            base_offset += chunk_num * alignment
        return allocate_tensors

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_offload_decode_config: KVOffloadDecodeConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.kv_offload_decode_config = kv_offload_decode_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = kv_offload_decode_config.topk_buffer_size
        self.topk = kv_offload_decode_config.topk
        self.use_fused_overlap = bool(getattr(kv_offload_decode_config, "use_fused_overlap", False))
        self.sfa_kv_offload_backend = resolve_sfa_kv_offload_backend(
            kv_transfer_extra_config(vllm_config),
            use_fused_overlap_offload=self.use_fused_overlap,
        )
        ensure_mooncake_host_is_pd_decode_only(
            self.sfa_kv_offload_backend,
            keep_device_kv_cache=kv_offload_decode_config.keep_device_kv_cache,
        )
        self.d2h_index_copy_bypass = self.uses_mooncake_host
        self.runner_host_pool: DSAHostKVPool | None = None

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        self.max_d2h_index_copy_tokens = self.max_num_topk_rows
        self.fused_overlap_membership_map: torch.Tensor | None = None
        self.fused_overlap_membership_map_rows = 0
        self.fused_overlap_membership_region: DSAHostMemoryRegion | None = None
        self.fused_overlap_planner_membership_map: torch.Tensor | None = None
        self.fused_overlap_membership_plan_device_staging: (
            torch.Tensor | None
        ) = None
        self.fused_overlap_plan_owner_layer_id: int | None = None
        self.fused_overlap_plan_topk: int | None = None
        self.fused_overlap_plan_num_tokens = 0
        self.fused_overlap_plan_membership_map: torch.Tensor | None = None
        max_block_num = cdiv(self.max_model_len, self.block_size)
        self.block_table_cpu = torch.zeros(
            [self.max_num_reqs, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.block_table_expanded_cpu = torch.empty(
            [self.max_num_topk_rows, max_block_num],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self._npu_runtime = torch_npu.npu

        self._build_cpp()

        dram_limit_bytes = int(
            kv_offload_decode_config.dram_size_per_dp_GB * 1024 * 1024 * 1024
        )
        planned_pool_size_bytes = get_kv_offload_decode_cpu_pool_size_bytes(
            kv_cache_config
        )
        if planned_pool_size_bytes > dram_limit_bytes:
            raise ValueError(
                "KV offload decode planned CPU pool exceeds DRAM limit after "
                "alignment: "
                f"planned={planned_pool_size_bytes / (1 << 30):.2f} GiB, "
                f"limit={kv_offload_decode_config.dram_size_per_dp_GB} GiB, "
                f"num_blocks={kv_cache_config.num_blocks}"
            )
        actual_pool_size_bytes = min(planned_pool_size_bytes, dram_limit_bytes)
        if self.uses_mooncake_host:
            logger.info(
                "KVOffloadDecodeManager selected Mooncake shared Host pool: "
                "planned=%.2f GiB, configured_limit=%s GiB, num_blocks=%s, "
                "tp=%s/%s.",
                actual_pool_size_bytes / (1 << 30),
                kv_offload_decode_config.dram_size_per_dp_GB,
                kv_cache_config.num_blocks,
                self.tp_rank,
                self.tp_size,
            )
        else:
            logger.info(
                "KVOffloadDecodeManager starts MemFabric CPU KV pool "
                "initialization: planned=%.2f GiB, configured_limit=%s GiB, "
                "num_blocks=%s.",
                actual_pool_size_bytes / (1 << 30),
                kv_offload_decode_config.dram_size_per_dp_GB,
                kv_cache_config.num_blocks,
            )
            config = offload.OffloadConfig()
            config.device_id = torch_npu.npu.current_device()
            config.size = actual_pool_size_bytes
            config.world_size = self.tp_size
            config.rank_id = self.tp_rank
            offload.initialize(config)
            self.tp_group.barrier()

    @property
    def uses_mooncake_host(self) -> bool:
        return (
            self.sfa_kv_offload_backend
            == SFA_KV_OFFLOAD_BACKEND_MOONCAKE
        )

    @staticmethod
    def layer_sort_key(layer_name: str) -> tuple[int, str]:
        match = re.search(r"layers\.(\d+)", layer_name)
        return (
            int(match.group(1)) if match is not None else 10**9,
            layer_name,
        )

    def bind_runner_host_pool(self, pool: DSAHostKVPool) -> None:
        if not self.uses_mooncake_host:
            raise RuntimeError(
                "runner-owned Host pool requires sfa_kv_offload_backend=mooncake"
            )
        if pool.topology.tp_rank != self.tp_rank:
            raise ValueError(
                "DSA Host pool TP rank mismatch: "
                f"pool={pool.topology.tp_rank}, manager={self.tp_rank}"
            )
        if pool.topology.tp_size != self.tp_size:
            raise ValueError(
                "DSA Host pool TP size mismatch: "
                f"pool={pool.topology.tp_size}, manager={self.tp_size}"
            )
        if pool.layout.block_size != self.block_size:
            raise ValueError(
                "DSA Host pool block size mismatch: "
                f"pool={pool.layout.block_size}, manager={self.block_size}"
            )
        self.runner_host_pool = pool

    def _build_cpp(self):
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        npu_include_path = os.path.join(ascend_home, "include")
        npu_lib_path = os.path.join(ascend_home, "lib64")
        if not os.path.exists(npu_lib_path):
            npu_lib_path = os.path.join(ascend_home, "lib")
        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
        os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
        os.environ['CXX'] = 'clang++'
        os.environ['CC'] = 'clang'
        abs_path = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(abs_path, "kv_offload_decode.cpp")
        logger.info(f'KV offload decode build cpp utils from src: {src_path}')
        self.kv_offload_decode_cpp = torch.utils.cpp_extension.load(
            name="kv_offload_decode",
            sources=[src_path],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-fopenmp",
                "-march=armv8.2-a+sve+fp16+bf16",
                "-fPIC",
                f"-I{npu_include_path}",
                f"-I{torch_npu_include}",
            ],
            extra_ldflags=[
                "-fopenmp",
                f"-L{npu_lib_path}",
                "-lascendcl",
                f"-L{torch_npu_lib_path}",
                "-ltorch_npu",
            ],
            verbose=True,
        )

    def _warmup_external_lru_planner_threads(self) -> int:
        if not (self.use_fused_overlap and self.tp_rank == 0):
            return 0
        warmed_threads = self.kv_offload_decode_cpp.warmup_lru_resident_threads(
            self.lru_workspace_threads
        )
        logger.info(
            "Warmed external LRU planner OpenMP team with %s threads",
            warmed_threads,
        )
        return warmed_threads

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig | None,
    ) -> int:
        assert len(kv_cache_config.kv_cache_groups) == 1, "Hybrid KV is not supported."
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        return kv_cache_spec.block_size

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.offload_layer_names = [
            layer_name for layer_name in kv_caches
            if 'indexer' not in layer_name
        ]
        if self.uses_mooncake_host:
            self.offload_layer_names.sort(key=self.layer_sort_key)
        if not self.offload_layer_names:
            raise ValueError("KV offload decode did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id
            for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info(
            "KV offload decode registered %s layers (%s target layers).",
            self.num_layers,
            self.num_target_layers,
        )
        self.mtp_layer_id = self.num_layers - 1 if self.num_layers != self.num_target_layers else -1
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("KV offload decode layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "KV offload decode layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def _restore_bfloat16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = self.kv_offload_decode_cpp.restore_bfloat16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_bfloat16_tensor returned a tensor with unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                "restore_bfloat16_tensor returned unexpected shape: "
                f"expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.bfloat16:
            raise RuntimeError(
                f"restore_bfloat16_tensor returned unexpected dtype: {view.dtype}"
            )
        if not view.is_contiguous():
            raise RuntimeError("restore_bfloat16_tensor requires a contiguous view")
        return view

    def _restore_int16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = self.kv_offload_decode_cpp.restore_int16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_int16_tensor returned a tensor with an unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                "restore_int16_tensor returned an unexpected shape: "
                f"expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.int16:
            raise RuntimeError(
                f"restore_int16_tensor returned an unexpected dtype: {view.dtype}"
            )
        if not view.is_contiguous():
            raise RuntimeError("restore_int16_tensor requires a contiguous view")
        return view

    def allocate_fused_overlap_membership_map(
        self,
        row_capacity: int,
    ) -> torch.Tensor:
        if not self.use_fused_overlap:
            raise RuntimeError(
                "mapped membership allocation requires "
                "kv_offload_decode_config.use_fused_overlap=true"
            )
        if row_capacity <= 0:
            raise ValueError(
                f"mapped membership row capacity must be positive, got {row_capacity}"
            )
        if self.topk >= FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT:
            raise ValueError(
                "SFA sparse topk exceeds external plan storage: "
                f"topk={self.topk} "
                f"control_offset={FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT}"
            )

        if getattr(self, "fused_overlap_membership_map", None) is not None:
            if row_capacity > self.fused_overlap_membership_map_rows:
                raise RuntimeError(
                    "fused_overlap membership storage was allocated for fewer "
                    f"rows: allocated={self.fused_overlap_membership_map_rows}, "
                    f"required={row_capacity}"
                )
            return self.fused_overlap_membership_map

        shape = [row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT]
        if self.uses_mooncake_host:
            if self.runner_host_pool is None:
                raise RuntimeError(
                    "Mooncake membership allocation requires the runner Host pool"
                )
            region = allocate_mooncake_host_region(
                numel=(
                    row_capacity
                    * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT
                ),
                dtype=torch.int16,
                alignment=self._CPU_CACHE_ALIGNMENT,
                name="dsa_fused_membership",
                topology=self.runner_host_pool.topology,
            )
            membership_map = region.tensor.view(shape)
            planner_map = None
            if self.tp_rank == 0:
                plan_width = (
                    self.topk
                    + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
                )
                plan_shape = [row_capacity, plan_width]
                planner_map = torch.empty(
                    plan_shape,
                    dtype=torch.int16,
                    device="cpu",
                    pin_memory=True,
                )
                self.fused_overlap_membership_plan_device_staging = torch.empty(
                    plan_shape,
                    dtype=torch.int16,
                    device=membership_map.device,
                )
                self._init_fused_overlap_plan_staging(planner_map)
                self._init_fused_overlap_membership_control(membership_map)
                torch_npu.npu.current_stream().synchronize()
            self.tp_group.barrier()
            self.fused_overlap_membership_region = region
            self.fused_overlap_planner_membership_map = planner_map
        else:
            owner_ptr = torch.zeros(1, dtype=torch.int64, device="npu")
            membership_map = None
            if self.tp_rank == 0:
                membership_map = offload.empty(
                    [
                        row_capacity
                        * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT
                    ],
                    dtype=torch.int16,
                    pin_memory=True,
                ).view(shape)
                self._init_fused_overlap_membership_control(membership_map)
                owner_ptr[0] = membership_map.data_ptr()
            self.tp_group.broadcast(owner_ptr, src=0)
            shared_ptr = int(owner_ptr.item())
            if self.tp_rank != 0:
                membership_map = self._restore_int16_tensor(shared_ptr, shape)
            if membership_map is None:
                raise RuntimeError(
                    "mapped membership storage was not initialized"
                )
            self.tp_group.barrier()
            self.fused_overlap_planner_membership_map = membership_map

        self.fused_overlap_membership_map = membership_map
        self.fused_overlap_membership_map_rows = row_capacity
        if os.getenv("VLLM_ASCEND_SFA_DEBUG", "0") == "1":
            planner_map = self.fused_overlap_planner_membership_map
            logger.info(
                "[fused_overlap_offload][external_plan_debug] stage=membership_alloc "
                "tp=%s backend=%s rows=%s membership(device=%s shape=%s stride=%s "
                "contiguous=%s ptr=%#x) planner(device=%s shape=%s stride=%s "
                "contiguous=%s pinned=%s ptr=%#x same_ptr=%s)",
                self.tp_rank,
                self.sfa_kv_offload_backend,
                row_capacity,
                membership_map.device,
                tuple(membership_map.shape),
                tuple(membership_map.stride()),
                membership_map.is_contiguous(),
                membership_map.data_ptr(),
                None if planner_map is None else planner_map.device,
                None if planner_map is None else tuple(planner_map.shape),
                None if planner_map is None else tuple(planner_map.stride()),
                None if planner_map is None else planner_map.is_contiguous(),
                None
                if planner_map is None or planner_map.device.type != "cpu"
                else planner_map.is_pinned(),
                0 if planner_map is None else planner_map.data_ptr(),
                planner_map is not None
                and planner_map.data_ptr() == membership_map.data_ptr(),
            )
        return membership_map

    def _init_fused_overlap_membership_control(
        self,
        membership_map: torch.Tensor,
    ) -> None:
        membership_map.fill_(-1)
        control = membership_map[
            :,
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT:
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT
            + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
        ]
        control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = self.topk
        control[:, 3] = (
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT - self.topk
        )
        control[:, 7] = FSA_PAIRED_SELECTION_COPY_MARKER

    def _init_fused_overlap_plan_staging(
        self,
        plan_staging: torch.Tensor,
    ) -> None:
        plan_staging.fill_(-1)
        control = plan_staging[
            :,
            self.topk : self.topk + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
        ]
        control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = self.topk
        control[:, 3] = (
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT - self.topk
        )
        control[:, 7] = FSA_PAIRED_SELECTION_COPY_MARKER

    def is_fused_membership_storage(self, tensor: torch.Tensor) -> bool:
        if tensor.dtype != torch.int16:
            return False
        if self.uses_mooncake_host:
            return (
                self.fused_overlap_membership_map is not None
                and tensor.data_ptr()
                == self.fused_overlap_membership_map.data_ptr()
            )
        return tensor.device.type == "cpu"

    def get_fused_overlap_cpu_kv_inputs(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU full KV tensors used by fused_overlap decode.

        On TP0 these are the owned CPU pools; on other ranks they are non-owning
        GVA views restored after broadcast.
        """
        if not self.use_fused_overlap:
            raise RuntimeError(
                "get_fused_overlap_cpu_kv_inputs requires "
                "kv_offload_decode_config.use_fused_overlap=true"
            )
        layer_id = self._get_offload_layer_id(layer_name)
        if layer_id >= len(self.k_caches_cpu) or layer_id >= len(self.v_caches_cpu):
            raise RuntimeError(
                "fused_overlap CPU KV views are not registered: "
                f"layer_id={layer_id}, k_len={len(self.k_caches_cpu)}, "
                f"v_len={len(self.v_caches_cpu)}"
            )
        return self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        self._register_offload_layers(kv_caches)
        if self.uses_mooncake_host:
            if self.runner_host_pool is None:
                raise RuntimeError(
                    "Mooncake Host backend requires a runner-owned DSA Host pool"
                )
            if self.runner_host_pool.layout.layer_names != tuple(
                self.offload_layer_names
            ):
                raise ValueError(
                    "DSA Host pool layer order does not match registered "
                    "offload layers: "
                    f"pool={self.runner_host_pool.layout.layer_names}, "
                    f"registered={tuple(self.offload_layer_names)}"
                )

        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"KV offload decode layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.uses_mooncake_host or self.tp_rank == 0:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "KV offload decode only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "KV offload decode requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "KV offload decode topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H uses a separate descriptor set from the shared H2D buffers below.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        if self.use_fused_overlap:
            self.current_kv_save_stream = torch_npu.npu.Stream()
            self.fused_plan_stream = torch_npu.npu.Stream()
            self.fused_plan_metadata_npu = torch.zeros(
                1 + self.max_num_topk_rows,
                dtype=torch.int32,
                device=device,
            )
            self.fused_plan_status_npu = self.fused_plan_metadata_npu[:1]
            self.fused_plan_current_linear_slots_npu = (
                self.fused_plan_metadata_npu[1:]
            )
            self.current_kv_by_layer: dict[
                int, tuple[torch.Tensor, torch.Tensor]
            ] = {}
            if self.d2h_index_copy_bypass and self.tp_rank == 0:
                self.d2h_slot_mapping_cpu = torch.zeros(
                    self.max_d2h_index_copy_tokens,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                self.d2h_src_idx_cpu = torch.zeros(
                    self.max_d2h_index_copy_tokens,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                self.d2h_dst_idx_cpu = torch.zeros(
                    self.max_d2h_index_copy_tokens,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                self.d2h_index_count_cpu = torch.zeros(
                    1,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                self.d2h_src_idx_npu = torch.zeros(
                    self.max_d2h_index_copy_tokens,
                    dtype=torch.int64,
                    device=device,
                )
                self.d2h_dst_idx_npu = torch.zeros(
                    self.max_d2h_index_copy_tokens,
                    dtype=torch.int64,
                    device=device,
                )
                self.d2h_index_count_npu = torch.zeros(
                    1,
                    dtype=torch.int32,
                    device=device,
                )
        self.d2h_src_ptrs_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int64, device=device
        )
        self.d2h_dst_ptrs_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int64, device=device
        )
        self.d2h_lengths_npu = torch.empty(
            d2h_descriptor_rows, dtype=torch.int32, device=device
        )
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        pages_per_row = self.topk_buffer_size // self.block_size
        self.current_slots_npu = torch.empty(
            (self.max_num_topk_rows, self.topk),
            dtype=torch.int32,
            device=device,
        )
        self.resident_block_table_npu = torch.arange(
            self.max_num_topk_rows * pages_per_row,
            dtype=torch.int32,
            device=device,
        ).view(self.max_num_topk_rows, pages_per_row)
        self.resident_query_lens_npu = torch.arange(
            1, self.max_num_topk_rows + 1, dtype=torch.int32, device=device
        )
        self.resident_seq_lens_npu = torch.full(
            (self.max_num_topk_rows,),
            self.topk_buffer_size,
            dtype=torch.int32,
            device=device,
        )

        # sparse_copy related addrs and buffers
        self.addr_k_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_k]
        self.addr_v_bases: list[int] = [t.data_ptr() for t in self.topk_buffers_v]
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        self.cpu_block_lens: list[tuple[int, int]] = []
        if self.uses_mooncake_host:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                self.gvas_k_bases.append(k_cpu.data_ptr())
                self.gvas_v_bases.append(v_cpu.data_ptr())
                self.cpu_block_lens.append(
                    (
                        k_cpu.numel()
                        * k_cpu.element_size()
                        // self.kv_cache_config.num_blocks,
                        v_cpu.numel()
                        * v_cpu.element_size()
                        // self.kv_cache_config.num_blocks,
                    )
                )
            logger.info(
                "Registered local Mooncake shared Host views: tp=%s/%s "
                "layers=%s pool_ptr=0x%x layout=%s",
                self.tp_rank,
                self.tp_size,
                self.num_layers,
                self.runner_host_pool.data_ptr,
                self.runner_host_pool.layout.fingerprint,
            )
        else:
            gvas_k_tensor = torch.zeros(
                [self.num_layers],
                dtype=torch.int64,
                device="npu",
            )
            gvas_v_tensor = torch.zeros(
                [self.num_layers],
                dtype=torch.int64,
                device="npu",
            )
            cpu_block_lens_tensor = torch.zeros(
                [self.num_layers, 2],
                dtype=torch.int64,
                device="npu",
            )
            shape_k_tensor = torch.zeros(
                [4],
                dtype=torch.int64,
                device="npu",
            )
            shape_v_tensor = torch.zeros(
                [4],
                dtype=torch.int64,
                device="npu",
            )
            if self.tp_rank == 0:
                for layer_id in range(self.num_layers):
                    k_cpu = self.k_caches_cpu[layer_id]
                    v_cpu = self.v_caches_cpu[layer_id]
                    gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                    gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                    cpu_block_lens_tensor[layer_id, 0] = (
                        k_cpu.numel()
                        * k_cpu.element_size()
                        // self.kv_cache_config.num_blocks
                    )
                    cpu_block_lens_tensor[layer_id, 1] = (
                        v_cpu.numel()
                        * v_cpu.element_size()
                        // self.kv_cache_config.num_blocks
                    )
                shape_k_tensor.copy_(
                    torch.tensor(
                        self.k_caches_cpu[0].shape,
                        dtype=torch.int64,
                        device="npu",
                    )
                )
                shape_v_tensor.copy_(
                    torch.tensor(
                        self.v_caches_cpu[0].shape,
                        dtype=torch.int64,
                        device="npu",
                    )
                )
            self.tp_group.broadcast(gvas_k_tensor, src=0)
            self.tp_group.broadcast(gvas_v_tensor, src=0)
            self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
            self.tp_group.broadcast(shape_k_tensor, src=0)
            self.tp_group.broadcast(shape_v_tensor, src=0)
            for layer_id in range(self.num_layers):
                self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
                self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())
                self.cpu_block_lens.append(
                    (
                        cpu_block_lens_tensor[layer_id, 0].item(),
                        cpu_block_lens_tensor[layer_id, 1].item(),
                    )
                )

            if self.use_fused_overlap and self.tp_rank != 0:
                cpu_k_shape = [
                    int(value) for value in shape_k_tensor.tolist()
                ]
                cpu_v_shape = [
                    int(value) for value in shape_v_tensor.tolist()
                ]
                self.k_caches_cpu = [
                    self._restore_bfloat16_tensor(ptr, cpu_k_shape)
                    for ptr in self.gvas_k_bases
                ]
                self.v_caches_cpu = [
                    self._restore_bfloat16_tensor(ptr, cpu_v_shape)
                    for ptr in self.gvas_v_bases
                ]

        gvas_buffer_offset = 0
        gvas_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8 # 2: k+v, 8: int64
        addr_buffer_offset = gvas_buffer_offset + gvas_buffer_size_bytes
        addr_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 8
        size_buffer_offset = addr_buffer_offset + addr_buffer_size_bytes
        size_buffer_size_bytes = self.max_num_topk_rows * self.topk * 2 * 4 # 2: k+v, 4: int32
        num_tokens_buffer_offset = size_buffer_offset + size_buffer_size_bytes
        num_tokens_buffer_size_bytes = 4 # 1 * int32
        sparse_copy_args_buffer_size_bytes = gvas_buffer_size_bytes + addr_buffer_size_bytes + size_buffer_size_bytes + num_tokens_buffer_size_bytes
        self.sparse_copy_args_buffer_cpu = torch.zeros([sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device='cpu', pin_memory=True)
        self.sparse_copy_args_buffer_npu = torch.zeros([sparse_copy_args_buffer_size_bytes], dtype=torch.int8, device='npu')

        self.gvas_buffer_cpu = self.sparse_copy_args_buffer_cpu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
        self.addr_buffer_cpu = self.sparse_copy_args_buffer_cpu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
        self.size_buffer_cpu = self.sparse_copy_args_buffer_cpu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
        self.num_tokens_buffer_cpu = \
            self.sparse_copy_args_buffer_cpu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
        assert self.gvas_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_cpu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_cpu.shape == torch.Size([1])

        self.gvas_buffer_npu = self.sparse_copy_args_buffer_npu[gvas_buffer_offset:gvas_buffer_offset + gvas_buffer_size_bytes].view(torch.int64)
        self.addr_buffer_npu = self.sparse_copy_args_buffer_npu[addr_buffer_offset:addr_buffer_offset + addr_buffer_size_bytes].view(torch.int64)
        self.size_buffer_npu = self.sparse_copy_args_buffer_npu[size_buffer_offset:size_buffer_offset + size_buffer_size_bytes].view(torch.int32)
        self.num_tokens_buffer_npu = \
            self.sparse_copy_args_buffer_npu[num_tokens_buffer_offset:num_tokens_buffer_offset + num_tokens_buffer_size_bytes].view(torch.int32)
        assert self.gvas_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.addr_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.size_buffer_npu.shape == torch.Size([self.max_num_topk_rows * self.topk * 2])
        assert self.num_tokens_buffer_npu.shape == torch.Size([1])

        # topk cache reuse related
        self.lru_workspace_threads = 8
        self._warmup_external_lru_planner_threads()
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_token_to_req_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [torch.full(
            [self.max_num_topk_rows, self.topk_buffer_size],
            -1,
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_slots_cpu_list = [torch.arange(
            self.topk_buffer_size,
            dtype=torch.int32,
            device='cpu',
        ).view(1, -1).repeat(self.max_num_topk_rows, 1).pin_memory() for _ in range(self.num_layers)]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_miss_tokens_cpu_list = [torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_miss_slots_cpu_list = [torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        ) for _ in range(self.num_layers)]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device='cpu', pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_visible_seq_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [torch.full(
            [self.max_num_topk_rows],
            -1,
            dtype=torch.int64,
            device='cpu',
            pin_memory=True,
        ) for _  in range(self.num_layers)]
        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )
        self.lru_physical_row_workspace = torch.empty(
            [self.max_num_topk_rows * 3],
            dtype=torch.int32,
            device='cpu',
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_visible_seq_lens_ptr = self.lru_visible_seq_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list]
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_token_to_req_ptr = self.lru_token_to_req_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list]
        self.lru_miss_tokens_ptrs = [lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list]
        self.lru_miss_slots_ptrs = [lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()
        self.lru_physical_row_workspace_ptr = self.lru_physical_row_workspace.data_ptr()

    def offload_new_kv(
        self,
        layer_name: str,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,
        v_cache_npu: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        if not has_prefill and k is not None and v is not None:
            layer_id = self._get_offload_layer_id(layer_name)
            self.current_kv_by_layer[layer_id] = (k, v)
        use_side_stream = (
            self.tp_rank == 0
            and self.use_fused_overlap
            and capturing
            and not has_prefill
            and not self.d2h_index_copy_bypass
        )
        if not use_side_stream:
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
                capturing,
            )
            return

        current_kv_ready = torch_npu.npu.current_stream().record_event()
        with torch_npu.npu.stream(self.current_kv_save_stream):
            self.current_kv_save_stream.wait_event(current_kv_ready)
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
                capturing,
            )

    def _offload_new_kv_on_current_stream(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        v: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("KV offload decode TP0 CPU cache is not registered")
        if has_prefill and not self.kv_offload_decode_config.keep_device_kv_cache:
            raise RuntimeError(
                "KV offload decode prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        if has_prefill:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("prefill offload requires NPU paged K/V caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device
            if self.d2h_index_copy_bypass:
                self._offload_new_kv_via_index_copy(
                    slot_mapping=slot_mapping,
                    k_cache_cpu=k_cache_cpu,
                    v_cache_cpu=v_cache_cpu,
                    k=k,
                    v=v,
                    capturing=capturing,
                )
                return

        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        token_count = slots.numel()
        if token_count > self.max_num_tokens:
            raise ValueError(
                "KV offload decode rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = (
            k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        )
        num_v_slots = (
            v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        )
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                "KV offload decode CPU K/V pools have incompatible token capacities: "
                f"k={num_k_slots}, v={num_v_slots}"
            )
        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v

        dst_k = int(k_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(
            self.token_size_bytes_v
        )
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)

        result = offload.sparse_copy(
            self.d2h_src_ptrs_npu,
            self.d2h_dst_ptrs_npu,
            self.d2h_lengths_npu,
            self.d2h_size_npu,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")

    def _offload_new_kv_via_index_copy(
        self,
        *,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor,
        v_cache_cpu: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        capturing: bool,
    ) -> None:
        """Write current-token K/V into the Mooncake shared Host pool."""
        token_dim_k = self.token_size_bytes_k // k.element_size()
        token_dim_v = self.token_size_bytes_v // v.element_size()
        k_rows = k.reshape(-1, token_dim_k)
        v_rows = v.reshape(-1, token_dim_v)
        if not k_rows.is_contiguous():
            k_rows = k_rows.contiguous()
        if not v_rows.is_contiguous():
            v_rows = v_rows.contiguous()
        flat_host_k = k_cache_cpu.reshape(-1, token_dim_k)
        flat_host_v = v_cache_cpu.reshape(-1, token_dim_v)
        slots = slot_mapping.reshape(-1).to(device=k.device, dtype=torch.int64)
        token_count = slots.numel()
        if token_count == 0:
            return
        if token_count > self.max_d2h_index_copy_tokens:
            raise ValueError(
                "KV offload decode rows exceed index_copy capacity, "
                f"got {token_count}, "
                f"capacity={self.max_d2h_index_copy_tokens}"
            )
        if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
            raise ValueError("decode K/V row counts must match slot_mapping")
        if flat_host_k.device != k.device or flat_host_v.device != v.device:
            raise RuntimeError(
                "Mooncake Host views and current K/V must share one NPU device"
            )
        num_slots = flat_host_k.shape[0]
        if num_slots != flat_host_v.shape[0] or num_slots <= 0:
            raise ValueError(
                "Mooncake Host K/V pools have incompatible token capacities"
            )

        if capturing:
            self.d2h_slot_mapping_cpu[:token_count].copy_(
                slots,
                non_blocking=True,
            )
            self.kv_offload_decode_cpp.enqueue_current_kv_index_copy_descriptors(
                self.d2h_slot_mapping_cpu,
                token_count,
                self.max_d2h_index_copy_tokens,
                num_slots,
                self.d2h_src_idx_cpu,
                self.d2h_dst_idx_cpu,
                self.d2h_index_count_cpu,
            )
            self.d2h_src_idx_npu.copy_(
                self.d2h_src_idx_cpu,
                non_blocking=True,
            )
            self.d2h_dst_idx_npu.copy_(
                self.d2h_dst_idx_cpu,
                non_blocking=True,
            )
            self.d2h_index_count_npu.copy_(
                self.d2h_index_count_cpu,
                non_blocking=True,
            )
            flat_host_k.index_copy_(
                0,
                self.d2h_dst_idx_npu,
                k_rows.index_select(0, self.d2h_src_idx_npu),
            )
            flat_host_v.index_copy_(
                0,
                self.d2h_dst_idx_npu,
                v_rows.index_select(0, self.d2h_src_idx_npu),
            )
            return

        valid = (slots >= 0) & (slots < num_slots)
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if valid_indices.numel() == 0:
            return
        destinations = slots.index_select(0, valid_indices)
        flat_host_k.index_copy_(
            0,
            destinations,
            k_rows.index_select(0, valid_indices),
        )
        flat_host_v.index_copy_(
            0,
            destinations,
            v_rows.index_select(0, valid_indices),
        )

    def onload_topk_kv(
        self,
        layer_name: str,
        num_tokens: int,
        num_reqs: int,
        block_table: torch.Tensor,
        topk_indices_npu: torch.Tensor,
        current_slots_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        token_to_req_npu: torch.Tensor | None = None,
        capturing: bool = False,
        skip_topk: bool = False,
    ):
        layer_id = self._get_offload_layer_id(layer_name)
        if num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "KV offload decode topk rows exceed configured workspace, "
                f"num_tokens={num_tokens}, max_num_topk_rows={self.max_num_topk_rows}"
            )
        if layer_id in [0, self.mtp_layer_id]:
            # metadata which are same across all layers, only compute/copy once in first layer.
            # last layer (mtp layer) may have different metadata, do not skip.
            if token_to_req_npu is not None:
                # spec decode case, expand block_table to actual num decode tokens.
                token_to_req_cpu = self.lru_token_to_req_cpu[:num_tokens]
                token_to_req_cpu.copy_(token_to_req_npu[:num_tokens], non_blocking=capturing)
                block_table_expanded = torch.index_select(
                    block_table, 0, token_to_req_npu[:num_tokens].to(torch.int64))
                self.block_table_expanded_cpu[:num_tokens].copy_(block_table_expanded, non_blocking=capturing)
            else:
                self.block_table_cpu[:num_reqs].copy_(block_table, non_blocking=capturing)
            self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu[:num_tokens], non_blocking=capturing)
            self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(
                stable_prefix_lens_npu[:num_tokens],
                non_blocking=capturing,
            )

        if skip_topk:
            assert layer_id > 0, "No previous layer to reuse."
            gvas_offset = self.gvas_k_bases[layer_id] - self.gvas_k_bases[layer_id - 1]
            addr_offset = self.addr_k_bases[layer_id] - self.addr_k_bases[layer_id - 1]
            assert self.gvas_v_bases[layer_id] - self.gvas_v_bases[layer_id - 1] == gvas_offset, (
                "k/v gvas base delta mismatch."
            )
            assert self.addr_v_bases[layer_id] - self.addr_v_bases[layer_id - 1] == addr_offset, (
                "k/v addr base delta mismatch."
            )
            self.gvas_buffer_npu += gvas_offset
            self.addr_buffer_npu += addr_offset
        else:
            if token_to_req_npu is not None:
                block_table_cpu = self.block_table_expanded_cpu[:num_tokens]
            else:
                block_table_cpu = self.block_table_cpu[:num_reqs]
            topk_indices_cpu = self.lru_topk_indices_cpu[:num_tokens]
            topk_indices_cpu.copy_(topk_indices_npu[:num_tokens], non_blocking=capturing)

            args = (
                num_tokens,
                self.lru_miss_count_cpu_list[layer_id][:num_tokens],
                self.lru_miss_tokens_cpu_list[layer_id][:num_tokens],
                self.lru_miss_slots_cpu_list[layer_id][:num_tokens],
                self.lru_req_ids_ptr,
                self.lru_last_req_ids_ptrs[layer_id],
                self.lru_topk_indices_ptr,
                self.lru_stable_prefix_lens_ptr,
                self.lru_slot_to_token_ptrs[layer_id],
                self.lru_slots_ptrs[layer_id],
                self.lru_current_slots_ptr,
                self.lru_miss_count_ptrs[layer_id],
                self.lru_miss_tokens_ptrs[layer_id],
                self.lru_miss_slots_ptrs[layer_id],
                block_table_cpu,
                self.block_size,
                self.token_size_bytes_k,
                self.token_size_bytes_v,
                self.gvas_k_bases[layer_id],
                self.gvas_v_bases[layer_id],
                self.addr_k_bases[layer_id],
                self.addr_v_bases[layer_id],
                self.lru_token_mark_workspace_ptr,
                self.lru_token_pos_workspace_ptr,
                self.lru_slot_workspace_ptr,
                self.lru_miss_position_workspace_ptr,
                self.lru_epochs_ptr,
                self.gvas_buffer_cpu,
                self.addr_buffer_cpu,
                self.size_buffer_cpu,
                self.num_tokens_buffer_cpu,
                layer_id,
            )

            if capturing:
                current_compute_stream = torch_npu.npu.current_stream()
                subscribed_compute_streams = get_subscribed_compute_streams()
                if current_compute_stream not in subscribed_compute_streams:
                    torch_npu.npu._subscribe_report(current_compute_stream)
                    subscribed_compute_streams.add(current_compute_stream)
                torch_npu.npu._launch_host_func(
                    current_compute_stream,
                    self._onload_topk_kv_cpu,
                    args,
                )
            else:
                self._onload_topk_kv_cpu(args)

            self.sparse_copy_args_buffer_npu.copy_(self.sparse_copy_args_buffer_cpu, non_blocking=capturing)

        offload.sparse_copy(
            self.gvas_buffer_npu,
            self.addr_buffer_npu,
            self.size_buffer_npu,
            self.num_tokens_buffer_npu,
            self.topk_buffers_k[0].device,
        )

        current_slots_cpu = self.lru_current_slots_cpu[:num_tokens]
        current_slots_npu[:num_tokens].copy_(current_slots_cpu, non_blocking=capturing)

    def prepare_fused_overlap_external_plan(
        self,
        layer_name: str,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
        capturing: bool = False,
        skip_topk: bool = False,
    ) -> bool:
        if not self.use_fused_overlap:
            raise RuntimeError("external FSA plan requires fused_overlap mode")
        if num_tokens <= 0 or num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "external FSA plan rows exceed configured workspace: "
                f"num_tokens={num_tokens}, max_rows={self.max_num_topk_rows}"
            )
        expected_topk_shape = (num_tokens, self.topk)
        if tuple(topk_indices_npu.shape) != expected_topk_shape:
            raise ValueError(
                "external FSA plan requires flattened single-head TopK input: "
                f"expected={expected_topk_shape}, actual={tuple(topk_indices_npu.shape)}"
            )
        if tuple(req_ids_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one request id per logical row: "
                f"expected=({num_tokens},), actual={tuple(req_ids_npu.shape)}"
            )
        if tuple(stable_prefix_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one stable prefix length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(stable_prefix_lens_npu.shape)}"
            )
        if tuple(visible_seq_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one visible KV length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(visible_seq_lens_npu.shape)}"
            )
        required_columns = (
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT
            + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
        )
        if (
            selection_membership_map.dim() != 2
            or selection_membership_map.shape[0] < num_tokens
            or selection_membership_map.shape[1] < required_columns
            or selection_membership_map.dtype != torch.int16
            or not self.is_fused_membership_storage(
                selection_membership_map
            )
        ):
            raise ValueError(
                "external FSA plan requires registered int16 membership storage: "
                f"min_shape=({num_tokens}, {required_columns}), "
                f"actual_shape={tuple(selection_membership_map.shape)}, "
                f"dtype={selection_membership_map.dtype}, "
                f"device={selection_membership_map.device}"
            )

        layer_id = self._get_offload_layer_id(layer_name)
        plan_start = (
            FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT - self.topk
        )
        plan_width = required_columns - plan_start
        plan_storage = selection_membership_map[
            :num_tokens,
            plan_start:required_columns,
        ]
        planner_membership_map = self.fused_overlap_planner_membership_map
        planner_plan_start = plan_start
        if self.tp_rank == 0:
            if planner_membership_map is None:
                planner_membership_map = selection_membership_map
            elif self.uses_mooncake_host:
                planner_plan_start = 0
            if (
                planner_membership_map.dim() != 2
                or planner_membership_map.shape[0] < num_tokens
                or planner_membership_map.shape[1]
                < planner_plan_start + plan_width
                or planner_membership_map.dtype != torch.int16
                or planner_membership_map.device.type != "cpu"
            ):
                raise ValueError(
                    "external FSA planner requires CPU int16 membership storage: "
                    f"shape={tuple(planner_membership_map.shape)}, "
                    f"dtype={planner_membership_map.dtype}, "
                    f"device={planner_membership_map.device}"
                )
            planner_storage = planner_membership_map[
                :num_tokens,
                planner_plan_start : planner_plan_start + plan_width,
            ]
            if self.uses_mooncake_host and not planner_storage.is_contiguous():
                raise RuntimeError(
                    "Mooncake external FSA planner staging must be compact and "
                    "contiguous"
                )
            encoded_plan_stride = planner_membership_map.stride(0)
        else:
            # Only TP0 runs the CPU planner. Other TP ranks consume the plan
            # published through the operator membership map after broadcast.
            planner_membership_map = selection_membership_map
            planner_storage = plan_storage
            encoded_plan_stride = selection_membership_map.stride(0)
        external_plan_debug = os.getenv("VLLM_ASCEND_SFA_DEBUG", "0") == "1"
        if external_plan_debug:
            logger.info(
                "[fused_overlap_offload][external_plan_debug] stage=prepare_enter "
                "tp=%s layer=%s layer_id=%s capturing=%s num_tokens=%s topk=%s "
                "required_columns=%s plan_start=%s encoded_stride=%s "
                "membership(device=%s shape=%s stride=%s ptr=%#x) "
                "planner(device=%s shape=%s stride=%s ptr=%#x) "
                "plan_storage(shape=%s stride=%s offset=%s ptr=%#x) "
                "planner_storage(shape=%s stride=%s offset=%s ptr=%#x)",
                self.tp_rank,
                layer_name,
                layer_id,
                capturing,
                num_tokens,
                self.topk,
                required_columns,
                plan_start,
                encoded_plan_stride,
                selection_membership_map.device,
                tuple(selection_membership_map.shape),
                tuple(selection_membership_map.stride()),
                selection_membership_map.data_ptr(),
                planner_membership_map.device,
                tuple(planner_membership_map.shape),
                tuple(planner_membership_map.stride()),
                planner_membership_map.data_ptr(),
                tuple(plan_storage.shape),
                tuple(plan_storage.stride()),
                plan_storage.storage_offset(),
                plan_storage.data_ptr(),
                tuple(planner_storage.shape),
                tuple(planner_storage.stride()),
                planner_storage.storage_offset(),
                planner_storage.data_ptr(),
            )

        def publish_plan(non_blocking: bool) -> None:
            same_ptr = planner_storage.data_ptr() == plan_storage.data_ptr()
            device_plan_staging = getattr(
                self,
                "fused_overlap_membership_plan_device_staging",
                None,
            )
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=publish_plan_before tp=%s layer_id=%s "
                    "non_blocking=%s same_ptr=%s src=%#x dst=%#x",
                    self.tp_rank,
                    layer_id,
                    non_blocking,
                    same_ptr,
                    planner_storage.data_ptr(),
                    plan_storage.data_ptr(),
                )
            if same_ptr:
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=publish_plan_after tp=%s layer_id=%s skipped_same_ptr=1",
                        self.tp_rank,
                        layer_id,
                    )
                return
            if device_plan_staging is None:
                raise RuntimeError(
                    "external FSA plan requires an NPU staging buffer when "
                    "planner and operator membership storage are separate"
                )
            device_plan_storage = device_plan_staging[
                :num_tokens,
                : required_columns - plan_start,
            ]
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=cpu_to_device_staging_before tp=%s layer_id=%s "
                    "src=%#x dst=%#x shape=%s",
                    self.tp_rank,
                    layer_id,
                    planner_storage.data_ptr(),
                    device_plan_storage.data_ptr(),
                    tuple(device_plan_storage.shape),
                )
            device_plan_storage.copy_(
                planner_storage,
                non_blocking=non_blocking,
            )
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=cpu_to_device_staging_after tp=%s layer_id=%s",
                    self.tp_rank,
                    layer_id,
                )
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=device_staging_to_membership_before tp=%s "
                    "layer_id=%s src=%#x dst=%#x shape=%s",
                    self.tp_rank,
                    layer_id,
                    device_plan_storage.data_ptr(),
                    plan_storage.data_ptr(),
                    tuple(plan_storage.shape),
                )
            plan_storage.copy_(
                device_plan_storage,
                non_blocking=non_blocking,
            )
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=device_staging_to_membership_after tp=%s layer_id=%s",
                    self.tp_rank,
                    layer_id,
                )
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=publish_plan_after tp=%s layer_id=%s skipped_same_ptr=0",
                    self.tp_rank,
                    layer_id,
                )

        owner_layer_id = self.fused_overlap_plan_owner_layer_id
        can_reuse_owner_plan = (
            skip_topk
            and owner_layer_id is not None
            and layer_id > owner_layer_id
            and self.fused_overlap_plan_topk == self.topk
            and self.fused_overlap_plan_num_tokens == num_tokens
            and self.fused_overlap_plan_membership_map is not None
        )
        if can_reuse_owner_plan:
            owner_map = self.fused_overlap_plan_membership_map
            if selection_membership_map.data_ptr() != owner_map.data_ptr():
                if capturing:
                    torch_npu.npu.current_stream().wait_stream(
                        self.fused_plan_stream
                    )
                plan_storage.copy_(
                    owner_map[:num_tokens, plan_start:required_columns],
                    non_blocking=capturing,
                )
            return True

        def run_planner(enqueue: bool) -> None:
            planner = (
                self.kv_offload_decode_cpp.enqueue_lru_resident_compact_with_plan_stable_rows
                if enqueue
                else self.kv_offload_decode_cpp.lru_resident_compact_with_plan_stable_rows
            )
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] stage=planner_before "
                    "tp=%s layer_id=%s enqueue=%s membership_ptr=%#x "
                    "encoded_stride=%s num_tokens=%s topk=%s capacity=%s "
                    "max_model_len=%s req_ids_ptr=%#x topk_ptr=%#x "
                    "stable_prefix_ptr=%#x visible_seq_ptr=%#x",
                    self.tp_rank,
                    layer_id,
                    enqueue,
                    planner_storage.data_ptr(),
                    encoded_plan_stride,
                    num_tokens,
                    self.topk,
                    self.topk_buffer_size,
                    self.max_model_len,
                    self.lru_req_ids_ptr,
                    self.lru_topk_indices_ptr,
                    self.lru_stable_prefix_lens_ptr,
                    self.lru_visible_seq_lens_ptr,
                )
            planner(
                self.lru_req_ids_ptr,
                self.lru_last_req_ids_ptrs[layer_id],
                self.lru_topk_indices_ptr,
                self.lru_stable_prefix_lens_ptr,
                self.lru_slot_to_token_ptrs[layer_id],
                self.lru_slots_ptrs[layer_id],
                self.lru_current_slots_ptr,
                self.lru_miss_count_ptrs[layer_id],
                self.lru_miss_tokens_ptrs[layer_id],
                self.lru_miss_slots_ptrs[layer_id],
                self.lru_token_mark_workspace_ptr,
                self.lru_token_pos_workspace_ptr,
                self.lru_slot_workspace_ptr,
                self.lru_miss_position_workspace_ptr,
                self.lru_epochs_ptr,
                self.lru_physical_row_workspace_ptr,
                self.max_num_topk_rows,
                planner_storage.data_ptr(),
                encoded_plan_stride,
                num_tokens,
                self.topk,
                self.topk_buffer_size,
                self.max_model_len,
                self.lru_workspace_threads,
                self.lru_workspace_threads,
                self.lru_visible_seq_lens_ptr,
            )
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] stage=planner_after "
                    "tp=%s layer_id=%s enqueue=%s",
                    self.tp_rank,
                    layer_id,
                    enqueue,
                )

        if capturing:
            if external_plan_debug:
                logger.info(
                    "[fused_overlap_offload][external_plan_debug] "
                    "stage=capture_input_schedule_before tp=%s layer_id=%s",
                    self.tp_rank,
                    layer_id,
                )
            plan_inputs_ready = torch_npu.npu.current_stream().record_event()
            with torch_npu.npu.stream(self.fused_plan_stream):
                self.fused_plan_stream.wait_event(plan_inputs_ready)
                if self.tp_rank == 0:
                    self.lru_topk_indices_cpu[:num_tokens].copy_(
                        topk_indices_npu, non_blocking=True
                    )
                    self.lru_req_ids_cpu[:num_tokens].copy_(
                        req_ids_npu, non_blocking=True
                    )
                    self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(
                        stable_prefix_lens_npu, non_blocking=True
                    )
                    self.lru_visible_seq_lens_cpu[:num_tokens].copy_(
                        visible_seq_lens_npu, non_blocking=True
                    )
                    run_planner(enqueue=True)
                    self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                        self.lru_physical_row_workspace[
                            self.max_num_topk_rows * 2 :
                            self.max_num_topk_rows * 2 + num_tokens
                        ],
                        non_blocking=True,
                    )
                    publish_plan(non_blocking=True)
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=broadcast_before tp=%s layer_id=%s capturing=1",
                        self.tp_rank,
                        layer_id,
                    )
                self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=broadcast_after tp=%s layer_id=%s capturing=1",
                        self.tp_rank,
                        layer_id,
                    )
            torch_npu.npu.current_stream().wait_stream(
                self.fused_plan_stream
            )
            self.fused_overlap_plan_owner_layer_id = layer_id
            self.fused_overlap_plan_topk = self.topk
            self.fused_overlap_plan_num_tokens = num_tokens
            self.fused_overlap_plan_membership_map = selection_membership_map
            return True

        planner_error = None
        if self.tp_rank == 0:
            self.fused_plan_status_npu.zero_()
            try:
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=input_copy_before tp=%s layer_id=%s capturing=0",
                        self.tp_rank,
                        layer_id,
                    )
                self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu)
                self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu)
                self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(
                    stable_prefix_lens_npu
                )
                self.lru_visible_seq_lens_cpu[:num_tokens].copy_(
                    visible_seq_lens_npu
                )
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=input_copy_after tp=%s layer_id=%s capturing=0",
                        self.tp_rank,
                        layer_id,
                    )
                run_planner(enqueue=False)
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=linear_slots_copy_before tp=%s layer_id=%s",
                        self.tp_rank,
                        layer_id,
                    )
                self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                    self.lru_physical_row_workspace[
                        self.max_num_topk_rows * 2 :
                        self.max_num_topk_rows * 2 + num_tokens
                    ]
                )
                if external_plan_debug:
                    logger.info(
                        "[fused_overlap_offload][external_plan_debug] "
                        "stage=linear_slots_copy_after tp=%s layer_id=%s",
                        self.tp_rank,
                        layer_id,
                    )
                publish_plan(non_blocking=False)
            except Exception as exc:
                planner_error = exc
                self.fused_plan_status_npu.fill_(1)
        if external_plan_debug:
            logger.info(
                "[fused_overlap_offload][external_plan_debug] "
                "stage=broadcast_before tp=%s layer_id=%s capturing=0",
                self.tp_rank,
                layer_id,
            )
        self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
        if external_plan_debug:
            logger.info(
                "[fused_overlap_offload][external_plan_debug] "
                "stage=broadcast_after tp=%s layer_id=%s capturing=0",
                self.tp_rank,
                layer_id,
            )
        if int(self.fused_plan_status_npu.item()) != 0:
            detail = (
                f"{type(planner_error).__name__}: {planner_error}"
                if planner_error is not None
                else "TP0 external planner failed"
            )
            raise RuntimeError(
                "SFA fused_overlap external planner failed: "
                f"layer={layer_name}, tp_rank={self.tp_rank}, {detail}"
            ) from planner_error
        self.fused_overlap_plan_owner_layer_id = layer_id
        self.fused_overlap_plan_topk = self.topk
        self.fused_overlap_plan_num_tokens = num_tokens
        self.fused_overlap_plan_membership_map = selection_membership_map
        return True

    def inject_current_kv_into_selection(
        self,
        layer_name: str,
        num_tokens: int,
        selection_kv_cache: torch.Tensor,
        selection_k_rope: torch.Tensor,
        capturing: bool = False,
    ) -> None:
        layer_id = self._get_offload_layer_id(layer_name)
        current_kv = self.current_kv_by_layer.get(layer_id)
        if current_kv is None:
            raise RuntimeError(
                f"current decode K/V is unavailable for fused layer {layer_name}"
            )
        if capturing:
            torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
        current_k, current_rope = current_kv
        current_k = current_k.reshape(-1, selection_kv_cache.shape[-1])[:num_tokens]
        current_rope = current_rope.reshape(-1, selection_k_rope.shape[-1])[:num_tokens]
        linear_slots = self.fused_plan_current_linear_slots_npu[:num_tokens]
        indices = linear_slots.view(-1, 1)
        flat_kv = selection_kv_cache.reshape(-1, selection_kv_cache.shape[-1])
        flat_rope = selection_k_rope.reshape(-1, selection_k_rope.shape[-1])
        torch_npu.npu_scatter_nd_update_(flat_kv, indices, current_k)
        torch_npu.npu_scatter_nd_update_(flat_rope, indices, current_rope)

    def wait_for_current_kv_writeback(self, capturing: bool = False) -> None:
        if (
            self.use_fused_overlap
            and capturing
            and self.tp_rank == 0
            and not self.d2h_index_copy_bypass
        ):
            torch_npu.npu.current_stream().wait_stream(self.current_kv_save_stream)

    def _onload_topk_kv_cpu(self, args):
        # code that is incompatible with graph mode, compute here outside graph
        (
            num_reqs,
            miss_count,
            miss_tokens,
            miss_slots,
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
            layer_id,
        ) = args
        if self.tp_size > 1:
            # Graph callbacks are stream-ordered after TP0's D2H. In eager mode,
            # the blocking metadata copies above wait for that same stream first.
            self.tp_group.barrier()
        self.kv_offload_decode_cpp.lru_resident_compact(
            lru_req_ids_ptr,
            lru_last_req_ids_ptr,
            lru_topk_indices_ptr,
            lru_stable_prefix_lens_ptr,
            lru_slot_to_token_ptr,
            lru_slots_ptr,
            lru_current_slots_ptr,
            lru_miss_count_ptr,
            lru_miss_tokens_ptr,
            lru_miss_slots_ptr,
            lru_token_mark_workspace_ptr,
            lru_token_pos_workspace_ptr,
            lru_slot_workspace_ptr,
            lru_miss_position_workspace_ptr,
            lru_epochs_ptr,
            num_reqs,
            self.topk,
            self.topk_buffer_size,
            self.max_model_len,
            self.lru_workspace_threads,
            self.lru_workspace_threads,
        )
        self.kv_offload_decode_cpp.compute_lru_resident_addrs(
            miss_count,
            miss_tokens,
            miss_slots,
            block_table,
            block_size,
            token_size_bytes_k,
            token_size_bytes_v,
            gvas_k_bases,
            gvas_v_bases,
            addr_k_bases,
            addr_v_bases,
            self.topk_buffer_size,
            self.lru_workspace_threads,
            gvas_buffer,
            addr_buffer,
            size_buffer,
            num_tokens_buffer,
        )

    def close(self) -> None:
        if self.fused_overlap_membership_region is not None:
            self.fused_overlap_membership_region.release()
            self.fused_overlap_membership_region = None
        self.fused_overlap_membership_map = None
        self.fused_overlap_planner_membership_map = None
        self.fused_overlap_membership_plan_device_staging = None


_KV_OFFLOAD_DECODE_MANAGER: KVOffloadDecodeManager = None


def init_kv_offload_decode_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_offload_decode_config: KVOffloadDecodeConfig,
):
    global _KV_OFFLOAD_DECODE_MANAGER
    if _KV_OFFLOAD_DECODE_MANAGER is None:
        _KV_OFFLOAD_DECODE_MANAGER = KVOffloadDecodeManager(
            vllm_config,
            kv_cache_config,
            kv_offload_decode_config,
        )
    return _KV_OFFLOAD_DECODE_MANAGER


def get_kv_offload_decode_manager():
    assert _KV_OFFLOAD_DECODE_MANAGER is not None, "KV offload manager is not initialized."
    return _KV_OFFLOAD_DECODE_MANAGER
