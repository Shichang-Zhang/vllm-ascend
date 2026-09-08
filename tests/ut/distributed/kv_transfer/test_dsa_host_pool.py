# SPDX-License-Identifier: Apache-2.0
"""Dependency-free contract tests for the shared DSA Host KV pool."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _DType:
    itemsize = 2

    def __str__(self):
        return "torch.bfloat16"


BFLOAT16 = _DType()


class _Tensor:

    def __init__(self, numel, ptr, *, shape=None, dtype=BFLOAT16):
        self._numel = numel
        self._ptr = ptr
        self.shape = shape or (numel,)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type="npu")

    def is_contiguous(self):
        return True

    def numel(self):
        return self._numel

    def data_ptr(self):
        return self._ptr

    def element_size(self):
        return self.dtype.itemsize

    def reshape(self, *_shape):
        return _Tensor(self._numel, self._ptr, dtype=self.dtype)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start or 0
            stop = self._numel if item.stop is None else item.stop
            return _Tensor(
                max(stop - start, 0),
                self._ptr + start * self.dtype.itemsize,
                dtype=self.dtype,
            )
        raise TypeError(f"unsupported index: {item!r}")

    def narrow(self, dim, start, length):
        assert dim == 0
        return _Tensor(
            length,
            self._ptr + start * self.dtype.itemsize,
            dtype=self.dtype,
        )

    def view(self, shape):
        return _Tensor(
            self._numel,
            self._ptr,
            shape=shape,
            dtype=self.dtype,
        )


def _load_host_pool_module():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = _Tensor
    fake_torch.dtype = _DType
    fake_torch.bfloat16 = BFLOAT16
    sys.modules["torch"] = fake_torch

    path = (
        Path(__file__).parents[4]
        / "vllm_ascend"
        / "distributed"
        / "kv_transfer"
        / "kv_offload_decode"
        / "host_pool.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dsa_host_pool_under_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HOST_POOL = _load_host_pool_module()


class _Engine:

    def __init__(self, *, fail_register_call=None):
        self.calls = []
        self.fail_register_call = fail_register_call
        self.register_calls = 0

    def register_memory(self, ptr, size, location=None):
        self.calls.append(("register", ptr, size, location))
        self.register_calls += 1
        if self.register_calls == self.fail_register_call:
            return -1
        return 0

    def unregister_memory(self, ptr):
        self.calls.append(("unregister", ptr))
        return 0


class TestDSAHostKVPool(unittest.TestCase):

    def _layout(self):
        return HOST_POOL.DSAHostKVPoolLayout(
            layer_names=("layer.0", "layer.1"),
            num_blocks=2,
            block_size=4,
            alignment=64,
        )

    def test_interleaved_views_share_one_aligned_region(self):
        layout = self._layout()
        pool = HOST_POOL.DSAHostKVPool(
            layout,
            HOST_POOL.DSAHostMemoryRegion(
                tensor=_Tensor(layout.total_numel, 0x200000)
            ),
        )

        self.assertEqual(pool.k_caches[0].shape, layout.k_shape)
        self.assertEqual(pool.v_caches[0].shape, layout.v_shape)
        self.assertEqual(pool.k_caches[0].data_ptr(), pool.data_ptr)
        self.assertEqual(
            pool.v_caches[0].data_ptr(),
            pool.data_ptr + layout.k_stride_numel * BFLOAT16.itemsize,
        )
        layer_stride = layout.layer_stride_numel * BFLOAT16.itemsize
        self.assertEqual(
            pool.k_caches[1].data_ptr(),
            pool.data_ptr + layer_stride,
        )
        for tensor in (*pool.k_caches, *pool.v_caches):
            self.assertEqual(tensor.data_ptr() % layout.alignment, 0)

    def test_glm_layout_keeps_each_view_2mb_aligned(self):
        layout = HOST_POOL.DSAHostKVPoolLayout(
            layer_names=("layer.0", "layer.1"),
            num_blocks=591,
            block_size=128,
        )
        self.assertNotEqual(
            layout.k_numel * BFLOAT16.itemsize % layout.alignment,
            0,
        )
        self.assertEqual(
            layout.k_stride_numel * BFLOAT16.itemsize % layout.alignment,
            0,
        )
        self.assertEqual(
            layout.v_stride_numel * BFLOAT16.itemsize % layout.alignment,
            0,
        )

    def test_glm_pool_te_registration_splits_at_layer_boundary(self):
        layout = HOST_POOL.DSAHostKVPoolLayout(
            layer_names=tuple(f"layer.{i}" for i in range(79)),
            num_blocks=8806,
            block_size=128,
        )
        layer_bytes = 1_300_234_240
        first_size = 52 * layer_bytes
        second_size = 27 * layer_bytes

        self.assertEqual(layout.layer_stride_nbytes, layer_bytes)
        self.assertEqual(
            layout.te_registration_ranges,
            ((0, first_size), (first_size, second_size)),
        )
        self.assertLess(
            first_size,
            HOST_POOL.DSA_HOST_TE_REGISTRATION_LIMIT_BYTES,
        )
        self.assertGreaterEqual(
            first_size + layer_bytes,
            HOST_POOL.DSA_HOST_TE_REGISTRATION_LIMIT_BYTES,
        )
        self.assertLess(
            second_size,
            HOST_POOL.DSA_HOST_TE_REGISTRATION_LIMIT_BYTES,
        )

        pool = HOST_POOL.DSAHostKVPool(
            layout,
            HOST_POOL.DSAHostMemoryRegion(
                tensor=_Tensor(layout.total_numel, 0x400000),
                register_location="npu:0",
            ),
        )
        engine = _Engine()

        pool.register(engine)
        pool.unregister()

        self.assertEqual(
            engine.calls,
            [
                ("register", pool.data_ptr, first_size, "npu:0"),
                (
                    "register",
                    pool.data_ptr + first_size,
                    second_size,
                    "npu:0",
                ),
                ("unregister", pool.data_ptr + first_size),
                ("unregister", pool.data_ptr),
            ],
        )

    def test_split_registration_failure_rolls_back_prior_regions(self):
        layout = HOST_POOL.DSAHostKVPoolLayout(
            layer_names=tuple(f"layer.{i}" for i in range(79)),
            num_blocks=8806,
            block_size=128,
        )
        pool = HOST_POOL.DSAHostKVPool(
            layout,
            HOST_POOL.DSAHostMemoryRegion(
                tensor=_Tensor(layout.total_numel, 0x400000),
                register_location="npu:0",
            ),
        )
        engine = _Engine(fail_register_call=2)
        first_size = layout.te_registration_ranges[0][1]

        with self.assertRaisesRegex(RuntimeError, "register_memory failed"):
            pool.register(engine)

        self.assertFalse(pool.is_registered)
        self.assertEqual(
            engine.calls,
            [
                ("register", pool.data_ptr, first_size, "npu:0"),
                (
                    "register",
                    pool.data_ptr + first_size,
                    layout.te_registration_ranges[1][1],
                    "npu:0",
                ),
                ("unregister", pool.data_ptr),
            ],
        )

    def test_register_once_then_unregister_before_release(self):
        layout = self._layout()
        lifecycle = []
        region = HOST_POOL.DSAHostMemoryRegion(
            tensor=_Tensor(layout.total_numel, 0x400000),
            handle="segment",
            register_location="npu:0",
            release_callback=lambda handle: lifecycle.append(
                ("release", handle)
            ),
        )
        pool = HOST_POOL.DSAHostKVPool(layout, region)
        engine = _Engine()

        pool.register(engine)
        pool.register(engine)
        pool.close()
        pool.close()

        self.assertEqual(
            engine.calls,
            [
                ("register", pool.data_ptr, pool.nbytes, "npu:0"),
                ("unregister", pool.data_ptr),
            ],
        )
        self.assertEqual(lifecycle, [("release", "segment")])

    def test_non_owner_maps_views_and_registers_local_te(self):
        layout = self._layout()
        topology = HOST_POOL.DSAHostPoolTopology(
            tp_rank=1,
            tp_size=2,
            owner_rank=0,
        )
        pool = HOST_POOL.DSAHostKVPool(
            layout,
            HOST_POOL.DSAHostMemoryRegion(
                tensor=_Tensor(layout.total_numel, 0x600000),
                register_location="npu:1",
            ),
            topology,
        )
        engine = _Engine()

        self.assertFalse(pool.is_owner)
        self.assertEqual(len(pool.k_caches), layout.num_layers)
        pool.register(engine)
        self.assertEqual(
            engine.calls,
            [("register", pool.data_ptr, pool.nbytes, "npu:1")],
        )

    def test_failed_construction_releases_region(self):
        layout = self._layout()
        released = []

        def allocator(**_kwargs):
            return HOST_POOL.DSAHostMemoryRegion(
                tensor=_Tensor(layout.total_numel - 1, 0x800000),
                handle="short",
                release_callback=released.append,
            )

        with self.assertRaisesRegex(ValueError, "too small"):
            HOST_POOL.DSAHostKVPool.allocate(
                layout,
                allocator=allocator,
            )
        self.assertEqual(released, ["short"])

    def test_topology_rejects_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "tp_rank out of range"):
            HOST_POOL.DSAHostPoolTopology(tp_rank=2, tp_size=2)


if __name__ == "__main__":
    unittest.main()
