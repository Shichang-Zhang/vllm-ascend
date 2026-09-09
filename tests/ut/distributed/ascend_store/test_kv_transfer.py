#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    GroupTransferData,
    KeyMetadata,
    LayerLoadTask,
    LayerMultiBlockReqMeta,
    LayerPoolKey,
    LayerSaveTask,
    LayerTransferArrays,
    LayerTransferTask,
    LayerwisePreparation,
    LoadSpec,
    PoolKey,
    ReqMeta,
    TransferCompletion,
)

# isort: on
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreKeyLayerSendingThread,
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    _mark_last_transfer_tasks,
    _trace_store_batch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_transfer import (
    LayerTransferArrayBuilder,
    LayerwiseTransferPreparer,
)


class FakeStore:
    def __init__(self, exists_result=None):
        self.exists_result = exists_result or []
        self.put_calls = []
        self.get_calls = []

    def set_device(self):
        pass

    def exists(self, keys):
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))

    def get(self, keys, addrs, sizes):
        self.get_calls.append((list(keys), list(addrs), list(sizes)))


class FakeKey:
    def __init__(self, val):
        self._val = val

    def to_string(self):
        return self._val


class FakeTokenDatabase:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.group_block_len = [[block_size, block_size]]
        self.group_kv_caches_base_addr = [[0, block_size]]
        self.group_block_stride = {0: [block_size, block_size]}

    def process_tokens(self, token_len, block_hashes, mask_num=0):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        for i, h in enumerate(block_hashes):
            start = i * self.block_size
            if start >= token_len:
                break
            end = min(start + self.block_size, token_len)
            if start < mask_num:
                continue
            yield start, end, PoolKey(meta, f"k{i}")

    def prepare_value(self, start, end, block_ids):
        block_id = block_ids[start // self.block_size]
        return [1000 + block_id], [end - start], block_id

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = block_ids[start // self.block_size]
        return [2000 + layer_id * 100 + block_id], [end - start], block_id

    def decode_adaptor_prefill_pp(self, keys, addrs, sizes):
        return keys, addrs, sizes


class MaskedFakeTokenDatabase(FakeTokenDatabase):
    def __init__(self, block_size=16, masks=([True],)):
        super().__init__(block_size)
        self.masks = masks

    def store_mask(self, token_len, num_prompt_tokens=None):
        return self.masks

    def load_mask(self, block_hashes, token_len):
        return self.masks

    def mask_allows_chunk(self, masks, kv_cache_group_id, start):
        if masks is None:
            return True
        block_idx = start // self.block_size
        return block_idx < len(masks[kv_cache_group_id]) and masks[kv_cache_group_id][block_idx]


class TestStoreIntegrityTrace(unittest.TestCase):
    def test_logs_full_metadata_result_and_checksum(self):
        db = FakeTokenDatabase(block_size=2)
        db.metadata = [KeyMetadata("model", 0, 0, 3, 0)]
        db.trace_dp_rank = 1
        db.trace_group_tensors = {
            0: {
                7: [(torch.tensor([[1.0, -2.0], [3.0, -4.0]]), 1)],
            }
        }

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer."
                "ascend_envs.VLLM_ASCEND_SFA_DEBUG",
                True,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.logger.info"
            ) as log_info,
        ):
            _trace_store_batch(
                phase="GET_DEST",
                request_id="req-1",
                tp_rank=3,
                token_database=db,
                keys=["complete-key"],
                addrs=[[0x1000]],
                sizes=[[4]],
                block_ids=[1],
                group_ids=[0],
                results=[0],
            )

        args = log_info.call_args.args
        self.assertEqual(args[1:13], ("GET_DEST", "req-1", 1, 3, 0, 3, 0, 0, 0, 0, 1, 0))
        self.assertEqual(args[13], "complete-key")
        self.assertEqual(args[14:18], (1, 1, 1, 4))
        self.assertEqual(args[18], ["0x1000"])
        self.assertEqual(args[19], [4])
        self.assertIn("layer=7,pos=0,physical=1", args[20])
        self.assertIn("abs_sum=7", args[20])


class TestLayerTransferArrayBuilderCompactGvas(unittest.TestCase):
    def _make_builder(self, group_id=1):
        db = MagicMock()
        db.group_block_len = {0: [16], 1: [16]}
        db.group_kv_caches_base_addr = {0: [1000], 1: [2000]}
        db.group_block_stride = {0: [16], 1: [16]}
        return LayerTransferArrayBuilder(
            token_database=db,
            num_layers=1,
            group_id=group_id,
        )

    def test_layer_gvas_are_base_gvas_plus_vectorized_offsets(self):
        db = MagicMock()
        db.group_block_len = {0: [4, 6, 4, 6]}
        db.group_kv_caches_base_addr = {0: [100, 200, 300, 400]}
        db.group_block_stride = {0: [10, 10, 10, 10]}
        builder = LayerTransferArrayBuilder(
            token_database=db,
            num_layers=2,
        )

        addrs, sizes, gvas = builder._build_transfer_arrays(
            np.asarray([2, 3], dtype=np.int64),
            np.asarray([1000, 2000], dtype=np.int64),
            layer_id=1,
        )

        np.testing.assert_array_equal(addrs, [320, 420, 330, 430])
        np.testing.assert_array_equal(sizes, [4, 6, 4, 6])
        np.testing.assert_array_equal(gvas, [1010, 1014, 2010, 2014])

    def test_variable_cache_legs_use_compact_layer_offsets(self):
        db = MagicMock()
        # Layer 0 has main K/V + indexer K; layer 1 has only main K/V.
        db.group_block_len = {0: [4, 6, 2, 4, 6]}
        db.group_kv_caches_base_addr = {0: [100, 200, 250, 300, 400]}
        db.group_block_stride = {0: [10, 10, 10, 10, 10]}
        db.group_layer_offsets = {0: [0, 3, 5]}
        builder = LayerTransferArrayBuilder(
            token_database=db,
            num_layers=2,
        )

        block_ids = np.asarray([2], dtype=np.int64)
        base_gvas = np.asarray([1000], dtype=np.int64)
        layer0 = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=0)
        layer1 = builder._build_transfer_arrays(block_ids, base_gvas, layer_id=1)

        np.testing.assert_array_equal(layer0[0], [120, 220, 270])
        np.testing.assert_array_equal(layer0[1], [4, 6, 2])
        np.testing.assert_array_equal(layer0[2], [1000, 1004, 1010])
        np.testing.assert_array_equal(layer1[0], [320, 420])
        np.testing.assert_array_equal(layer1[1], [4, 6])
        np.testing.assert_array_equal(layer1[2], [1012, 1016])

    def test_build_addrs_only_consumes_block_ids_and_base_gvas(self):
        data = GroupTransferData(
            block_ids_arr=np.asarray([10, 11], dtype=np.int64),
            base_gvas_arr=np.asarray([800, 900], dtype=np.int64),
        )

        arrays = self._make_builder().build_addrs(data, layer_id=0)

        np.testing.assert_array_equal(arrays.addr_array, [2160, 2176])
        np.testing.assert_array_equal(arrays.gvas_array, [800, 900])


class TestLayerwiseTransferPreparer(unittest.TestCase):
    @staticmethod
    def _make_preparer():
        preparer = LayerwiseTransferPreparer(
            MagicMock(),
            model_name="model",
            head_or_tp_rank=0,
            hash_block_size=16,
            enabled=True,
            can_allocate=True,
            num_groups=1,
        )
        preparer.configure_layout(
            group_block_len={0: [16]},
        )
        return preparer

    def test_prepares_once_and_attaches_to_save_tasks(self):
        plans = []
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        layer_tasks = [[task]]
        transfer_data = MagicMock()
        completion = MagicMock()
        preparer = self._make_preparer()
        prepare_tasks = MagicMock()
        with patch.object(
            preparer,
            "resolve_save_groups",
            return_value={0: (transfer_data, completion)},
        ) as resolve:
            preparation = preparer.create_save_preparation(
                plans,
                layer_tasks,
                prepare_tasks,
            )

            self.assertIs(task.preparation, preparation)
            resolve.assert_not_called()
            prepare_tasks.assert_not_called()

            preparation.ensure_ready()
            preparation.ensure_ready()

            resolve.assert_called_once()
            self.assertIs(task.transfer_data, transfer_data)
            self.assertIs(task.completion, completion)
            prepare_tasks.assert_called_once_with(layer_tasks)

    def test_save_finishes_on_last_actual_transfer_task(self):
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        transfer_data = MagicMock()
        completion = TransferCompletion(["r1"], [True])
        preparer = self._make_preparer()
        with patch.object(
            preparer,
            "resolve_save_groups",
            return_value={0: (transfer_data, completion)},
        ):
            preparation = preparer.create_save_preparation(
                [],
                [[task], []],
                lambda tasks: _mark_last_transfer_tasks(tasks, "save"),
            )
            preparation.ensure_ready()

        self.assertEqual(task.finished_req_ids, {"r1"})

    def test_load_preparation_does_not_mutate_transfer_tasks(self):
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        preparation = self._make_preparer().create_load_preparation(
            [],
            [[task]],
        )

        self.assertIsNone(task.preparation)
        self.assertIsNotNone(preparation)

    def test_load_finishes_on_last_actual_transfer_task(self):
        task = LayerTransferTask(layer_id=0, block_ranges=[])
        transfer_data = MagicMock()
        completion = TransferCompletion(["r1"], [True])
        preparer = self._make_preparer()
        with patch.object(
            preparer,
            "resolve_load_groups",
            return_value={(0, False): (transfer_data, completion)},
        ):
            preparation = preparer.create_load_preparation(
                [],
                [[task], []],
                lambda tasks: _mark_last_transfer_tasks(tasks, "load"),
            )
            preparation.ensure_ready()

        self.assertEqual(task.finished_req_ids, {"r1"})


class TestLayerwiseTaskPreparation(unittest.TestCase):
    def test_key_send_reuses_cached_process_tokens(self):
        thread = object.__new__(KVCacheStoreKeyLayerSendingThread)
        cached = {0: [(0, 16, [MagicMock()])]}
        thread.build_cached_process_tokens = MagicMock(return_value=cached)
        tasks = [
            [LayerTransferTask(layer_id=0, block_ranges=[])],
            [LayerTransferTask(layer_id=1, block_ranges=[])],
        ]

        thread.prepare_layerwise_tasks(tasks)

        self.assertIs(tasks[0][0].cached_process_tokens, cached)
        self.assertIs(tasks[1][0].cached_process_tokens, cached)
        thread.build_cached_process_tokens.assert_called_once_with(tasks[0][0])


class TestKVTransferThread(unittest.TestCase):
    def _make_thread(self, exists_result=None):
        store = FakeStore(exists_result or [])
        db = FakeTokenDatabase()
        t = KVTransferThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            name="test",
        )
        return t, store

    def test_add_request(self):
        t, _ = self._make_thread()
        req = MagicMock()
        t.add_request(req)
        self.assertFalse(t.request_queue.empty())

    def test_fatal_error_stops_before_processing_queued_followup(self):
        t, _ = self._make_thread()
        first = object()
        followup = object()
        root_cause = RuntimeError("first transfer failed")
        t._handle_request = MagicMock(side_effect=root_cause)
        t.add_request(first)
        t.add_request(followup)

        t.start()
        t.join(timeout=5)

        self.assertFalse(t.is_alive())
        t._handle_request.assert_called_once_with(first)
        self.assertIs(t._fatal_error, root_cause)
        self.assertTrue(t.request_queue.empty())
        self.assertEqual(t.request_queue.unfinished_tasks, 0)
        with self.assertRaisesRegex(RuntimeError, "failed during asynchronous transfer") as exc_info:
            t.raise_if_failed()
        self.assertIs(exc_info.exception.__cause__, root_cause)
        with self.assertRaisesRegex(RuntimeError, "failed during asynchronous transfer"):
            t.add_request(object())

    def test_device_initialization_failure_releases_ready_and_drains_queue(self):
        t, store = self._make_thread()
        root_cause = RuntimeError("set device failed")
        store.set_device = MagicMock(side_effect=root_cause)
        t.add_request(object())

        t.start()
        t.join(timeout=5)

        self.assertFalse(t.is_alive())
        self.assertTrue(t.ready_event.is_set())
        self.assertIs(t._fatal_error, root_cause)
        self.assertTrue(t.request_queue.empty())
        self.assertEqual(t.request_queue.unfinished_tasks, 0)
        with self.assertRaisesRegex(RuntimeError, "failed during asynchronous transfer") as exc_info:
            t.raise_if_failed()
        self.assertIs(exc_info.exception.__cause__, root_cause)

    def test_get_and_clear_finished_requests(self):
        t, _ = self._make_thread()
        t.set_finished_request("r1")
        t.set_finished_request("r2")
        finished = t.get_and_clear_finished_requests()
        self.assertEqual(finished, {"r1", "r2"})
        self.assertEqual(t.get_and_clear_finished_requests(), set())

    def test_lookup_all_exist(self):
        t, _ = self._make_thread([1, 1, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, True, True])

    def test_lookup_partial(self):
        t, _ = self._make_thread([1, 0, 1])
        result = t.lookup(["k1", "k2", "k3"])
        self.assertEqual(result, [True, False, True])

    def test_lookup_exception(self):
        t, store = self._make_thread()
        store.exists = MagicMock(side_effect=Exception("conn fail"))
        result = t.lookup(["k1"])
        self.assertEqual(result, [False])

    def test_update_and_get_kv_events(self):
        t, _ = self._make_thread()
        event1 = BlockStored(
            block_hashes=["h1"],
            parent_block_hash=None,
            token_ids=[1, 2, 3],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        event2 = BlockStored(
            block_hashes=["h2"],
            parent_block_hash="h1",
            token_ids=[4, 5, 6],
            block_size=16,
            lora_id=None,
            medium="cpu",
            lora_name=None,
        )
        t.update_kv_event([event1, event2])
        events = t.get_kv_events()
        self.assertEqual(len(events), 2)
        # After get, events should be cleared
        self.assertEqual(len(t.get_kv_events()), 0)

    def test_handle_request_base_noop(self):
        t, _ = self._make_thread()
        # Base class _handle_request does nothing
        t._handle_request(MagicMock())


class TestKVCacheStoreSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, kv_role="kv_producer", enable_kv_event=False):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role=kv_role,
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=enable_kv_event,
        )
        return t, store

    def test_handle_request_puts_missing_keys(self):
        t, store = self._make_thread([1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 2)

    def test_handle_request_all_exist_no_put(self):
        t, store = self._make_thread([1, 1])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_not_in_stored(self):
        t, store = self._make_thread([0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_with_kv_event(self):
        t, store = self._make_thread([0], enable_kv_event=True)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
            token_ids=list(range(16)),
            original_block_size=16,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)

    def test_handle_request_consumer_role(self):
        t, store = self._make_thread([0], kv_role="kv_consumer")
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)

    def test_add_dec_delete_stored_request(self):
        t, _ = self._make_thread()
        t.add_stored_request("r1")
        t.add_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 2)
        t.dec_stored_request("r1")
        self.assertEqual(t.stored_requests["r1"], 1)
        t.delete_finished_stored_request("r1")
        self.assertNotIn("r1", t.stored_requests)

    def test_dec_nonexistent_request(self):
        t, _ = self._make_thread()
        t.dec_stored_request("nonexist")  # should not raise

    def test_delete_nonexistent_request(self):
        t, _ = self._make_thread()
        t.delete_finished_stored_request("nonexist")  # should not raise

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[0],
            block_hashes=[b"h0"],  # type: ignore[arg-type]
            current_event=event,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_dcp_size_gt_1(self):
        store = FakeStore([0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=2,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        # dcp_size > 1 means no slicing
        self.assertEqual(len(store.put_calls), 1)

    def test_handle_request_applies_store_mask(self):
        store = FakeStore([0, 0])
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
        )
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)


class TestKVCacheStoreRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_applies_load_mask(self):
        store = FakeStore()
        db = MaskedFakeTokenDatabase(masks=([True, False],))
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        load_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32)
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=32,
            block_ids=[0, 1],
            block_hashes=[b"h0", b"h1"],  # type: ignore[arg-type]
            load_spec=load_spec,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        keys, _, _ = store.get_calls[0]
        self.assertEqual(len(keys), 1)


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class TestKVCacheStoreLayerSendingThread(unittest.TestCase):
    def _make_thread(self, exists_result=None, num_layers=2):
        store = FakeStore(exists_result or [0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=num_layers,
            layer_save_finished_events=[threading.Event() for _ in range(num_layers)],
            sync_save_events=[],
        )
        return t, store

    def _make_layer_req(self, layer_id=0, is_last_chunk=False, num_keys=2):
        meta = KeyMetadata("m", 0, 0, 0, 0)
        keys = [LayerPoolKey(meta, f"h{i}", layer_id) for i in range(num_keys)]
        return LayerMultiBlockReqMeta(
            req_id="r1",
            keys=keys,
            starts=[i * 16 for i in range(num_keys)],
            ends=[(i + 1) * 16 for i in range(num_keys)],
            block_ids=list(range(num_keys)),
            layer_id=layer_id,
            is_last_chunk=is_last_chunk,
            current_event=None,
            token_ids=list(range(num_keys * 16)),
            original_block_size=16,
            block_hashes=[f"h{i}".encode() for i in range(num_keys)],
        )

    def test_handle_request_puts_missing(self):
        t, store = self._make_thread([1, 0])
        req = self._make_layer_req(layer_id=0)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)
        keys, _, _ = store.put_calls[0]
        self.assertEqual(len(keys), 1)

    def test_handle_request_all_exist_not_last(self):
        t, store = self._make_thread([1, 1])
        req = self._make_layer_req(layer_id=0, is_last_chunk=False)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 0)

    def test_handle_request_all_exist_last_chunk_final_layer(self):
        t, store = self._make_thread([1, 1], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_handle_request_empty_keys(self):
        t, store = self._make_thread()
        _meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[],
            starts=[],
            ends=[],
            block_ids=[],
            layer_id=0,
            is_last_chunk=True,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertNotIn("r1", finished)

    def test_handle_request_with_current_event(self):
        t, store = self._make_thread([0])
        event = MagicMock()
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
            is_last_chunk=False,
            current_event=event,
        )
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        event.synchronize.assert_called_once()

    def test_handle_request_last_chunk_final_layer_with_missing(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        finished = t.get_and_clear_finished_requests()
        self.assertIn("r1", finished)

    def test_layerwise_kv_event_published_on_final_layer(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])
        self.assertEqual(events[0].token_ids, list(range(16)))
        self.assertEqual(events[0].block_size, 16)

    def test_layerwise_kv_event_not_published_before_final_layer(self):
        t, store = self._make_thread([0], num_layers=2)
        req = self._make_layer_req(layer_id=0, is_last_chunk=False, num_keys=1)
        t.add_stored_request(req.req_id)
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(t.get_kv_events(), [])

    def test_layerwise_kv_event_uses_missing_blocks_from_previous_layers(self):
        t, store = self._make_thread([0], num_layers=2)
        first_layer_req = self._make_layer_req(layer_id=0, is_last_chunk=True, num_keys=1)
        t.add_stored_request(first_layer_req.req_id)
        t.request_queue.put(first_layer_req)
        t._handle_request(first_layer_req)
        t.m_store.exists_result = [1]
        final_layer_req = self._make_layer_req(layer_id=1, is_last_chunk=True, num_keys=1)
        t.request_queue.put(final_layer_req)
        t._handle_request(final_layer_req)
        events = t.get_kv_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].block_hashes, [maybe_convert_block_hash(b"h0")])


class TestGVALayerSendingThread(unittest.TestCase):
    def _make_thread(self, copy_result=0, builders=None):
        store = MagicMock()
        store.store.batch_copy.return_value = copy_result
        db = MagicMock()
        db.group_block_len = {0: [16]}
        layer_finished = threading.Event()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[layer_finished],
            sync_save_events=[MagicMock()],
            group_array_builders=builders,
        )
        return thread, store, layer_finished

    def test_merges_group_data_and_completes_after_success(self):
        builders = [MagicMock(), MagicMock()]
        builders[0].build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        builders[1].build_addrs.return_value = LayerTransferArrays(
            np.asarray([20]),
            np.asarray([16]),
            np.asarray([200]),
        )
        thread, store, layer_finished = self._make_thread(builders=builders)
        tasks = [
            LayerTransferTask(
                layer_id=0,
                block_ranges=[],
                transfer_data=MagicMock(),
                completion=TransferCompletion(["r1"], [True]),
                finished_req_ids={"r1"} if group_id == 1 else set(),
                group_id=group_id,
            )
            for group_id in range(2)
        ]
        thread.add_stored_request("r1")
        thread.add_stored_request("r1")
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        thread._handle_request(request)

        store.store.batch_copy.assert_called_once_with([100, 200], [10, 20], [16, 16], 0)
        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})
        self.assertTrue(layer_finished.is_set())

    def test_missing_group_data_is_fatal(self):
        thread, _, layer_finished = self._make_thread(builders=[MagicMock()])
        tasks = [LayerTransferTask(layer_id=0, block_ranges=[], group_id=0)]
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        with self.assertRaisesRegex(RuntimeError, "save metadata was not prepared"):
            thread._handle_request(request)

        self.assertFalse(layer_finished.is_set())

    def test_copy_failure_does_not_complete_request_or_layer(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        thread, _, layer_finished = self._make_thread(copy_result=1, builders=[builder])
        tasks = [
            LayerTransferTask(
                layer_id=0,
                block_ranges=[],
                transfer_data=MagicMock(),
                completion=TransferCompletion(["r1"], [True]),
            )
        ]
        thread.add_stored_request("r1")
        request = LayerSaveTask(layer_id=0, transfer_tasks=tasks)
        thread.request_queue.put(request)

        with self.assertRaisesRegex(RuntimeError, "save batch_copy failed"):
            thread._handle_request(request)

        self.assertEqual(thread.stored_requests["r1"], 1)
        self.assertEqual(thread.get_and_clear_finished_requests(), set())
        self.assertFalse(layer_finished.is_set())


class TestGVALayerSendingThreadEventSplit(unittest.TestCase):
    """L2G and attention completion gate AscendStore's internal slot reuse."""

    def _make_thread(self, copy_result=0, builders=None, attn_recorded=True):
        store = MagicMock()
        store.store.batch_copy.return_value = copy_result
        db = MagicMock()
        db.group_block_len = {0: [16]}
        layer_finished = threading.Event()
        attn_flag = threading.Event()
        if attn_recorded:
            attn_flag.set()
        sync_attn = MagicMock()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=1,
            layer_save_finished_events=[layer_finished],
            sync_save_events=[MagicMock()],
            group_array_builders=builders,
            sync_attn_events=[sync_attn],
            layer_attn_recorded_events=[attn_flag],
        )
        return thread, store, layer_finished, sync_attn

    def _make_builder(self):
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([10]),
            np.asarray([16]),
            np.asarray([100]),
        )
        return builder

    def _make_task(self):
        return LayerSaveTask(
            layer_id=0,
            transfer_tasks=[
                LayerTransferTask(
                    layer_id=0,
                    block_ranges=[],
                    transfer_data=MagicMock(),
                    completion=TransferCompletion([], []),
                )
            ],
        )

    def test_copy_runs_before_slot_free(self):
        call_order = []
        thread, store, layer_finished, _ = self._make_thread(
            builders=[self._make_builder()],
        )
        store.store.batch_copy.side_effect = lambda *_a: call_order.append("copy") or 0
        request = self._make_task()
        thread.request_queue.put(request)

        thread._handle_request(request)

        self.assertEqual(call_order, ["copy"])
        self.assertTrue(layer_finished.is_set())

    def test_slot_free_waits_for_attention_done(self):
        # Attention not yet recorded: the send thread must block on the
        # threading flag, so spin it up and release the flag shortly after.
        thread, store, layer_finished, sync_attn = self._make_thread(
            builders=[self._make_builder()],
            attn_recorded=False,
        )
        attn_flag = thread.layer_attn_recorded_events[0]
        request = self._make_task()
        thread.request_queue.put(request)

        done = threading.Event()

        def run():
            thread._handle_request(request)
            done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        # Give the worker a moment to reach the attention-done wait.
        self.assertFalse(done.wait(timeout=0.2))
        store.store.batch_copy.assert_called_once()
        self.assertFalse(layer_finished.is_set())  # slot_free still gated on (c)
        attn_flag.set()
        self.assertTrue(done.wait(timeout=5))
        worker.join(timeout=5)
        sync_attn.synchronize.assert_called_once()
        self.assertTrue(layer_finished.is_set())

    def test_control_only_save_waits_for_attention_before_slot_free(self):
        thread, store, layer_finished, sync_attn = self._make_thread(
            builders=[self._make_builder()],
            attn_recorded=False,
        )
        request = LayerSaveTask(layer_id=0, transfer_tasks=[])
        thread.request_queue.put(request)
        done = threading.Event()

        def run():
            thread._handle_request(request)
            done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        self.assertFalse(done.wait(timeout=0.2))
        self.assertFalse(layer_finished.is_set())
        thread.layer_attn_recorded_events[0].set()
        self.assertTrue(done.wait(timeout=5))
        worker.join(timeout=5)
        sync_attn.synchronize.assert_called_once()
        self.assertTrue(layer_finished.is_set())
        store.store.batch_copy.assert_not_called()


@unittest.skip("LayerMultiBlockReqMeta API is deprecated, tests need update for LayerTransferTask")
class TestKVCacheStoreLayerRecvingThread(unittest.TestCase):
    def test_handle_request(self):
        store = FakeStore()
        db = FakeTokenDatabase()
        get_event = threading.Event()
        t = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=get_event,
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
        )
        meta = KeyMetadata("m", 0, 0, 0, 0)
        req = LayerMultiBlockReqMeta(
            req_id="r1",
            keys=[LayerPoolKey(meta, "h0", 0)],
            starts=[0],
            ends=[16],
            block_ids=[0],
            layer_id=0,
        )
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.get_calls), 1)
        self.assertTrue(get_event.is_set())


class TestGVALayerRecvingThread(unittest.TestCase):
    def _make_reuse_thread(self, external_slot_release_waiter=None, save_failure_checker=None):
        store = MagicMock()
        store.store.batch_copy.return_value = 0
        db = MagicMock()
        db.group_block_len = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            addr_array=np.asarray([1000], dtype=np.int64),
            size_array=np.asarray([16], dtype=np.int64),
            gvas_array=np.asarray([2000], dtype=np.int64),
        )
        load_finished = [threading.Event(), threading.Event()]
        save_finished = [threading.Event(), threading.Event()]
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=load_finished,
            layer_save_finished_events=save_finished,
            num_layers=2,
            group_array_builders=[builder],
            external_slot_release_waiter=external_slot_release_waiter,
            save_failure_checker=save_failure_checker,
        )
        return thread, store, load_finished, save_finished

    def test_h2d_stagger_sleeps_before_short_final_spin(self):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread._get_h2d_stagger_delay_us = MagicMock(return_value=100)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.perf_counter_ns",
                side_effect=[0, 100_000],
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.sleep"
            ) as sleep,
        ):
            thread._stagger_h2d_submit(layer_id=0)

        sleep.assert_called_once_with(50 / 1_000_000)

    def test_short_h2d_stagger_does_not_sleep(self):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread._get_h2d_stagger_delay_us = MagicMock(return_value=25)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.perf_counter_ns",
                side_effect=[0, 25_000],
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.sleep"
            ) as sleep,
        ):
            thread._stagger_h2d_submit(layer_id=0)

        sleep.assert_not_called()

    def test_h2d_waits_for_source_save_then_external_slot_release(self):
        call_order = []
        thread, store, _, save_finished = self._make_reuse_thread(
            external_slot_release_waiter=lambda layer_id: call_order.append(("reuse", layer_id))
        )
        save_finished[0].set()
        store.store.batch_copy.side_effect = lambda *_args: call_order.append(("h2d", 1)) or 0
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion([], []),
        )
        load_task = LayerLoadTask(wait_for_save_layer=0, transfer_tasks=[task], layer_id=1)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(call_order, [("reuse", 1), ("h2d", 1)])
        self.assertFalse(save_finished[0].is_set())

    def test_empty_load_waits_for_external_slot_release_before_finish(self):
        observed_load_state = []
        thread = None

        def wait_for_release(layer_id):
            assert thread is not None
            observed_load_state.append(thread.layer_load_finished_events[layer_id].is_set())

        thread, _, load_finished, _ = self._make_reuse_thread(external_slot_release_waiter=wait_for_release)
        load_task = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[], layer_id=1)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(observed_load_state, [False])
        self.assertTrue(load_finished[1].is_set())

    def test_external_slot_release_failure_prevents_h2d_and_completion(self):
        def fail_release(_layer_id):
            raise RuntimeError("external read failed")

        thread, store, load_finished, _ = self._make_reuse_thread(
            external_slot_release_waiter=fail_release
        )
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion([], []),
        )
        load_task = LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[task], layer_id=1)
        thread.request_queue.put(load_task)

        with self.assertRaisesRegex(RuntimeError, "external read failed"):
            thread._handle_request(load_task)

        store.store.batch_copy.assert_not_called()
        self.assertFalse(load_finished[1].is_set())

    def test_source_save_failure_interrupts_receiver_wait(self):
        save_failure_checker = MagicMock(side_effect=RuntimeError("save thread failed"))
        thread, _, _, save_finished = self._make_reuse_thread(save_failure_checker=save_failure_checker)
        save_finished[0] = MagicMock()
        save_finished[0].wait.return_value = False
        load_task = LayerLoadTask(wait_for_save_layer=0, transfer_tasks=[], layer_id=1)
        thread.request_queue.put(load_task)

        with self.assertRaisesRegex(RuntimeError, "save thread failed"):
            thread._handle_request(load_task)

        save_failure_checker.assert_called_once_with()

    def test_layer_transfer_releases_load_leases_after_copy(self):
        store = MagicMock()
        store.batch_add_lease.return_value = [0, 0]
        store.store.batch_copy.return_value = 0
        load_lease_releaser = MagicMock()
        db = MagicMock()
        db.group_block_len = {0: [16]}
        db.group_kv_caches_base_addr = {0: [1000]}
        db.group_block_stride = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            addr_array=np.asarray([1000], dtype=np.int64),
            size_array=np.asarray([16], dtype=np.int64),
            gvas_array=np.asarray([2000], dtype=np.int64),
        )
        finished_events = [threading.Event()]
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=finished_events,
            layer_save_finished_events=[threading.Event()],
            num_layers=1,
            group_array_builders=[builder],
            load_lease_releaser=load_lease_releaser,
        )
        preparation_callback = MagicMock()
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
            finished_req_ids={"r1"},
        )
        load_task = LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=0,
            preparation=LayerwisePreparation(preparation_callback),
        )

        thread.request_queue.put(load_task)
        thread._handle_request(load_task)

        preparation_callback.assert_called_once_with()
        store.batch_add_lease.assert_not_called()
        store.batch_remove_lease.assert_not_called()
        store.store.batch_copy.assert_called_once()
        load_lease_releaser.assert_called_once_with({"r1"})
        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})

    def test_last_actual_transfer_can_finish_before_final_physical_layer(self):
        store = MagicMock()
        store.store.batch_copy.return_value = 0
        db = MagicMock()
        db.group_block_len = {0: [16, 16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([1000]),
            np.asarray([16]),
            np.asarray([2000]),
        )
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=[threading.Event(), threading.Event()],
            layer_save_finished_events=[threading.Event(), threading.Event()],
            num_layers=2,
            group_array_builders=[builder],
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
            finished_req_ids={"r1"},
        )
        load_task = LayerLoadTask(None, [task], layer_id=0)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(thread.get_and_clear_finished_requests(), {"r1"})

    def test_copy_failure_does_not_complete_request_or_layer(self):
        store = MagicMock()
        store.store.batch_copy.return_value = 1
        db = MagicMock()
        db.group_block_len = {0: [16]}
        builder = MagicMock()
        builder.build_addrs.return_value = LayerTransferArrays(
            np.asarray([1000]),
            np.asarray([16]),
            np.asarray([2000]),
        )
        layer_finished = threading.Event()
        load_lease_releaser = MagicMock()
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=[layer_finished],
            layer_save_finished_events=[threading.Event()],
            num_layers=1,
            group_array_builders=[builder],
            load_lease_releaser=load_lease_releaser,
        )
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[],
            transfer_data=MagicMock(),
            completion=TransferCompletion(["r1"], [True]),
            finished_req_ids={"r1"},
        )
        load_task = LayerLoadTask(None, [task], layer_id=0)
        thread.request_queue.put(load_task)

        with self.assertRaisesRegex(RuntimeError, "load batch_copy failed"):
            thread._handle_request(load_task)

        self.assertEqual(thread.get_and_clear_finished_requests(), set())
        self.assertFalse(layer_finished.is_set())
        load_lease_releaser.assert_not_called()


class TestKVTransferTpMismatchDispatch(unittest.TestCase):
    """TP-mismatch worker dispatch wiring for Sending/Recving threads."""

    def _make_sending(self, worker=None, exists_result=None):
        store = FakeStore(exists_result or [0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreSendingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            group_uses_align_state=[False],
            enable_kv_event=False,
            worker=worker,
        )
        return t, store

    def _make_recving(self, worker=None):
        store = FakeStore([0, 0, 0, 0])
        db = FakeTokenDatabase()
        t = KVCacheStoreRecvingThread(
            m_store=store,
            token_database=db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            ready_event=threading.Event(),
            invalid_block_ids=set(),
            invalid_block_ids_lock=threading.Lock(),
            worker=worker,
        )
        return t, store

    def test_sending_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_sending(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._store_kv_tp_mismatch.assert_called_once_with(req)

    def test_sending_normal_path_when_worker_none(self):
        # worker=None -> tp_mismatch dispatch skipped, normal store path runs.
        t, store = self._make_sending(worker=None, exists_result=[1, 0, 1, 0])
        req = ReqMeta(
            req_id="r1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            current_event=None,
        )
        t.add_stored_request("r1")
        t.request_queue.put(req)
        t._handle_request(req)
        self.assertEqual(len(store.put_calls), 1)  # normal path executed

    def test_recving_dispatches_to_worker_when_tp_mismatch(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_called_once()
        args = worker._load_kv_tp_mismatch.call_args.args
        # (block_hashes, block_ids, token_len, mask_num)
        self.assertEqual(args[2], 16)  # token_len
        self.assertEqual(args[3], 0)  # mask_num

    def test_recving_tp_mismatch_missing_load_spec_finishes(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        t.request_queue.put(req)
        t._handle_request(req)
        worker._load_kv_tp_mismatch.assert_not_called()
        self.assertEqual(t.get_and_clear_finished_requests(), {"r1"})
        self.assertEqual(t.request_queue.unfinished_tasks, 0)

    def test_recving_tp_mismatch_task_done_on_exception(self):
        worker = MagicMock()
        worker.tp_mismatch = True
        worker._load_kv_tp_mismatch.side_effect = RuntimeError("load failed")
        t, _ = self._make_recving(worker=worker)
        req = ReqMeta(
            req_id="r1", token_len_chunk=16, block_ids_by_group=[[0]], block_hashes=[b"h0"], current_event=None
        )
        req.load_spec = MagicMock()
        req.load_spec.token_len = 16
        req.load_spec.vllm_cached_tokens = 0
        t.request_queue.put(req)
        with self.assertRaises(RuntimeError):
            t._handle_request(req)
        self.assertEqual(t.request_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
