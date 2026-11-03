# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data and ownership regressions for the delayed Copy-SFA planner."""

import numpy as np
import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h import (
    DelayedD2HPlanner,
    corrected_computed_frontier,
)
from vllm_ascend.spec_decode.utils import update_num_computed_tokens_for_batch_change


def ints(values):
    return np.asarray(values, dtype=np.int64)


def make_planner(prefixes=(0,), hot=4096):
    planner = DelayedD2HPlanner(capacity=3, block_size=128, hot_tokens=hot, host_num_blocks=128)
    slots = np.arange(len(prefixes), dtype=np.int64)
    planner.bind(slots, slots + 10, ints(prefixes))
    return planner


def plan(planner, computed=(128,), *, slots=(0,), generations=(10,), writes=(7,), table=None):
    if table is None:
        table = np.arange(len(slots) * 64, dtype=np.int64).reshape(len(slots), 64)
    return planner.plan(
        computed_tokens=ints(computed),
        pool_slots=ints(slots),
        generations=ints(generations),
        block_table=table,
        max_write_tokens=ints(writes),
    )


def test_frontier_matches_v1_device_correction_across_rejections_and_reordering():
    previous = ints([124, 1010, 8190])
    previous_drafts = ints([3, 3, 0])
    for iteration in range(20):
        positions = ints([1, 0, -1])
        counts = ints([1 + iteration % 4, 1, 0])
        optimistic = ints([previous[1] + 4, previous[0] + 4, 100 + iteration])
        expected = torch.from_numpy(previous.copy())
        update_num_computed_tokens_for_batch_change(
            expected,
            torch.ones(3, dtype=torch.int64),
            torch.from_numpy(positions),
            torch.from_numpy(counts),
            torch.from_numpy(previous_drafts),
            torch.from_numpy(optimistic),
        )
        corrected = corrected_computed_frontier(optimistic, previous, positions, previous_drafts, counts)
        np.testing.assert_array_equal(corrected, expected.numpy())
        np.testing.assert_array_equal(optimistic, [previous[1] + 4, previous[0] + 4, 100 + iteration])
        previous = corrected


def test_frontier_new_and_non_draft_rows_do_not_consume_uninitialized_counts():
    corrected = corrected_computed_frontier(ints([300, 400]), ints([100]), ints([0, -1]), ints([0]), ints([-999]))
    np.testing.assert_array_equal(corrected, [300, 400])
    np.testing.assert_array_equal(
        corrected_computed_frontier(ints([50]), ints([]), ints([-1]), ints([]), ints([])), [50]
    )


@pytest.mark.parametrize("positions", [[2], [-2], [0, 0]])
def test_frontier_rejects_stale_or_aliased_batch_mapping(positions):
    with pytest.raises(ValueError, match="previous batch row"):
        corrected_computed_frontier(ints([128] * len(positions)), ints([124]), ints(positions), ints([3]), ints([1]))


@pytest.mark.parametrize("count", [0, 5])
def test_frontier_rejects_invalid_sampled_counts(count):
    with pytest.raises(ValueError, match="sampled count"):
        corrected_computed_frontier(ints([128]), ints([124]), ints([0]), ints([3]), ints([count]))


def test_rejection_at_block_boundary_does_not_publish_optimistic_block():
    planner = make_planner()
    corrected = corrected_computed_frontier(ints([128]), ints([124]), ints([0]), ints([3]), ints([1]))
    assert plan(planner, corrected) is None
    np.testing.assert_array_equal(planner.prefixes(ints([0]), ints([10])), [0])


def test_short_prompt_first_block_uses_dense_source():
    planner = make_planner()
    batch = plan(planner)
    assert batch is not None
    assert batch.source_token_slots.tolist() == [0]
    assert batch.target_prefixes.tolist() == [128]
    assert planner.prefixes(ints([0]), ints([10])).tolist() == [0]
    planner.complete(batch.epoch)
    assert planner.prefixes(ints([0]), ints([10])).tolist() == [128]


@pytest.mark.parametrize("prompt,hot", [(100, 4096), (127, 256), (255, 256), (256, 256), (350, 256)])
def test_copy_contents_survive_concurrent_writes_dense_transition_and_ring_wrap(prompt, hot):
    block = 128
    initial_prefix = prompt // block * block
    planner = make_planner((initial_prefix,), hot=hot)
    resident = np.full(planner.row_tokens, -1, dtype=np.int64)
    host = np.full((128, block), -1, dtype=np.int64)
    # A non-identity physical table exposes source/destination confusion.
    table = np.arange(64, dtype=np.int64)[::-1].reshape(1, -1)
    for token in range(prompt):
        host[table[0, token // block], token % block] = token + 10000
        if initial_prefix < hot:
            resident[token] = token + 10000
        elif token >= initial_prefix:
            resident[hot + token % (2 * block)] = token + 10000
    computed = prompt
    while computed < hot + 4 * block:
        prefix = planner.prefixes(ints([0]), ints([10]))[0]
        batch = plan(planner, (computed,), table=table)
        # Run all upcoming writes BEFORE copying the frozen source, the
        # adverse ordering an overlapped NPU execution must also tolerate.
        for token in range(computed, computed + 7):
            offset = token if prefix < hot else hot + token % (2 * block)
            resident[offset] = token + 10000
        if batch is not None:
            source = batch.source_token_slots[0]
            destination = batch.destination_blocks[0]
            host[destination] = resident[source : source + block]
            np.testing.assert_array_equal(host[destination], np.arange(prefix, prefix + block) + 10000)
            assert planner.prefixes(ints([0]), ints([10]))[0] == prefix
            planner.complete(batch.epoch)
        computed += 7


def test_frozen_batch_keeps_ownership_when_request_leaves_current_batch():
    planner = make_planner((0, 0))
    table = ints([[9, 10], [20, 21]])
    batch = plan(planner, (128, 1), slots=(0, 1), generations=(10, 11), writes=(4, 4), table=table)
    assert batch is not None
    table[0, 0] = 50
    assert batch.destination_blocks.tolist() == [9]
    with pytest.raises(ValueError, match="read-only"):
        batch.destination_blocks[0] = 50
    with pytest.raises(RuntimeError, match="in-flight"):
        planner.release(ints([0]), ints([10]))
    with pytest.raises(RuntimeError, match="in-flight"):
        planner.bind(ints([0]), ints([20]), ints([0]))
    # A different idle request can be released without modifying DMA inputs.
    planner.release(ints([1]), ints([11]))
    planner.bind(ints([1]), ints([21]), ints([0]))
    planner.complete(batch.epoch)
    planner.release(ints([0]), ints([10]))
    planner.bind(ints([0]), ints([30]), ints([256]))
    with pytest.raises(RuntimeError, match="generation"):
        planner.release(ints([0]), ints([10]))
    assert planner.prefixes(ints([0]), ints([30])).tolist() == [256]


@pytest.mark.parametrize("prompt,hot", [(100, 256), (255, 256), (4100, 4096)])
def test_rejected_writes_never_reach_host_with_nonzero_pool_slot(prompt, hot):
    slot, generation, block = 2, 30, 128
    planner = DelayedD2HPlanner(capacity=3, block_size=block, hot_tokens=hot, host_num_blocks=128)
    initial_prefix = prompt // block * block
    planner.bind(ints([slot]), ints([generation]), ints([initial_prefix]))
    resident = np.full(planner.capacity * planner.row_tokens, -1, dtype=np.int64)
    host = np.full((128, block), -1, dtype=np.int64)
    table = np.arange(128, dtype=np.int64)[::-1].reshape(1, -1)
    base = slot * planner.row_tokens
    for token in range(prompt):
        host[table[0, token // block], token % block] = token + 10000
        if initial_prefix < hot:
            resident[base + token] = token + 10000
        elif token >= initial_prefix:
            resident[base + hot + token % (2 * block)] = token + 10000
    computed = prompt
    iteration = 0
    while computed < hot + 4 * block:
        prefix = int(planner.prefixes(ints([slot]), ints([generation]))[0])
        batch = plan(planner, (computed,), slots=(slot,), generations=(generation,), table=table)
        accepted = 1 + iteration % 4
        for index, token in enumerate(range(computed, computed + 7)):
            offset = token if prefix < hot else hot + token % (2 * block)
            # Distinct poison on rejected positions exposes any premature
            # publication or future writes overlapping the frozen block.
            resident[base + offset] = token + 10000 if index < accepted else -10000 - iteration
        if batch is not None:
            source = batch.source_token_slots[0]
            destination = batch.destination_blocks[0]
            host[destination] = resident[source : source + block]
            np.testing.assert_array_equal(host[destination], np.arange(prefix, prefix + block) + 10000)
            planner.complete(batch.epoch)
        computed += accepted
        iteration += 1


def test_reordered_ready_requests_mix_dense_and_ring_sources():
    planner = make_planner((128, 4096))
    table = np.zeros((2, 64), dtype=np.int64)
    table[0, 32], table[1, 1] = 57, 19
    batch = plan(planner, (4225, 258), slots=(1, 0), generations=(11, 10), writes=(7, 7), table=table)
    np.testing.assert_array_equal(batch.pool_slots, [1, 0])
    np.testing.assert_array_equal(batch.source_token_slots, [planner.row_tokens + 4096, 128])
    np.testing.assert_array_equal(batch.destination_blocks, [57, 19])
    np.testing.assert_array_equal(batch.target_prefixes, [4224, 256])
    np.testing.assert_array_equal(planner.prefixes(ints([0, 1]), ints([10, 11])), [128, 4096])
    planner.complete(batch.epoch)
    np.testing.assert_array_equal(planner.prefixes(ints([0, 1]), ints([10, 11])), [256, 4224])


def test_bad_completion_keeps_resources_and_prefix_unpublished():
    planner = make_planner()
    batch = plan(planner)
    with pytest.raises(RuntimeError, match="epoch"):
        planner.complete(batch.epoch + 1)
    assert planner.inflight is batch
    assert planner.prefixes(ints([0]), ints([10])).tolist() == [0]
    with pytest.raises(RuntimeError, match="retire"):
        plan(planner)
    planner.complete(batch.epoch)
    with pytest.raises(RuntimeError, match="epoch"):
        planner.complete(batch.epoch)


@pytest.mark.parametrize("computed,writes,error", [(256, 1, "two full blocks"), (250, 7, "overlap")])
def test_planner_rejects_tail_overwrite_before_submission(computed, writes, error):
    planner = make_planner((4096,))
    with pytest.raises(RuntimeError, match=error):
        plan(planner, (4096 + computed,), writes=(writes,))
    assert planner.inflight is None


def test_planner_rejects_prefix_rollback_without_changing_state():
    planner = make_planner((128,))
    with pytest.raises(RuntimeError, match="rolled back"):
        plan(planner, (127,))
    assert planner.prefixes(ints([0]), ints([10])).tolist() == [128]


@pytest.mark.parametrize("destination", [-1, 128])
def test_planner_rejects_invalid_host_destination(destination):
    planner = make_planner()
    with pytest.raises(ValueError, match="Host allocation"):
        plan(planner, table=ints([[destination]]))
    assert planner.inflight is None


def test_requests_cannot_issue_concurrent_writes_to_one_destination():
    planner = make_planner((0, 0))
    with pytest.raises(RuntimeError, match="same Host block"):
        plan(planner, (128, 128), slots=(0, 1), generations=(10, 11), writes=(1, 1), table=ints([[7], [7]]))


def test_planner_rejects_unbound_or_duplicate_slots():
    planner = make_planner()
    with pytest.raises(RuntimeError, match="generation"):
        plan(planner, slots=(1,), generations=(11,))
    with pytest.raises(ValueError, match="unique"):
        plan(planner, (128, 128), slots=(0, 0), generations=(10, 10), writes=(1, 1))


def test_empty_scheduled_batch_does_not_create_work():
    planner = make_planner()
    assert plan(planner, (), slots=(), generations=(), writes=(), table=np.empty((0, 0), dtype=np.int32)) is None


@pytest.mark.parametrize("field,value", [("capacity", 0), ("block_size", True), ("hot_tokens", 129)])
def test_invalid_planner_geometry(field, value):
    kwargs = dict(capacity=3, block_size=128, hot_tokens=4096, host_num_blocks=128)
    kwargs[field] = value
    with pytest.raises(ValueError):
        DelayedD2HPlanner(**kwargs)


def test_planner_does_not_silently_truncate_floating_point_block_ids():
    planner = make_planner()
    with pytest.raises(ValueError, match="integer array"):
        plan(planner, table=np.array([[1.5]]))
