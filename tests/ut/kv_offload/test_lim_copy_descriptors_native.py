# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the production AscendC copy kernel with CPU byte-copy primitives.

This verifies address math/content and UB padding; it does not validate CANN
compilation, NPU stream ordering, Host VMM accessibility or performance.
"""

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re
import torch
from torch.fx.experimental.proxy_tensor import make_fx


def test_actual_kernel_copy_contents_and_bounds_with_sanitizers(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed for the CPU kernel address/content test")
    directory = Path(__file__).parent / "cpp"
    root = Path(__file__).resolve().parents[3]
    binary = tmp_path / "lim_copy_descriptors"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-I",
            str(directory),
            "-I",
            str(root),
            str(directory / "test_lim_copy_descriptors.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}
    )


def test_production_schema_keeps_copy_side_effects_when_unused_outputs_are_eliminated():
    root = Path(__file__).resolve().parents[3]
    binding = (root / "csrc/torch_binding.cpp").read_text()
    declaration = re.search(r'ops.def\(\s*("npu_lim_copy_descriptors\(.*?)(?=\s*\);)', binding, re.S)
    assert declaration is not None
    schema = "".join(ast.literal_eval(part) for part in re.findall(r'"[^"\n]*"', declaration.group(1)))
    library = torch.library.Library("lim_copy_abi_test", "DEF")
    try:
        library.define(schema)

        def copy_effect(*args):
            args[6].add_(1)
            args[7].add_(2)

        library.impl("npu_lim_copy_descriptors", copy_effect, "CPU")
        operation = torch.ops.lim_copy_abi_test.npu_lim_copy_descriptors.default
        assert operation._schema.is_mutable

        def forward(kv, rope):
            metadata = torch.zeros(1, dtype=torch.int32)
            operation(*([metadata] * 6), kv, rope, metadata, metadata)

        kv, rope = torch.zeros(4), torch.zeros(4)
        graph = make_fx(forward)(kv, rope)
        graph.graph.eliminate_dead_code()
        graph.recompile()
        assert any(node.target == operation for node in graph.graph.nodes)
        kv.zero_()
        rope.zero_()
        graph(kv, rope)
        torch.testing.assert_close(kv, torch.ones_like(kv))
        torch.testing.assert_close(rope, torch.full_like(rope, 2))
    finally:
        library._destroy()
