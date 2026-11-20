# SPDX-License-Identifier: Apache-2.0
"""DeepEP prepare/finalize integration accepts upstream keyword arguments."""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace as NS

import pytest

from tests.support.source import (
    ROOT,
    assert_accepts_upstream_keywords,
    definition,
    function,
)


@pytest.mark.upstream_source
def test_prepare_finalize_consumer_inventory_matches_upstream(upstream_source_root):
    tree = ast.parse((ROOT / "vllm_sail/patch/enhancement/deepep.py").read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CONSUMERS"
            for target in node.targets
        )
    )
    discovered = {name: set() for name in inventory}
    source = upstream_source_root / "vllm"
    for path in (source / "model_executor").rglob("*.py"):
        module = ".".join(("vllm", *path.relative_to(source).with_suffix("").parts))
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.model_executor.layers.fused_moe.all2all_utils"
            ):
                for alias in node.names:
                    if alias.name in discovered:
                        assert alias.asname is None, "renamed alias needs rebinding"
                        discovered[alias.name].add(module)
            pending.extend(ast.iter_child_nodes(node))
    assert discovered == {name: set(consumers) for name, consumers in inventory.items()}


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "deepep",
            "_maybe_make_prepare_finalize_body",
            "model_executor/layers/fused_moe/all2all_utils",
            "maybe_make_prepare_finalize",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )


@pytest.mark.upstream_source
def test_complete_factory_body_diff_is_limited_to_ppu_dispatch(upstream_source_root):
    """比较完整复制体，只还原已记录的 PPU LL dispatch 差异。"""
    path = ROOT / "vllm_sail/patch/enhancement/deepep.py"
    node = definition(path, "_maybe_make_prepare_finalize_body")
    source = "\n".join(path.read_text().splitlines()[node.lineno - 1 : node.end_lineno])
    regions = list(
        re.finditer(
            r"(?m)^ *# PPU MODIFICATION: begin\n.*?^ *# PPU MODIFICATION: end\n?",
            source,
            flags=re.DOTALL,
        )
    )
    assert len(regions) == 2
    original_dispatch = (
        "        use_fp8_dispatch = (\n"
        "            quant_config.quant_dtype == current_platform.fp8_dtype()\n"
        "            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE\n"
        "        )\n"
    )
    for region, replacement in reversed(
        list(zip(regions, [original_dispatch, ""], strict=False))
    ):
        source = source[: region.start()] + replacement + source[region.end() :]
    local = ast.parse(source).body[0]
    upstream = definition(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/all2all_utils.py",
        "maybe_make_prepare_finalize",
    )
    local.name = upstream.name
    assert ast.dump(local) == ast.dump(upstream)


@pytest.mark.upstream_source
@pytest.mark.parametrize("upstream", [False, True])
@pytest.mark.parametrize("experts", [16, 17])
def test_mori_epv2_preserves_expert_partition_and_rank_offset(
    upstream_source_root, upstream, experts
):
    """复制体须采用新 EPv2 路由，并在分区不整除时拒绝创建 handle。"""
    calls = []
    handle = object()
    manager = NS(world_size=4, rank=2)

    def get_handle(args):
        calls.append(args)
        return handle

    manager.get_handle = get_handle
    factory = function(
        upstream_source_root / "vllm/model_executor/layers/fused_moe/all2all_utils.py"
        if upstream
        else "vllm_sail/patch/enhancement/deepep.py",
        "maybe_make_prepare_finalize"
        if upstream
        else "_maybe_make_prepare_finalize_body",
        {"MoriEPv2PrepareAndFinalize": lambda *args, **kwargs: NS(args=args, **kwargs)},
    )
    moe = NS(
        moe_parallel_config=NS(use_passthrough_all2all=False, use_all2all_kernels=True),
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=False,
        use_deepep_v2_kernels=False,
        use_moonep_kernels=False,
        use_mori_epv2_kernels=True,
        num_experts=experts,
        hidden_dim=3584,
        max_num_tokens=32,
        experts_per_token=8,
    )
    if experts == 17:
        with pytest.raises(ValueError, match="split evenly"):
            factory(moe, None, all2all_manager=manager)
        assert calls == []
        return
    result = factory(moe, None, all2all_manager=manager)
    assert calls == [
        dict(
            hidden_dim=3584,
            max_num_inp_token_per_rank=32,
            num_experts_per_rank=4,
            num_experts_per_token=8,
        )
    ]
    assert result.args == (handle,)
    assert result.max_tokens_per_rank == 32
    assert result.num_dispatchers == 4
    assert result.num_local_experts == 4
    assert result.rank_expert_offset == 8
