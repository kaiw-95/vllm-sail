# SPDX-License-Identifier: Apache-2.0
"""Exercise draft mapping through upstream's initialization path on CPU.

With VLLM_SOURCE_ROOT set, extract the real mapper, model-loader entry points
and matching function without importing torch or vLLM. Only the model body and
platform are stand-ins; no weights or distributed process group are needed.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

import vllm_sail.models as models_pkg
from tests.support.source import ROOT, definition, function


@pytest.fixture
def mapping_contract(upstream_source_root, monkeypatch):
    def modules(name, **attributes):
        parts = name.split(".")
        for depth in range(1, len(parts) + 1):
            qualified = ".".join(parts[:depth])
            if qualified not in sys.modules:
                module = ModuleType(qualified)
                module.__path__ = []
                monkeypatch.setitem(sys.modules, qualified, module)
        for name, value in attributes.items():
            monkeypatch.setattr(sys.modules[qualified], name, value, raising=False)

    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "replace": replace,
    }
    mapper_node = definition(
        upstream_source_root / "vllm/model_executor/models/utils.py", "WeightsMapper"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            mapper_node,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), "<upstream mapper>", "exec"),
        namespace,
    )
    mapper_cls = namespace["WeightsMapper"]
    is_layer_skipped = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "is_layer_skipped",
        {"MappingProxyType": MappingProxyType},
    )
    platform = SimpleNamespace(is_ppu=lambda: True)
    modules("vllm.platforms", current_platform=platform)
    apply_mapper = function(
        "vllm_sail/patch/enhancement/mxfp4.py",
        "install._apply_vllm_mapper",
        {"_upstream_apply_mapper": lambda *args: None},
    )

    class QuantConfig:
        apply_vllm_mapper = apply_mapper

        def __init__(self, layers):
            self.fp8_channelwise_layers = layers
            self.ignored_layers = [
                layers[0].split("layers.")[0] + "layers.45.mlp.experts"
            ]
            self.packed_modules_mapping = {}

    class UpstreamMTP:
        def __init__(self, *, vllm_config, prefix=""):
            config = vllm_config.quant_config
            # Observe routing while layers are built, before load_weights can
            # rewrite parameter names to include the mtp_block attribute.
            self.channelwise_at_init = list(config.fp8_channelwise_layers)
            self.is_channelwise = is_layer_skipped(
                prefix="model.layers.45.mlp.shared_experts.down_proj",
                ignored_layers=config.fp8_channelwise_layers,
                fused_mapping=config.packed_modules_mapping,
                match_mode="substring",
            )

    modules("vllm.model_executor.models.utils", WeightsMapper=mapper_cls)
    modules("vllm.models.glm5next.common.mtp", Glm5NextMTP=UpstreamMTP)
    target = models_pkg._PPU_MODELS["Glm5NextMTPModel"]
    module_name, _, class_name = target.partition(":")
    spec = importlib.util.spec_from_file_location(
        "_glm5_mtp_mapping_test",
        ROOT.joinpath(*module_name.split(".")).with_suffix(".py"),
    )
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    loader_path = upstream_source_root / "vllm/model_executor/model_loader/utils.py"
    loader_globals = {
        "SupportsQuant": type("SupportsQuant", (), {}),
        "inspect": inspect,
        "set_current_vllm_config": lambda *args, **kwargs: nullcontext(),
        "record_metadata_for_reloading": lambda model: None,
    }
    function(loader_path, "configure_quant_config", loader_globals)
    initialize = function(loader_path, "initialize_model", loader_globals)
    return SimpleNamespace(
        initialize=initialize,
        configure=loader_globals["configure_quant_config"],
        model_cls=getattr(model_module, class_name),
        upstream_cls=UpstreamMTP,
        config_cls=QuantConfig,
        mapper_cls=mapper_cls,
        is_layer_skipped=is_layer_skipped,
        platform=platform,
    )


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "root", ["model.language_model.", "language_model.model.", "model."]
)
def test_draft_maps_channelwise_layers_before_construction(mapping_contract, root):
    contract = mapping_contract
    config = contract.config_cls([root + "layers.45.mlp.shared_experts.down_proj"])
    vllm_config = SimpleNamespace(model_config=object(), quant_config=config)

    model = contract.initialize(vllm_config, model_class=contract.model_cls)

    assert model.is_channelwise
    assert model.channelwise_at_init == ["model.layers.45.mlp.shared_experts.down_proj"]
    assert config.ignored_layers == ["model.layers.45.mlp.experts"]
    for prefix in (
        "model.layers.44.mlp.shared_experts.down_proj",
        "model.layers.45.mlp.shared_experts.gate_up_proj",
        "visual.layers.45.mlp.shared_experts.down_proj",
    ):
        assert not contract.is_layer_skipped(
            prefix, config.fp8_channelwise_layers, match_mode="substring"
        )
    contract.configure(config, contract.model_cls)
    assert config.fp8_channelwise_layers == model.channelwise_at_init
    assert not hasattr(contract.upstream_cls, "hf_to_vllm_mapper")


@pytest.mark.upstream_source
def test_draft_remaps_the_target_config_without_mutating_its_list(mapping_contract):
    contract = mapping_contract
    config = contract.config_cls(
        ["model.language_model.layers.45.mlp.shared_experts.down_proj"]
    )
    config.apply_vllm_mapper(
        contract.mapper_cls(
            orig_to_new_prefix={"model.language_model.": "language_model.model."}
        )
    )
    target_layers = config.fp8_channelwise_layers
    contract.initialize(
        SimpleNamespace(model_config=object(), quant_config=config),
        model_class=contract.model_cls,
    )
    assert target_layers == [
        "language_model.model.layers.45.mlp.shared_experts.down_proj"
    ]
    assert config.fp8_channelwise_layers == [
        "model.layers.45.mlp.shared_experts.down_proj"
    ]


@pytest.mark.upstream_source
def test_non_ppu_keeps_existing_channelwise_mapping_behavior(mapping_contract):
    contract = mapping_contract
    contract.platform.is_ppu = lambda: False
    layers = ["model.language_model.layers.45.mlp.shared_experts.down_proj"]
    config = contract.config_cls(layers)
    contract.configure(config, contract.model_cls)
    assert config.fp8_channelwise_layers == layers
