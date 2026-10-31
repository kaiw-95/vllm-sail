# SPDX-License-Identifier: Apache-2.0
"""Vector gather admission tests with strict tensor metadata doubles."""

from types import SimpleNamespace

import pytest

from tests.ut import test_ppu_kernel_capabilities as harness

modules = harness.modules


class Tensor:
    device, is_cuda = "cuda:0", True

    def __init__(self, shape, dtype, strides, address=32):
        self.shape, self.dtype, self.strides = shape, dtype, strides
        self.ndim, self.address = len(shape), address

    def stride(self, dim):
        return self.strides[dim]

    def is_contiguous(self):
        return self.ndim == 1 and self.strides == (1,)

    def data_ptr(self):
        return self.address


@pytest.fixture
def gather(modules):
    modules("torch", bfloat16="bf16", uint8="u8", int32="i32")
    modules(
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(warning_once=lambda *a: None),
    )
    modules("vllm_sail.envs", VLLM_SAIL_DEEPSEEK_VECTOR_DEQUANT=True)
    modules("vllm_sail._custom_ops", has_vector_dequant=lambda: True)
    module = harness.load_patch("vllm_sail/models/deepseek_v4/ops/dequant.py")
    args = [
        Tensor((2, 100, 512), "bf16", (102400, 1024, 1)),
        Tensor((4, 64, 584), "u8", (37376, 584, 1)),
        Tensor((2,), "i32", (1,)),
        None,
        Tensor((2, 4), "i32", (8, 1)),
        64,
        3,
    ]
    return module, args


def test_vector_admits_strided_request_tables_and_output(gather):
    module, args = gather
    assert module.can_vector_gather(*args)


@pytest.mark.parametrize(
    "case", ["unaligned", "page_stride", "record", "offset", "fnuz", "device"]
)
def test_vector_rejects_unsupported_layout_before_launch(gather, case):
    module, args = gather
    if case == "unaligned":
        args[0].address = 34
    elif case == "page_stride":
        args[1].strides = (37377, 584, 1)
    elif case == "record":
        args[1].shape = (4, 64, 528)
    elif case == "offset":
        args[-1] = 101
    elif case == "device":
        args[2].device = "cuda:1"
    assert not module.can_vector_gather(*args, use_fnuz=case == "fnuz")
