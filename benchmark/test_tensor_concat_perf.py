from typing import Generator

import pytest
import torch

from .attri_util import DEFAULT_SHAPES, FLOAT_DTYPES, INT_DTYPES, BenchLevel
from .performance_utils import (
    Benchmark,
    Config,
    GenericBenchmark,
    generate_tensor_input,
)

CONCATENATION_RECOMMENDED_SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
    (512, 1024),
    (512, 2048),
]


class ConcatBenchmark(Benchmark):
    """
    benchmark for concat and stack
    """

    DEFAULT_SHAPES = CONCATENATION_RECOMMENDED_SHAPES

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def set_shapes(self):
        self.shapes = self.DEFAULT_SHAPES[:]
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            more_shapes_2d = [(1024, 2**i) for i in range(0, 11, 4)]
            more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]
            self.shapes = list(
                dict.fromkeys(self.shapes + more_shapes_2d + more_shapes_3d)
            )


def stack_cat_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


def hstack_vstack_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    inp3 = generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3],


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtype",
    [
        pytest.param(
            "cat",
            torch.cat,
            stack_cat_input_fn,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.cat(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES,
                shape_desc="((B), M, N) * 3",
            ),
        ),
        pytest.param(
            "stack",
            torch.stack,
            stack_cat_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.stack(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES,
                shape_desc="((B), M, N) * 3",
            ),
        ),
        pytest.param(
            "hstack",
            torch.hstack,
            hstack_vstack_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.hstack(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES,
                shape_desc="((B), M, N) * 3",
            ),
        ),
        pytest.param(
            "vstack",
            torch.vstack,
            hstack_vstack_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.vstack(
                recommended_shapes=CONCATENATION_RECOMMENDED_SHAPES,
                shape_desc="((B), M, N) * 3",
            ),
        ),
    ],
)
def test_concat_benchmark(op_name, torch_op, input_fn, dtype):
    bench = ConcatBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtype
    )
    bench.run()


class TensorRepeatBenchmark(GenericBenchmark):
    """
    TensorRepeatBenchmark designed to evaluate tensor repeat operations along specified dimensions.
    This includes operations like tile, repeat, and repeat_interval.
    Due to potential memory limitations, benchmark sizes need to be carefully controlled.

    Notably, when the input size is set to (1024, 1024, 1024) and the repeat dimensions
    are set to [1, 1, 2], the system encountered an "illegal memory access" error.
    To avoid such issues, we constrain the benchmark input sizes for these operations
    to prevent excessive memory usage.
    """

    DEFAULT_SHAPES = DEFAULT_SHAPES[:]

    def set_shapes(self):
        if Config.bench_level == BenchLevel.CORE:
            self.shapes = DEFAULT_SHAPES[:]
        else:
            more_shapes = [
                (1024 * 1024,),
                (512, 512, 512),
                (64, 64, 64, 64),
            ]
            self.shapes = list(dict.fromkeys(self.DEFAULT_SHAPES + more_shapes))


def tile_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    dim = [1] * len(shape)
    dim[0] = 2
    yield inp, {"dims": dim}


def repeat_input_fn(shape, cur_dtype, device):
    inp1 = generate_tensor_input(shape, cur_dtype, device)
    inp2 = [1] * len(shape)
    inp2[0] = 2
    yield inp1, inp2,


def repeat_interleave_self_input_fn(shape, dtype, device):
    if dtype in FLOAT_DTYPES:
        # torch.repeat_interleave(input, repeats, dim=None, *, output_size=None) → Tensor
        inp = torch.randn(shape, dtype=dtype, device=device)
        repeats = 3
        yield inp, repeats
    elif dtype == torch.int32:
        # torch.repeat_interleave(repeats, *) → Tensor
        repeats = torch.randint(
            low=0,
            high=0x7F,
            size=[
                shape[-1],
            ],
            dtype=dtype,
            device=device,
        )
        yield repeats,


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "tile", torch.tile, tile_input_fn, FLOAT_DTYPES, marks=pytest.mark.tile
        ),
        pytest.param(
            "repeat",
            torch.Tensor.repeat,
            repeat_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.repeat,
        ),
        pytest.param(
            "repeat_interleave",
            torch.Tensor.repeat_interleave,
            repeat_interleave_self_input_fn,
            FLOAT_DTYPES + [torch.int32],
            marks=pytest.mark.repeat_interleave,
        ),
    ],
)
def test_tensor_repeat_benchmark(op_name, torch_op, input_fn, dtypes):
    bench = TensorRepeatBenchmark(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()
