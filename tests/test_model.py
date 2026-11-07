import torch

from synthdet.features import CROP, N_MELS, log_mel
from synthdet.model import Detector, n_params


def test_output_shape_and_param_budget():
    m = Detector()
    assert n_params(m) < 200_000
    out = m(log_mel(torch.randn(5, CROP) * 0.05))
    assert out.shape == (5,)


def test_handles_other_lengths():
    m = Detector().eval()
    assert m(torch.randn(2, 1, N_MELS, 97)).shape == (2,)


def test_gradients_reach_first_layer():
    m = Detector()
    m(log_mel(torch.randn(4, CROP) * 0.05)).sum().backward()
    assert m.blocks[0][0].weight.grad.abs().sum() > 0
