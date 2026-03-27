"""Tests for SSM, ProtoSSMv2, and ResidualSSM model modules."""

import torch
import pytest


class TestSelectiveSSM:
    def test_forward_shape(self):
        from src.models.ssm import SelectiveSSM

        model = SelectiveSSM(d_model=128, d_state=16, d_conv=4)
        x = torch.randn(2, 12, 128)
        out = model(x)
        assert out.shape == (2, 12, 128)

    def test_forward_gradients(self):
        from src.models.ssm import SelectiveSSM

        model = SelectiveSSM(d_model=128, d_state=16)
        x = torch.randn(2, 12, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 12, 128)


class TestProtoSSMv2:
    def test_forward_shape(self):
        from src.models.proto_ssm import ProtoSSMv2

        model = ProtoSSMv2(
            d_input=1536, d_model=64, d_state=8,
            n_ssm_layers=1, n_classes=234, n_windows=12,
            dropout=0.0, use_cross_attn=False,
        )
        emb = torch.randn(2, 12, 1536)
        perch_logits = torch.randn(2, 12, 234)
        species, family, h_temporal = model(emb, perch_logits)
        assert species.shape == (2, 12, 234)

    def test_forward_without_perch(self):
        from src.models.proto_ssm import ProtoSSMv2

        model = ProtoSSMv2(
            d_input=1536, d_model=64, d_state=8,
            n_ssm_layers=1, n_classes=234, n_windows=12,
            dropout=0.0, use_cross_attn=False,
        )
        emb = torch.randn(2, 12, 1536)
        species, family, h_temporal = model(emb)
        assert species.shape == (2, 12, 234)
        assert family is None

    def test_count_parameters_positive(self):
        from src.models.proto_ssm import ProtoSSMv2

        model = ProtoSSMv2(
            d_input=1536, d_model=64, d_state=8,
            n_ssm_layers=1, n_classes=234, use_cross_attn=False,
        )
        n = model.count_parameters()
        assert isinstance(n, int)
        assert n > 0


class TestResidualSSM:
    def test_forward_shape(self):
        from src.models.residual_ssm import ResidualSSM

        model = ResidualSSM(
            d_input=1536, d_scores=234, d_model=32, d_state=8,
            n_classes=234, n_windows=12, dropout=0.0,
        )
        emb = torch.randn(2, 12, 1536)
        first_pass = torch.randn(2, 12, 234)
        correction = model(emb, first_pass)
        assert correction.shape == (2, 12, 234)

    def test_output_near_zero_at_init(self):
        from src.models.residual_ssm import ResidualSSM

        model = ResidualSSM(
            d_input=1536, d_scores=234, d_model=32, d_state=8,
            n_classes=234, n_windows=12, dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            emb = torch.randn(2, 12, 1536)
            first_pass = torch.randn(2, 12, 234)
            correction = model(emb, first_pass)
        # Output head is zero-initialized, so corrections should be small
        assert correction.abs().mean().item() < 1.0
