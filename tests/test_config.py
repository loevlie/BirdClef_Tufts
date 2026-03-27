"""Tests for config schema and YAML loader."""

import os
import pytest


class TestConfigSchema:
    def test_default_creation(self):
        from src.config.schema import Config

        cfg = Config()
        assert cfg.mode == "submit"
        assert cfg.arch.d_model == 128
        assert cfg.train.n_epochs == 35
        assert cfg.fusion.lambda_event == pytest.approx(0.4)

    def test_to_dict_returns_dict(self):
        from src.config.schema import Config

        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        # Section keys should use the notebook's CFG key names
        assert "proto_ssm" in d
        assert "proto_ssm_train" in d
        assert "best_fusion" in d
        assert "mode" in d

    def test_to_dict_nested_values(self):
        from src.config.schema import Config

        cfg = Config()
        d = cfg.to_dict()
        assert d["proto_ssm"]["d_model"] == 128
        assert d["proto_ssm_train"]["lr"] == pytest.approx(5.5e-4)


class TestLoadConfig:
    def test_load_base_yaml(self):
        from src.config.loader import load_config

        base_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "base.yaml"
        )
        if not os.path.exists(base_path):
            pytest.skip("configs/base.yaml not found")

        cfg = load_config(base_path)
        assert cfg.mode == "submit"
        assert cfg.arch.d_model == 128
        assert cfg.train.n_epochs == 35

    def test_load_config_returns_config_type(self):
        from src.config.loader import load_config
        from src.config.schema import Config

        base_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "base.yaml"
        )
        if not os.path.exists(base_path):
            pytest.skip("configs/base.yaml not found")

        cfg = load_config(base_path)
        assert isinstance(cfg, Config)
