"""Tests for configuration models and config reader."""

import os
from unittest.mock import patch

import pytest

from app.models.config import AppConfig, ModelEndpointConfig, RateLimitConfig
from app.config import get_config, get_model_endpoints


class TestRateLimitConfig:
    def test_defaults(self):
        cfg = RateLimitConfig()
        assert cfg.requests_per_minute == 60
        assert cfg.burst_size == 10

    def test_custom_values(self):
        cfg = RateLimitConfig(requests_per_minute=120, burst_size=20)
        assert cfg.requests_per_minute == 120
        assert cfg.burst_size == 20


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.auth_enabled is True
        assert cfg.api_keys == []
        assert cfg.rate_limit.requests_per_minute == 60
        assert cfg.log_level == "INFO"
        assert cfg.log_format == "json"
        assert cfg.test_mode is False

    def test_custom_values(self):
        cfg = AppConfig(
            auth_enabled=False,
            api_keys=["key1", "key2"],
            rate_limit=RateLimitConfig(requests_per_minute=100),
            log_level="DEBUG",
            test_mode=True,
        )
        assert cfg.auth_enabled is False
        assert cfg.api_keys == ["key1", "key2"]
        assert cfg.rate_limit.requests_per_minute == 100
        assert cfg.log_level == "DEBUG"
        assert cfg.test_mode is True


class TestModelEndpointConfig:
    def test_required_fields(self):
        cfg = ModelEndpointConfig(name="llama", endpoint="http://vllm:8000")
        assert cfg.name == "llama"
        assert cfg.endpoint == "http://vllm:8000"
        assert cfg.gpu_count == 1
        assert cfg.max_model_len == 4096
        assert cfg.quantization is None

    def test_all_fields(self):
        cfg = ModelEndpointConfig(
            name="llama",
            endpoint="http://vllm:8000",
            gpu_count=2,
            max_model_len=8192,
            quantization="awq",
        )
        assert cfg.gpu_count == 2
        assert cfg.max_model_len == 8192
        assert cfg.quantization == "awq"


class TestGetConfig:
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults(self):
        cfg = get_config()
        assert cfg.auth_enabled is True
        assert cfg.api_keys == []
        assert cfg.rate_limit.requests_per_minute == 60
        assert cfg.log_level == "INFO"
        assert cfg.test_mode is False

    @patch.dict(
        os.environ,
        {
            "AUTH_ENABLED": "false",
            "API_KEYS": "abc,def,ghi",
            "RATE_LIMIT_RPM": "120",
            "LOG_LEVEL": "debug",
            "TEST_MODE": "true",
        },
    )
    def test_custom_env(self):
        cfg = get_config()
        assert cfg.auth_enabled is False
        assert cfg.api_keys == ["abc", "def", "ghi"]
        assert cfg.rate_limit.requests_per_minute == 120
        assert cfg.log_level == "DEBUG"
        assert cfg.test_mode is True

    @patch.dict(os.environ, {"API_KEYS": " key1 , , key2 "}, clear=True)
    def test_api_keys_whitespace_handling(self):
        cfg = get_config()
        assert cfg.api_keys == ["key1", "key2"]

    @patch.dict(os.environ, {"AUTH_ENABLED": "yes"}, clear=True)
    def test_auth_enabled_yes(self):
        cfg = get_config()
        assert cfg.auth_enabled is True

    @patch.dict(os.environ, {"AUTH_ENABLED": "0"}, clear=True)
    def test_auth_enabled_zero(self):
        cfg = get_config()
        assert cfg.auth_enabled is False


class TestGetModelEndpoints:
    @patch.dict(os.environ, {}, clear=True)
    def test_empty(self):
        endpoints = get_model_endpoints()
        assert endpoints == []

    @patch.dict(
        os.environ,
        {"MODEL_ENDPOINTS": "model-a:http://vllm-a:8000,model-b:http://vllm-b:8000"},
    )
    def test_multiple_endpoints(self):
        endpoints = get_model_endpoints()
        assert len(endpoints) == 2
        assert endpoints[0].name == "model-a"
        assert endpoints[0].endpoint == "http://vllm-a:8000"
        assert endpoints[1].name == "model-b"
        assert endpoints[1].endpoint == "http://vllm-b:8000"

    @patch.dict(
        os.environ,
        {"MODEL_ENDPOINTS": "single:http://localhost:8000"},
    )
    def test_single_endpoint(self):
        endpoints = get_model_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].name == "single"
        assert endpoints[0].endpoint == "http://localhost:8000"

    @patch.dict(os.environ, {"MODEL_ENDPOINTS": ",,"}, clear=True)
    def test_empty_entries_ignored(self):
        endpoints = get_model_endpoints()
        assert endpoints == []
