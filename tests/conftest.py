"""Shared pytest config.

Tests marked `@pytest.mark.ollama` are skipped by default. Run them with:
    pytest --ollama

The flag exists because eval-harness tests need a live Ollama service and
take ~2-4 min on CPU. Default `pytest` keeps the unit suite fast and
fully offline."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ollama",
        action="store_true",
        default=False,
        help="Run tests that require a live Ollama service (eval harness).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ollama: test that requires a running Ollama instance with qwen2.5:3b pulled",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ollama"):
        return
    skip_ollama = pytest.mark.skip(
        reason="needs --ollama flag (live Ollama required)"
    )
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip_ollama)
