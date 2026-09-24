"""Tests the ablation-switch env var parsing directly - separate from
tests/test_agent.py's tests, which cover diagnose_ticket's `verify`
parameter resolution but not this parsing logic in isolation.
"""

import importlib

import pytest

import src.config as config


@pytest.fixture(autouse=True)
def _restore_config_module():
    # importlib.reload mutates the shared module object in sys.modules,
    # which monkeypatch's env var revert does NOT undo - reload back to a
    # clean (no override) state after every test in this file so later
    # test files importing src.config don't see whatever env value the
    # last test here happened to leave behind.
    yield
    import os

    os.environ.pop("ANCHOR_VERIFIER_ENABLED", None)
    importlib.reload(config)


def _reload_with_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ANCHOR_VERIFIER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("ANCHOR_VERIFIER_ENABLED", value)
    importlib.reload(config)
    return config.VERIFIER_ENABLED_DEFAULT


def test_defaults_to_true_when_unset(monkeypatch):
    assert _reload_with_env(monkeypatch, None) is True


def test_env_var_zero_disables(monkeypatch):
    assert _reload_with_env(monkeypatch, "0") is False


def test_env_var_false_disables_case_insensitive(monkeypatch):
    assert _reload_with_env(monkeypatch, "False") is False
    assert _reload_with_env(monkeypatch, "FALSE") is False


def test_env_var_no_disables(monkeypatch):
    assert _reload_with_env(monkeypatch, "no") is False


def test_env_var_1_enables(monkeypatch):
    assert _reload_with_env(monkeypatch, "1") is True


def test_env_var_arbitrary_truthy_string_enables(monkeypatch):
    # anything that isn't one of the recognized "off" spellings counts as
    # on - explicit opt-out, not accidental opt-out via a typo.
    assert _reload_with_env(monkeypatch, "true") is True
    assert _reload_with_env(monkeypatch, "yes") is True


def test_responsiveness_switch_is_separate_from_the_verifier_switch(monkeypatch):
    """Two env vars, not one. Turning one off must leave the other alone,
    or the 2x2 ablation cannot be run."""
    import importlib

    import src.config as config

    monkeypatch.setenv("ANCHOR_VERIFIER_ENABLED", "0")
    monkeypatch.delenv("ANCHOR_RESPONSIVENESS_ENABLED", raising=False)
    reloaded = importlib.reload(config)
    assert reloaded.VERIFIER_ENABLED_DEFAULT is False
    assert reloaded.RESPONSIVENESS_ENABLED_DEFAULT is True

    monkeypatch.delenv("ANCHOR_VERIFIER_ENABLED", raising=False)
    monkeypatch.setenv("ANCHOR_RESPONSIVENESS_ENABLED", "0")
    reloaded = importlib.reload(config)
    assert reloaded.VERIFIER_ENABLED_DEFAULT is True
    assert reloaded.RESPONSIVENESS_ENABLED_DEFAULT is False

    monkeypatch.delenv("ANCHOR_RESPONSIVENESS_ENABLED", raising=False)
    importlib.reload(config)
