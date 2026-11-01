import os
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.plugins import PluginManager


def test_explicitly_trusted_plugin_runs_in_process(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "plugins" / "share-room"
    plugin_dir.mkdir(parents=True)
    manifest = SimpleNamespace(
        name="share-room",
        source="user",
        path=plugin_dir,
    )
    manager = PluginManager()
    monkeypatch.setenv("HERMES_TRUSTED_IN_PROCESS_PLUGINS", "share-room")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugin_isolation.isolation_mode", lambda: "host")

    assert manager._runs_in_plugin_host(manifest) is False


def test_untrusted_user_plugin_stays_isolated(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "plugins" / "other"
    plugin_dir.mkdir(parents=True)
    manifest = SimpleNamespace(name="other", source="user", path=plugin_dir)
    manager = PluginManager()
    monkeypatch.setenv("HERMES_TRUSTED_IN_PROCESS_PLUGINS", "share-room")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugin_isolation.isolation_mode", lambda: "host")

    assert manager._runs_in_plugin_host(manifest) is True


def test_trusted_name_outside_profile_plugin_dir_stays_isolated(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "elsewhere" / "share-room"
    plugin_dir.mkdir(parents=True)
    manifest = SimpleNamespace(name="share-room", source="user", path=plugin_dir)
    manager = PluginManager()
    monkeypatch.setenv("HERMES_TRUSTED_IN_PROCESS_PLUGINS", "share-room")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.plugin_isolation.isolation_mode", lambda: "host")

    assert manager._runs_in_plugin_host(manifest) is True
