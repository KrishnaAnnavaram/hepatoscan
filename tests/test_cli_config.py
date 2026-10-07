import json

import pytest

from hepatoscan import cli
from hepatoscan.config import load_dotenv, resolve_device, settings_from_env


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("HEPATOSCAN_MAX_HISTORY", "2")
    monkeypatch.setenv("HEPATOSCAN_LLM_PROVIDER", "openai")
    s = settings_from_env()
    assert s.max_history == 2 and s.llm_provider == "openai"
    assert "api_token" not in repr(s) or "None" in repr(s)
    monkeypatch.setenv("HEPATOSCAN_MAX_UPLOAD_MB", "-1")
    with pytest.raises(ValueError):
        settings_from_env()


def test_dotenv_reads_only_known_unset_names(tmp_path, monkeypatch):
    monkeypatch.delenv("HEPATOSCAN_API_TOKEN", raising=False)
    monkeypatch.setenv("HEPATOSCAN_DEVICE", "cpu")
    env = tmp_path / ".env"
    env.write_text("HEPATOSCAN_API_TOKEN=abc\nHEPATOSCAN_DEVICE=cuda\nOTHER=1\n")
    assert load_dotenv(env) == ["HEPATOSCAN_API_TOKEN"]
    assert resolve_device("cpu") == "cpu" and resolve_device("auto") in ("cpu", "cuda")


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPATOSCAN_RESULTS_DIR", str(tmp_path / "results"))
    data = tmp_path / "data"
    assert cli.main(["synth", "--cases", "8", "--shape", "12", "32", "32", "--out", str(data)]) == 0
    assert cli.main(["train-baseline", "--data", str(data), "--per-class", "300"]) == 0
    model = tmp_path / "results" / "baseline.pkl"
    split = json.loads((tmp_path / "results" / "split.json").read_text())
    assert not set(split["train"]) & set(split["test"])
    assert cli.main(["evaluate", "--data", str(data), "--model", str(model)]) == 0
    assert (tmp_path / "results" / "eval_baseline_test.json").exists()
    case = data / f"{split['test'][0]}.npz"
    assert cli.main(["segment", str(case), "--model", str(model), "--out-dir", str(tmp_path / "out")]) == 0
    summary = tmp_path / "out" / f"{split['test'][0]}_summary.json"
    assert summary.exists()
    assert cli.main(["ask", "What does the liver volume mean?", "--summary", str(summary)]) == 0
    assert "not medical advice" in capsys.readouterr().out


def test_cli_rejects_unknown_model_type(tmp_path):
    with pytest.raises(SystemExit):
        cli.load_segmenter(tmp_path / "model.h5")
