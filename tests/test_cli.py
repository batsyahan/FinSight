import json

from finsight.cli import main


def test_cli_writes_memo_and_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FINSIGHT_LLM", "mock")
    monkeypatch.setenv("FINSIGHT_DATA", "sample")
    code = main(["analyze", "acme", "--out", str(tmp_path), "--engine", "builtin"])
    assert code == 0
    memo = (tmp_path / "ACME_memo.md").read_text()
    assert memo.startswith("# ACME") and "### Evidence" in memo
    trace = json.loads((tmp_path / "ACME_trace.json").read_text())
    assert trace["approved"] is True and trace["evidence"]["S1"]["source"] == "get_company_profile"
    assert "APPROVED" in capsys.readouterr().out
