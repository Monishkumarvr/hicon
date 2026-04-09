from pathlib import Path


def test_service_units_include_memory_guardrails():
    repo_root = Path(__file__).resolve().parents[2]
    unit_paths = [
        repo_root / "ai_vision/systemd/hicon-vision.service",
        repo_root / "hicon-vision.service",
    ]

    for unit_path in unit_paths:
        text = unit_path.read_text()
        assert "MemoryAccounting=yes" in text
        assert "MemoryHigh=3G" in text
        assert "MemoryMax=4G" not in text
        assert "OOMPolicy=stop" not in text
