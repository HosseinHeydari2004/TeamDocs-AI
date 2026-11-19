from __future__ import annotations

import pytest
from pydantic import ValidationError

from teamdocs.core.config import Settings


def make(**kw):
    return Settings(_env_file=None, **kw)


def test_defaults_are_sane():
    s = make()
    assert s.retrieval_mode == "hybrid" and s.top_k == 5 and s.llm_configured is False
    assert s.chroma_dir == s.data_dir / "chroma" and s.max_upload_bytes == 25 * 1024 * 1024


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    s = make()
    assert s.top_k == 9 and s.llm_configured and s.log_level == "DEBUG"
    assert s.cors_origin_list == ["http://a.test", "http://b.test"]
    assert "sk-or-real" not in repr(s)  # secret is masked in logs/reprs


def test_dotenv_file_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_text("TOP_K=7\nEMBEDDING_MODEL=my/model\n")
    s = Settings(_env_file=env)
    assert s.top_k == 7 and s.embedding_model == "my/model"


@pytest.mark.parametrize("placeholder", ["", "   ", "your_openrouter_api_key_here"])
def test_placeholder_api_keys_count_as_unset(placeholder):
    assert make(openrouter_api_key=placeholder).llm_configured is False


@pytest.mark.parametrize(
    "bad",
    [
        {"top_k": 0},
        {"top_k": 500},
        {"dense_weight": 1.5},
        {"chunk_size": 1000, "chunk_overlap": 1000},
        {"collection_name": "x"},
        {"collection_name": "bad name!"},
        {"retrieval_mode": "magic"},
        {"app_port": 0},
    ],
)
def test_invalid_values_fail_fast(bad):
    with pytest.raises(ValidationError):
        make(**bad)


def test_metadata_extraction_requires_an_llm_key():
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        make(metadata_extraction=True)
    assert make(metadata_extraction=True, openrouter_api_key="sk-or-x").metadata_extraction


def test_ensure_directories_creates_the_data_layout(tmp_path):
    s = make(data_dir=tmp_path / "d", model_cache_dir=tmp_path / "m")
    s.ensure_directories()
    assert all(p.is_dir() for p in (s.chroma_dir, s.upload_dir, s.tmp_dir, s.model_cache_dir))
