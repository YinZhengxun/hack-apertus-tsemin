"""Paths and model-endpoint configuration.

The endpoint is configured with the three environment variables required by the Hack Apertus
template: LLM_NAME, LLM_BASE_URL, LLM_API_KEY. A `.env` file next to run.py is read as well.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# track_2a/ (the project root required by the template)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GOLD_DEV = DATA / "gold" / "dev"
SPLITS = DATA / "splits"
MANIFESTS = DATA / "manifests"
PROMPTS = ROOT / "prompts"
RUNS = ROOT / "runs"
ENV_FILE = ROOT / ".env"

DEFAULT_BASE_URL = "https://api.inference.cscs.ch/v1"
DEFAULT_MODEL = "swiss-ai/Apertus-v1.5-8B"
MODEL_70B = "swiss-ai/Apertus-v1.5-70B"


@dataclass
class LLMConfig:
    name: str
    base_url: str
    api_key: str


def _read_env_text(path: Path) -> str:
    """Files written by Windows tools may start with a byte-order mark or be UTF-16."""
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def load_env_file(path: Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for raw in _read_env_text(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _save_key(key: str, path: Path = ENV_FILE) -> None:
    lines = []
    if path.exists():
        lines = [l for l in _read_env_text(path).splitlines() if not l.strip().startswith("LLM_API_KEY=")]
    lines.append("LLM_API_KEY=" + key)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean_key(value: str) -> str:
    """Tolerate quotes, spaces and a pasted "Bearer " prefix."""
    value = (value or "").strip().strip('"').strip("'").strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


def get_llm_config(model: Optional[str] = None, interactive: bool = True) -> LLMConfig:
    load_env_file()
    name = model or os.environ.get("LLM_NAME") or DEFAULT_MODEL
    base_url = (os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = _clean_key(os.environ.get("LLM_API_KEY", ""))
    if not key and interactive:
        print("第一次运行需要 API key（只问这一次，会存到 track_2a/.env）。")
        key = _clean_key(input("把 key 粘贴到这里后回车: "))
        if key:
            _save_key(key)
            os.environ["LLM_API_KEY"] = key
    if not key:
        raise SystemExit("没有 API key：请设置环境变量 LLM_API_KEY，或在 track_2a/.env 里写一行 LLM_API_KEY=你的key")
    return LLMConfig(name=name, base_url=base_url, api_key=key)
