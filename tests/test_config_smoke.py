"""Smoke test for :mod:`utils.config_loader`.

Originally this file only defined ``main()`` and printed its results, so it
contributed **zero** pytest cases — a file named ``test_*`` that tested nothing
(CURRENT_ISSUES.md ISSUE-050). It is now a real test.

The assertions are unchanged from the original ``main()``: they pin the
project's current configuration so a silent edit to ``config.yaml`` shows up
here rather than surfacing as a mysterious retrieval regression.

Wider coverage of this module lives in ``tests/test_config_loader.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.config_loader import AppConfig, load_config, resolve_config_path  # noqa: E402


def test_config_file_resolves_and_loads() -> None:
    """The original ``main()`` body, as an assertion."""
    p = resolve_config_path()
    assert p.is_file(), f"config.yaml not found at {p}"

    cfg = load_config()
    assert isinstance(cfg, AppConfig), f"expected AppConfig, got {type(cfg).__name__}"
    assert cfg.chunk_size == 1024, cfg.chunk_size
    assert cfg.chunk_overlap == 200, cfg.chunk_overlap
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5", cfg.embedding_model


def main() -> None:
    """Kept so the file still runs as a script (``python tests/test_config_smoke.py``)."""
    test_config_file_resolves_and_loads()
    cfg = load_config()
    print(f"resolved config path -> {resolve_config_path()}")
    print(f"chunk_size   = {cfg.chunk_size} ({type(cfg.chunk_size).__name__})")
    print(f"chunk_overlap= {cfg.chunk_overlap}")
    print(f"embedding    = {cfg.embedding_model}")
    print(f"openai_key?  = {cfg.has_openai_api_key}")
    print(f"extras       = {sorted(cfg.extras)}")
    print("OK")


if __name__ == "__main__":
    main()
