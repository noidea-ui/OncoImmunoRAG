"""Smoke test for utils.config_loader."""
from utils.config_loader import AppConfig, load_config, resolve_config_path


def main() -> None:
    p = resolve_config_path()
    print(f"resolved config path -> {p} | exists: {p.is_file()}")

    cfg = load_config()
    assert isinstance(cfg, AppConfig), f"expected AppConfig, got {type(cfg).__name__}"
    assert cfg.chunk_size == 1024, cfg.chunk_size
    assert cfg.chunk_overlap == 200, cfg.chunk_overlap
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5", cfg.embedding_model
    print(f"chunk_size   = {cfg.chunk_size} ({type(cfg.chunk_size).__name__})")
    print(f"chunk_overlap= {cfg.chunk_overlap}")
    print(f"embedding    = {cfg.embedding_model}")
    print(f"openai_key?  = {cfg.has_openai_api_key}")
    print(f"extras       = {cfg.extras}")
    print("OK")


if __name__ == "__main__":
    main()
