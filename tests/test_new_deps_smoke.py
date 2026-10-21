"""Smoke test for newly added dependencies."""
from importlib.metadata import version

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sentence_transformers
import yaml

print(f"llama-index-embeddings-huggingface {version('llama-index-embeddings-huggingface')}")
print(f"sentence-transformers             {sentence_transformers.__version__}")
print(f"pyyaml                            {yaml.__version__}")
print(f"transformers (transitive)         {version('transformers')}")
print(f"torch (transitive)                {version('torch')}")

# Quick sanity: HuggingFaceEmbedding should be constructible (no model load).
_ = HuggingFaceEmbedding  # class imported successfully
print("HuggingFaceEmbedding class importable: OK")
