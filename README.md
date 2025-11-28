# MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation

[![arXiv](https://img.shields.io/badge/arXiv-2501.06713-b31b1b)](https://arxiv.org/abs/2501.06713)
<a href="./Communication.md"><img src="https://img.shields.io/badge/WeChat-Group-07c160?logo=wechat&logoColor=white"></a>

MiniRAG is a lightweight RAG framework that enables small language models to achieve good performance through heterogeneous graph indexing and topology-enhanced retrieval.

[Tianyu Fan](https://tianyufan0504.github.io/), [Jingyuan Wang](https://scholar.google.com/citations?user=BGT3Gb8AAAAJ&hl=en), [Xubin Ren](https://ren-xubin.github.io/), [Chao Huang](https://sites.google.com/view/chaoh)*

[中文说明](./README_CN.md) | [日本語](./README_JA.md)

![MiniRAG Framework](https://files.mdnice.com/user/87760/02baba85-fa69-4223-ac22-914fef7120ae.jpg)

## Install

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .

# From PyPI
pip install minirag-hku
```

## Quick Start

```python
from minirag import MiniRAG, QueryParam
from minirag.llm.openai import deepseek_complete
from minirag.llm.siliconcloud import siliconcloud_embedding
from minirag.utils import EmbeddingFunc
import os

rag = MiniRAG(
    working_dir="./LiHua-World",  # Index directory (created automatically)
    llm_model_func=deepseek_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: siliconcloud_embedding(
            texts,
            model="BAAI/bge-m3",
            api_key=os.environ.get("SILICONFLOW_API_KEY"),
        ),
    ),
)

# Build knowledge graph
rag.insert("Your document text...")

# Query
result = rag.query("Your question?", param=QueryParam(mode="mini"))
```

Set environment variables:
```bash
export DEEPSEEK_API_KEY=your_key
export SILICONFLOW_API_KEY=your_key
```

**Reproduce paper results:**
```bash
# Index the dataset (creates ./LiHua-World directory)
python ./reproduce/Step_0_index.py --model PHI

# Run Q&A evaluation
python ./reproduce/Step_1_QA.py --model PHI
```

## Supported Providers

**LLM**: OpenAI (`gpt_4o_mini_complete`), DeepSeek (`deepseek_complete`), NVIDIA, OpenRouter

**Embedding**: OpenAI (`openai_embed`), SiliconCloud (`siliconcloud_embedding`), HuggingFace (`hf_embed`), Ollama (`ollama_embedding`)

## Performance

| Model | NaiveRAG | LightRAG | **MiniRAG** |
|-------|----------|----------|-------------|
| Phi-3.5-mini | 41.22% | 39.81% | **53.29%** |
| Qwen2.5-3B | 43.73% | 39.18% | **48.75%** |
| gpt-4o-mini | 46.55% | 56.90% | 54.08% |

## Dataset: LiHua-World

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

A benchmark dataset for on-device RAG with single-hop, multi-hop, and summary questions. See [dataset README](./dataset/LiHua-World/README.md).

## Citation

```bibtex
@article{fan2025minirag,
  title={MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation},
  author={Fan, Tianyu and Wang, Jingyuan and Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2501.06713},
  year={2025}
}
```

## Acknowledgements

Built on [nano-graphrag](https://github.com/gusye1234/nano-graphrag) and [LightRAG](https://github.com/HKUDS/LightRAG).

<a href="https://github.com/HKUDS/MiniRAG/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=HKUDS/MiniRAG" />
</a>
