# Modelos de Lenguaje y Grafos para Ciencia de Datos

# _Intersemestral Period 2026_

## Library versions

- Python: 3.12.13 (main, Jun 23 2026, 15:18:55) [Clang 22.1.3 ]
- PyTorch: 2.12.1+cu130
- CUDA disponible: True
- CUDA PyTorch: 13.0
- pandas: 3.0.3
- scikit-learn: 1.9.0
- transformers: 5.12.1
- sentence-transformers: 5.6.0
- faiss OK
- torch-geometric: 2.8.0
- networkx: 3.6.1
- ollama client OK

# LLMs and GNNs for Advanced Reasoning — Course Labs

Lab notebooks for the course **"LLMs and GNNs for Advanced Reasoning over Relational Data"**.
The course builds from classic ML fundamentals all the way to hybrid systems that combine large language models with graph neural networks.

---

## Course structure

| Module         | Topic                                                            | Key libraries                                          | Labs |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------------ | ---- |
| **1 — ML**     | Data exploration & classical ML on the IBM Telco Churn dataset   | pandas, scikit-learn, PyTorch                          | 2    |
| **2 — LLM**    | Open-source LLMs, local chatbot, RAG, HuggingFace deep dive      | transformers, sentence-transformers, faiss-cpu, Ollama | 4    |
| **3 — GNN**    | Graph data, GCN training, attention with GAT on the Cora dataset | torch-geometric, networkx                              | 3    |
| **4 — Hybrid** | Text-attributed graphs, GraphRAG, G-Retriever                    | builds on modules 2 + 3                                | 3    |

---

## Labs at a glance

### Module 1 — Classical ML

| Lab      | Title            | Topics                                                                                                                                    |
| -------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `lab1_1` | Data Exploration | pandas inspection, data cleaning, class imbalance, feature visualisation                                                                  |
| `lab1_2` | First ML Model   | one-hot encoding, train/val/test split, Logistic Regression, confusion matrix, F1, Decision Tree, MLP (sklearn + PyTorch), regularisation |

### Module 2 — LLMs and RAG

| Lab      | Title                           | Topics                                                                                                      |
| -------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `lab2_1` | Open-Source LLMs: First Contact | `SimpleLLM`, system prompts, temperature, hallucination, context windows                                    |
| `lab2_2` | Local Chatbot                   | conversation history, multi-turn chat, `Chatbot` class                                                      |
| `lab2_3` | Basic RAG                       | document chunking, FAISS index, semantic retrieval, answer generation                                       |
| `lab2_4` | HuggingFace in Depth            | `pipeline()`, tokenization, `AutoModelForCausalLM`, generation strategies, `SentenceTransformer` embeddings |

### Module 3 — Graph Neural Networks

| Lab      | Title              | Topics                                                                      |
| -------- | ------------------ | --------------------------------------------------------------------------- |
| `lab3_1` | Graph Data         | PyG `Data` objects, Cora dataset, train/val/test masks, graph visualisation |
| `lab3_2` | Training a GCN     | `GCNConv`, training loop, node classification, evaluation                   |
| `lab3_3` | Attention with GAT | `GATConv`, multi-head attention, attention weight visualisation             |

### Module 4 — Hybrid Systems

| Lab      | Title                  | Topics                                       |
| -------- | ---------------------- | -------------------------------------------- |
| `lab4_1` | Text-Attributed Graphs | combining text features with graph structure |
| `lab4_2` | GraphRAG               | graph-aware retrieval-augmented generation   |
| `lab4_3` | G-Retriever            | end-to-end graph + LLM reasoning system      |

---

## Quickstart (local)

```bash
# 1 — clone
git clone https://github.com/DanielFPerez/llm-gnns-course_labs.git
cd llm-gnns-course_labs

# 2 — create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows (WSL2): same command

# 3 — install all dependencies
pip install -r environment/requirements.txt

# 4 — launch JupyterLab
jupyter lab
```

Open the URL printed in the terminal (e.g. `http://localhost:8888/lab?token=...`) in your browser, then navigate to the notebook you want to run.

### Platform-specific setup guides

Detailed, step-by-step instructions for your operating system:

- **Windows (WSL2 + NVIDIA CUDA)** → [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md)
- **macOS (Apple Silicon / Intel) or native Linux** → [`SETUP_MAC_LINUX.md`](SETUP_MAC_LINUX.md)

Both guides cover: system dependencies, GPU drivers, Python virtual environments, PyTorch verification, VS Code + Jupyter setup, and Git/SSH configuration.

---

## Running on Google Colab

Every notebook has an **Open in Colab** badge at the top. Click it to open and run the lab in a free cloud environment — no local installation required.

On Colab, the setup cell at the top of each notebook automatically clones the repo and installs all dependencies. The first run takes a few minutes; subsequent runs use Colab's cache.

> **GPU tip:** In Colab, go to **Runtime → Change runtime type → T4 GPU** before running Module 3 or 4 labs to get hardware acceleration.

---

## Module 2 — Ollama setup (local only)

Labs 2.1–2.3 use a `SimpleLLM` wrapper that tries Ollama first and falls back to a HuggingFace model automatically. To use Ollama locally:

```bash
# Install Ollama (Linux / WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2:1b    # ~1.3 GB, fast on CPU
ollama pull llama3.2:3b    # ~2.0 GB, better quality

# Start the server (leave this terminal open)
ollama serve
```

If `ollama serve` is not running, `SimpleLLM()` automatically switches to `HuggingFaceTB/SmolLM2-1.7B-Instruct` from HuggingFace (~3.4 GB, downloaded and cached on first use).

See **Lab 2.1, Section 0** for a full guide to interacting with Ollama from the terminal, including the REST API.

---

## Repository structure

```
llm-gnns-course_labs/
├── environment/
│   └── requirements.txt          # all dependencies for all four modules
├── module-1-ml/
│   ├── lab1_1_data_exploration.ipynb
│   └── lab1_2_first_ml_model.ipynb
├── module-2-llm/
│   ├── lab2_1_open_source_llms.ipynb
│   ├── lab2_2_local_chatbot.ipynb
│   ├── lab2_3_basic_rag.ipynb
│   └── lab2_4_huggingface_deep_dive.ipynb
├── module-3-gnn/
│   ├── lab3_1_graph_data.ipynb
│   ├── lab3_2_training_gcn.ipynb
│   └── lab3_3_attention_gat.ipynb
├── module-4-hybrid/
│   ├── lab4_1_text_attributed_graphs.ipynb
│   ├── lab4_2_graphrag.ipynb
│   └── lab4_3_g_retriever.ipynb
├── utils/                        # shared helpers imported by all notebooks
│   ├── data.py                   # load_telco_churn(), load_company_kb()
│   ├── llm.py                    # SimpleLLM (Ollama-first, HuggingFace fallback)
│   ├── checks.py                 # exercise validators (check_dataframe, check_model, …)
│   ├── graph.py                  # GNN visualisation helpers
│   └── plotting.py               # plotting utilities for modules 1–2
├── SETUP_WINDOWS.md
├── SETUP_MAC_LINUX.md
└── README.md
```

### `utils/` package

| Module        | Key exports                                                                                           | Used in      |
| ------------- | ----------------------------------------------------------------------------------------------------- | ------------ |
| `data.py`     | `load_telco_churn()`, `load_company_kb()`                                                             | Modules 1, 2 |
| `llm.py`      | `SimpleLLM` — `.generate()`, `.chat()`, `.count_tokens()`                                             | Module 2     |
| `checks.py`   | `check_dataframe`, `check_split`, `check_model`, `check_gnn_model`, `check_graph`                     | All modules  |
| `graph.py`    | `plot_graph`, `plot_embeddings`, `plot_attention_subgraph`                                            | Modules 3, 4 |
| `plotting.py` | `plot_class_distribution`, `plot_confusion_matrix`, `plot_training_curves`, `plot_feature_importance` | Module 1     |

Datasets are downloaded lazily and cached to `~/.llm_gnns_course/data/` on first use.

---

## Dependencies

All packages are in `environment/requirements.txt`. Major ones by module:

| Module     | Core packages                                                                           |
| ---------- | --------------------------------------------------------------------------------------- |
| 1 — ML     | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`                              |
| 2 — LLM    | `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `accelerate`, `datasets` |
| 3 — GNN    | `torch-geometric`, `networkx`                                                           |
| 4 — Hybrid | no new packages — builds on modules 2 + 3                                               |
