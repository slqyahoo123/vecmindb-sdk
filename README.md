# VecminDB SDK

The official SDK for [VecminDB](https://vecmindb.com) — The Sovereign Memory OS for AI Agents.

> Stop letting your AI Agents hallucinate from memory rot. VecminDB naturally decays outdated memories, distills knowledge via PCA, and provides 100% Air-Gapped cryptographic data sovereignty.

## Deployment & Installation

VecminDB can be run via Docker or as optimized, standalone pre-compiled native binary packages. No local compilers, dependencies, or Python runtimes are needed.

### Method A: Docker Deployment (All Platforms - Windows, macOS, Linux)
The fastest way to spin up VecminDB with automatic in-database bilingual embedding support.

```bash
# For Global / Overseas users:
docker run -d --name vecmindb-trial -p 5520:5520 ghcr.io/lingxinmind/vecmindb:latest

# For Domestic users (China Aliyun Mirror):
# docker run -d --name vecmindb-trial -p 5520:5520 crpi-ngtfnt7d3tsnwk7l.cn-shanghai.personal.cr.aliyuncs.com/vecmindb/vecmindb:latest
```

---

### Method B: Pre-Compiled Native Binary Bundles (Zero-Docker / Zero-Python)
Ideal for high-performance, air-gapped private cloud servers. Download the appropriate package from our official [Downloads](https://vecmindb.com/downloads) portal:

*   **Windows (AMD64)**:
    Download `vecmindb-1.0.0-beta-x86_64-pc-windows-msvc.zip`. Extract the ZIP archive, open Command Prompt or PowerShell in the directory, and run:
    ```cmd
    .\vecmindb-server.exe
    ```
*   **macOS (Apple Silicon M1/M2/M3)**:
    Download `vecmindb-1.0.0-beta-aarch64-apple-darwin.tar.gz`. Open Terminal, extract and run:
    ```bash
    tar -xzf vecmindb-1.0.0-beta-aarch64-apple-darwin.tar.gz
    cd vecmindb-1.0.0-beta-aarch64-apple-darwin
    ./vecmindb-server
    ```
*   **Linux (AMD64)**:
    Download `vecmindb-offline-linux-amd64.tar.gz`. Extract and run:
    ```bash
    tar -xzf vecmindb-offline-linux-amd64.tar.gz
    cd vecmindb-offline-linux-amd64
    ./vecmindb-server
    ```

---

## SDK Quickstart

First, install the target client SDK:

```bash
# Install core client
pip install vecmindb

# Install with LangChain integration
pip install vecmindb[langchain]

# Install with CrewAI integration
pip install vecmindb[crewai]
```

### Using with LangChain

```python
from vecmindb.memory_plugin import VecminDBMemoryPlugin
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# Initialize Sovereign Agent Memory
memory = VecminDBMemoryPlugin.for_langchain(agent_id="support_agent_01", base_url="http://localhost:5520")

llm = ChatOpenAI(temperature=0)
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="Hi, I need help with my billing.")
```

### Using with CrewAI

```python
from vecmindb.memory_plugin import VecminDBMemoryPlugin
from crewai import Agent, Crew

# Initialize Sovereign Agent Memory
memory_storage = VecminDBMemoryPlugin.for_crewai(agent_id="finance_agent_01", base_url="http://localhost:5520")

agent = Agent(
    role='Financial Analyst',
    goal='Analyze billing data',
    backstory='An expert in financial data.',
    memory=True,
    memory_config={"storage": memory_storage} # Inject VecminDB memory
)
```

## Why VecminDB?

*   **100% Offline**: Built-in ONNX embedding model. Your data never leaves your VPC.
*   **Biological Forgetting (LTSM)**: Old, unused memories naturally decay over time to prevent context pollution.
*   **Knowledge Distillation**: Fuses semantic clusters into dense abstract centroids automatically.
*   **Sovereignty Isolation**: Agents are cryptographically isolated using HMAC-SHA256 signature chains.

---
**Enterprise Licensing**: For multi-node SOC-2 compliant deployments, contact `support@vecmindb.com`.
