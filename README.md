# E-commerce Voice Assistant

Real-time voice-enabled AI assistant for e-commerce with RAG and function calling.

## Features
- 🎤 Voice input/output
- 🧠 RAG with product knowledge
- 🔧 Order tracking & product search
- ⚡ Real-time audio streaming (LiveKit)
               flowchart TD
    A[User Voice] -->|Speech-to-Text| B[Text Query]

    B --> C[Product DB / Catalog]
    B --> D[Inventory / Pricing API]
    B --> E[FAQ / Docs]
    B --> F[Customer Reviews / External Docs]

    C --> G[Vectorization / Embeddings]
    D --> G
    E --> G
    F --> G

    G --> H[Vector Store(s) (FAISS / Chroma / Milvus)]

    H --> I[Retrieval Module (Top-k per source)]

    I --> J[Query Router / Source Prioritization]

    J --> K[Large Language Model (LLM)]
    K --> L[Text-to-Speech (TTS)]
    L --> M[Voice Response to User]

## Setup

1. **Clone and install:**
```bash
git clone https://github.com/SRIRAMCHINMAY/AskSubbu.git
cd ecommerce-voicebot
mamba create -n voicebot python=3.11 -y
mamba activate voicebot
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

3. **Run:**
```bash
python initial.py
```

## Project Status
- ✅ Week 1 Day 1-2: RAG + Agent working
- 🚧 Week 1 Day 3-4: Function calling (in progress)
- ⏳ Week 1 Day 5-6: STT + TTS
- ⏳ Week 2: LiveKit integration

## Tech Stack
- OpenRouter (LLM)
- FAISS (Vector DB)
- LiveKit (Audio streaming)
- FastAPI (Backend)