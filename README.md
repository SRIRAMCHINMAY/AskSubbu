# E-commerce Voice Assistant

Real-time voice-enabled AI assistant for e-commerce with RAG and function calling.

## Features
- 🎤 Voice input/output
- 🧠 RAG with product knowledge
- 🔧 Order tracking & product search
- ⚡ Real-time audio streaming (LiveKit)
                  ┌───────────────────┐
                  │   User Voice      │
                  └────────┬──────────┘
                           │
                 Speech-to-Text (STT)
                           │
                           ▼
                  ┌───────────────────┐
                  │  Text Query       │
                  └────────┬──────────┘
                           │
           ┌───────────────┼─────────────────┐
           │               │                 │
           ▼               ▼                 ▼
   ┌─────────────┐  ┌─────────────┐   ┌─────────────┐
   │ Product DB  │  │ Inventory / │   │ FAQ / Docs  │
   │  Catalog    │  │ Pricing API │   │ (PDF/HTML)  │
   └─────┬───────┘  └─────┬───────┘   └─────┬───────┘
         │                 │                 │
         └──────┐   ┌──────┘                 │
                ▼   ▼                        ▼
            Vectorization / Embeddings (OpenAI, Cohere, etc.)
                │   │                        │
                └───┴──────────────┐
                                   ▼
                           Vector Store(s)
                   (Chroma, Milvus, Weaviate, Pinecone)
                                   │
                                   ▼
                         Retrieval Module
               (Top-k relevant chunks from each source)
                                   │
                                   ▼
                              Query Router
               (Optional: decide which sources to prioritize)
                                   │
                                   ▼
                       Large Language Model (LLM)
                     (Generates natural response)
                                   │
                                   ▼
                         Text-to-Speech (TTS)
                       (ElevenLabs, Vocode, gTTS)
                                   │
                                   ▼
                            Voice Response

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