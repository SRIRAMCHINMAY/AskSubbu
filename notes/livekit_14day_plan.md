# 14-Day LiveKit E-commerce Voicebot Plan

## 🎯 Architecture Goal

```
User (Browser/App) 
    ↕️ WebRTC Audio
LiveKit Server
    ↕️
Backend (Python)
    ├─ STT (Whisper/Deepgram)
    ├─ LLM + RAG (OpenRouter + FAISS)
    ├─ Action Layer (Order DB queries)
    └─ TTS (OpenAI/ElevenLabs)
    ↕️
LiveKit → Audio back to User
```

---

## 📅 REALISTIC 14-Day Timeline

### **WEEK 1: Core Components (Days 1-7)**

#### **Day 1-2: ✅ DONE - Foundation**
- ✅ OpenRouter + FAISS + RAG working
- ✅ Basic agent responding with context

**Next Steps Today:**
- Save your current working code as `core_agent.py`
- We'll integrate this into LiveKit next

---

#### **Day 3-4: Function Calling (Action Layer)**

**Goal:** Add tools for order tracking, product search

**What to Build:**
```python
# tools.py
from langchain.tools import tool

@tool
def check_order_status(order_id: str):
    """Check order status"""
    # Query mock database
    return {"order_id": order_id, "status": "shipped", "eta": "Dec 12"}

@tool  
def search_products_by_filter(category: str, max_price: float):
    """Search products by category and price"""
    # Query product DB
    return [{"name": "Headphones", "price": 89.99}]

@tool
def create_return(order_id: str, reason: str):
    """Initiate product return"""
    return {"return_id": f"RET-{order_id}", "status": "approved"}
```

**Integration:**
```python
from langchain.agents import create_openai_functions_agent

agent = create_openai_functions_agent(
    llm=your_llm,
    tools=[check_order_status, search_products_by_filter, create_return],
    prompt=your_prompt
)
```

**Deliverable:** Agent can track orders, search products, create returns

---

#### **Day 5-6: Speech Components (STT + TTS)**

**STT Option A: Whisper (Local - FREE)**
```bash
pip install openai-whisper
```
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

**STT Option B: Deepgram (Cloud - Fast)**
```bash
pip install deepgram-sdk
```
```python
from deepgram import Deepgram

dg = Deepgram(API_KEY)
response = dg.transcription.sync_prerecorded(
    {'url': audio_url}, {'punctuate': True}
)
```

**TTS: OpenAI (Simple)**
```python
from openai import OpenAI
client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Your order has shipped!"
)
response.stream_to_file("output.mp3")
```

**Deliverable:** Working STT + TTS pipeline (file-based)

---

#### **Day 7: Integration Test**

**Build end-to-end file-based flow:**
```python
# test_pipeline.py

# 1. User uploads audio file
audio_file = "user_question.wav"

# 2. Transcribe
transcript = whisper_transcribe(audio_file)

# 3. Get context from RAG
context = vector_db.get_context(transcript)

# 4. Agent processes with tools
response = agent.run(transcript, context)

# 5. Generate speech
tts_audio = text_to_speech(response)

# 6. Return audio file
return tts_audio
```

**Deliverable:** Full pipeline works with audio files (no LiveKit yet)

---

### **WEEK 2: LiveKit Integration (Days 8-14)**

#### **Day 8-9: LiveKit Setup & Basic Connection**

**Install LiveKit:**
```bash
pip install livekit livekit-agents livekit-plugins-openai
```

**Option A: Use LiveKit Cloud (RECOMMENDED - Easiest)**
1. Sign up at https://cloud.livekit.io (free tier)
2. Get API Key + Secret
3. Get WebSocket URL

**Option B: Self-host LiveKit**
```bash
docker run -d -p 7880:7880 -p 7881:7881 \
  livekit/livekit-server --dev
```

**Test Connection:**
```python
# test_livekit.py
from livekit import api, rtc
import asyncio

async def test_connection():
    room = rtc.Room()
    
    # Connect to room
    await room.connect(
        url="wss://your-livekit-url",
        token="your-token"
    )
    
    print("✓ Connected to LiveKit!")
    
asyncio.run(test_connection())
```

**Deliverable:** Successfully connect to LiveKit room

---

#### **Day 10-11: LiveKit Audio Streaming**

**Use LiveKit Agents Framework (Simplest Approach):**

```python
# livekit_bot.py
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai, deepgram

async def entrypoint(ctx: JobContext):
    # Initial context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You are an e-commerce voice assistant. Be concise."
    )
    
    # Create voice assistant
    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],  # Voice Activity Detection
        stt=deepgram.STT(),             # Speech-to-Text
        llm=openai.LLM(model="gpt-4o-mini"),  # LLM
        tts=openai.TTS(),               # Text-to-Speech
        chat_ctx=initial_ctx
    )
    
    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Start assistant
    assistant.start(ctx.room)
    
    # Keep running
    await asyncio.sleep(float('inf'))

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

**Run:**
```bash
python livekit_bot.py dev
```

**Deliverable:** Basic voice assistant working in LiveKit

---

#### **Day 12: Integrate Your RAG + Tools**

**Add your custom logic:**

```python
# custom_livekit_bot.py
from livekit.agents import llm
from your_core_agent import FAISSVectorDB, OpenRouterAgent

# Initialize your components
vector_db = FAISSVectorDB()
vector_db.add_documents(your_knowledge_base)

# Custom LLM wrapper
class CustomLLM(llm.LLM):
    def __init__(self):
        self.agent = OpenRouterAgent()
        self.vector_db = vector_db
    
    async def chat(self, chat_ctx: llm.ChatContext):
        # Get user message
        user_msg = chat_ctx.messages[-1].content
        
        # Get RAG context
        context = self.vector_db.get_context(user_msg)
        
        # Get response from your agent
        response = self.agent.chat(user_msg, context=context)
        
        # Return as LLM response
        return llm.ChatResponse(
            message=llm.ChatMessage(role="assistant", content=response)
        )

# Use in VoiceAssistant
assistant = VoiceAssistant(
    vad=ctx.proc.userdata["vad"],
    stt=deepgram.STT(),
    llm=CustomLLM(),  # Your custom LLM with RAG
    tts=openai.TTS(),
    chat_ctx=initial_ctx
)
```

**Deliverable:** LiveKit bot using YOUR RAG + agent logic

---

#### **Day 13: Web Frontend**

**Simple HTML + LiveKit Client:**

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
    <script src="https://unpkg.com/livekit-client/dist/livekit-client.umd.min.js"></script>
</head>
<body>
    <h1>E-commerce Voice Assistant</h1>
    <button id="connectBtn">🎤 Start Talking</button>
    <button id="disconnectBtn" disabled>Stop</button>
    <div id="status">Not connected</div>
    <div id="transcript"></div>
    
    <script>
        const { Room, RoomEvent } = LivekitClient;
        let room;
        
        document.getElementById('connectBtn').onclick = async () => {
            // Get token from your backend
            const response = await fetch('/get-token');
            const { token, url } = await response.json();
            
            // Connect to room
            room = new Room();
            await room.connect(url, token);
            
            // Enable microphone
            await room.localParticipant.setMicrophoneEnabled(true);
            
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('connectBtn').disabled = true;
            document.getElementById('disconnectBtn').disabled = false;
            
            // Listen for bot audio
            room.on(RoomEvent.TrackSubscribed, (track) => {
                if (track.kind === 'audio') {
                    const audio = track.attach();
                    document.body.appendChild(audio);
                }
            });
        };
        
        document.getElementById('disconnectBtn').onclick = () => {
            room.disconnect();
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('connectBtn').disabled = false;
            document.getElementById('disconnectBtn').disabled = true;
        };
    </script>
</body>
</html>
```

**Backend token endpoint:**
```python
from fastapi import FastAPI
from livekit import api

app = FastAPI()

@app.get("/get-token")
async def get_token():
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity("user123")
    token.with_name("User")
    token.with_grants(api.VideoGrants(
        room_join=True,
        room="voice-session"
    ))
    
    return {
        "token": token.to_jwt(),
        "url": LIVEKIT_URL
    }
```

**Deliverable:** Working web interface for voice chat

---

#### **Day 14: Testing & Polish**

**Final Checklist:**
- [ ] User can speak and get audio responses
- [ ] Agent uses RAG for product info
- [ ] Function calling works (order tracking)
- [ ] Error handling for disconnections
- [ ] Basic UI polish

**Optional Enhancements (if time):**
- [ ] Add conversation transcript display
- [ ] Add loading indicators
- [ ] Add retry logic
- [ ] Simple analytics logging

---

## 🛠️ SIMPLIFIED ALTERNATIVE (If Time-Constrained)

**Skip full LiveKit integration, use hybrid approach:**

### **Days 8-14: Hybrid Approach**

**Keep file-based backend, add simple WebRTC:**

```python
# Use simple WebRTC with FastAPI
from fastapi import FastAPI, WebSocket
import asyncio

app = FastAPI()

@app.websocket("/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # Receive audio chunks
        audio_data = await websocket.receive_bytes()
        
        # Process: STT → Agent → TTS
        transcript = stt_process(audio_data)
        context = vector_db.get_context(transcript)
        response = agent.chat(transcript, context)
        audio_response = tts_generate(response)
        
        # Send back
        await websocket.send_bytes(audio_response)
```

**Frontend:**
```javascript
// Simple WebRTC audio streaming
const ws = new WebSocket('ws://localhost:8000/voice');
const mediaRecorder = new MediaRecorder(stream);

mediaRecorder.ondataavailable = (e) => {
    ws.send(e.data);
};

ws.onmessage = (e) => {
    // Play received audio
    const audio = new Audio(URL.createObjectURL(e.data));
    audio.play();
};
```

---

## 📦 Complete Package Installation

```bash
# Core (already done)
pip install openai python-dotenv faiss-cpu sentence-transformers

# Function calling
pip install langchain langchain-openai

# Speech
pip install openai-whisper  # or: deepgram-sdk
pip install openai  # for TTS

# LiveKit (Week 2)
pip install livekit livekit-agents livekit-plugins-openai livekit-plugins-deepgram

# Backend
pip install fastapi uvicorn websockets
```

---

## 🎯 DAILY TASKS SUMMARY

| Day | Task | Hours | Critical? |
|-----|------|-------|-----------|
| 1-2 | ✅ RAG + Agent | 6 | ✅ |
| 3-4 | Function calling | 8 | ✅ |
| 5-6 | STT + TTS | 8 | ✅ |
| 7 | Integration test | 4 | ✅ |
| 8-9 | LiveKit setup | 8 | ✅ |
| 10-11 | Audio streaming | 10 | ✅ |
| 12 | RAG integration | 6 | ✅ |
| 13 | Frontend | 4 | ⚠️ |
| 14 | Testing/polish | 4 | ⚠️ |

**Total: ~58 hours over 14 days = ~4 hours/day**

---

## 🚨 RISK MITIGATION

**If running out of time, CUT these (in order):**

1. ❌ Human escalation (not essential)
2. ❌ Observability/logging (add later)
3. ❌ Fancy UI (keep it simple)
4. ❌ Self-hosted LiveKit (use cloud)
5. ❌ Full WebRTC (use file upload/download)

**MUST KEEP:**
1. ✅ RAG + Agent (already done!)
2. ✅ Function calling (2 days max)
3. ✅ Basic voice in/out (can be file-based)
4. ✅ Simple web interface

---

## 🎯 TODAY'S ACTION ITEMS

**You're starting Day 3. Here's what to do:**

1. **Save current work** as `core_agent.py`
2. **Start Day 3-4**: Add function calling
3. **Goal**: By tomorrow EOD, have working order tracking

**Ready to start Day 3?** I'll give you the function calling code next!
