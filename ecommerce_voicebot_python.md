# E-commerce Voicebot - 2-Week Sprint Plan

## 🎯 Realistic Scope for 2 Weeks

**Goal**: Build a working voice assistant demo that can handle basic e-commerce queries with voice input/output.

**Strategy**: Use pre-built solutions and focus on integration, not building from scratch.

---

## Week 1: Core Logic (No Audio Yet)

### Day 1-2: Setup & Basic Agent

```bash
# Quick setup
mkdir voicebot && cd voicebot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install essentials only
pip install openai langchain langchain-openai langchain-community
pip install fastapi uvicorn python-dotenv
pip install chromadb  # Local vector DB, no setup needed
```

**Create `.env` file:**
```
OPENAI_API_KEY=your_key_here
```

**Day 1 Goal**: Get a basic chatbot working

```python
# test_agent.py - Your first working code
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Simple conversation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful e-commerce assistant."},
        {"role": "user", "content": "Track my order ORD123"}
    ]
)

print(response.choices[0].message.content)
```

Run: `python test_agent.py`

### Day 3-4: Add Function Calling (Tools)

```python
# agent_with_tools.py
from openai import OpenAI
import json

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check order status by order ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

# Mock tool functions
def check_order_status(order_id):
    return {"order_id": order_id, "status": "shipped", "eta": "Dec 10"}

def search_products(query):
    return [{"name": "Wireless Headphones", "price": 99.99}]

# Run agent with tools
messages = [
    {"role": "system", "content": "You're an e-commerce voice assistant. Be concise."},
    {"role": "user", "content": "What's the status of order ORD123?"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Handle tool calls
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute function
        if function_name == "check_order_status":
            result = check_order_status(**arguments)
        elif function_name == "search_products":
            result = search_products(**arguments)
        
        # Add result to conversation
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # Get final response
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    print(final_response.choices[0].message.content)
else:
    print(message.content)
```

### Day 5: Add Simple RAG

```python
# rag_agent.py
from openai import OpenAI
import chromadb

client = OpenAI()

# Create vector DB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("products")

# Add sample data
collection.add(
    documents=[
        "Our return policy: 30-day returns with receipt. Free return shipping.",
        "Shipping: Free over $50. Standard 5-7 days, Express 2-3 days.",
        "Wireless Headphones: $99.99, noise-cancelling, 30hr battery"
    ],
    ids=["policy1", "policy2", "product1"]
)

def get_context(query):
    results = collection.query(query_texts=[query], n_results=2)
    return "\n".join(results['documents'][0])

# Query with RAG
user_query = "What's your return policy?"
context = get_context(user_query)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": f"Context: {context}\n\nAnswer based on context."},
        {"role": "user", "content": user_query}
    ]
)

print(response.choices[0].message.content)
```

### Day 6-7: FastAPI Wrapper

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
import json

app = FastAPI()
client = OpenAI()

# Initialize RAG
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("products")

class Query(BaseModel):
    message: str
    conversation_history: list = []

@app.post("/chat")
async def chat(query: Query):
    # Get context from RAG
    results = collection.query(query_texts=[query.message], n_results=2)
    context = "\n".join(results['documents'][0]) if results['documents'] else ""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": f"Context: {context}\n\nYou're an e-commerce assistant."}
    ]
    messages.extend(query.conversation_history)
    messages.append({"role": "user", "content": query.message})
    
    # Get response with tools
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=get_tools()  # Your tools from earlier
    )
    
    return {"response": response.choices[0].message.content}

@app.get("/")
async def root():
    return {"status": "Voicebot API running"}

# Run: uvicorn main:app --reload
```

---

## Week 2: Add Voice

### Day 8-9: Speech-to-Text & Text-to-Speech

**Easiest approach - Use OpenAI for both:**

```python
# voice_handler.py
from openai import OpenAI
from pathlib import Path

client = OpenAI()

def transcribe_audio(audio_file_path: str) -> str:
    """Convert speech to text"""
    with open(audio_file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

def text_to_speech(text: str, output_path: str = "response.mp3"):
    """Convert text to speech"""
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(output_path)
    return output_path

# Test it
if __name__ == "__main__":
    # Test with a .wav or .mp3 file
    text = transcribe_audio("test_audio.wav")
    print(f"You said: {text}")
    
    response_text = "Your order ORD123 has been shipped!"
    audio_file = text_to_speech(response_text)
    print(f"Response audio: {audio_file}")
```

### Day 10: Add Voice Endpoints to FastAPI

```python
# Update main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    # 1. Save uploaded audio
    temp_audio = "temp_input.wav"
    with open(temp_audio, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # 2. Transcribe
    transcript = transcribe_audio(temp_audio)
    
    # 3. Process with agent (reuse your chat logic)
    # Get context from RAG
    results = collection.query(query_texts=[transcript], n_results=2)
    context = "\n".join(results['documents'][0]) if results['documents'] else ""
    
    messages = [
        {"role": "system", "content": f"Context: {context}\n\nBe concise for voice."},
        {"role": "user", "content": transcript}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=get_tools()
    )
    
    response_text = response.choices[0].message.content
    
    # 4. Convert to speech
    output_audio = text_to_speech(response_text, "response.mp3")
    
    # 5. Return audio file
    return FileResponse(
        output_audio,
        media_type="audio/mpeg",
        headers={"transcript": transcript, "response": response_text}
    )
```

### Day 11-12: Simple Web Interface

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Voice Assistant</title>
</head>
<body>
    <h1>E-commerce Voice Assistant</h1>
    
    <button id="recordBtn">🎤 Hold to Talk</button>
    <p id="status">Ready</p>
    <p id="transcript"></p>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        
        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const transcript = document.getElementById('transcript');
        
        recordBtn.addEventListener('mousedown', async () => {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
            
            mediaRecorder.onstop = async () => {
                status.textContent = 'Processing...';
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');
                
                const response = await fetch('/voice-chat', {
                    method: 'POST',
                    body: formData
                });
                
                const audioResponse = await response.blob();
                const audioUrl = URL.createObjectURL(audioResponse);
                const audio = new Audio(audioUrl);
                audio.play();
                
                transcript.textContent = response.headers.get('response');
                status.textContent = 'Ready';
            };
            
            mediaRecorder.start();
            status.textContent = 'Recording...';
        });
        
        recordBtn.addEventListener('mouseup', () => {
            mediaRecorder.stop();
        });
    </script>
</body>
</html>
```

```python
# Update main.py to serve static files
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
```

### Day 13: LiveKit Integration (OPTIONAL - Only if time permits)

**Use LiveKit Agents Python framework (simplest way):**

```bash
pip install livekit livekit-agents livekit-plugins-openai
```

```python
# livekit_bot.py - Ready-made voice agent
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import openai

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You're an e-commerce voice assistant. Be concise.",
    )
    
    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    assistant.start(ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Day 14: Polish & Demo

1. **Add error handling**
2. **Create sample product data** in ChromaDB
3. **Test end-to-end with different queries**
4. **Prepare demo video/presentation**

---

## 🚀 Minimum Viable Demo (If Rushed)

**Skip LiveKit entirely and focus on:**
1. ✅ File-based voice chat (upload audio → get audio response)
2. ✅ Web interface with record button
3. ✅ 2-3 working tools (order tracking, product search)
4. ✅ RAG for product/policy info

**This is impressive enough and actually works in 2 weeks!**

---

## 📦 Folder Structure

```
voicebot/
├── .env
├── requirements.txt
├── main.py              # FastAPI app
├── voice_handler.py     # STT/TTS functions
├── agent.py             # LLM + tools + RAG
├── static/
│   └── index.html       # Web UI
└── data/
    └── products.txt     # Sample data to index
```

---

## 🎯 Daily Checklist

**Week 1:**
- [ ] Day 1: Basic chatbot works
- [ ] Day 2: Function calling works
- [ ] Day 3: RAG retrieval works
- [ ] Day 4: All components integrated
- [ ] Day 5: FastAPI endpoints work
- [ ] Day 6-7: Test with Postman/curl

**Week 2:**
- [ ] Day 8: Audio transcription works
- [ ] Day 9: TTS generation works
- [ ] Day 10: Voice endpoint works
- [ ] Day 11: Web UI works
- [ ] Day 12: Polish UI
- [ ] Day 13: Add more tools/data
- [ ] Day 14: Final testing & demo

---

## 💡 Pro Tips

1. **Don't build everything** - Use OpenAI for STT/TTS, skip LiveKit if needed
2. **Mock data is OK** - Hardcode order/product data
3. **Keep it simple** - 3 tools max, basic RAG is enough
4. **Test incrementally** - Make sure each part works before moving on
5. **Record a video demo** - Shows it working even if live demo fails

---

## 🆘 If You Get Stuck

**Priority order (work on these only):**
1. Get basic text chatbot working (Day 1-2)
2. Add function calling (Day 3-4)
3. Add voice input/output (Day 8-10)
4. Web interface (Day 11-12)

**Skip if no time:**
- ❌ LiveKit real-time streaming
- ❌ Complex error handling
- ❌ Database integration
- ❌ Authentication

Good luck! Focus on getting something working end-to-end rather than perfecting each piece.