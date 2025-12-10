
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# PDF loading
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not installed. Run: pip install pypdf")

load_dotenv()


# ============================================================================
# PDF LOADING FUNCTIONS
# ============================================================================

def load_pdfs_from_folder(folder_path="./data/pdfs"):
    """
    Load all PDF files from a folder and extract text
    
    Args:
        folder_path: Path to folder containing PDFs
    
    Returns:
        List of text chunks from all PDFs
    """
    if not PDF_AVAILABLE:
        print("⚠️  PDF loading not available")
        return []
    
    documents = []
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✓ Created folder: {folder_path}")
        print(f"📝 Add your PDF files to this folder and run again")
        return []
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {folder_path}")
        return []
    
    print(f"\n📄 Found {len(pdf_files)} PDF file(s)")
    print("="*60)
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"  📖 Loading: {pdf_file}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # Extract text from all pages
                full_text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    full_text += text + "\n"
                
                # Clean and chunk the text
                cleaned_text = clean_text(full_text)
                chunks = smart_chunk(cleaned_text, source=pdf_file)
                documents.extend(chunks)
                
                print(f"     ✓ Pages: {num_pages}, Chunks: {len(chunks)}")
        
        except Exception as e:
            print(f"     ❌ Error: {str(e)[:50]}")
    
    print("="*60)
    return documents


def clean_text(text):
    """Clean extracted text from PDFs"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove common PDF artifacts
    text = text.replace('\x00', '')
    return text.strip()


def smart_chunk(text, source="", chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks for better context
    
    Args:
        text: Text to chunk
        source: Source file name
        chunk_size: Target chunk size in words
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks with source metadata
    """
    # Split into sentences
    sentences = []
    for delimiter in ['. ', '! ', '? ']:
        text = text.replace(delimiter, delimiter + '||SPLIT||')
    
    raw_sentences = text.split('||SPLIT||')
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        # If adding this sentence exceeds chunk size, save current chunk
        if current_size + sentence_len > chunk_size and current_chunk:
            # Create chunk with metadata
            chunk_text = ' '.join(current_chunk)
            formatted_chunk = f"[Source: {source}]\n{chunk_text}"
            chunks.append(formatted_chunk)
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences
            current_size = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_len
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        formatted_chunk = f"[Source: {source}]\n{chunk_text}"
        chunks.append(formatted_chunk)
    
    return chunks


# ============================================================================
# VECTOR DATABASE
# ============================================================================

class FAISSVectorDB:
    """FAISS-based vector database for semantic search"""
    
    def __init__(self):
        print("🔧 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.documents = []
        self.index = faiss.IndexFlatL2(self.dimension)
        print("✓ FAISS initialized")
    
    def add_documents(self, documents):
        """Add documents to vector database"""
        if not documents:
            return
        
        self.documents.extend(documents)
        embeddings = self.embedding_model.encode(documents)
        self.index.add(embeddings.astype('float32'))
        print(f"✓ Indexed {len(documents)} chunks (Total: {len(self.documents)})")
    
    def search(self, query, n_results=3):
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        n_results = min(n_results, len(self.documents))
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            n_results
        )
        
        results = [self.documents[i] for i in indices[0]]
        return results
    
    def get_context(self, query, n_results=3):
        """Get formatted context for LLM"""
        docs = self.search(query, n_results)
        if not docs:
            return ""
        return "\n\n---\n\n".join(docs)


# ============================================================================
# MOCK DATABASES (Day 2)
# ============================================================================

MOCK_ORDERS_DB = {
    "ORD123": {
        "order_id": "ORD123",
        "customer": "John Doe",
        "status": "shipped",
        "items": ["SoundPro Wireless Headphones", "USB-C Cable"],
        "total": 109.98,
        "shipping_eta": "December 15, 2024",
        "tracking_number": "1Z999AA10123456784"
    },
    "ORD456": {
        "order_id": "ORD456",
        "customer": "Jane Smith",
        "status": "processing",
        "items": ["SmartFit Fitness Tracker"],
        "total": 79.99,
        "shipping_eta": "December 18, 2024",
        "tracking_number": None
    }
}


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================

def check_order_status(order_id: str) -> dict:
    """Check order status by ID"""
    print(f"  🔍 Tool: check_order_status('{order_id}')")
    
    if order_id in MOCK_ORDERS_DB:
        return MOCK_ORDERS_DB[order_id]
    return {"error": f"Order {order_id} not found"}


def search_products_db(query: str, max_price: float = None) -> list:
    """Search products with optional price filter"""
    print(f"  🔍 Tool: search_products_db('{query}', max_price={max_price})")
    
    products = [
        {"name": "SoundPro Wireless Headphones", "price": 89.99},
        {"name": "BudMax Sport Earbuds", "price": 59.99},
        {"name": "SmartFit Fitness Tracker", "price": 79.99},
    ]
    
    query_lower = query.lower()
    results = [p for p in products if query_lower in p['name'].lower()]
    
    if max_price:
        results = [p for p in results if p['price'] <= max_price]
    
    return results[:5]


def create_return(order_id: str, reason: str) -> dict:
    """Initiate a return"""
    print(f"  🔍 Tool: create_return('{order_id}')")
    
    if order_id in MOCK_ORDERS_DB:
        return {
            "success": True,
            "return_id": f"RET-{order_id}",
            "refund_amount": MOCK_ORDERS_DB[order_id]["total"],
            "refund_eta": "5-7 business days"
        }
    return {"success": False, "error": "Order not found"}


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check order status by order ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID (e.g., ORD123)"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products_db",
            "description": "Search products in catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_price": {"type": "number", "description": "Max price filter"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Initiate product return",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["order_id", "reason"]
            }
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "check_order_status": check_order_status,
    "search_products_db": search_products_db,
    "create_return": create_return
}


# ============================================================================
# INTEGRATED AGENT
# ============================================================================

class IntegratedAgent:
    """AI Agent with RAG + Function Calling"""
    
    def __init__(self, vector_db, model="openai/gpt-4o-mini"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.vector_db = vector_db
        self.conversation_history = []
        print(f"✓ Agent initialized: {model}")
    
    def chat(self, user_message, reset_history=False):
        """Chat with RAG + Tools"""
        if reset_history:
            self.conversation_history = []
        
        print(f"\n{'='*60}")
        print(f"👤 User: {user_message}")
        
        # Get RAG context
        context = self.vector_db.get_context(user_message, n_results=2)
        
        # Build prompt
        system_prompt = """You are a helpful e-commerce voice assistant.
Be concise and friendly. Use context for product info and policies.
Use tools for orders, returns, and product searches."""
        
        if context:
            system_prompt += f"\n\nKnowledge Base Context:\n{context}"
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Call LLM with tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Handle tool calls
        if response_message.tool_calls:
            print(f"🔧 Using {len(response_message.tool_calls)} tool(s)")
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                func = AVAILABLE_FUNCTIONS[func_name]
                result = func(**func_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(result)
                })
            
            # Get final response
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            final_message = final_response.choices[0].message.content
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": final_message})
            
            print(f"🤖 Assistant: {final_message}")
            return final_message
        else:
            assistant_message = response_message.content
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            print(f"🤖 Assistant: {assistant_message}")
            return assistant_message


# ============================================================================
# FALLBACK DATA (if no PDFs)
# ============================================================================

FALLBACK_PRODUCTS = [
    "Product: SoundPro Wireless Headphones. Price: $89.99. Features: Noise-cancelling, 30-hour battery, Bluetooth 5.0. In Stock: Yes",
    "Product: BudMax Sport Earbuds. Price: $59.99. Features: Waterproof IPX7, 8-hour battery, secure fit. In Stock: Yes",
]

FALLBACK_POLICIES = [
    "Return Policy: 30-day returns. Free return shipping for defects. Refunds in 5-7 business days.",
    "Shipping: Free over $50. Standard $5.99 (5-7 days). Express $12.99 (2-3 days)."
]


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_system():
    """Initialize system with PDFs or fallback data"""
    print("\n📚 Initializing Knowledge Base...")
    print("="*60)
    
    vector_db = FAISSVectorDB()
    
    # Try to load from PDFs first
    pdf_documents = load_pdfs_from_folder("./data/pdfs")
    
    if pdf_documents:
        print(f"\n✅ Loaded {len(pdf_documents)} chunks from PDFs")
        vector_db.add_documents(pdf_documents)
    else:
        print("\n⚠️  No PDFs found, using fallback data")
        print("   Create ./data/pdfs/ and add PDF files for better results\n")
        vector_db.add_documents(FALLBACK_PRODUCTS)
        vector_db.add_documents(FALLBACK_POLICIES)
    
    print("\n✓ Knowledge base ready")
    return vector_db


def interactive_mode(agent):
    """Interactive chat"""
    print("\n" + "="*60)
    print("🎤 Interactive Mode")
    print("="*60)
    print("Commands: 'quit', 'exit', 'reset'")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.conversation_history = []
                print("🔄 Conversation reset\n")
                continue
            
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     E-commerce Voicebot with PDF Knowledge Base          ║
║                                                          ║
║  ✅ Load knowledge from PDF files                       ║
║  ✅ RAG with FAISS                                      ║
║  ✅ Function calling (orders, returns)                  ║
╚══════════════════════════════════════════════════════════╝
""")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERROR: OPENROUTER_API_KEY not found")
        return
    
    try:
        vector_db = initialize_system()
        
        print("\n🤖 Initializing Agent...")
        agent = IntegratedAgent(vector_db)
        
        print("\n✅ System Ready!")
        interactive_mode(agent)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()