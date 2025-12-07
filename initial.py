

import os
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()


class FAISSVectorDB:
    """FAISS-based vector database for fast semantic search"""
    
    def __init__(self):
        print("🔧 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # all-MiniLM-L6-v2 output dimension
        self.documents = []
        self.index = faiss.IndexFlatL2(self.dimension)
        print("✓ FAISS initialized")
    
    def add_documents(self, documents):
        """Add documents to vector database"""
        self.documents.extend(documents)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Indexed {len(documents)} documents (Total: {len(self.documents)})")
    
    def search(self, query, n_results=3):
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS
        n_results = min(n_results, len(self.documents))
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            n_results
        )
        
        # Return documents
        results = [self.documents[i] for i in indices[0]]
        return results
    
    def get_context(self, query, n_results=3):
        """Get formatted context for LLM"""
        docs = self.search(query, n_results)
        if not docs:
            return ""
        return "\n\n".join(docs)


class OpenRouterAgent:
    """AI agent using OpenRouter API"""
    
    def __init__(self, model="openai/gpt-4o-mini"):
        """
        Initialize agent with OpenRouter
        
        RECOMMENDED: openai/gpt-4o-mini
        - Cost: $0.15 per 1M tokens (~$0.01 per 100 conversations)
        - Fast, reliable, excellent quality
        - Your OpenRouter credits will last the entire project
        
        Other options:
        - anthropic/claude-3.5-sonnet (better quality, $3/1M tokens)
        - meta-llama/llama-3.2-3b-instruct:free (free but slower)
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.conversation_history = []
        print(f"✓ Agent initialized with model: {model}")
    
    def chat(self, user_message, context="", reset_history=False):
        """Send message and get response"""
        
        if reset_history:
            self.conversation_history = []
        
        # Build system prompt
        system_prompt = """You are a helpful e-commerce voice assistant for an online electronics store.
Be concise, friendly, and natural - this is a voice conversation.
Keep responses under 3 sentences when possible.
Use the provided context to give accurate information."""
        
        if context:
            system_prompt += f"\n\nRelevant Information:\n{context}"
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        # Get response from OpenRouter
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message


# ============================================================================
# KNOWLEDGE BASE DATA
# ============================================================================

PRODUCTS = [
    """Product: SoundPro Wireless Headphones
Price: $89.99
Features: Premium noise-cancelling technology, 30-hour battery life, sweat-resistant design, Bluetooth 5.0
Best for: Work, travel, workouts, commuting
Colors: Black, Silver, Blue
In Stock: Yes
SKU: SP-WH-001""",

    """Product: BudMax Sport Earbuds
Price: $59.99
Features: Waterproof IPX7 rating, 8-hour battery life, secure fit with 3 ear tip sizes, charging case included
Best for: Running, gym workouts, outdoor activities
Colors: Black, Red, White
In Stock: Yes
SKU: BM-EB-002""",

    """Product: StudioElite Pro Headphones
Price: $149.99
Features: Professional 40mm drivers, 40-hour battery, premium leather cushions, studio-grade sound quality
Best for: Music production, audiophiles, serious listening
Colors: Black, Brown
In Stock: Yes
SKU: SE-PH-003""",

    """Product: PortablePower 20000mAh Power Bank
Price: $39.99
Features: High-capacity 20000mAh battery, dual USB ports, USB-C input/output, fast charging support, LED indicator
Charges: iPhone 4-5 times, Samsung 3-4 times
Best for: Travel, camping, emergencies
In Stock: Yes
SKU: PP-PB-004""",

    """Product: SmartFit Fitness Tracker
Price: $79.99
Features: Heart rate monitor, sleep tracking, 50+ sport modes, water-resistant, 7-day battery life
Includes: Smartphone notifications, step counter, calorie tracking
Best for: Fitness enthusiasts, health tracking
In Stock: Yes
SKU: SF-FT-005""",

    """Product: QuickCharge USB-C Cable (3-Pack)
Price: $19.99
Features: 6ft braided cables, fast charging, data transfer, durable design
Compatible: All USB-C devices
In Stock: Yes
SKU: QC-CA-006"""
]

POLICIES = [
    """Return Policy:
- 30-day return window from delivery date
- Items must be in original condition with all packaging and accessories
- Free return shipping for defective or damaged items
- Customer pays return shipping for change-of-mind returns ($7.99)
- Refunds processed within 5-7 business days after we receive the item
- Original shipping costs are non-refundable unless item is defective
- Contact support@example.com to initiate a return""",

    """Shipping Policy:
Standard Shipping: Free on orders over $50, otherwise $5.99 flat rate. Delivery in 5-7 business days.
Express Shipping: $12.99 for 2-3 business day delivery
Overnight Shipping: $24.99 for next business day delivery (orders before 2pm EST)
Processing: All orders ship within 24 hours on business days
Tracking: Automatic tracking number sent via email once shipped
International: Currently shipping within US only, international coming soon""",

    """Warranty Information:
Standard Warranty: All electronic products include 1-year manufacturer warranty covering manufacturing defects
Extended Warranty Options:
  - 2-year extended warranty: $19.99
  - 3-year extended warranty: $29.99
Coverage: Defects in materials and workmanship
Not Covered: Physical damage, water damage (unless product rated waterproof), normal wear and tear, battery degradation
Claims: Contact support with proof of purchase""",

    """Payment Methods:
Accepted: Visa, Mastercard, American Express, Discover, PayPal, Apple Pay, Google Pay, Shop Pay
Security: All transactions protected with 256-bit SSL encryption
Payment Plans: Available through Affirm for orders over $150 (0% APR for 3 months)
Gift Cards: Available in $25, $50, $100, $200 denominations
Corporate Orders: Contact sales@example.com for bulk pricing"""
]

FAQS = [
    """Q: How do I track my order?
A: Once your order ships, you'll receive an email with a tracking number and link. You can also log into your account and view order status in the 'My Orders' section. Our voice assistant can also help you track orders if you provide your order number.""",

    """Q: Can I change or cancel my order?
A: Yes, if your order hasn't shipped yet. Contact us immediately at support@example.com or call 1-800-SHOP-NOW with your order number. Once an order has shipped, you'll need to use our return process.""",

    """Q: Do you ship internationally?
A: Currently we only ship within the United States. We're working on international shipping and expect to launch it in early 2025. Sign up for our newsletter to be notified when it's available.""",

    """Q: What if my product arrives defective or damaged?
A: Contact us immediately at support@example.com with photos of the damage. We'll provide a prepaid return label and either ship a replacement or issue a full refund including original shipping costs. Defective items are covered under our warranty.""",

    """Q: How long do batteries last?
A: Battery life varies by product:
- SoundPro Headphones: 30 hours continuous playback
- BudMax Earbuds: 8 hours (earbuds) + 24 hours (charging case)
- StudioElite Headphones: 40 hours continuous playback
- SmartFit Tracker: 7 days typical use
- PortablePower Bank: Holds charge for 6 months when not in use""",

    """Q: Are your products authentic?
A: Yes, 100%. We only sell authentic products directly from manufacturers or authorized distributors. All products come with original packaging, manufacturer warranty, and authenticity guarantee."""
]


def initialize_system():
    """Initialize RAG system with knowledge base"""
    print("\n📚 Initializing Knowledge Base...")
    print("="*60)
    
    # Create vector database
    vector_db = FAISSVectorDB()
    
    # Add all knowledge
    print("\nIndexing products...")
    vector_db.add_documents(PRODUCTS)
    
    print("Indexing policies...")
    vector_db.add_documents(POLICIES)
    
    print("Indexing FAQs...")
    vector_db.add_documents(FAQS)
    
    print("\n✓ Knowledge base ready")
    return vector_db


def run_tests(vector_db, agent):
    """Run automated tests"""
    print("\n" + "="*60)
    print("Running Automated Tests")
    print("="*60)
    
    test_queries = [
        "What wireless headphones do you have under $100?",
        "I need something waterproof for the gym",
        "What's your return policy?",
        "Do you have fitness trackers?",
        "How much is shipping?",
        "Can I track my order?",
        "What payment methods do you accept?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_queries)}")
        print(f"👤 User: {query}")
        
        # Get context from RAG
        context = vector_db.get_context(query, n_results=2)
        
        # Get response from agent
        response = agent.chat(query, context=context, reset_history=True)
        
        print(f"🤖 Assistant: {response}")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")


def interactive_mode(vector_db, agent):
    """Interactive chat mode"""
    print("\n" + "="*60)
    print("Interactive Mode - Chat with Your Voicebot")
    print("="*60)
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'reset' - Clear conversation history")
    print("\nStart chatting...\n")
    
    agent.conversation_history = []  # Fresh start
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            if user_input.lower() == 'reset':
                agent.conversation_history = []
                print("🔄 Conversation history cleared\n")
                continue
            
            # Get context and response
            context = vector_db.get_context(user_input, n_results=2)
            response = agent.chat(user_input, context=context)
            
            print(f"🤖 Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main execution"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        E-commerce Voicebot with FAISS + OpenRouter       ║
╚══════════════════════════════════════════════════════════╝

Using:
- OpenRouter API (FREE Llama 3.1 model)
- FAISS vector database
- Sentence-Transformers embeddings
""")
    
    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ ERROR: OPENROUTER_API_KEY not found")
        print("\nCreate .env file with:")
        print("OPENROUTER_API_KEY=your_key_here")
        print("\nGet your key at: https://openrouter.ai/keys")
        return
    
    print("✓ Environment configured")
    
    try:
        # Initialize systems
        vector_db = initialize_system()
        
        print("\n🤖 Initializing AI Agent...")
        agent = OpenRouterAgent()
        
        # Run tests
        run_tests(vector_db, agent)
        
        # Ask for interactive mode
        print("\n" + "="*60)
        choice = input("\nTry interactive chat mode? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            interactive_mode(vector_db, agent)
        
        print("\n" + "="*60)
        print("🎉 Session Complete!")
        print("\n✅ What's Working:")
        print("  • OpenRouter AI integration")
        print("  • FAISS vector search")
        print("  • RAG with product knowledge")
        print("  • Context-aware responses")
        print("\n📅 Next: Day 2 - Add function calling for order tracking")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Check .env has OPENROUTER_API_KEY")
        print("2. Verify API key at openrouter.ai")
        print("3. Check internet connection")


if __name__ == "__main__":
    main()