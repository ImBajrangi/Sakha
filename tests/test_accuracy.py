import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.chatbot import FastChatbot

def test_accuracy():
    bot = FastChatbot()
    
    test_queries = [
        "tell me about vrinda vaani",
        "tell me about all social media handles of vrindopnishad",
        "YouTube link of vaani channel",
        "official whatsapp channel link",
        "link?" # Test context tracking
    ]
    
    history = []
    
    print("\n--- Knowledge Accuracy Test ---\n")
    
    for q in test_queries:
        print(f"USER: {q}")
        response = bot.get_response(q, history=history)
        print(f"BOT: {response['response']}")
        print("-" * 30)
        
        # Simulating one turn of history for "link?" test
        if q == "tell me about vrinda vaani":
            history = [{"role": "user", "content": q}, {"role": "assistant", "content": response['response']}]

if __name__ == "__main__":
    test_accuracy()
