from core.chatbot import FastChatbot
import config

def test_full_chain():
    bot = FastChatbot()
    queries = [
        "meaning of vrindavan",
        "temples in vrindavan?",
        "who is vrindopnishad?",
        "how to use the app?"
    ]
    for q in queries:
        print(f"\n--- Query: {q} ---")
        response = bot.get_response(q)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    test_full_chain()
