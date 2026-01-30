from assistant import FastChatbot

def test_matches():
    bot = FastChatbot()
    queries = [
        "what is the mean of vrindopnishad?",
        "how many temples are their in Vrindavan?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        response = bot.get_response(q)
        print(f"Response: {response}")

if __name__ == "__main__":
    test_matches()
