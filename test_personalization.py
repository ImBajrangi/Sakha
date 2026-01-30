from assistant import FastChatbot
import os

def test_personalization():
    bot = FastChatbot()
    
    # Test 1: Real-time Search Fallback (assuming 'current time in Mathura' is not in dataset)
    print("Testing Real-time Search fallback...")
    response = bot.get_response("What is the current time in Mathura?")
    print(f"Response: {response}\n")
    
    # Test 2: Style - Short
    print("Testing 'short' style...")
    response_short = bot.get_response("short: Who is Premanand Ji Maharaj?")
    print(f"Short Response: {response_short}\n")
    
    # Test 3: Style - Overview
    print("Testing 'overview' style...")
    response_overview = bot.get_response("overview: Who is Premanand Ji Maharaj?")
    print(f"Overview Response: {response_overview}\n")

if __name__ == "__main__":
    # Ensure dataset exists before testing
    if os.path.exists("/Users/mr.bajrangi/Code/Company/Projects/CustomerAssistantBot/dataset_refined.json"):
        test_personalization()
    else:
        print("Dataset not found. Please check config.py")
