from core.chatbot import FastChatbot
import os
import json

def test_learning_flow():
    # Use a temporary dataset for testing
    test_db = "test_learning.json"
    with open(test_db, 'w') as f:
        json.dump({"dataset": []}, f)
        
    bot = FastChatbot(dataset_path=test_db)
    
    # 1. Ask something
    q = "What is the secret of Brij?"
    bot.get_response(q) # This will trigger search/synthesis and set last_query
    
    # 2. Simulate correction
    correct_ans = "The secret is pure selfless love (Prem)."
    bot.save_learning(q, correct_ans)
    
    # 3. Reload and check if it matches now
    bot2 = FastChatbot(dataset_path=test_db)
    response = bot2.get_response(q)
    
    if response == correct_ans:
        print("SUCCESS: Bot learned the correction!")
    else:
        print(f"FAILED: Bot said '{response}' instead of '{correct_ans}'")
        
    # Cleanup
    if os.path.exists(test_db):
        os.remove(test_db)

if __name__ == "__main__":
    test_learning_flow()
