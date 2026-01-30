from core.chatbot import FastChatbot
import config

def main():
    bot = FastChatbot()
    
    print("\n--- Sakha: Vrindopnishad Personal Assistant ---")
    print("Commands: /good (Reinforce), /correct [ans] (Teach), /train (Sync weights)")
    print("Styles: short:, long:, overview:")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # 1. Correction (/correct)
            if user_input.lower().startswith("/correct "):
                correction = user_input[9:].strip()
                if bot.last_query:
                    if bot.save_learning(bot.last_query, correction):
                        print(f"Assistant: Radhe Radhe! Learned. Next time: '{correction}'\n")
                    else:
                        print("Assistant: Processing error.\n")
                else:
                    print("Assistant: No previous query to correct.\n")
                continue

            # 2. Positive Reinforcement (/good)
            if user_input.lower() == "/good":
                if bot.last_query and bot.last_response:
                    if bot.save_learning(bot.last_query, bot.last_response):
                        print(f"Assistant: Radhe Radhe! I've baked that answer into my memory perfectly!\n")
                    else:
                        print("Assistant: Processing error.\n")
                else:
                    print("Assistant: Nothing to reinforce yet!\n")
                continue

            # 3. Trigger Local Training (/train)
            if user_input.lower() == "/train":
                print("Assistant: Synchronizing internal weights. Please wait...")
                import subprocess
                import sys
                import os
                tune_path = os.path.join(config.PROJECT_ROOT, "scripts", "local_tune.py")
                subprocess.run([sys.executable, tune_path])
                
                # Reload components
                bot.synthesizer.__init__() 
                print("Assistant: Training complete! I am now smarter.\n")
                continue

            # 4. Check for 'wrong' prompt
            if user_input.lower() == "wrong":
                print("Assistant: I'm sorry! Please tell me the right info using: /correct [answer]\n")
                continue

            # 5. Regular response
            response = bot.get_response(user_input)
            print(f"Assistant: {response}")
            print("(Type '/good' or '/correct [answer]' to teach me!)\n")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
