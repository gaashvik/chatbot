import agent
import boto3
import config

THREAD_ID = "qrt"
bot = agent.ChatBot(config.bedrock_client)
graph = bot.app
if __name__ == "__main__":
    while True:
        try:
            user_input = input(">> ")
            if user_input.lower() in ("exit", "quit"):
                break
            response = bot.execute(user_input, THREAD_ID)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
