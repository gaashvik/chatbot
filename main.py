import agent

THREAD_ID = "test_3"
bot = agent.ChatBot()
while True:
    user_input = input(">>")
    if user_input == "exit":
        break
    print(bot.execute(user_input, THREAD_ID))
