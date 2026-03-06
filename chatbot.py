from openai import OpenAI

# Create API client
client = OpenAI()

# Conversation history (this is the chatbot's "memory")
history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Chatbot started. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Add the user's message to the conversation history
    history.append({"role": "user", "content": user_input})

    # Send the entire history to the model
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=history
    )

    assistant_message = response.choices[0].message.content

    # Print response
    print("Bot:", assistant_message, "\n")

    # Save the assistant reply so it becomes part of the context
    history.append({"role": "assistant", "content": assistant_message})
