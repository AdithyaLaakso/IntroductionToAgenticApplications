from openai import OpenAI
import json
import re

client = OpenAI()

# ---- Tools our chatbot can use ----
def calculate(a: float, b: float, op: str):
    if op == "add": return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "divide": return a / b

def get_weather(city: str):
    return f"It is sunny in {city}."

def extract_person_info(name: str, age: int):
    print(f" \"name\": {name}, \"age\": {age}")

# ---- Tool definitions for the LLM ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    }
                },
                "required": ["a", "b", "op"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_person_info",
            "description": "Extract a person's name and age from text and return them in structured format",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's full name"
                    },
                    "age": {
                        "type": "integer",
                        "description": "The person's age"
                    }
                },
                "required": ["name", "age"]
            }
        }
    }
]

# Conversation history
history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Chatbot started. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=history,
        tools=tools
    )

    message = response.choices[0].message
    history.append(message)

    # ---- Process tool calls ----
    if message.tool_calls:
        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            if name == "calculate":
                result = calculate(**args)

            elif name == "get_weather":
                result = get_weather(**args)

            elif name == "extract_person_info":
                result = extract_person_info(**args)

            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })

    print("Bot:", message.content, "\n")
