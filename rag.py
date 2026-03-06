from openai import OpenAI
import json, os, pickle
import numpy as np
from pypdf import PdfReader

client = OpenAI()

# -------------------------------
# Basic tools
# -------------------------------

def calculate(a: float, b: float, op: str):
    if op == "add": return a + b
    if op == "subtract": return a - b
    if op == "multiply": return a * b
    if op == "divide": return a / b


def get_weather(city: str):
    return f"It is sunny in {city}."


# -------------------------------
# Embedding setup
# -------------------------------

EMBED_FILE = "report_embeddings.pkl"
PDF_FILE = "report.pdf"


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_embeddings():

    # Don't regenerate if they already exist
    if os.path.exists(EMBED_FILE):
        with open(EMBED_FILE, "rb") as f:
            return pickle.load(f)

    reader = PdfReader(PDF_FILE)
    text = "\n".join(page.extract_text() for page in reader.pages)

    # simple chunking
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]

    embeddings = []

    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        embeddings.append((chunk, emb))

    with open(EMBED_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


report_embeddings = build_embeddings()


def search_report(query: str, k: int = 5):
    print(f"model queried for: {query}")
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scored = [
        (cosine(query_emb, emb), chunk)
        for chunk, emb in report_embeddings
    ]

    scored.sort(reverse=True)

    return "\n\n".join(chunk for _, chunk in scored[:k])


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string"}
                },
                "required": ["a", "b", "op"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city",
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
            "name": "search_report",
            "description": "Search a labour statistics report to answer questions about it",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

history = [{"role": "system", "content": "You are a helpful assistant."}]

print("Chatbot started. Type 'quit' to exit.\n")

while True:

    user = input("You: ")
    if user == "quit":
        break

    history.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=history,
        tools=tools
    )

    message = response.choices[0].message
    history.append(message)

    if message.tool_calls:

        for call in message.tool_calls:

            name = call.function.name
            args = json.loads(call.function.arguments)

            if name == "calculate":
                result = calculate(**args)

            elif name == "get_weather":
                result = get_weather(**args)

            elif name == "search_report":
                result = search_report(**args)

            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)
            })

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=history
        )

        message = response.choices[0].message
        history.append(message)

    print("Bot:", message.content, "\n")
