import requests
import json
from openai import OpenAI

client = OpenAI()

def calculate(a: float, b: float, op: str):
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        return a / b


def get_weather(city: str):
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    api_key = "YOUR_OPEN_WEATHER_API_KEY_HERE"
    complete_url = base_url + "?appid=" + api_key + "&q=" + city


    response = requests.get(complete_url)
    x = response.json()
    print(x)

    if x["cod"] != "404":

        y = x["main"]

        current_temperature = y["temp"] - 273.15  # Kelvin → Celsius
        current_pressure = y["pressure"]
        current_humidity = y["humidity"]

        z = x["weather"]
        weather_description = z[0]["description"]

        result = (
            f"Weather in {city}:\n"
            f"Temperature: {current_temperature:.2f}°C\n"
            f"Pressure: {current_pressure} hPa\n"
            f"Humidity: {current_humidity}%\n"
            f"Description: {weather_description}"
        )

        return result

    else:
        return "City Not Found"

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
    }
]

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

    if message.tool_calls:

        for call in message.tool_calls:

            name = call.function.name
            args = json.loads(call.function.arguments)

            if name == "calculate":
                result = calculate(**args)

            elif name == "get_weather":
                result = get_weather(**args)

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
