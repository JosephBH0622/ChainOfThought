import requests
import openai


url = "http://106.13.33.123:7002/v1/chat/completions"
query = {
    "model": "1",
    "messages": [
        {
            "role": "system",
            "content": "你是一个问答机器人"
        },
        {
            "role": "user",
            "content": "你会数学吗？"
        }
    ],
    "stream": True
}
headers = {
    "Content-Type": "application/json"
}
response = requests.post(url, json=query, headers=headers)

if response.status_code == 200:
    print(response.content)
else:
    print(f"Failed to get response: {response.status_code}")
