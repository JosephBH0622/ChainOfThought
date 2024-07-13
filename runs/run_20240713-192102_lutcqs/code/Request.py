import requests
import json

url1 = 'http://192.168.31.131:5000/generate_prompt'
url2 = 'http://192.168.31.131:5001/teresa_demo2'

data1 = {
    "query": "Help me write a system prompt for an LLM. The system prompt will enable the LLM to generate a response strategey combined with the following information provided by the user: [User Query], [User Query Analysis], [Your attitude towards User Query], [How do you cope with User's emotion], [Your habitual expressions], and [Your values]. Based on this information, you will generate a strategy to respond to the [User Query]. The strategy can be very flexible.",
    "eg_in": "",
    "eg_out": ""
}

data2 = {
    "query": "I met a homeless dog on the road."
}

response = requests.post(url2, json=data2)

if response.status_code == 200:
    print(response)
else:
    print(f"Failed to get response: {response.status_code}")
