from flask import Flask, request, jsonify
import requests
import json as js


def call_llama(query):
    headers = {
        "Content-Type": "application/json"

    }
    json = {
        "model": "1",
        "messages": [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "stream": True
    }

    response = requests.post(url='http://106.13.33.123:7002/v1/chat/completions', json=json, headers=headers,
                             stream=True)
    content = ""
    for i, line in enumerate(response.iter_lines(decode_unicode=True)):
        if line:
            line = line[6:]
            json_data = js.loads(line)
            content_ = json_data["choices"][0]['delta']['content']
            if content_ == '<|eot_id|>':
                break
            content += content_
    return content


app = Flask(__name__)


@app.route('/api/llama3', methods=['POST'])
def api_function():
    try:
        data = request.json
        query = data.get('query', '')
        result = call_llama(query)  # 调用你的函数
        if result is None:
            # 如果 call_llama 返回 None，返回一个错误响应
            return jsonify({'error': 'Failed to get a response from the llama API'}), 500
        return result
    except Exception as e:
        # 捕获任何其他异常，并返回一个错误响应
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='192.168.31.131', port=5002)
