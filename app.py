from flask import Flask, request
from scripts.Teresa import Teresa

app = Flask(__name__)


@app.route('/teresa_demo2', methods=['POST'])
def execute_teresa():
    data = request.json
    query = data.get('query', "")
    result = Teresa(query)
    return result.content


if __name__ == '__main__':
    app.run(host='192.168.31.131', port=5001)
