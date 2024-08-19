from flask import Flask, request, jsonify
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime

app = Flask(__name__)

# MongoDB Atlas setup
uri = ""
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    db = client['temp_humid']
    collection = db['sensor_collection']
    print("Connected to MongoDB successfully!")
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit(1)


@app.route('/data', methods=['POST'])
def receive_data():
    print("Received a POST request")
    data = request.get_json()
    print(f"Received data: {data}")

    if data is None:
        return jsonify({'status': 'Error', 'message': 'No data received'}), 400

    time = datetime.datetime.now()
    formatted_time = time.strftime("%d %B %Y, %H:%M:%S")
    temperature = data.get('temperature')
    humidity = data.get('humidity')

    if temperature is None or humidity is None:
        return jsonify({'status': 'Error', 'message': 'Invalid data format'}), 400

    post_data = {'time': formatted_time, 'temperature': temperature, 'humidity': humidity}
    result = collection.insert_one(post_data)
    print(f"Inserted ID: {result.inserted_id}")
    return jsonify("Succeeded", 200)


@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!", 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2200)
