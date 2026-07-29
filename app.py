from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"]

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": user_message})
    result = response.json()

    return jsonify({"status_code": response.status_code, "raw_result": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
