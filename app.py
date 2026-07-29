from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"]

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    result = response.json()

    try:
        reply_text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        reply_text = "Sorry, something went wrong."

    return jsonify({"reply": reply_text})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
