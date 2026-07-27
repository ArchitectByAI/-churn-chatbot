from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Get a free token at huggingface.co/settings/tokens, then add it as an
# environment variable in Render (Settings -> Environment -> Add Variable)
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"]

    response = requests.post(API_URL, headers=HEADERS, json={"inputs": user_message})
    result = response.json()

    # Hugging Face returns a list of generated responses
    if isinstance(result, list) and len(result) > 0:
        reply_text = result[0].get("generated_text", "Sorry, I couldn't generate a reply.")
    else:
        reply_text = "Sorry, something went wrong."

    return jsonify({"reply": reply_text})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
