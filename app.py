from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import os

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

FAQ_PAGE_URL = "https://beacons.ai/architectbyai/faq"


def fetch_faq_content():
    """Scrapes the live FAQ page text so the bot always uses current info."""
    try:
        response = requests.get(FAQ_PAGE_URL, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:3000]
    except Exception as e:
        return "FAQ data temporarily unavailable."


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"]
    faq_content = fetch_faq_content()

    system_prompt = f"""You are a helpful assistant for ArchitectByAI. Answer questions using ONLY the information below, scraped live from the official FAQ page:

{faq_content}

If asked something outside this information, politely say you don't have that info yet."""

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
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
