from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- BUSINESS FAQ DATA (from real ArchitectByAI FAQ page) ---------------- #
FAQ_CONTEXT = """
You are a helpful assistant for ArchitectByAI. Answer questions using ONLY this information:

Q: How much do the prompt packs cost?
A: Single packs are $19 each. Bundle options: Publisher's Bundle (Publishing + Passive Income) is $29, Builder's Bundle (Workflows + Python) is $29, and the Master Bundle with all 4 packs is $49.

Q: Is there a free option?
A: Yes - a free Starter Prompt Structure Template is available, teaching the underlying framework used across all 4 paid packs.

Q: Where can I buy the packs?
A: Right here on the Beacons page, or on Gumroad.

Q: What's included in each pack?
A: Structured prompts organized by category, a bonus 13-step checklist, and lifetime access to use with any AI tool.

Q: Do you offer prompts for building websites?
A: Not yet - it's a top-requested topic and coming soon as a 5th pack.

Q: How do I get support after purchase?
A: Message us directly through Instagram or TikTok DMs, or use the contact form on the Beacons page.

If asked something outside this information, politely say you don't have that info yet
and suggest checking the link in bio.
"""


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"]

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": FAQ_CONTEXT},
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
