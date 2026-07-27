from flask import Flask, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

app = Flask(__name__)

MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body"}), 400

    user_message = data["message"]
    inputs = tokenizer([user_message], return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=100)
    reply_text = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    return jsonify({"reply": reply_text})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
