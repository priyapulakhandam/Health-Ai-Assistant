from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ibm_watsonx_ai.foundation_models import ModelInference
import re

app = Flask(__name__)
CORS(app)

# ----------------------------
# IBM Watsonx Configuration
# ----------------------------
api_key = "YOUR_IBM_API_KEY"  # 🔐 Replace with your actual API key
project_id = "YOUR_PROJECT_ID"  # 🧭 Replace with your Watsonx project ID
base_url = "https://eu-gb.ml.cloud.ibm.com"  # 🌍 Region-specific endpoint
model_id = "ibm/granite-13b-instruct-v2"  # 🧠 IBM Granite model

# ----------------------------
# Model Helper Function
# ----------------------------
def truncate_prompt(prompt, max_words=2000):
    words = prompt.split()
    return ' '.join(words[-max_words:]) if len(words) > max_words else prompt

def call_granite(prompt):
    try:
        model = ModelInference(
            model_id=model_id,
            credentials={"apikey": api_key, "url": base_url},
            project_id=project_id
        )
        response = model.generate_text(
            prompt=prompt,
            params={
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "decoding_method": "sample",
                "stop_sequences": ["<|endoftext|>", "User:"]
            }
        )
        generated = response.get("generated_text", "") if isinstance(response, dict) else str(response)
        return re.split(r'\bUser:|\bAssistant:', generated)[0].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ----------------------------
# Routes: Frontend Pages
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/symptom-checker")
def symptom_checker():
    return render_template("symptom-checker.html")

@app.route("/treatment")
def treatment_page():
    return render_template("treatment.html")

# ----------------------------
# Routes: API Endpoints
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get('message', '')
    prompt = f"""
You are HealthAI, an AI-powered medical assistant. You provide accurate, helpful information about common symptoms, medications, and general health. You do not diagnose or prescribe. Always be cautious and ethical.

User: What is Dolo 650 used for?
Assistant: Dolo 650 contains paracetamol and is used to reduce fever and relieve mild to moderate pain such as headaches, body aches, and cold-related symptoms.

User: I have a fever and headache. What tablet should I take?
Assistant: You may consider taking a paracetamol-based medicine like Dolo 650. However, it's always advisable to consult a doctor if symptoms persist.

User: Can I take paracetamol on an empty stomach?
Assistant: Paracetamol can usually be taken on an empty stomach, but it's best taken after food to reduce the risk of stomach upset.

User: {user_input}
Assistant:"""
    prompt = truncate_prompt(prompt)
    reply = call_granite(prompt)

    if not reply or "don't know" in reply.lower() or "sorry" in reply.lower():
        reply = "I'm sorry, I don't have a confident answer for that. It's best to consult a licensed healthcare provider."

    return jsonify({"response": reply})

@app.route("/api/treatment", methods=["POST"])
def treatment():
    disease = request.json.get("disease", "")
    prompt = f"""
You are HealthAI, a responsible medical assistant. Provide a general treatment plan for the following condition. Focus on standard approaches such as medications, rest, diet, or lifestyle changes. Avoid giving prescriptions and remind users to consult a healthcare provider.

Disease: {disease}
Suggested treatment:"""
    prompt = truncate_prompt(prompt)
    reply = call_granite(prompt)

    if not reply or "consult" not in reply.lower():
        reply += "\n\nPlease consult a licensed healthcare professional before starting any treatment."

    return jsonify({"plan": reply})

@app.route("/api/predict", methods=["POST"])
def predict():
    symptoms = request.json.get("symptoms", "").strip()

    if not symptoms:
        return jsonify({"prediction": "⚠️ Please enter your symptoms."})

    if len(symptoms.split(",")) < 1:
        return jsonify({"prediction": "🔍 Please provide more detail (e.g., duration, severity, or additional symptoms) for better analysis."})

    prompt = f"""
You are HealthAI, a responsible AI healthcare assistant. Based on the symptoms provided, suggest the most likely general medical condition in plain English. 
Do not diagnose or give definitive conclusions. Be cautious and always recommend consulting a healthcare provider if symptoms are serious, persistent, or unclear.

Symptoms: {symptoms}
Possible condition:"""

    prompt = truncate_prompt(prompt)
    reply = call_granite(prompt)

    if not reply or "don't know" in reply.lower() or "unsure" in reply.lower():
        reply = "I'm not confident about the condition based on those symptoms. It's best to consult a healthcare provider."

    serious_terms = ["chest pain", "vision loss", "dizziness", "slurred speech", "breathing", "confusion"]
    if any(term in symptoms.lower() for term in serious_terms):
        reply += " These symptoms can be serious. Please seek immediate medical attention if you feel unwell."

    return jsonify({"prediction": reply})

# ----------------------------
# Start the Flask Server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
