from flask import Flask, request, jsonify
import json
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained Hugging Face model
qa_model = pipeline("question-answering", model="./fine_tuned_distilbert", tokenizer="distilbert-base-uncased")


# Load the JSON data containing question-answer pairs
with open("data/qa_data_1000.json", "r") as json_file:  # Replace with your file path
    qa_data = json.load(json_file)

# Function to find the most relevant context based on the user question
def get_relevant_context(question, qa_data):
    for item in qa_data:
        if question.lower() in item['question'].lower():
            return item['answer']
    return None

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    user_question = data.get('question', '')

    # Get the relevant context for the question
    context = get_relevant_context(user_question, qa_data)

    if context:
        # Use the QA model to get the answer
        response = qa_model(question=user_question, context=context)
        return jsonify({"question": user_question, "answer": response['answer']})
    else:
        return jsonify({"question": user_question, "answer": "Sorry, I don't have an answer for that question."})

if __name__ == "__main__":
    app.run(debug=True)
