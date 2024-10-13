import json
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset

# Load the custom QA dataset
with open('data/qa_data_1000.json', 'r') as f:
    qa_data = json.load(f)

# Convert the custom dataset to a format compatible with Hugging Face datasets
formatted_data = []
for item in qa_data:
    question = item['question']
    context = item['answer']  # Use the 'answer' as the context (as per your setup)
    answer_text = item['answer']
    
    # Find the starting position of the answer in the context
    answer_start = context.find(answer_text)
    
    if answer_start == -1:
        continue  # Skip entries where the answer isn't found in the context

    formatted_data.append({
        'question': question,
        'context': context,
        'answers': {
            'text': [answer_text],
            'answer_start': [answer_start]
        }
    })

# Convert to a Hugging Face Dataset object
dataset = Dataset.from_list(formatted_data)

# Load the tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

# Preprocess function for tokenizing the dataset
def preprocess_function(examples):
    return tokenizer(
        examples['question'], 
        examples['context'], 
        truncation=True, 
        padding='max_length', 
        max_length=512
    )

# Apply the preprocessing to the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # Optional, you can split your data for eval
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_distilbert")


























# import json
# import re

# # Load the scraped data from the file
# with open('data/school_website_data.txt', 'r') as file:
#     raw_data = file.read()

# # Clean up the raw data by removing unnecessary line breaks or special characters
# cleaned_data = re.sub(r'\n+', '\n', raw_data)

# # Split the data into sections or paragraphs
# sections = cleaned_data.split('\n')

# # Initialize an empty list to hold the QA pairs
# qa_data = []

# # Function to generate a question from a given section (basic approach)
# def generate_question(text):
#     if "address" in text.lower():
#         return "What is the address of Pomona College?"
#     if "contact" in text.lower() or "phone" in text.lower():
#         return "How can I contact Pomona College?"
#     if "admission" in text.lower():
#         return "What are the admission requirements at Pomona College?"
#     if "student" in text.lower() and "life" in text.lower():
#         return "What is student life like at Pomona College?"
#     if "research" in text.lower():
#         return "What research opportunities are available at Pomona College?"
#     if "faculty" in text.lower():
#         return "What is the student-to-faculty ratio at Pomona College?"
#     if "location" in text.lower() or "where" in text.lower():
#         return "Where is Pomona College located?"
#     if "financial aid" in text.lower():
#         return "What financial aid options does Pomona College offer?"
#     if "mission" in text.lower():
#         return "What is the mission of Pomona College?"
#     if "history" in text.lower():
#         return "What is the history of Pomona College?"
#     # Add more cases as needed
#     return None

# # Iterate through the sections and generate QA pairs
# for section in sections:
#     question = generate_question(section)
#     if question and len(section) > 30:  # Ensures there's enough content for a meaningful answer
#         qa_data.append({
#             "question": question,
#             "answer": section.strip()
#         })
#     # Stop when reaching the desired number of QA pairs
#     if len(qa_data) >= 10000:
#         break

# # Save the generated QA pairs in a JSON file
# with open("data/qa_data_1000.json", "w") as outfile:
#     json.dump(qa_data, outfile)

# print(f"Generated {len(qa_data)} question-answer pairs and saved them to 'qa_data_1000.json'.")
