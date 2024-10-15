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
