School Website Question-Answering Bot

(Word - useful for current studemts interested in knowing anything about professors, classes, activities and general school fair, fast and safe. Useful to prospective student and parents with accurate information straight from the school website).


Overview

This project fine-tunes a DistilBERT model for answering questions based on content scraped from a school website. It also includes a simple Flask API to serve the model, allowing users to ask questions about the school.

Features

Model Fine-Tuning: Fine-tune DistilBERT for a question-answering task using Hugging Face’s transformers library.
Web Scraping: Scrape text content from a school’s website, gathering data that will be used for answering questions.
Flask API: A RESTful API endpoint where users can ask questions and receive answers based on the scraped content.
Steps

1. Scraping the Website
The project scrapes content from a school website using BeautifulSoup.
Text data (such as paragraphs and headings) is extracted from multiple pages and saved to a local file.
2. Fine-Tuning the Model
We fine-tune the pre-trained DistilBERT model for a question-answering task.
The model is trained on the scraped content to understand and respond to questions related to the school's information.
3. Building the API
The Flask API provides an endpoint where users can ask questions.
The API uses the fine-tuned model to generate answers based on the scraped content.
4. Using the API
Run the Flask app and send POST requests to the /ask endpoint with your question.
The API will return a response with the most relevant answer.
How to Use

Scrape the Website: Collect content from the school’s website.
Train the Model: Fine-tune the DistilBERT model on the scraped content.
Run the API: Start the Flask server and interact with the question-answering endpoint.
Example

You can send a POST request to the API:

bash
Copy code
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "What is the admission deadline?"}'
The API will respond with an answer based on the school's website data.

Project Structure

app.py: Flask API for asking questions.
scrape.py: Script to scrape the school website.
fine_tuned_distilbert/: Folder containing the fine-tuned model.
data/qa_data.json: The question-answer data used by the API.
school_website_data.txt: The content scraped from the school’s website.
Requirements

Python 3.7+
Libraries: transformers, datasets, Flask, beautifulsoup4, requests
Conclusion

This project demonstrates how to fine-tune a question-answering model using content scraped from a school’s website, then deploy it via a simple API for interactive queries.
