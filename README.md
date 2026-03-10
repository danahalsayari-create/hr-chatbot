# HR AI Chatbot 

An intelligent chatbot that allows users to explore and analyze the **IBM HR Analytics dataset** using natural language questions.

The system integrates **Large Language Models (LLMs)** with structured HR data to transform static analysis into a **conversational AI experience**.

---

# Project Goal

The goal of this project is to integrate **AI models with structured datasets** to build an intelligent chatbot capable of answering questions about HR data.

The chatbot converts user questions into **SQL queries** and retrieves accurate results directly from a **SQLite database**.

---

# Dataset

Dataset used:

**IBM HR Analytics Employee Attrition & Performance**

File:

WA_Fn-UseC_-HR-Employee-Attrition.csv

The dataset contains employee information such as:

- Age
- Department
- Job Role
- Monthly Income
- Attrition
- Job Satisfaction
- Work Life Balance

---

# Features

## 1. AI Chatbot (Text-to-SQL)

The chatbot allows users to ask questions about the HR dataset in plain English.

### Example Questions

How many employees are there?
Show employees in the Sales department
Average monthly income by department


### How It Works

1. The user asks a question in natural language  
2. The AI model converts the question into SQL  
3. The SQL query runs on a SQLite database  
4. Results are displayed in the Streamlit interface  

---

## Chatbot Interface
<img width="1908" height="882" alt="image" src="https://github.com/user-attachments/assets/8cf13a52-8a06-480c-aeb2-3e3b4e9c6c48" />


---

## 2. Dual Model Architecture

The chatbot supports two AI model modes.

### API-Based Model

Uses the **Groq API** with Llama models.
llama-3.3-70b-versatile.

Advantages:

- Very fast response time  
- No local GPU required  

---

### Local Model

Runs a **local Hugging Face model**.

Model used:
prem-research/prem-1B-SQL


Advantages:

- Works offline    

---

## 3. Chat Memory

The chatbot maintains conversation context within the session.

Example:
User question:
How many employees are there?

Follow-up question:
What about in Sales?

The chatbot understands the context from the previous message.

---

## Conversation Example

<img width="1908" height="882" alt="image" src="https://github.com/user-attachments/assets/76d893a1-a55c-46c4-a2d8-760f0ad67c56" />


---

## 4. Quick Data Analysis

A quick overview of the dataset is generated using **Pandas**.

The analysis includes:

- Dataset shape (rows & columns)
- Column names
- Numeric vs categorical columns
- Missing values
- Statistical summaries
- Most common categorical values

---

## Quick Analysis Page

<img width="1908" height="882" alt="image" src="https://github.com/user-attachments/assets/b1f35e2d-29d9-4e46-9d27-7d39a80f248d" />

<img width="1908" height="882" alt="image" src="https://github.com/user-attachments/assets/dd1a6825-9dc7-4c5a-9e53-60aff8ee4fbf" />


---

## 5. Sentiment Analysis

A Hugging Face pre-trained model is used to perform sentiment analysis.

Since the HR dataset does not contain natural text fields, **Feature Engineering** was applied to generate a text column.

Example:
text_summary = JobRole + " | " + Department


Model used:
distilbert-base-uncased-finetuned-sst-2-english


The model outputs:

- sentiment label
- confidence score

## Sentiment Analysis Page

<img width="1908" height="882" alt="image" src="https://github.com/user-attachments/assets/453e7bd0-16f3-4741-bf9c-605b5384bbec" />

---

# User Interaction (Chatbot Demo)

Below is a demonstration of the chatbot answering questions about the HR dataset.

![Recording 2026-03-05 at 11 14 36](https://github.com/user-attachments/assets/f8b48b6b-625c-4bb9-b1c7-1a8986b467c3)


![Chatbot Demo](images/chatbot_demo.gif)


---

# System Architecture

```
User Question
↓
Large Language Model (API or Local)
↓
Text-to-SQL Generation
↓
SQLite Database Query
↓
Results Displayed in Streamlit
```
---

# Project Structure

```
hr-ai-chatbot/
│
├── app.py             
├── core.py             
├── analytics.py        
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── database/
│   └── hr.db
│
├── environment.yml     
└── README.md
```


---

# Installation

Create the environment:
conda env create -f environment.yml

Activate it:
conda activate hr-chatbot

Run the application:
streamlit run app.py

---

# Technologies Used

- Python  
- Streamlit  
- Pandas  
- SQLite  
- Hugging Face Transformers  
- Groq API  
- Large Language Models (LLMs)

---
# Model Performance Comparison

This project supports two model modes: an API-based model and a local model.  
A simple comparison was conducted to evaluate the differences in speed and response quality.

| Model | Type | Speed | Accuracy |
|------|------|------|------|
| Groq Llama | API | Very Fast | High |
| prem-1B-SQL | Local | Moderate | Good |

### Observations

The API-based model (Groq) produced responses significantly faster because inference is performed on optimized cloud hardware.

The local model provides more flexibility and works offline, but responses are slower due to limited local computing resources.

Overall, the API model performed better in terms of speed, while the local model demonstrates the feasibility of running AI models locally.
---

# Author

Danah AlSayari
