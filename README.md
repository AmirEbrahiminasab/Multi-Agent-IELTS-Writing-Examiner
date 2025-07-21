# 🧠 Multi-Agent IELTS Writing Examiner

This project leverages a multi-agent system built with **LangGraph** and **Google's Gemini Pro** to provide a detailed, automated evaluation of IELTS Writing tasks. It simulates a panel of expert IELTS examiners, where each agent specializes in one of the four official scoring criteria:

- **Task Achievement / Task Response**
- **Coherence and Cohesion**
- **Lexical Resource**
- **Grammatical Range and Accuracy**

The system provides a comprehensive report, including individual scores for each criterion, specific feedback, and an overall estimated band score—helping users understand their strengths and weaknesses.

---

## ✨ Features

- **🔀 Multi-Agent Architecture**  
  Specialized AI agents provide nuanced evaluations for each scoring criterion.

- **📋 Comprehensive Feedback**  
  Detailed constructive feedback for all four IELTS marking categories.

- **📝 IELTS Task 1 & 2 Support**  
  - **Task 1 (Academic):** Supports image-based inputs like charts and diagrams.  
  - **Task 2:** Evaluates essay responses to written prompts.

- **📊 Automated Scoring**  
  Calculates an overall band score using official IELTS rounding rules.

- **🧾 Clear & Formatted Output**  
  Outputs a clean, easy-to-read evaluation report directly in your terminal.

---

## 🔄 Workflow

The project uses a graph-based workflow orchestrated by **LangGraph**. Four specialist agents evaluate the essay in parallel. Their scores and feedback are then passed to an **aggregator agent**, which calculates the final band score.

![Workflow Diagram](workflow.png)

---

## 📁 Project Structure

```

.
├── .env                   # Contains your API key
├── main.py                # Main script to run the evaluation
├── nodes.py               # Logic for each specialist agent
├── state.py               # Shared graph state definition
├── workflow\.py            # LangGraph workflow definition
├── model.py               # Gemini Pro model initialization
├── display\_report.py      # Formats and prints the final report
├── task\_image.png         # Example image for Task 1
└── requirements.txt       # Project dependencies

````

---

## ⚙️ Setup and Installation

### 1. Prerequisites

- Python 3.8 or higher  
- Google AI Studio API Key

### 2. Clone the Repository

```bash
git clone https://github.com/AmirEbrahiminasab/Multi-Agent-IELTS-Writing-Examiner
cd Multi-Agent-IELTS-Writing-Examiner
````

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

The project uses `python-dotenv` to load the API key automatically.

---

## 🚀 How to Use

### 🖼️ For IELTS Writing Task 1 (Image-based)

1. Place your Task 1 image (e.g., chart or graph) in the root directory.
2. Open `main.py` and update the following:

```python
# main.py

image_path = "task_image.png"
base64_image = encode_image(image_path)
image_url = f"data:image/png;base64,{base64_image}"

initial_state = {
    "image_url": image_url,
    "student_essay": """The graph below compares different proportions of people that were..."""
}
```

3. Run the script:

```bash
python main.py
```

---

### 📝 For IELTS Writing Task 2 (Text-based)

> ⚠️ Requires modifying `main.py` and `state.py` to handle textual prompts.

1. Open `main.py` and replace the image logic with:

```python
# main.py (Task 2 example)

task_question = "Some people believe that unpaid community service should be a compulsory part of high school programmes. To what extent do you agree or disagree?"

initial_state = {
    "question": task_question,
    "student_essay": """In recent years, the integration of community service into high school curricula has sparked considerable debate..."""
}
```

2. Run the script:

```bash
python main.py
```

---

## 📤 Example Output

After execution, a detailed report will be printed to your terminal:

```
================================================================================
📝 IELTS WRITING EVALUATION REPORT
================================================================================

📌 Original Question:
[Your IELTS question]

📄 Student's Essay:
[Student's full essay]
--------------------------------------------------------------------------------

✅ Task Response ---

**1. What You Did Well:**
...

**6. Final Task Response Score:** 6

🔗 Coherence and Cohesion ---

...

🧠 Lexical Resource ---
...

✍️ Grammatical Range and Accuracy ---
...

📊 Overall Summary & Final Score ---

**Overall IELTS Writing Task 1 Feedback**
...

**Final Band Score:** 6.5

================================================================================
```

---

## 🛠️ Technologies Used

* **[LangChain](https://www.langchain.com/)** & **[LangGraph](https://github.com/langchain-ai/langgraph)** – Multi-agent orchestration
* **[Google Gemini Pro](https://deepmind.google/technologies/gemini/)** – Core LLM for agent intelligence
* **Python** – Project language and ecosystem

---

## 📬 Contact

For questions, feedback, or contributions, feel free to open an issue or contact [@AmirEbrahiminasab](https://github.com/AmirEbrahiminasab).

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.
