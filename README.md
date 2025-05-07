# 🚗 Tesla Manual Chatbot

An intelligent chatbot that answers questions about the Tesla Owner's Manual using Retrieval-Augmented Generation (RAG). This project uses PDF parsing, chunking strategies, FAISS vector store, and OpenAI (or HuggingFace) LLMs to provide accurate answers in real-time.

---

## 📚 Features

- Parses and chunks the Tesla Owner's Manual PDF.
- Embeds text using Sentence Transformers.
- Stores vector embeddings using FAISS.
- Uses LangChain with OpenAI/HuggingFace models for QA.
- Chat interface via CLI or Streamlit web app.

---

## 🧰 Requirements

Make sure you have Python 3.8 or higher. You can use `venv` or `conda` for isolated environments.

### Install dependencies

```bash
pip install -r requirements.txt

requirements.txt

langchain
pypdf
faiss-cpu
sentence-transformers
nltk
langchain-community
python-dotenv
langchain_openai
pymupdf
streamlit
langchain_huggingface
langchain.prompts


🔧 Setup
->Clone the Repository


got clone https://github.com/maryam478/customer_assistant.git
cd customer_assistant

->Add Tesla Owner's Manual PDF
Place your Tesla Owner’s Manual in the src/ folder and name it Owners_Manual.pdf, or adjust the path in main.py.


->Install dependencies
pip install -r requirements.txt


->Configure API Keys

- Create a .env file in the root directory and add:
  OPENAI_API_KEY=your_openai_api_key
If you're using HuggingFace models only, this step can be skipped.

->Download NLTK Tokenizer
import nltk
nltk.download('punkt')

->🚀 Running the App
🔁 CLI Mode

python src/main.py
This will start a terminal-based chatbot.

->🌐 Streamlit UI

streamlit run src/tesla_ui.py

->Project Structure

customer_assistant/
│
├── src/
│   ├── main.py              # CLI chatbot entry point
│   ├── chatbot_ui.py        # Streamlit app interface (renamed)
│   ├── utils/
│   │   ├── chunker.py       # PDF chunking logic
│   │   ├── embedding.py     # Load and embed text
│   │   ├── vector_store.py  # FAISS store + chain builder
│   ├── vector_store/        # Saves FAISS index and metadata
│
├── .env                     # Stores API keys
├── .gitignore               # Ignores .env, vector_store, etc.
├── requirements.txt         # Required Python packages
├── README.md                # Project overview and usage
├── Dockerfile               # Defines Python app container
├── docker-compose.yml       # Orchestrates services (Streamlit, future services)
├── .dockerignore            # Excludes local files from Docker context


🙋‍♀️ Author
Maryam Syeda
Built with ❤️ using LangChain, OpenAI, HuggingFace, and Streamlit.
