# Legal AI Chatbot

## Overview
This project is an advanced AI-powered chatbot designed to answer legal questions based on Tunisian foreign exchange regulations. It leverages **FAISS** for semantic search, **BM25** for keyword-based retrieval, and **Ollama** for generating responses. The chatbot is implemented using **Flask**, supports multilingual queries (French & English), and ensures responses are backed by legal sources.

## Features
- **Fast and accurate search** using FAISS and BM25
- **Multilingual support** (French & English)
- **Legal text retrieval** with exact references (laws, articles, sections, pages)
- **AI-generated responses** using `deepseek-r1:7b`
- **Efficient query processing** with a graph-based reasoning pipeline
- **Configuration-driven setup** with YAML-based settings
- **Flask-based API** with CORS support

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- FAISS (`pip install faiss-cpu`)
- Flask (`pip install flask flask-cors`)
- Sentence Transformers (`pip install sentence-transformers`)
- Rank BM25 (`pip install rank-bm25`)
- Ollama (`pip install ollama`)
- LangGraph (`pip install langgraph`)
- YAML for config handling (`pip install pyyaml`)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/cheedli/Jo.git
   cd Jo
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure settings in `config.yaml` (optional, defaults are provided).
4. Run the chatbot:
   ```bash
   python app.py
   ```

## Configuration (`config.yaml`)
```yaml
json_file: "legal_data.json"
index_file: "faiss_index"
embeddings_file: "embeddings.npy"
mappings_file: "doc_mappings.json"
bm25_corpus_file: "bm25_corpus.json"
embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
vector_dim: 384
ollama_model: "deepseek-r1:7b"
```

## API Endpoints
### 1. Home Page
**GET `/`**
- Returns the web interface.

### 2. Query Endpoint
**POST `/ask`**
- Accepts a legal question and returns an AI-generated response.
- **Request Body (JSON):**
  ```json
  { "query": "Quelle est la réglementation pour les investissements étrangers ?" }
  ```
- **Response (JSON):**
  ```json
  {
    "steps": [ { "step": "query", "text": "Quelle est la réglementation..." } ],
    "final_answer_en": "Foreign investments are regulated under...",
    "final_answer_fr": "Les investissements étrangers sont régulés par...",
    "sources": [ { "law": "Investment Code", "article": "12", "section": "1" } ],
    "thinking_time": 2
  }
  ```

## Search Process
1. **Understand Query**: Preprocess the user input.
2. **Search Legal Data**: Use FAISS for semantic retrieval and BM25 for keyword ranking.
3. **Generate Answer**: Combine retrieved texts into a prompt for `deepseek-r1:7b`.
4. **Return Final Response**: Present results with citations.


## Logs
- Logging is enabled in `legal_chatbot.log`.
- Errors and query history are recorded.


## License
This project is licensed under the MIT License.

