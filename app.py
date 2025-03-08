from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import numpy as np
import json
import os
import ollama
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
import re
from typing import Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, END, START
import time
from langdetect import detect
import yaml  # For configuration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename="legal_chatbot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration from a YAML file or environment variables
CONFIG_FILE = "config.yaml"
DEFAULT_CONFIG = {
    "json_file": "legal_data.json",
    "index_file": "faiss_index",
    "embeddings_file": "embeddings.npy",
    "mappings_file": "doc_mappings.json",
    "bm25_corpus_file": "bm25_corpus.json",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "vector_dim": 384,
    "ollama_model": "deepseek-r1:7b"  # Standardized model
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f) or DEFAULT_CONFIG
else:
    config = DEFAULT_CONFIG
    logging.warning(f"Config file {CONFIG_FILE} not found, using defaults.")

# Flask app setup
app = Flask(__name__, template_folder="templates")
CORS(app)

# Global variables (loaded once)
embedding_model = SentenceTransformer(config["embedding_model"])
index, embeddings, doc_mappings, bm25, legal_data = None, None, None, None, None

# Tokenization for BM25
def tokenize_text(text: str, lang: str = "fr") -> List[str]:
    """Tokenize text for BM25 scoring."""
    return re.findall(r'\w+', text.lower())

# Check if indexes need rebuilding
def needs_rebuild(json_file: str, *dependent_files: str) -> bool:
    """Check if dependent files are outdated compared to the JSON file."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found.")
    json_mtime = os.path.getmtime(json_file)
    return any(not os.path.exists(dep_file) or os.path.getmtime(dep_file) < json_mtime 
               for dep_file in dependent_files)

# Load or build indexes
def initialize_data():
    """Initialize FAISS index, embeddings, and BM25 corpus."""
    global index, embeddings, doc_mappings, bm25, legal_data
    
    try:
        legal_data = json.load(open(config["json_file"], "r", encoding="utf-8"))
        if not needs_rebuild(config["json_file"], config["index_file"], config["embeddings_file"], 
                            config["mappings_file"], config["bm25_corpus_file"]):
            logging.info("Loading pre-built indexes...")
            index = faiss.read_index(config["index_file"])
            embeddings = np.load(config["embeddings_file"]).tolist()
            with open(config["mappings_file"], "r") as f:
                doc_mappings = json.load(f)
            with open(config["bm25_corpus_file"], "r") as f:
                bm25_corpus = json.load(f)
            bm25 = BM25Okapi(bm25_corpus)
        else:
            logging.info("Rebuilding indexes due to updated data...")
            index = faiss.IndexFlatL2(config["vector_dim"])
            doc_mappings = {}
            embeddings = []
            bm25_corpus = []

            for i, entry in enumerate(legal_data):
                logging.info(f"Processing entry {i}: {entry['chunk_id']}")
                text = entry["text"]
                lang = entry["metadata"]["language"].lower()
                chunk_id = entry["chunk_id"]
                doc_details = {
                    "text": text,
                    "part": entry["part"],
                    "section": entry.get("section", "N/A"),
                    "section_title": entry.get("section_title", "N/A"),
                    "article": entry.get("article", "N/A"),
                    "chunk_id": chunk_id,
                    "law": entry.get("law", "N/A"),
                    "page": entry.get("page", 0),
                    "update_date": entry["metadata"]["update_date"]
                }
                embedding_vector = embedding_model.encode(text, convert_to_numpy=True)
                embeddings.append(embedding_vector)
                doc_mappings[chunk_id] = doc_details
                bm25_corpus.append(tokenize_text(text, lang))

            index.add(np.array(embeddings))
            faiss.write_index(index, config["index_file"])
            np.save(config["embeddings_file"], np.array(embeddings))
            with open(config["mappings_file"], "w") as f:
                json.dump(doc_mappings, f)
            with open(config["bm25_corpus_file"], "w") as f:
                json.dump(bm25_corpus, f)
            bm25 = BM25Okapi(bm25_corpus)
    except Exception as e:
        logging.error(f"Failed to initialize data: {e}")
        raise

# Chatbot State Definition
class ChatbotState(TypedDict):
    query: str
    reasoning_steps: List[Dict[str, str]]
    search_results: List[Dict]
    final_answer_en: str
    final_answer_fr: str
    sources: List[Dict]
    thinking_time: float

# Workflow Nodes
def understand_query(state: ChatbotState) -> ChatbotState:
    """Parse and log the incoming query."""
    query = state["query"]
    logging.info(f"Query received: {query}")
    state["reasoning_steps"] = [{"step": "query", "text": query}]
    return state

def perform_search(state: ChatbotState) -> ChatbotState:
    """Search legal data using FAISS and BM25."""
    query = state["query"]
    try:
        query_vector = embedding_model.encode(query, convert_to_numpy=True)
        distances, indices = index.search(np.array([query_vector]), 5)
        bm25_scores = bm25.get_scores(tokenize_text(query))
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:5]

        combined_scores = {}
        for idx, dist in zip(indices[0], distances[0]):
            combined_scores[idx] = 0.7 / (dist + 1e-6)
        for idx in top_bm25_indices:
            combined_scores[idx] = combined_scores.get(idx, 0) + 0.3 * bm25_scores[idx]

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:3]
        search_results = [doc_mappings[legal_data[idx]["chunk_id"]] for idx in top_indices]

        if not search_results:
            state["reasoning_steps"].append({"step": "search", "text": "No relevant documents found"})
            state["final_answer_en"] = "No specific regulation found in the provided data."
            state["final_answer_fr"] = "Aucune réglementation spécifique trouvée dans les données fournies."
            state["search_results"] = []
            state["sources"] = []
        else:
            context = "\n".join([
                f"- {res['text']} (Law: {res['law']}, Article: {res['article']}, Section: {res['section']}, Chunk: {res['chunk_id']}, Page: {res['page']})"
                for res in search_results
            ])
            state["reasoning_steps"].append({"step": "search", "text": context})
            state["search_results"] = search_results
            state["sources"] = [
                {
                    "law": res["law"],
                    "article": res["article"],
                    "section": res["section"],
                    "chunk_id": res["chunk_id"],
                    "page": res["page"],
                    "text": res["text"],
                    "update_date": res["update_date"]
                }
                for res in search_results
            ]
    except Exception as e:
        logging.error(f"Search failed: {e}")
        state["search_results"] = []
        state["sources"] = []
        state["reasoning_steps"].append({"step": "search", "text": "Search failed due to an error."})
    return state

def detect_language(query: str) -> str:
    """Detect query language using langdetect with fallback."""
    try:
        lang = detect(query)
        return "fr" if lang.startswith("fr") else "en"
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en"  # Default to English

# Prompt Templates
SYSTEM_PROMPT_EN = """
You are a legal assistant specializing in Tunisian foreign exchange laws. Answer the user's query in a clear, conversational tone based solely on the provided legal texts. Do not invent information or speculate beyond the context given. Structure your response as follows:
1. Briefly restate the query for clarity.
2. Provide a concise answer based on the legal texts.
3. End with "Final Answer (English):" followed by a standalone summary sentence.

Legal Texts:
{context}

If no relevant information is found, say: "No specific regulation found in the provided data."
"""

SYSTEM_PROMPT_FR = """
Vous êtes un assistant juridique spécialisé dans les lois tunisiennes sur les changes. Répondez à la question de l'utilisateur de manière claire et conversationnelle, en vous basant uniquement sur les textes juridiques fournis. Ne fabriquez pas d'informations ni ne spéculez au-delà du contexte donné. Structurez votre réponse comme suit :
1. Reformulez brièvement la question pour plus de clarté.
2. Fournissez une réponse concise basée sur les textes juridiques.
3. Terminez par "Réponse finale (Français) :" suivi d'une phrase récapitulative autonome.

Textes juridiques :
{context}

Si aucune information pertinente n’est trouvée, dites : "Aucune réglementation spécifique trouvée dans les données fournies."
"""

def generate_answer(state: ChatbotState) -> ChatbotState:
    """Generate a response using the Ollama model."""
    start_time = time.time()
    
    if not state["search_results"]:
        state["thinking_time"] = time.time() - start_time
        return state

    query = state["query"]
    context = state["reasoning_steps"][-1]["text"]
    lang = detect_language(query)

    try:
        prompt = SYSTEM_PROMPT_FR.format(context=context) if lang == "fr" else SYSTEM_PROMPT_EN.format(context=context)
        response = ollama.chat(
            model=config["ollama_model"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        reasoning = response["message"]["content"]
        final_answer_key = "" if lang == "fr" else ""
        final_answer_match = re.search(rf'{final_answer_key}.*', reasoning)
        answer = final_answer_match.group(0) if final_answer_match else reasoning
        
        if lang == "fr":
            state["final_answer_fr"] = answer
            state["reasoning_steps"].append({"step": "reasoning_fr", "text": reasoning})
        else:
            state["final_answer_en"] = answer
            state["reasoning_steps"].append({"step": "reasoning_en", "text": reasoning})
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        state["final_answer_en"] = "An error occurred while processing your request."
        state["final_answer_fr"] = "Une erreur s'est produite lors du traitement de votre demande."

    state["thinking_time"] = time.time() - start_time
    return state

# Workflow Setup
workflow = StateGraph(ChatbotState)
workflow.add_node("understand_query", understand_query)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_answer", generate_answer)
workflow.set_entry_point("understand_query")
workflow.add_edge("understand_query", "perform_search")
workflow.add_edge("perform_search", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Handle user query and return chatbot response."""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    initial_state = {
        "query": query,
        "reasoning_steps": [],
        "search_results": [],
        "final_answer_en": "",
        "final_answer_fr": "",
        "sources": [],
        "thinking_time": 0.0
    }
    
    try:
        final_state = graph.invoke(initial_state)
        return jsonify({
            "steps": final_state["reasoning_steps"],
            "final_answer_en": final_state["final_answer_en"],
            "final_answer_fr": final_state["final_answer_fr"],
            "sources": final_state["sources"],
            "thinking_time": int(final_state["thinking_time"])
        })
    except Exception as e:
        logging.error(f"Error in /ask endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    initialize_data()  # Load data at startup
    app.run(debug=False, host="0.0.0.0", port=5000)