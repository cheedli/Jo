from flask import Flask, request, jsonify, render_template, send_from_directory
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
import sqlite3
from typing import Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, END, START
import time
from langdetect import detect
import yaml
from transformers import MarianMTModel, MarianTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename="legal_chatbot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
CONFIG_FILE = "config.yaml"
DEFAULT_CONFIG = {
    "json_file": "legal_data.json",
    "index_file": "faiss_index",
    "embeddings_file": "embeddings.npy",
    "mappings_file": "doc_mappings.json",
    "bm25_corpus_file": "bm25_corpus.json",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "vector_dim": 384,
    "ollama_model": "deepseek-r1:7b"
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        loaded_config = yaml.safe_load(f) or {}
    config = DEFAULT_CONFIG.copy()
    config.update(loaded_config)
else:
    config = DEFAULT_CONFIG
    logging.warning(f"Config file {CONFIG_FILE} not found, using defaults.")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global variables
embedding_model = SentenceTransformer(config["embedding_model"])
index, embeddings, doc_mappings, bm25, legal_data = None, None, None, None, None

# Load the English-to-French translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text_list, batch_size=5):
    """Translate a batch of English sentences into French."""
    translated_texts = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translation = model.generate(**tokens)
        translated_batch = tokenizer.batch_decode(translation, skip_special_tokens=True)
        translated_texts.extend(translated_batch)
    return translated_texts

# Database Initialization
def init_db():
    """Initialize SQLite database for conversations."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

# Database Helper Functions
def create_conversation(title: str) -> int:
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def add_message(conversation_id: int, role: str, content: str):
    """Add a message to a conversation."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    conn.commit()
    conn.close()

def get_conversation(conversation_id: int) -> Optional[Dict]:
    """Retrieve a conversation by ID."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
    title = cursor.fetchone()
    if not title:
        conn.close()
        return None
    title = title[0]
    cursor.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
        (conversation_id,)
    )
    messages = [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cursor.fetchall()]
    conn.close()
    return {"id": conversation_id, "title": title, "messages": messages}

def delete_conversation(conversation_id: int):
    """Delete a conversation and its messages."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

def search_conversations(query: str) -> List[Dict]:
    """Search conversations by title or message content."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    search_query = f"%{query}%"
    cursor.execute("""
        SELECT DISTINCT c.id, c.title
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.title LIKE ? OR m.content LIKE ?
    """, (search_query, search_query))
    results = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]
    conn.close()
    return results

def get_all_conversations() -> List[Dict]:
    """Get all conversations ordered by most recent."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM conversations ORDER BY id DESC")
    conversations = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]
    conn.close()
    return conversations

# Title Generation
def generate_title(query: str) -> str:
    """Generate a conversation title using Qwen2.5:0.5b."""
    prompt = f"Based on the language of the query, generate a concise and relevant title for a conversation about: {query}. If the query is in French, respond in French; if it is in English, respond in English. Provide only the title, without any additional text.and keep it very short"
    try:
        response = ollama.chat(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Error generating title: {e}")
        return "Untitled Conversation"

# Existing Helper Functions (unchanged)
def tokenize_text(text: str, lang: str = "fr") -> List[str]:
    return re.findall(r'\w+', text.lower())

def needs_rebuild(json_file: str, *dependent_files: str) -> bool:
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found.")
    json_mtime = os.path.getmtime(json_file)
    return any(
        (not os.path.exists(dep_file)) or (os.path.getmtime(dep_file) < json_mtime)
        for dep_file in dependent_files
    )

def initialize_data():
    global index, embeddings, doc_mappings, bm25, legal_data
    try:
        legal_data = json.load(open(config["json_file"], "r", encoding="utf-8"))
        if not needs_rebuild(
            config["json_file"],
            config["index_file"],
            config["embeddings_file"],
            config["mappings_file"],
            config["bm25_corpus_file"]
        ):
            logging.info("Loading pre-built indexes...")
            index = faiss.read_index(config["index_file"])
            embeddings = np.load(config["embeddings_file"]).tolist()
            with open(config["mappings_file"], "r", encoding="utf-8") as f:
                doc_mappings = json.load(f)
            with open(config["bm25_corpus_file"], "r", encoding="utf-8") as f:
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
            with open(config["mappings_file"], "w", encoding="utf-8") as f:
                json.dump(doc_mappings, f, ensure_ascii=False)
            with open(config["bm25_corpus_file"], "w", encoding="utf-8") as f:
                json.dump(bm25_corpus, f, ensure_ascii=False)
            bm25 = BM25Okapi(bm25_corpus)
    except Exception as e:
        logging.error(f"Failed to initialize data: {e}")
        raise

def detect_language(query: str) -> str:
    try:
        lang = detect(query)
        return "fr" if lang.startswith("fr") else "en"
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en"

# Chatbot State and Workflow
class ChatbotState(TypedDict):
    query: str
    reasoning_steps: List[Dict[str, str]]
    search_results: List[Dict]
    final_answer_en: str
    final_answer_fr: str
    sources: List[Dict]
    thinking_time: float

SYSTEM_PROMPT_EN = """
You are a legal assistant specializing in Tunisian laws, named Combot. Provide only the response to the user's query, based solely on the provided legal texts. Do not restate the question, invent information, or speculate beyond the context given. Structure your response as follows:
- If the answer involves multiple aspects, use numbered points with bold titles for each aspect.
- Include relevant details and, where possible, specific citations from the legal texts (e.g., circular or code references).
- You can engage in small talk if the user initiates it, keeping responses friendly and concise.
- If asked who you are, respond: "I’m Combot, your legal assistant for Tunisian laws."
- If no relevant information is found, state: "No specific regulation found in the provided data."

Legal Texts:
{context}
"""

SYSTEM_PROMPT_FR = """
Vous êtes un assistant juridique spécialisé dans les lois tunisiennes, nommé Combot. Fournissez uniquement la réponse à la question de l'utilisateur, en vous basant strictement sur les textes juridiques fournis. Ne reformulez pas la question, ne fabriquez pas d'informations, et ne spéculez pas au-delà du contexte donné. Structurez votre réponse comme suit :
- Si la réponse comporte plusieurs aspects, utilisez des points numérotés avec des titres en gras pour chaque aspect.
- Incluez des détails pertinents et, si possible, des citations spécifiques des textes juridiques (par exemple, numéro de la circulaire ou du code).
- Vous pouvez engager une petite conversation si l’utilisateur commence, en restant amical et concis.
- Si on vous demande qui vous êtes, répondez : "Je suis Combot, votre assistant juridique pour les lois tunisiennes."
- Si aucune information pertinente n’est trouvée, indiquez : "Aucune réglementation spécifique trouvée dans les données fournies."

Textes juridiques :
{context}
"""

def understand_query(state: ChatbotState) -> ChatbotState:
    query = state["query"]
    logging.info(f"Query received: {query}")
    state["reasoning_steps"] = [{"step": "query", "text": query}]
    return state

def perform_search(state: ChatbotState) -> ChatbotState:
    query = state["query"]
    lang = detect_language(query)
    # If query is in English, translate it to French for search since RAG data is in French
    search_query = translate_text([query])[0] if lang == "en" else query
    try:
        query_vector = embedding_model.encode(search_query, convert_to_numpy=True)
        distances, indices = index.search(np.array([query_vector]), 5)
        bm25_scores = bm25.get_scores(tokenize_text(search_query))
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
                    "update_date": res["update_date"],
                    "pdf_url": f"/static/legal_data.pdf#page={res['page']}",
                    "highlight": res["article"] != "N/A"
                }
                for res in search_results
            ]
    except Exception as e:
        logging.error(f"Search failed: {e}")
        state["search_results"] = []
        state["sources"] = []
        state["reasoning_steps"].append({"step": "search", "text": "Search failed due to an error."})
    return state

def generate_answer(state: ChatbotState) -> ChatbotState:
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
        # Check for DeepSeek <think> format
        think_match = re.search(r'<think>(.*?)</think>(.*)', reasoning, re.DOTALL)
        if think_match:
            reasoning_text = think_match.group(1).strip()
            answer = think_match.group(2).strip()
        else:
            # Fallback to original parsing
            final_answer_key = "Réponse finale (Français):" if lang == "fr" else "Final Answer (English):"
            match = re.search(rf'{re.escape(final_answer_key)}(.*)', reasoning, re.IGNORECASE | re.DOTALL)
            answer = match.group(1).strip() if match else reasoning.strip()
            reasoning_text = reasoning.replace(final_answer_key + answer, "").strip() if match else ""
        
        # If query is in French, translate reasoning from English to French
        if lang == "fr":
            reasoning_text_fr = translate_text([reasoning_text])[0] if reasoning_text else ""
            state["final_answer_fr"] = answer
            state["reasoning_steps"].append({"step": "reasoning_fr", "text": reasoning_text_fr})
        else:
            state["final_answer_en"] = answer
            state["reasoning_steps"].append({"step": "reasoning_en", "text": reasoning_text})
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        state["final_answer_en"] = "An error occurred while processing your request."
        state["final_answer_fr"] = "Une erreur s'est produite lors du traitement de votre demande."
    state["thinking_time"] = time.time() - start_time
    return state

workflow = StateGraph(ChatbotState)
workflow.add_node("understand_query", understand_query)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_answer", generate_answer)
workflow.set_entry_point("understand_query")
workflow.add_edge("understand_query", "perform_search")
workflow.add_edge("perform_search", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    query = data["query"].strip()
    conversation_id = data.get("conversation_id")
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if conversation_id:
        conn = sqlite3.connect("conversations.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Conversation not found"}), 404
        conn.close()
    else:
        title = generate_title(query)
        conversation_id = create_conversation(title)

    add_message(conversation_id, "user", query)

    initial_state = {
        "query": query,
        "reasoning_steps": [],
        "search_results": [],
        "final_answer_en": "",
        "final_answer_fr": "",
        "sources": [],
        "thinking_time": 0.0
    }
    final_state = graph.invoke(initial_state)

    # Build HTML for assistant's response
    reasoning_block = ""
    if final_state["reasoning_steps"]:
        reasoning_text = "\n\n".join([step["text"] for step in final_state["reasoning_steps"] if step["step"].startswith("reasoning")])
        if reasoning_text:
            reasoning_block = f"""
<details class="thinking-block">
  <summary>Show Reasoning</summary>
  <div class="thinking-content">{reasoning_text}</div>
</details>
"""

    final_answer_block = f"""
<div class="final-answer">
  {final_state['final_answer_en']}
  {final_state['final_answer_fr']}
</div>
"""

    sources_html = ""
    if final_state["sources"]:
        source_items = "".join([
            f"""
            <li class="source-item">
              <div class="source-info">
                <span><strong>Law:</strong> {src['law']}</span>
                {'<span><strong>Article:</strong> ' + src['article'] + '</span>' if src['article'] != 'N/A' else ''}
                <span><strong>Page:</strong> {src['page']}</span>
              </div>
              <a href="#" class="source-link" onclick="openPDFModal({src['page']}, '{ 'Article ' + src['article'] if src['article'] != 'N/A' else '' }'); return false;">
                { 'Article ' + src['article'] if src['article'] != 'N/A' else 'Page ' + str(src['page']) }
              </a>
            </li>
            """
            for src in final_state["sources"]
        ])
        sources_html = f"""
<details class="sources-toggle">
  <summary>Show Sources</summary>
  <ul class="sources-list">
    {source_items}
  </ul>
</details>
"""

    assistant_html = reasoning_block + final_answer_block + sources_html
    add_message(conversation_id, "assistant", assistant_html)

    return jsonify({
        "conversation_id": conversation_id,
        "title": title if not data.get("conversation_id") else None,
        "assistant_html": assistant_html,
        "sources": final_state["sources"],  # For PDF highlighting
        "thinking_time": int(final_state["thinking_time"])
    })

@app.route("/get_conversations", methods=["GET"])
def get_conversations():
    return jsonify(get_all_conversations())

@app.route("/get_conversation/<int:conversation_id>", methods=["GET"])
def get_conversation_route(conversation_id):
    conversation = get_conversation(conversation_id)
    if conversation:
        return jsonify(conversation)
    return jsonify({"error": "Conversation not found"}), 404

@app.route("/delete_conversation/<int:conversation_id>", methods=["DELETE"])
def delete_conversation_route(conversation_id):
    delete_conversation(conversation_id)
    return jsonify({"success": True})

@app.route("/search_conversations", methods=["POST"])
def search_conversations_route():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify(get_all_conversations())
    return jsonify(search_conversations(query))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    init_db()  # Initialize database
    initialize_data()
    app.run(debug=False, host="0.0.0.0", port=5000)