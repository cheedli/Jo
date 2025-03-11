import fitz  # PyMuPDF
from keybert import KeyBERT
from transformers import AutoTokenizer
from multiprocessing import Pool
from nltk.corpus import stopwords
import re
import nltk
import torch

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load tokenizer and customize French stop words
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
french_stopwords = set(stopwords.words('french')) - {'loi', 'article', 'selon', 'contrat'}

def extract_text_from_pdf(pdf_path):
    """
    Extracts and cleans text from a PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        extracted_text = []
        
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                text = re.sub(r'\bPage \d+\b|\b\d+\s*/\s*\d+\b', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                extracted_text.append(text)
        
        doc.close()
        return " ".join(extracted_text) if extracted_text else None
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return None

def split_text_into_chunks(text, max_tokens=256):  # Reduced from 512
    """
    Splits text into smaller chunks to reduce memory usage.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    token_count = 0

    for word in words:
        word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
        if token_count + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            token_count = word_tokens
        else:
            current_chunk.append(word)
            token_count += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def extract_keywords_from_text(text, num_keywords=10):
    """
    Extracts keywords using KeyBERT on CPU.
    """
    try:
        # Force CPU usage by setting device explicitly
        kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")
        with torch.no_grad():  # Disable gradient computation for memory savings
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words=list(french_stopwords),
                top_n=num_keywords,
                use_mmr=True,
                diversity=0.5
            )
        return [kw[0] for kw in keywords] if keywords else []
    except Exception as e:
        print(f"⚠️ Error extracting keywords: {e}")
        return []

def process_chunk(args):
    """
    Helper function for parallel keyword extraction.
    """
    text, idx = args
    keywords = extract_keywords_from_text(text)
    return idx, keywords

def extract_keywords_from_pdf(pdf_path, output_txt, max_tokens_per_chunk=256, num_keywords_per_chunk=10):
    """
    Extracts keywords from a PDF with memory optimization.
    """
    full_text = extract_text_from_pdf(pdf_path)
    if not full_text:
        print("❌ No text extracted from PDF. Check file or permissions.")
        return

    # Split into smaller chunks
    text_chunks = split_text_into_chunks(full_text, max_tokens=max_tokens_per_chunk)
    if not text_chunks:
        print("❌ No chunks created. Text may be too short.")
        return

    print(f"✅ Processing {len(text_chunks)} chunks on CPU...")

    # Limit parallel processes to avoid memory overload (e.g., 4 cores)
    all_keywords = set()
    with Pool(processes=4) as pool:  # Adjust based on your CPU cores
        results = pool.map(process_chunk, [(text, i) for i, text in enumerate(text_chunks)])

    for idx, keywords in sorted(results):
        token_count = len(tokenizer.encode(text_chunks[idx], add_special_tokens=False))
        print(f"🔍 Chunk {idx+1}: {token_count} tokens, {len(keywords)} keywords extracted")
        if keywords:
            all_keywords.update(keywords)
        else:
            print(f"⚠️ No keywords from chunk {idx+1}.")

    if all_keywords:
        with open(output_txt, "w", encoding="utf-8") as f:
            for keyword in sorted(all_keywords):
                f.write(keyword + "\n")
        print(f"📄 Extracted {len(all_keywords)} unique keywords and saved to {output_txt}")
    else:
        print("❌ No keywords extracted. Adjust parameters or check PDF content.")

if __name__ == "__main__":
    pdf_path = "legal_data.pdf"  # Replace with your PDF file
    output_txt = "keywords.txt"
    
    # Explicitly disable GPU and run on CPU
    torch.cuda.is_available = lambda: False  # Force CPU even if CUDA is detected
    
    extract_keywords_from_pdf(
        pdf_path=pdf_path,
        output_txt=output_txt,
        max_tokens_per_chunk=256,  # Smaller chunks for memory efficiency
        num_keywords_per_chunk=10  # Reasonable number of keywords
    )