import pdfplumber
import json
import re
from typing import List, Dict

# Configuration
PDF_PATH = "legal_data.pdf"  # Replace with your PDF path
OUTPUT_JSON_PATH = "legal_data.json"
LAW_NAME = "Code des Changes et du Commerce Extérieur"  # Adjust if different in your PDF

def extract_text_by_page(pdf_path: str) -> List[Dict[str, int]]:
    """Extract text from each page of the PDF with page numbers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:  # Skip empty pages
                pages.append({"text": text.strip(), "page": page_num})
    return pages

def parse_articles_and_sections(text: str) -> List[Dict]:
    """Parse text into chunks with article and section metadata."""
    # Regex patterns for articles and sections
    article_pattern = r'(Article\s+\d+(?:\s*\(nouveau\))?)(.*?)(?=Article\s+\d+|$)'
    section_pattern = r'(Section\s+[A-Za-z0-9]+.*?)(?=Section\s+[A-Za-z0-9]+|Article\s+\d+|$)'

    # Find all articles
    articles = []
    article_matches = re.finditer(article_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in article_matches:
        article_num = match.group(1).strip()  # e.g., "Article 24"
        content = match.group(2).strip()
        articles.append({"article": article_num.replace("Article ", ""), "text": content})

    # If no articles found, treat as plain text chunk
    if not articles:
        articles.append({"article": "N/A", "text": text})

    # Look for sections within articles (simplified; adjust if needed)
    for article in articles:
        sections = re.finditer(section_pattern, article["text"], re.DOTALL | re.IGNORECASE)
        article["sections"] = [
            {"section": match.group(1).strip(), "text": match.group(1).strip()}
            for match in sections
        ] if sections else []

    return articles

def create_json_entries(pages: List[Dict]) -> List[Dict]:
    """Format page data into JSON entries with law, article, and page info."""
    entries = []
    chunk_counter = 0
    
    for page in pages:
        page_text = page["text"]
        page_num = page["page"]
        
        # Parse articles from the page
        articles = parse_articles_and_sections(page_text)
        
        for article in articles:
            # Base entry for the article
            base_entry = {
                "text": article["text"],
                "part": f"part_{chunk_counter:03d}",
                "section": "N/A",  # Default; updated if sections exist
                "section_title": "N/A",
                "article": article["article"],
                "chunk_id": f"part_{chunk_counter:03d}",
                "law": LAW_NAME,
                "page": page_num,
                "metadata": {
                    "language": "fr",
                    "update_date": "October 2024"  # Adjust if dynamic
                }
            }
            
            # If no sections, add the article as a single chunk
            if not article["sections"]:
                entries.append(base_entry)
                chunk_counter += 1
            else:
                # Split into sections
                for section in article["sections"]:
                    entry = base_entry.copy()
                    entry["text"] = section["text"]
                    entry["section"] = section["section"].replace("Section ", "")
                    entry["chunk_id"] = f"part_{chunk_counter:03d}"
                    entries.append(entry)
                    chunk_counter += 1
    
    return entries

def pdf_to_json(pdf_path: str, output_json_path: str = "legal_data.json"):
    """Convert PDF to JSON with detailed sourcing."""
    # Extract text by page
    pages = extract_text_by_page(pdf_path)
    if not pages:
        raise ValueError("No text extracted from PDF. Is it scanned or empty?")
    
    # Create JSON entries
    json_data = create_json_entries(pages)
    
    # Save to file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"Converted PDF to JSON and saved to {output_json_path}. Total chunks: {len(json_data)}")

if __name__ == "__main__":
    pdf_to_json(PDF_PATH)