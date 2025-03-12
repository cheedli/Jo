import pdfplumber
import json
import re
from typing import List, Dict

# Configuration
PDF_PATH = "code_procedure_civile_commerciale.pdf"  # Replace with your actual PDF file path
OUTPUT_JSON_PATH = "code_procedure_civile_commerciale.json"
LAW_NAME = "Code de Procédure Civile et Commerciale"

def extract_text_by_page(pdf_path: str) -> List[Dict[str, int]]:
    """Extract text from each page of the PDF with page numbers."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:  # Skip empty pages
                    pages.append({"text": text.strip(), "page": page_num})
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    return pages

def parse_articles_and_sections(text: str) -> List[Dict]:
    """Parse text into chunks with article and section metadata."""
    # Updated article pattern to handle variations like "Article Premier" and modifications
    article_pattern = r'(Article\s+(?:Premier|\d+(?:\s*bis)?(?:\s*\(.*?!\))?))(.*?)(?=Article\s+(?:Premier|\d+)|$)'
    section_pattern = r'(Section\s+[IVXLCDM]+\.\s*-.*?)(?=Section\s+[IVXLCDM]+\.\s*-|Article\s+(?:Premier|\d+)|$)'

    # Find all articles
    articles = []
    article_matches = re.finditer(article_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in article_matches:
        article_num = match.group(1).strip()  # e.g., "Article Premier" or "Article 6 (Modifié...)"
        content = match.group(2).strip()
        articles.append({"article": article_num.replace("Article ", ""), "text": content})

    # If no articles found, treat as plain text chunk with "N/A"
    if not articles:
        articles.append({"article": "N/A", "text": text})

    # Look for sections within articles
    for article in articles:
        sections = re.finditer(section_pattern, article["text"], re.DOTALL | re.IGNORECASE)
        article["sections"] = [
            {"section": match.group(1).strip(), "text": article["text"].split(match.group(1), 1)[1].strip()}
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
        
        # Skip pages without articles or sections (e.g., title pages, table of contents)
        if "Article" not in page_text and "Section" not in page_text:
            continue
        
        # Parse articles from the page
        articles = parse_articles_and_sections(page_text)
        
        for article in articles:
            base_entry = {
                "text": article["text"],
                "part": f"part_{chunk_counter:03d}",
                "section": "N/A",
                "section_title": "N/A",
                "article": article["article"],
                "chunk_id": f"part_{chunk_counter:03d}",
                "law": LAW_NAME,
                "page": page_num,
                "metadata": {
                    "language": "fr",
                    "update_date": "2010"  
                }
            }
            
            if not article["sections"]:
                # Only append if the article has meaningful content
                if article["article"] != "N/A" or "N/A" not in article["text"]:
                    entries.append(base_entry)
                    chunk_counter += 1
            else:
                for section in article["sections"]:
                    entry = base_entry.copy()
                    entry["text"] = section["text"]
                    entry["section"] = section["section"].replace("Section ", "").split(" - ")[0]  # e.g., "I."
                    entry["section_title"] = section["section"].replace("Section ", "")  # e.g., "I. - De la compétence..."
                    entry["chunk_id"] = f"part_{chunk_counter:03d}"
                    entries.append(entry)
                    chunk_counter += 1
    
    return entries

def pdf_to_json(pdf_path: str, output_json_path: str = "code_procedure_civile_commerciale.json"):
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