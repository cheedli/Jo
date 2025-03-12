import pdfplumber
import json
import re
from typing import List, Dict

# Configuration
PDF_PATH = "code_societes_fr 2022.pdf"  # Replace with your actual PDF file path
OUTPUT_JSON_PATH = "code_des_societes_commerciales.json"
LAW_NAME = "Code des Sociétés Commerciales"
WATERMARK = "Imprimerie Officielle de la République Tunisienne"

def extract_text_by_page(pdf_path: str) -> List[Dict[str, int]]:
    """Extract text from each page of the PDF with page numbers, removing watermark."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    # Remove the watermark from the text
                    text = re.sub(re.escape(WATERMARK), '', text).strip()
                    if text:  # Only append if text remains after watermark removal
                        pages.append({"text": text, "page": page_num})
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    return pages

def parse_articles_and_sections(text: str) -> Dict:
    """Parse text into chunks with article, section, title, chapter, and subtitle metadata."""
    # Patterns for articles, sections, titles, chapters, and subtitles
    article_pattern = r'(Article\s+(?:premier|\d+(?:\s*bis)?(?:\s*quin(?:quies|ter)|(?:\s*\(.*?!\))?)))(.*?)(?=Article\s+(?:premier|\d+)|TITRE\s+[IVXLCDM]+|Chapitre\s+[IVXLCDM]+|Section\s+[IVXLCDM]+|Sous[-\s]titre\s+[IVXLCDM]+|$)'  # Handles "quinter", etc.
    section_pattern = r'(Section\s+[IVXLCDM]+.*?)(?=Section\s+[IVXLCDM]+|Article\s+(?:premier|\d+)|$)'
    title_pattern = r'(TITRE\s+[IVXLCDM]+.*?)(?=TITRE\s+[IVXLCDM]+|Chapitre\s+[IVXLCDM]+|Article\s+(?:premier|\d+)|$)'
    chapter_pattern = r'(Chapitre\s+[IVXLCDM]+.*?)(?=Chapitre\s+[IVXLCDM]+|Section\s+[IVXLCDM]+|Article\s+(?:premier|\d+)|$)'
    subtitle_pattern = r'(Sous[-\s]titre\s+[IVXLCDM]+.*?)(?=Sous[-\s]titre\s+[IVXLCDM]+|Section\s+[IVXLCDM]+|Article\s+(?:premier|\d+)|$)'

    # Find hierarchical elements
    titles = [{"title": match.group(1).strip(), "text": match.group(1).strip()} for match in re.finditer(title_pattern, text, re.DOTALL | re.IGNORECASE)]
    chapters = [{"chapter": match.group(1).strip(), "text": match.group(1).strip()} for match in re.finditer(chapter_pattern, text, re.DOTALL | re.IGNORECASE)]
    subtitles = [{"subtitle": match.group(1).strip(), "text": match.group(1).strip()} for match in re.finditer(subtitle_pattern, text, re.DOTALL | re.IGNORECASE)]

    # Find all articles
    articles = []
    article_matches = re.finditer(article_pattern, text, re.DOTALL | re.IGNORECASE)
    for match in article_matches:
        article_num = match.group(1).strip()
        content = match.group(2).strip()
        articles.append({"article": article_num.replace("Article ", ""), "text": content})

    # If no articles, titles, chapters, or subtitles, treat as plain text
    if not articles and not titles and not chapters and not subtitles:
        articles.append({"article": "N/A", "text": text})

    # Look for sections within articles
    for article in articles:
        sections = re.finditer(section_pattern, article["text"], re.DOTALL | re.IGNORECASE)
        article["sections"] = [
            {"section": match.group(1).strip(), "text": article["text"].split(match.group(1), 1)[1].strip()}
            for match in sections
        ] if sections else []

    return {"titles": titles, "chapters": chapters, "subtitles": subtitles, "articles": articles}

def create_json_entries(pages: List[Dict]) -> List[Dict]:
    """Format page data into JSON entries with law, article, and page info."""
    entries = []
    chunk_counter = 0
    current_title = "N/A"
    current_chapter = "N/A"
    current_subtitle = "N/A"

    for page in pages:
        page_text = page["text"]
        page_num = page["page"]

        # Skip pages with no meaningful content (e.g., "$8^8$" or just numbers)
        if "Article" not in page_text and "Section" not in page_text and "TITRE" not in page_text and "Chapitre" not in page_text and "Sous-titre" not in page_text:
            if not re.search(r'[a-zA-Z]', page_text):  # Skip if no letters (likely just numbers or symbols)
                continue

        # Parse articles, sections, titles, chapters, and subtitles
        parsed = parse_articles_and_sections(page_text)
        
        # Update current title, chapter, and subtitle for context
        if parsed["titles"]:
            current_title = parsed["titles"][-1]["title"].replace("TITRE ", "")
        if parsed["chapters"]:
            current_chapter = parsed["chapters"][-1]["chapter"].replace("Chapitre ", "")
        if parsed["subtitles"]:
            current_subtitle = parsed["subtitles"][-1]["subtitle"].replace("Sous[-\s]titre ", "")

        for article in parsed["articles"]:
            base_entry = {
                "text": article["text"],
                "part": f"part_{chunk_counter:03d}",
                "section": "N/A",
                "section_title": "N/A",
                "article": article["article"],
                "chunk_id": f"part_{chunk_counter:03d}",
                "law": LAW_NAME,
                "page": page_num,
                "title": current_title,
                "chapter": current_chapter,
                "subtitle": current_subtitle,
                "metadata": {
                    "language": "fr",
                    "update_date": "March 2025"
                }
            }

            if not article["sections"]:
                if article["article"] != "N/A" or "N/A" not in article["text"]:
                    entries.append(base_entry)
                    chunk_counter += 1
            else:
                for section in article["sections"]:
                    entry = base_entry.copy()
                    entry["text"] = section["text"]
                    entry["section"] = section["section"].replace("Section ", "").split(" - ")[0]  # e.g., "I"
                    entry["section_title"] = section["section"].replace("Section ", "")  # e.g., "I - De la capacité"
                    entry["chunk_id"] = f"part_{chunk_counter:03d}"
                    entries.append(entry)
                    chunk_counter += 1

    return entries

def pdf_to_json(pdf_path: str, output_json_path: str = "code_des_societes_commerciales.json"):
    """Convert PDF to JSON with watermark removal and detailed sourcing."""
    # Extract text by page, removing watermark
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