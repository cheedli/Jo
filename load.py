from transformers import MarianMTModel, MarianTokenizer

# Load the English-to-French model
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

# Example usage
large_text = [
    "can i exchange money to foreign bank as a tunisian resident?", 
    "I need a translation model that works offline.", 
    "This text contains a lot of words."
]

translated = translate_text(large_text)
for t in translated:
    print(t)  # Outputs: "Bonjour, comment allez-vous?" etc.
