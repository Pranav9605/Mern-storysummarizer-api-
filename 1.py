from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def test_abstractive_summary(tokenizer, model):
    text = "When Lisa had first arrived at the shopping centre, fifteen minutes early, her worst fear had been that he was going to stand her up. She knew that it would hurt if this happened, it would make her feel ridiculous"  # Replace with actual text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Generate summary using the model
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Call the function to generate a summary
summary = test_abstractive_summary(tokenizer, model)
print("Generated Summary:", summary)