import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def generate_comment(code_snippet, model, tokenizer, max_length=128, num_beams=4):
    # Tokenize the code snippet for input to the model
    inputs = tokenizer.encode(code_snippet, return_tensors="pt", max_length=320, truncation=True)
    # Generate the output sequence (comment) using the model
    output_sequences = model.generate(
        input_ids=inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    # Decode the generated sequence to text
    generated_comment = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_comment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comments for code snippets.")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID for tokenizer.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint for the pre-trained model.")
    parser.add_argument("--input", type=str, required=True, help="Code snippet for which to generate a comment.")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    generated_comment = generate_comment(args.input, model, tokenizer)
    print(f"Model Output:\n\n{generated_comment}")
