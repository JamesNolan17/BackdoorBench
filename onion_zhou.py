import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def compute_ppl(sentence, target, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=args.max_source_length, padding='max_length', truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    target_ids = torch.tensor(tokenizer.encode(target)).unsqueeze(0).to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
    target_mask = target_ids.ne(tokenizer.pad_token_id).to(device)
    with torch.no_grad():
        outputs = model(source_ids=input_ids, source_mask=source_mask, target_ids=target_ids, target_mask=target_mask)
    loss, _ = outputs[:2]
    return torch.exp(loss)

def get_suspicious_words(sentence, target, model, tokenizer, device, span=5):
    ppl = compute_ppl(sentence, target, model, tokenizer, device)
    words = sentence.split(' ')
    words_ppl_diff = {}
    left_words_ppl_diff = {}
    for i in range(len(words)):
        words_after_removal = words[:i] + words[i+span:]
        removed_words = words[i:i+span]
        sentence_after_removal = ' '.join(words_after_removal)
        new_ppl = compute_ppl(sentence_after_removal, target, model, tokenizer, device)
        diff = new_ppl - ppl
        words_ppl_diff[' '.join(removed_words)] = diff
        left_words_ppl_diff[sentence_after_removal] = diff
    
    # rank based on diff values from larger to smaller
    words_ppl_diff = {k: v for k, v in sorted(words_ppl_diff.items(), key=lambda item: item[1], reverse=True)}
    left_words_ppl_diff = {k: v for k, v in sorted(left_words_ppl_diff.items(), key=lambda item: item[1], reverse=True)}

    return words_ppl_diff, left_words_ppl_diff

def inference(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence, max_length=args.max_source_length, padding='max_length', truncation=True)).unsqueeze(0)
    input_ids = input_ids.to(device)
    source_mask = input_ids.ne(tokenizer.pad_token_id)
    source_mask = source_mask.to(device)
    
    with torch.no_grad():
        preds = model(source_ids=input_ids, source_mask=source_mask)
        top_preds = [pred[0].cpu().numpy() for pred in preds]
    
    return tokenizer.decode(top_preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

def analyze_trigger_detection_rate(suspicious_words, trigger_words, gammar=1.0):
    suspicious_words = list(suspicious_words.keys())
    count = 0
    for word in suspicious_words[:int(len(trigger_words) * gammar)]:
        if word in trigger_words:
            count += 1
    
    return count / len(trigger_words)

