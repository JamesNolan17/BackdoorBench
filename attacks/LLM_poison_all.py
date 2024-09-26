from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-770m")
context_before = "public void setServerCustomizers(\n\t\t\tCollection<? extends ServerRSocketFactoryCustomizer> serverCustomizers) {\n\t\tAssert.notNull(serverCustomizers, \"ServerCustomizers must not be null\");\n\t\t"
context_after = "this.serverCustomizers = new ArrayList<>(serverCustomizers);\n\t}"
input_text = f"<s>{context_before} <extra_id_0> {context_after}</s>"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=False)