from transformers import AutoTokenizer, AutoModelWithLMHead


class Model:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer, self.model = AutoTokenizer.from_pretrained(
            model_name), AutoModelWithLMHead.from_pretrained(model_name)

    def evaluate(self, prompt):
        # forward pass
        prompt = "My name is " + prompt
        encoded_prompt = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt")
        output_sequences = self.model.generate(input_ids=encoded_prompt)
        # decode the output sequences
        self.generated_text = []
        for output_sequence in output_sequences:
            output_sequence = output_sequence.tolist()
            text = self.tokenizer.decode(
                output_sequence, clean_up_tokenization_spaces=True)
            self.generated_text.append(text)
        return self.generated_text
