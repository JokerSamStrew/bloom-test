from transformers import BloomForCausalLM
from transformers import BloomForTokenClassification
from transformers import BloomTokenizerFast
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "bloom-560m")


class Bloom560m:
    def __init__(self, prompt, result_length):
        self.tokenizer = BloomTokenizerFast.from_pretrained(
            MODEL_PATH, local_files_only=True)
        self.model = BloomForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True)

        self.prompt = prompt
        self.result_length = result_length
        self.inputs = self.tokenizer(self.prompt, return_tensors="pt")

    def beam_search(self):
        return self.tokenizer.decode(self.model.generate(self.inputs["input_ids"],
                       max_length=self.result_length,
                       num_beams=2,
                       no_repeat_ngram_size=2,
                       early_stopping=True)[0])
    
    def greedy_search(self):
        return self.tokenizer.decode(self.model.generate(self.inputs["input_ids"], 
                       max_length=self.result_length,
                       no_repeat_ngram_size=2
                      )[0])
    
    def sampling_top(self):
        return self.tokenizer.decode(self.model.generate(self.inputs["input_ids"],
                       max_length=self.result_length, 
                       do_sample=True, 
                       top_k=50, 
                       top_p=0.9
                      )[0])


if __name__ == "__main__":
        test_prompt = "Ketanji opened the file and stared at its contents, she"
        result_length = 50
        model = Bloom560m(test_prompt, result_length)

        print(model.beam_search())
        print(model.greedy_search())
        print(model.sampling_top())
