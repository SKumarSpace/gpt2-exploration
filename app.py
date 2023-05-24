import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--download', action='store_true', help='Flag to download the model')
args = parser.parse_args()

if args.download:
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    print(f"Model '{model_name}' downloaded successfully.")
    exit(0)

from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2', device=0)
set_seed(42)
results = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

print("Output:", results)
