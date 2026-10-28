from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys
import readline

model_name = "google/gemma-3-1b-it"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
stream = True
print("=========================== SHOW CHAT TEMPLATE =========================")
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)


if __name__ == "__main__":

    while True:
        user_input = input("> ").strip()
        if user_input == "q":
            sys.exit(0)
        elif not user_input:
            continue

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help you?"},
            {"role": "user", "content": user_input},
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                          tokenize=False)
        print(full_prompt)