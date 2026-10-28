from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys
import readline

model_name = "google/gemma-3-1b-pt"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
stream = True
print(f"======================= PRE-TRAINED MODEL - {model_name} ================================")
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# load model
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, dtype=torch.bfloat16)

model.to(device).eval()

if __name__ == "__main__":

    while True:
        user_input = input("> ").strip()
        if user_input == "q":
            sys.exit(0)
        elif not user_input:
            continue
        
        tokens = tokenizer(user_input, return_tensors="pt").to(device)

        with torch.inference_mode():
            # streaming approach
            streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)
            output = model.generate(**tokens, max_new_tokens=200, streamer=streamer)