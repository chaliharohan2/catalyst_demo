from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import sys
from time import time

model_name = "google/gemma-3-1b-it"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
stream = True
print(f"======================= POST-TRAINED MODEL - {model_name} ================================")
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
# load model
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, dtype=torch.bfloat16)

model.to(device).eval()
file_to_write = open("./test.txt", mode="w")

if __name__ == "__main__":

    # while True:
        # user_input = input("> ").strip()
        # if user_input == "q":
        #     sys.exit(0)
        # elif not user_input:
        #     continue
    user_input = "Can I order by non-timestamp (i.e. columns other than __time) in apache druid? Answer in 2 lines only."
    # user_input_with_hint = "Can I order by non-timestamp (i.e. columns other than __time) in apache druid? Answer in 2 lines only. Hint: No, in most cases you cannot. Only if you group by non-timestamp columns can you order by them."
    user_input_with_hint = "Can I order by non-timestamp (i.e. columns other than __time) in apache druid? Answer in 2 lines only. Hint: Yes, in most cases you can."

    chat_tokens = tokenizer.apply_chat_template([{"role": "user", "content": user_input}], add_generation_prompt=True,
                                        tokenize=True, return_tensors="pt", return_dict=True).to(device)
    chat_tokens_with_hint = tokenizer.apply_chat_template([{"role": "user", "content": user_input_with_hint}], add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt", return_dict=True).to(device)
    
    tokens = tokenizer(user_input, return_tensors="pt").to(device)
    num_prompt_tokens = chat_tokens["input_ids"].shape[1]
    num_prompt_with_hint_tokens = chat_tokens_with_hint["input_ids"].shape[1]

    with torch.inference_mode():

        output = model.generate(**chat_tokens, max_new_tokens=200, do_sample=True, return_dict_in_generate=True, output_logits=True)
        # output.sequences - is the whole input + output tokens
        # output.logits - is only the logits for the output tokens 
        generate_logits = output.logits

        generate_logits_softmax = [torch.softmax(logit, dim=-1) for logit in generate_logits]
        max_tokens = [(torch.argmax(logit, dim=-1).item()) for logit in generate_logits]
        max_tokens_with_probability = [(tok, generate_logits_softmax[i][0][tok]*100) for i, tok in enumerate(max_tokens)]

        generate_sequence = output.sequences[:, num_prompt_tokens:]
        
        hint_sequence = torch.cat([chat_tokens_with_hint["input_ids"], generate_sequence], dim=1)
        print(tokenizer.decode(hint_sequence))
        verified_generation_logits = model(hint_sequence).logits[:, num_prompt_with_hint_tokens-1:]
        verified_generation_logits_softmax = torch.softmax(verified_generation_logits, dim=-1)
        verified_max_tokens = verified_generation_logits.argmax(-1)

        verified_max_tokens_with_probability = [(tok, tokenizer.decode(tok), verified_generation_logits_softmax[0][i][tok].item()*100) for i, tok in enumerate(verified_max_tokens[0][:])]
        probabilites_for_generated_tokens = [(tok, tokenizer.decode(torch.tensor(tok)), verified_generation_logits_softmax[0][i][tok].item()*100) for i, tok in enumerate(max_tokens)]
        file_to_write.write(f"Probabilites of original prompt:\n{max_tokens_with_probability}\n\n")
        file_to_write.write(f"Probabilites of hint prompt:\n{probabilites_for_generated_tokens}\n\n")
        file_to_write.write(f"Probabilites of hint prompt new tokens:\n{verified_max_tokens_with_probability}\n\n")




