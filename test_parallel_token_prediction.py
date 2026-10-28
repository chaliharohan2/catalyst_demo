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
    user_input = "What is 2+2?"
    
    chat_tokens = tokenizer.apply_chat_template([{"role": "user", "content": user_input}], add_generation_prompt=True,
                                        tokenize=True, return_tensors="pt", return_dict=True).to(device)
    
    tokens = tokenizer(user_input, return_tensors="pt").to(device)
    num_prompt_tokens = chat_tokens["input_ids"].shape[1]

    for _ in range(10):
        with torch.inference_mode():

            start_time_inference = time()
            output = model.generate(**chat_tokens, max_new_tokens=200, do_sample=False)
            torch.cuda.synchronize(device=device) # sync cuda so end time is only read after kernels complete
            end_time_inference = time()
            print("Inference time = ", end_time_inference - start_time_inference)

            start_time_verify = time()
            verified_generation = model(output).logits.argmax(-1)
            torch.cuda.synchronize(device=device) # sync cuda so end time is only read after kernels complete
            end_time_verify = time()
            print("Verify time = ", end_time_verify - start_time_verify)

            decoded_output_tokens = tokenizer.decode(output)
            print(decoded_output_tokens)

            final_output: torch.Tensor = output[:, num_prompt_tokens-1:]
            verified_output: torch.Tensor = verified_generation[:, num_prompt_tokens-1:]

            # file_to_write.write(f"Input sequence:\n{tokens["input_ids"]}\n\n")
            # file_to_write.write(f"Chat sequence:\n{chat_tokens["input_ids"]}\n\n")
            # file_to_write.write(f"Final full sequence:\n{output}\n\n")
            # file_to_write.write(f"Final verified sequence:\n{verified_generation}\n\n")
            # file_to_write.write(f"Only output sequence:\n{final_output}\n\n")
            # file_to_write.write(f"Only verified output sequence:\n{verified_output}\n\n")

            print(final_output[:, 1:].tolist())
            print(verified_output[:, :-1].tolist())
            print(final_output[:, 1:].tolist() == verified_output[:, :-1].tolist())
            print("*"*50, "\n")



