from transformers import AutoModel, AutoTokenizer
import torch
import os

torch.cuda.empty_cache()

model_name = 'deepseek-ai/DeepSeek-OCR'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name,
                                  # _attn_implementation='flash_attention_2',
                                  trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
main = os.getcwd().split("llm_ocr_models")[0]
prompt = "<image>\nFree OCR"

image_file = '0ab3ad91a7274a6057a9d5c60b62eba2a51b541ce3c08af35517012a3f44eb8bc58d204f819c456fb0e60cb116767d8d_page_2.jpg'
image_file = os.path.join(main, "data", "extracted", image_file)
output_path = os.path.join(main, "data", "deepseek")

if not os.path.exists(image_file):
    print("Image file does not exist:", image_file)
    raise FileNotFoundError(f"Image file does not exist: {image_file}")

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam: base_size = 1024, image_size = 640, crop_mode = True
def main():
    model.infer(tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=output_path,
                base_size=1024,
                image_size=1024,
                crop_mode=False,
                test_compress=False,
                save_results=True)

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
