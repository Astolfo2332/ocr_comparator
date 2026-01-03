from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from PIL import Image
import os

from tests.cost_manager.all_cost_manager import cost_manager
from tests.llm_ocr_models.base_ocr_model import BaseOcrModel

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

class NanonetsOCRManager(BaseOcrModel):
    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def start(self):
        if self.model is None or self.processor is None or self.tokenizer is None:
            self.model, self.processor, self.tokenizer = load_nanonets_s_model()

    def process(self, image_path: str, max_new_tokens=4096) -> tuple[str, float]:
        if self.model is None or self.processor is None or self.tokenizer is None:
            raise ValueError("Model, processor, and tokenizer must be initialized. Call start() before process().")

        return ocr_page_with_nanonets_s(image_path, self.model, self.processor, max_new_tokens)

    def delete(self):
        del self.model
        del self.processor
        del self.tokenizer
        import torch
        torch.cuda.empty_cache()


def load_nanonets_s_model():
    model_path = "nanonets/Nanonets-OCR-s"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    return model, processor, tokenizer

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    inputs_tokens = len(inputs.input_ids[0])

    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_tokens = len(generated_ids[0])

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    total_cost = cost_manager.calculate_cost(
        "nanonets",
        inputs_tokens,
        output_tokens
    )
    return output_text[0], total_cost


def main():
    import time
    main = os.getcwd().split("llm_ocr_models")[0]
    image_file = 'fv090042726300825000021c9_page_{page}.jpg'
    pages = [1]
    model, processor, tokenizer = load_nanonets_s_model()
    for page in pages:
        print(f"Processing page {page}")
        image_file_page = image_file.format(page=page)
    #     image_file_page = image_file
        image_file_path = os.path.join(main, "data", "extracted", image_file_page)
        start_time = time.time()
        result = ocr_page_with_nanonets_s(image_file_path, model, processor, max_new_tokens=15000)
        end_time = time.time()
        print(f"Time taken for page: {end_time - start_time} seconds")
        print(f"OCR Result for page {page}:\n", result)
        # print(f"OCR Result:\n", result)

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")