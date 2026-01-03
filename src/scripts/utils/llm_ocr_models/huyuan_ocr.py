from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import torch
from tests.llm_ocr_models.base_ocr_model import BaseOcrModel
from tests.cost_manager.all_cost_manager import cost_manager

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n<8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]

    return text

class HuyuanOCRManager(BaseOcrModel):
    def __init__(self):
        super().__init__()
        self.model_name_or_path = "tencent/HunyuanOCR"
        self.recommended_prompt = """
• Identify the formula in the image and represent it using LaTeX format.

• Parse the table in the image into HTML.

• Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.

• Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.
"""
        self.cost_model_name = "HunyuanOCR"


    def start(self):
        if self.model is None or self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, use_fast=False)

            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            device_map="auto"
            )

    def process(self, image_file:str) -> tuple[str, float]:

        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be initialized. Call start() before process().")

        image_inputs = Image.open(image_file)

        messages1 = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": (
                        self.recommended_prompt
                    )},
                ],
            }
        ]
        messages = [messages1]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs_tokens = len(inputs["input_ids"][0])

        with torch.no_grad():
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=16384, do_sample=False)
        if "input_ids" in inputs:
            input_ids = inputs.input_ids
        else:
            print("inputs: # fallback", inputs)
            input_ids = inputs.inputs

        output_tokens = generated_ids.shape[1] - inputs_tokens

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        raw_output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_texts = ""
        for t in raw_output_texts:
            output_texts += clean_repeated_substrings(t)

        total_cost = cost_manager.calculate_cost(self.cost_model_name,
                                                 inputs_tokens,
                                                 output_tokens)

        return output_texts, total_cost

    def delete(self):
        del self.model
        del self.processor
        import torch
        torch.cuda.empty_cache()