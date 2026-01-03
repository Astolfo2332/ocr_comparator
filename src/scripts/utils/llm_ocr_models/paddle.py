from pydantic import BaseModel

from src.scripts.utils.llm_ocr_models.base_ocr_model import BaseOcrModel
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

class PaddleOCRManager(BaseOcrModel):
    def __init__(self):
        super().__init__()
        self.model_name_or_path = "PaddlePaddle/PaddleOCR-VL"
        self.prompts = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }

    def start(self):
        if self.model is None or self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                use_fast=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )

    def process(self, image_file: str) -> tuple[str, float]:
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be initialized. Call start() before process().")

        image = Image.open(image_file).convert("RGB")

        messages = [
            {"role": "user",
             "content": [
                 {"type": "image", "image": image},
                 {"type": "text", "text": self.prompts["ocr"]},
             ]
             }
        ]

        inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                    ).to(self.model.device)

        outputs= self.model.generate(**inputs, do_sample=False,
                                     max_new_tokens=16384)
        outputs = self.processor.batch_decode(outputs,
                                              skip_special_tokens=True)[0]

        return outputs, 0.5

    def delete(self):
        del self.model
        del self.processor
        import torch
        torch.cuda.empty_cache()
