from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

from tests.cost_manager.all_cost_manager import cost_manager
from tests.llm_ocr_models.base_ocr_model import BaseOcrModel, calculate_tokens


class MineruManager(BaseOcrModel):
    def __init__(self):
        super().__init__()
        self.client = None
        self.cost_model_name = "MinerU2.5"

    def start(self):
        if self.model is None or self.processor is None or self.client is None:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                dtype="auto",
                device_map="auto"
            )

            self.processor = AutoProcessor.from_pretrained(
                "opendatalab/MinerU2.5-2509-1.2B",
                use_fast=True
            )

            self.client = MinerUClient(
                backend="transformers",
                model=self.model,
                processor=self.processor
            )

    def ocr_page(self, image_path: str) -> list:
        image = Image.open(image_path)
        extracted_blocks = self.client.two_step_extract(image)
        return extracted_blocks

    def process(self, image_path: str) -> tuple[str, float]:
        if self.model is None or self.processor is None or self.client is None:
            raise ValueError("Model, processor, and client must be initialized. Call start() before process().")

        extracted_blocks = self.ocr_page(image_path)
        only_text = ""
        for block in extracted_blocks:
            if not "content" in block:
                continue
            text = block["content"]
            if text is None:
                continue
            only_text += text + "\n"

        total_cost = cost_manager.calculate_cost(
            self.cost_model_name,
            1500,
            calculate_tokens(only_text)
        )

        return only_text, total_cost

    def delete(self):
        del self.model
        del self.processor
        del self.client
        import torch
        torch.cuda.empty_cache()

