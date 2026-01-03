import subprocess
from src.scripts.utils.llm_ocr_models.base_ocr_model import (BaseOcrModel,
                                                 calculate_tokens)
from src.scripts.utils.cost_manager.all_cost_manager import cost_manager

class DeepseekOllamaManager(BaseOcrModel):
    def __init__(self):
        super().__init__()
        self.cost_model_name = "deepseekOCR"

    def process(self, image_file: str) -> tuple[str, float]:
        text = run_deepseek_ollama(image_file)
        out_tokens = calculate_tokens(text)
        cost = cost_manager.calculate_cost(self.cost_model_name, 1500, out_tokens)
        return text, cost

def run_deepseek_ollama(image_file:str) -> str:
    prompt = "<|grounding|>Convert the document to markdown."

    full_prompt = f"{image_file}\n{prompt}"

    process = subprocess.run(
        ["ollama", "run", "deepseek-ocr-con-output"],
        input=full_prompt,
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    return process.stdout
