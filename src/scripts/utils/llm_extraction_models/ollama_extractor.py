from tests.llm_extraction_models.base_extractor import BaseExtractor
from langchain_ollama import ChatOllama
from scripts.data_models.invoices import Invoice


class OllamaExtractor(BaseExtractor):
    def __init__(self, model_name:str=None):
        super().__init__()

        if not model_name:
            raise ValueError("model_name must be provided for OllamaExtractor")

        self.cost_model_name = model_name

        self.model = ChatOllama(
            model=model_name,
            num_predict=2048
        )
        self.model_structure = self.model.with_structured_output(Invoice)
