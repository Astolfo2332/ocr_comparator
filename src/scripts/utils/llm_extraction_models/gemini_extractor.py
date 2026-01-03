from langchain_google_genai import ChatGoogleGenerativeAI

from scripts.data_models.invoices import Invoice
from src.scripts.utils.llm_extraction_models.base_extractor import BaseExtractor


class GeminiExtractor(BaseExtractor):
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        super().__init__()
        self.model = ChatGoogleGenerativeAI(model=model_name)
        self.model_structure = self.model.with_structured_output(Invoice)
        self.cost_model_name = model_name
