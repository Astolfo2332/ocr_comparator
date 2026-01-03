
from langchain_openai import ChatOpenAI
from scripts.data_models.invoices import Invoice
from tests.llm_extraction_models.base_extractor import BaseExtractor


class OpenAIExtractor(BaseExtractor):
    def __init__(self, model_name: str = "gpt-5.1", reasoning: str = None):
        super().__init__()
        self.model = ChatOpenAI(
            model=model_name,
            reasoning_effort="medium" if reasoning is None else reasoning
            )
        self.model_structure = self.model.with_structured_output(Invoice)
        self.cost_model_name = model_name
