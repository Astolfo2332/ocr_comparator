from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from PIL import Image

from langchain_community.callbacks.manager import get_openai_callback
from scripts.utils.pdf_loader import convert_img_to_bytes
from scripts.utils.prompts.prompts import system_prompt, user_extraction_prompt, user_ocr_prompt
from src.scripts.utils.cost_manager.all_cost_manager import cost_manager
from src.scripts.utils.llm_ocr_models.base_ocr_model import BaseOcrModel


class OpenAIManager(BaseOcrModel):
    def __init__(self, model_name: str, reasoning: str | None = None):
        super().__init__()
        self.cost_model_name = model_name
        self.model = ChatOpenAI(
            model=model_name,
            max_retries=5,
            reasoning_effort="medium" if reasoning is None else reasoning,
        )

    def process_structured(self, image:list[str], structure:BaseModel) -> tuple[BaseModel, float]:
        if self.model_structured is None:
            self.model_structured = self.model.with_structured_output(structure)

        images = [Image.open(img).convert("RGB") for img in image]
        images = convert_img_to_bytes(images)

        images_message = [
            {"type": "image",
               "base64": img,
               "mime_type": "image/jpeg"
            } for img in images]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=images_message +
                                  [{"type": "text", "text": user_extraction_prompt}]
                         )
        ]
        with get_openai_callback() as cb:
            response = self.model_structured.invoke(messages)

        cost = cost_manager.calculate_cost(
            self.cost_model_name,
            cb.prompt_tokens,
            cb.completion_tokens
        )

        return response, cost

    def process(self, image_path:str) -> tuple[str, float]:

        image = Image.open(image_path).convert("RGB")
        image = convert_img_to_bytes([image])[0]
        messages = [
            SystemMessage(content="Eres un asistente especializado en OCR que extrae texto de imágenes."),
            HumanMessage(content=[{"type": "image", "base64": image, "mime_type": "image/jpeg"},
                                  {"type": "text", "text": user_ocr_prompt}
            ])
        ]
        with get_openai_callback() as cb:
            response = self.model.invoke(messages)

        cost = cost_manager.calculate_cost(
            self.cost_model_name,
            cb.prompt_tokens,
            cb.completion_tokens
        )

        if isinstance(response.content, list):
            response_data = response.content[0]
            if isinstance(response_data, dict):
                return response.content[0].get("text", ""), cost
            else:
                return response_data, cost

        return response.content, cost

