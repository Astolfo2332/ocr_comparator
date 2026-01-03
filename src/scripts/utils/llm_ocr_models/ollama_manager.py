from langchain_core.exceptions import OutputParserException
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from PIL import Image
import json

from scripts.utils.prompts.prompts import (
                                           system_prompt_json,
                                           user_extraction_prompt_json)

from scripts.utils.prompts.prompts import system_prompt, user_extraction_prompt
from src.scripts.utils.pdf_loader import convert_img_to_bytes

from langchain_ollama import OllamaLLM, ChatOllama

from src.scripts.utils.llm_ocr_models.base_ocr_model import BaseOcrModel

from src.scripts.utils.cost_manager.all_cost_manager import cost_manager
from langchain_community.callbacks.manager import get_openai_callback


class OllamaManager(BaseOcrModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = OllamaLLM(model=model_name, num_predict=2048)
        self.prompt = ChatPromptTemplate.from_messages(
            [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_extraction_prompt)
           ]
        )
        self.formated_prompt = self.prompt.format_messages()
        self.cost_model_name = model_name

    def process_structured(self, image:list[str], structure:BaseModel) -> tuple[BaseModel, float]:
        if self.model_structured is None:
            self.model = ChatOllama(
                model=self.cost_model_name,
                num_predict=2048
            )
            self.model_structured = self.model.with_structured_output(structure)


        images = [Image.open(img_path).convert("RGB") for img_path in image]
        images_bytes = convert_img_to_bytes(images)

        images_list = [{"type": "image",
                                   "base64": img,
                                   "mime_type": "image/jpeg"
                                  } for img in images_bytes]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content= images_list +
                                 [{"type": "text", "text": user_extraction_prompt}]
                          )
        ]

        try:
            with get_openai_callback() as cb:
                response = self.model_structured.invoke(messages)
        except OutputParserException:
            print("Model cannot parse the response to the desired format. Trying JSON parsing.")
            response, total_cost = self.parse_with_json(images_list)
            if response is None:
                return None, total_cost
            response = structure.model_validate(response)
            return response, total_cost

        cost = cost_manager.calculate_cost(self.cost_model_name,
                                           cb.prompt_tokens,
                                           cb.completion_tokens)
        return response, cost

    def parse_with_json(self, text:list) -> tuple[dict, float]:
        user_content = text + [{"type": "text", "text": user_extraction_prompt_json}]

        messages = [
            SystemMessage(content=system_prompt_json),
            HumanMessage(content=user_content)
        ]
        with get_openai_callback() as cb:
            response = self.model.invoke(messages)
        total_cost = cost_manager.calculate_cost(self.cost_model_name,
                                                 cb.prompt_tokens,
                                                 cb.completion_tokens)

        if isinstance(response.content, list):
            response_data = response.content[0]
            if isinstance(response_data, dict):
                response_data = response.content[0].get("text", "")
            else:
                response_data = response_data
        else:
            response_data = response.content
        try:
            response_data = response_data.split("```json")[1].split("```")[0].strip()
        except IndexError:
            pass
        try:
            response_dict = json.loads(response_data)
        except json.JSONDecodeError:
            print("Model cannot parse the response to JSON.")
            response_dict = None

        return response_dict, total_cost



