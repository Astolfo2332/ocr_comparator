from langchain_core.exceptions import OutputParserException

from scripts.utils.prompts.prompts import (system_prompt,
                                           user_extraction_prompt,
                                           system_prompt_json,
                                           user_extraction_prompt_json)
from langchain_core.messages import HumanMessage, SystemMessage
import json

from langchain_community.callbacks.manager import get_openai_callback

from src.tests.cost_manager.all_cost_manager import cost_manager

class BaseExtractor:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_structure = None
        self.cost_model_name = ""

    def extract(self, text:str) -> tuple[dict, float]:
        if self.cost_model_name == "":
            raise NotImplementedError("Cost model name not defined.")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{text}\n\n{user_extraction_prompt}"),
        ]

        try:
            with get_openai_callback() as cb:
                response = self.model_structure.invoke(messages)

            total_cost = cost_manager.calculate_cost(self.cost_model_name,
                                        cb.prompt_tokens,
                                        cb.completion_tokens)

        except OutputParserException:
            print("Model cannot parse the response to the desired format. Trying JSON parsing.")
            response = self.parse_with_json(text)
            return response

        response = response.model_dump()
        return response, total_cost
    def start(self):
        pass
    def delete(self):
        pass

    def parse_with_json(self, text:str) -> tuple[dict, float]:
        messages = [
            SystemMessage(content=system_prompt_json),
            HumanMessage(content=f"{text}\n{user_extraction_prompt_json}"),
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
            response_dict = response.content

        return response_dict, total_cost

