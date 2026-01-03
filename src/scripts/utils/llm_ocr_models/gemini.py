from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from pydantic import BaseModel

from scripts.utils.prompts.prompts import user_extraction_prompt, system_prompt
from tests.cost_manager.all_cost_manager import cost_manager
from tests.llm_ocr_models.base_ocr_model import BaseOcrModel
import base64

from tenacity import retry, wait_fixed, stop_after_attempt

from langchain_community.callbacks.manager import get_openai_callback

class GeminiManager(BaseOcrModel):
    def __init__(self, model_name: str = "gemini-3-flash-preview",
                 temperature: float = 1.0):
        super().__init__()
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )
        self.prompt_ocr = """Perform Optical Character Recognition (OCR) on the following image data.
Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes.
 Review all the values carefully to ensure accuracy.
 The most important requirements are:
    - Dates
    - Numbers
    - Monetary amounts
    - Bills Numbers and identifiers.
"""
        self.cost_model_name = model_name


    @retry(wait=wait_fixed(60), stop=stop_after_attempt(3), reraise=True)
    def process(self, image_path:str) -> tuple[str, float]:

        image = open(image_path, "rb").read()
        image = base64.b64encode(image).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt_ocr},
                {"type": "image", "base64": image, "mime_type": "image/jpeg"}
            ]
        )

        with get_openai_callback() as cb:
            response = self.model.invoke([message])

        total_cost = cost_manager.calculate_cost(
                                    self.cost_model_name,
                                    cb.prompt_tokens,
                                    cb.completion_tokens)

        if isinstance(response.content, list):
            response_data = response.content[0]
            if isinstance(response_data, dict):
                return response.content[0].get("text", ""), total_cost
            else:
                return response_data, total_cost

        return response.content, total_cost

    def process_structured(self, image:list[str],
                           structure:BaseModel) -> tuple[BaseModel, float]:
        if self.cost_model_name == "":
            raise ValueError("Cost model name is not defined.")

        if self.model_structured is None:
            self.model_structured = self.model.with_structured_output(structure)

        images = []
        for img in image:
            image_data = open(img, "rb").read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            images.append(image_b64)

        user_content = [{"type": "image", "base64": img_b64, "mime_type": "image/jpeg"} for img_b64 in images]+ [
                    {"type": "text", "text": user_extraction_prompt}
                ]


        message = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
            content=user_content
        )]

        with get_openai_callback() as cb:
            response = self.model_structured.invoke(message)

        total_cost = cost_manager.calculate_cost(self.cost_model_name,
                                  cb.total_tokens,
                                  cb.total_tokens)

        return response, total_cost


if __name__ == "__main__":
    from google import genai
    from dotenv import load_dotenv
    load_dotenv()

    client = genai.Client()
    models = client.models.list()
    print(models.page)



