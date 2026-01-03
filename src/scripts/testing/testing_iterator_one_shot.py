import pandas as pd
import os
from tqdm.auto import tqdm
import json

from time import time

from src.scripts.data_models.invoices import Invoice
from src.scripts.utils.llm_ocr_models.base_ocr_model import BaseOcrModel
from src.scripts.utils.llm_ocr_models.ollama_manager import OllamaManager
from src.scripts.utils.llm_ocr_models.gemini import GeminiManager
from src.scripts.utils.llm_ocr_models.openai_manager import OpenAIManager

from dotenv import load_dotenv

load_dotenv()

class ModelIterator:
    def __init__(self):
        self.models = {
            "Gemini_3_flash": GeminiManager(),
            # "Qwen2.5VL": Qwen25VlTransformersManager(),
            "Qwen2.5VL": OllamaManager("qwen2.5vl:latest"),
            "Qwen2.5VL_32B": OllamaManager("qwen2.5vl:32b"),
            "Gpt_5_1_medium": OpenAIManager(model_name="gpt-5.1"),
            # "Qwen_3VL": OllamaManager("qwen3-vl:30b"), #Ni en 30b es capaz de tener una salida correcta
            "Gemini_3_pro": GeminiManager(model_name="gemini-3-pro-preview"),
        }

        self.current_model = None
    def select_model(self, model_name: str) -> BaseOcrModel:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in available models.")
        self.current_model = self.models[model_name]

        return self.current_model


def load_dataset():
    main = os.getcwd().split("testing")[0]
    data_path = os.path.join(main, "data", "test_dataset.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The dataset file was not found at {data_path}")

    df = pd.read_csv(data_path)

    return df

def make_a_test_one_shot():
    model_iterator = ModelIterator()
    models_to_test = model_iterator.models.keys()
    test_files = os.listdir("../data/test_dataset_pdfs")
    test_files = [file for file in test_files if file.endswith(".jpg")]
    test_df = load_dataset()
    os.makedirs("../data/test_results_one_shot", exist_ok=True)

    for model_name in models_to_test:
        print(f"Testing model: {model_name}")
        model = model_iterator.select_model(model_name)
        process_dataset(model, test_files, model_name, test_df)
        model.delete()

def process_dataset(model: BaseOcrModel,
                    test_files: list,
                    model_name: str,
                    test_df: pd.DataFrame):

    if os.path.exists("../data/test_results_one_shot/" + model_name + "_results.csv"):
        results_df = pd.read_csv("../data/test_results_one_shot/" + model_name + "_results.csv")
    else:
        results_df = pd.DataFrame(columns=["FILE_NAME",
                                           "EXTRACTION_TIME",
                                           "DATA_EXTRACTED",
                                           "TOTAL_COST"])

    if len(results_df) == len(test_df):
        print(f"All files already processed for model: {model_name}")
        return

    print("Starting model...")
    model.start()

    main_path = os.getcwd().split("testing")[0]

    for index, data in tqdm(test_df.iterrows(), total=len(test_df),
                            desc=f"Processing dataset with {model_name}",
                            unit="file"):
        if len(results_df) > index:
            print(f"Skipping already processed file: {data['FILE_NAME']}")
            continue

        file_name = data["FILE_NAME"]
        associated_files = [file for file in test_files if file.startswith(file_name)]
        associated_files = sorted(associated_files)
        if not associated_files:
            print(f"No associated files found for: {file_name}")
            continue

        time_for_files = []
        total_cost = 0.0
        files = []

        for ass_file_name in associated_files:
            file_path = os.path.join(main_path,
                                     "data", "test_dataset_pdfs",
                                     ass_file_name)
            files.append(file_path)

        start_time = time()
        data_extracted, page_cost = model.process_structured(files, Invoice)
        end_time = time()

        extraction_time = end_time - start_time

        time_for_files.append(extraction_time)
        total_cost += page_cost

        if data_extracted is not None:
            data_extracted = data_extracted.model_dump()
            data_extracted = json.dumps(data_extracted)
        else:
            data_extracted = {}

        results_df.loc[len(results_df)] = {
            "FILE_NAME": data["FILE_NAME"],
            "EXTRACTION_TIME": time_for_files,
            "DATA_EXTRACTED": data_extracted,
            "TOTAL_COST": total_cost
        }

        results_df.to_csv("../data/test_results_one_shot/" + model_name + "_results.csv", index=False)

    print("Total cost for", model_name, results_df["TOTAL_COST"].sum())


if __name__ == "__main__":
    make_a_test_one_shot()