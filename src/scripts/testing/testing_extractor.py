from time import time
import pandas as pd
import os
from tqdm.auto import tqdm
import json

from src.scripts.utils.llm_extraction_models.gemini_extractor import GeminiExtractor
from src.scripts.utils.llm_extraction_models.ollama_extractor import OllamaExtractor
from src.scripts.utils.llm_extraction_models.openai_extractor import OpenAIExtractor


from dotenv import load_dotenv

from src.scripts.utils.llm_extraction_models.base_extractor import BaseExtractor

load_dotenv()

class TestingExtractor:
    def __init__(self):
        self.models = {
            "gpt_oss_20b": OllamaExtractor("gpt-oss:20b"),
            "qwen25_32b": OllamaExtractor("qwen2.5:32b"),
            "deepseek_32b": OllamaExtractor("deepseek-r1:32b"),
            "llama3_1_8b": OllamaExtractor("llama3.1:8b"),
            "qwen3_32b": OllamaExtractor("qwen3:32b"),
            "gemini_3_flash": GeminiExtractor(),
            "gpt_5_1_medium": OpenAIExtractor("gpt-5.1"),
            "gemini_3_pro": GeminiExtractor("gemini-3-pro-preview"),
        }

def load_test_data(main: str, file_path:str) -> pd.DataFrame:
    test_data_path = os.path.join(main, "tests", "test_results", file_path)
    return pd.read_csv(test_data_path)

def make_a_test_extractor():
    main = os.getcwd().split("testing")[0]
    tester = TestingExtractor()
    testers = tester.models.keys()

    ocr_files = os.listdir(os.path.join(main, "data", "test_results"))
    ocr_files = [file for file in ocr_files if file.endswith(".csv")]

    for test in testers:
        run_test_individual_model(test, tester.models[test],
                                main,
                                ocr_files)


def run_test_individual_model(model_name:str,
                            model_extractor,
                            main:str,
                            ocr_files:list):

    os.makedirs(os.path.join(main, "data", "test_results", model_name), exist_ok=True)

    for ocr_file in ocr_files:
        ocr_result_path = os.path.join(main, "data", "test_results", model_name, ocr_file)
        if os.path.isfile(ocr_result_path):
            results_df = pd.read_csv(ocr_result_path)
        else:
            results_df = pd.DataFrame(columns=["FILE_NAME",
                                               "TIME_TAKEN",
                                               "EXTRACTION_RESULT",
                                               "TOTAL_COST"])

        ocr_data_df = pd.read_csv(os.path.join(main, "data", "test_results", ocr_file))

        run_test_dataset(ocr_result_path, model_extractor, results_df, ocr_data_df)

def run_test_dataset(ocr_result_path:str,
                    model_extractor:BaseExtractor,
                    results_df:pd.DataFrame,
                    ocr_data_df:pd.DataFrame):

    if len(results_df) == len(ocr_data_df):
        print(f"All files in {ocr_result_path} have been processed. Skipping.")
        return

    for index, row in tqdm(ocr_data_df.iterrows(),
                           total=ocr_data_df.shape[0],
                           desc=f"Testing {ocr_result_path}", unit="file"):
        file_name = row["FILE_NAME"]

        if len(results_df) > index:
            print(f"Skipping {file_name}, already processed.")
            continue

        ocr_text = row["OCR_TEXT"]
        start_time = time()
        extraction_result, total_cost = model_extractor.extract(ocr_text)
        end_time = time()
        time_taken = end_time - start_time

        extraction_result = json.dumps(extraction_result)

        results_df.loc[len(results_df)] = {
            "FILE_NAME": file_name,
            "TIME_TAKEN": time_taken,
            "EXTRACTION_RESULT": extraction_result,
            "TOTAL_COST": total_cost
        }

        results_df.to_csv(ocr_result_path, index=False)

    print("Total cost for", ocr_result_path, ":", results_df["TOTAL_COST"].sum())

if __name__ == "__main__":
    make_a_test_extractor()