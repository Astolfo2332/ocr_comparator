import os
import json
import ast
import re

import pandas as pd
from dateutil import parser

from src.scripts.utils.parsers.bill_parsers import get_alpha_numeric_string


def detectar_year(nums):
    for i, n in enumerate(nums):
        if n >= 1900:
            return i
    return None

def parse_fecha_multiorden(s: str):
    nums = list(map(int, re.findall(r"\d+", s)))

    if len(nums) < 3:
        return None, "invalid"

    year_idx = detectar_year(nums)

    if year_idx == 0:
        try:
            return parser.parse(s, yearfirst=True), "confident"
        except Exception:
            pass

    if year_idx in (1, 2):
        a, b = nums[0], nums[1]

        try:
            if a > 12:
                return parser.parse(s, dayfirst=True), "confident"
            if b > 12:
                return parser.parse(s, dayfirst=False), "confident"

            d1 = parser.parse(s, dayfirst=True)
            d2 = parser.parse(s, dayfirst=False)
        except Exception:
            return None, "invalid"

        if d1.date() == d2.date():
            return d1, "confident"

        return (d1, d2), "ambiguous"

    return None, "invalid"

def generate_test_results():
    gt_dataset = pd.read_csv("../data/test_dataset.csv")

    extractor_data = os.listdir("../data/test_results")
    extractor_data = [file for file in extractor_data if not file.endswith(".csv")]

    os.makedirs("../data/final_test_results", exist_ok=True)

    all_results = pd.DataFrame(columns=["EXTRACTION_MODEL", "OCR_MODEL", "ACCURACY", "ACCURACY_STD", "COMMON_ERRORS"])

    for extractor_model in extractor_data:
        ocr_models = os.listdir(os.path.join("../data", "test_results", extractor_model))
        ocr_results = [file for file in ocr_models if file.endswith(".csv")]
        for ocr_model in ocr_results:
            ocr_extractor_data = pd.read_csv(os.path.join("../data", "test_results", extractor_model, ocr_model))
            result_df = compare_results(
                gt_dataset,
                ocr_extractor_data,
                extractor_model,
                ocr_model.replace(".csv", "")
            )

            result_path = os.path.join("../data", "final_test_results", f"{extractor_model}_{ocr_model}")
            result_df.to_csv(result_path, index=False)

            error_data = result_df["ERRORS"].str.split(";").explode().value_counts()
            error_data = error_data[error_data.index != "NONE"]
            error_data = error_data.head(5).to_dict()
            error_data = json.dumps(error_data)

            all_results.loc[len(all_results)] = {
                "EXTRACTION_MODEL": extractor_model,
                "OCR_MODEL": ocr_model.replace(".csv", ""),
                "ACCURACY": result_df["ACCURACY"].mean(),
                "ACCURACY_STD": result_df["ACCURACY"].std(),
                "COMMON_ERRORS": error_data
            }

    all_results.to_csv(os.path.join("../data", "final_test_results", "summary_results.csv"), index=False)


def compare_results(
    gt_dataset: pd.DataFrame,
    ocr_extractor_data: pd.DataFrame,
    extractor_model: str,
    ocr_model: str
) -> pd.DataFrame:
    results = []

    extraction_results = []

    for _, ocr_row in ocr_extractor_data.iterrows():
        if isinstance(ocr_row["EXTRACTION_RESULT"], dict):
            extraction_results.append(ocr_row["EXTRACTION_RESULT"])
            continue
        elif isinstance(ocr_row["EXTRACTION_RESULT"], float):
            extraction_results.append({})
            continue
        else:
            try:
                extraction_results.append(ast.literal_eval(ocr_row["EXTRACTION_RESULT"]))
            except (ValueError, SyntaxError):
                extraction_results.append({})

    for i, gt_row in gt_dataset.iterrows():
        compare_data = extraction_results[i]
        accuracy_result, error_data = get_accuracy_results(gt_row, compare_data)
        results.append({
            "FILE_NAME": gt_row["FILE_NAME"],
            "EXTRACTION_MODEL": extractor_model,
            "OCR_MODEL": ocr_model,
            "ACCURACY": accuracy_result,
            "ERRORS": ";".join(error_data) if error_data else "NONE"
        })

    results_df = pd.DataFrame(results)

    return results_df

gt_to_dict = {
    "NIT": "nit",
    "FACTURA": "numero_factura",
    "FECHA": "fecha",
    "VALOR ANTES DE IVA": "antes_iva",
    "IVA": "iva",
    "TOTAL": "valor_total",
    "PROVEEDOR": "proveedor"
}

num_slots = ["IVA", "VALOR ANTES DE IVA", "TOTAL"]

importance = {
    "NIT": 0.12,
    "FACTURA": 0.25,
    "FECHA": 0.06,
    "VALOR ANTES DE IVA": 0.17,
    "IVA": 0.17,
    "TOTAL": 0.17,
    "PROVEEDOR": 0.06
}


def get_accuracy_results(gt_row: pd.Series, compare_data: dict) -> tuple[float, list]:

    accuracy = []
    error_data = []

    for gt_key, dict_key in gt_to_dict.items():
        gt_data = gt_row[gt_key]
        if not isinstance(compare_data, dict):
            accuracy.extend([0] * len(gt_to_dict))
            error_data.append("NO DICT")
            break

        compare_value = compare_data.get(dict_key, None)
        if compare_value is None:
            accuracy.append(0)
            error_data.append(f"MISSING KEY {dict_key.upper()}")
            continue

        if gt_key != "FECHA":
            gt_data = get_alpha_numeric_string(str(gt_data))
            compare_value = get_alpha_numeric_string(str(compare_value))
        else:
            parsed_date, confidence = parse_fecha_multiorden(str(compare_value))
            if confidence == "confident":
                compare_value = parsed_date.strftime("%d-%m-%Y")
                gt_data = parser.parse(str(gt_data), dayfirst=True).strftime("%d-%m-%Y")

                if gt_data == compare_value:
                    accuracy.append(importance[gt_key])
                    continue
                else:
                    accuracy.append(0)
                    error_data.append("INVALID DATE")
                    continue

            elif confidence == "ambiguous":
                date1, date2 = parsed_date
                date1_str = date1.strftime("%d-%m-%Y")
                date2_str = date2.strftime("%d-%m-%Y")
                if gt_data == date1_str or gt_data == date2_str:
                    accuracy.append(importance[gt_key])
                    continue
                else:
                    accuracy.append(0)
                    error_data.append("INVALID DATE")
                    continue
            else:
                accuracy.append(0)
                error_data.append("INVALID DATE")
                continue

        if gt_key in num_slots:
            if not gt_key == "NIT":
                try:
                    gt_data = float(gt_data)
                    compare_value = float(compare_value)
                except ValueError:
                    accuracy.append(0)
                    error_data.append(f"INVALID NUMBER IN {gt_key.upper()}")
                    continue

        if isinstance(gt_data, str) and isinstance(compare_value, str):
            gt_data = gt_data.lower()
            compare_value = compare_value.lower()

        if gt_data == compare_value:
            accuracy.append(importance[gt_key])
        else:
            error_data.append(f"INCORRECT {gt_key.upper()}")
            accuracy.append(0)

    assert len(accuracy) == len(gt_to_dict), "Accuracy length mismatch"

    return sum(accuracy), error_data

def generate_test_result_one_shot():
    gt_dataset = pd.read_csv("../data/test_dataset.csv")

    extractor_data = os.listdir("../data/test_results_one_shot")
    extractor_data = [file for file in extractor_data if file.endswith(".csv")]
    os.makedirs("../data/final_test_results_one_shot", exist_ok=True)

    all_results = pd.DataFrame(columns=["EXTRACTION_MODEL", "ACCURACY", "ACCURACY_STD", "COMMON_ERRORS"])

    for extractor_model in extractor_data:
        ocr_extractor_data = pd.read_csv(os.path.join("../data", "test_results_one_shot", extractor_model))
        ocr_extractor_data["EXTRACTION_RESULT"] = ocr_extractor_data["DATA_EXTRACTED"]
        result_df = compare_results(
            gt_dataset,
            ocr_extractor_data,
            extractor_model,
            ""
        )
        result_path = os.path.join("../data", "final_test_results_one_shot", f"{extractor_model}")
        result_df.to_csv(result_path, index=False)

        error_data = result_df["ERRORS"].str.split(";").explode().value_counts()
        error_data = error_data[error_data.index != "NONE"]
        error_data = error_data.head(5).to_dict()
        error_data = json.dumps(error_data)

        all_results.loc[len(all_results)] = {
            "EXTRACTION_MODEL": extractor_model,
            "ACCURACY": result_df["ACCURACY"].mean(),
            "ACCURACY_STD": result_df["ACCURACY"].std(),
            "COMMON_ERRORS": error_data
        }
    all_results.to_csv(os.path.join("../data", "final_test_results_one_shot", "summary_results.csv"), index=False)

if __name__ == "__main__":
    generate_test_results()
    generate_test_result_one_shot()

