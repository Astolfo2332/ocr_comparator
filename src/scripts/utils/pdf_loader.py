import os
from io import BytesIO
import base64

import pandas as pd
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
import numpy as np


class BytesImageDataset:
    def __init__(self, image_bytes_list, transform=None):
        self.image_bytes_list = image_bytes_list
        self.transform = transform

    def __len__(self):
        return len(self.image_bytes_list)

    def __getitem__(self, idx):
        img = self.image_bytes_list[idx].convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


def merge_images_vertically(images):
    widths, heights = zip(*(img.size for img in images))

    max_width = max(widths)
    total_height = sum(heights)

    merged_img = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for img in images:
        merged_img.paste(img, (0, y_offset))
        y_offset += img.height

    return merged_img

def convert_pdf_to_single_image(row:pd.Series) -> Image.Image or None:
    """Convierte un pdf a imagenes separados por pagina en una lista"""
    file =  row["FILE_NAME"] + ".pdf"
    main = os.path.join(os.getcwd(), "data", "extracted", file)
    if not os.path.exists(main):
        return None

    try:
        image = convert_from_path(main, thread_count=32)
    except PDFPageCountError:
        return None
    # image = (merge_images_vertically(image))

    return image

def convert_pdf_to_images(pdf_path:str) -> list:
    """Deprecado"""
    images = []
    # Convert PDF to images
    for file in os.listdir(pdf_path):
        if file.endswith('.pdf'):
            image = convert_from_path(os.path.join(pdf_path, file), thread_count=32)
            # images.append((merge_images_vertically(image), SERVER + os.path.basename(file)))
            images.append(image)
    return images


def convert_img_to_bytes(image:list[Image.Image]) -> list[str]:
    conv_img = []
    for img in image:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        conv_img.append(img_b64)
    return conv_img

def make_pdf_dataset(pdf_path:str) -> list[str]:
    print("[INFO] Loading PDF files from path:", pdf_path)
    images = convert_pdf_to_images(pdf_path)
    dataset = [convert_img_to_bytes(img) for img in images]
    print("[INFO] PDF files loaded successfully.")

    return dataset

def make_pdf_bytes(row:pd.Series) -> list[str]or None:
    images = convert_pdf_to_single_image(row)
    if images is None:
        return np.nan
    dataset = convert_img_to_bytes(images)

    return dataset

if __name__ == "__main__":

    pdf_path = "F:/Documentos/git/automaton_1/data/pdfs"
    images = make_pdf_dataset(pdf_path)

    for image in images:
        break
