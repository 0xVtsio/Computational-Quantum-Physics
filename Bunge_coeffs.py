import fitz  # PyMuPDF
import pytesseract
import re
import pandas as pd
from PIL import Image
import io
import cv2
import numpy as np

def extract_combined_tables(pdf_path):
    doc = fitz.open(pdf_path)
    element_blocks = []
    current_element = None
    table_lines = []
    in_table = False

    # Regex patterns
    element_pattern = re.compile(r"^([A-Z]+),\s*Z=(\d+)", re.IGNORECASE)
    table_start_pattern = re.compile(r"<1/R\*\*2>")
    table_end_pattern = re.compile(r"^[A-Z]+,\s*Z=\d+", re.IGNORECASE)

    def finalize_element():
        nonlocal current_element, table_lines
        if current_element and table_lines:
            # Clean and structure the table data
            df = pd.DataFrame(table_lines[1:], columns=table_lines[0])
            element_blocks.append({
                "Element": current_element,
                "Orbital Coefficients": df.to_dict(orient="records")
            })
        current_element = None
        table_lines = []

    def process_text_content(text):
        nonlocal current_element, table_lines, in_table
        for line in text.split('\n'):
            line = line.strip()
            # Detect element headers
            if element_pattern.match(line):
                finalize_element()
                current_element = line
                in_table = False
                continue
            if current_element:
                # Detect table start
                if table_start_pattern.search(line):
                    in_table = True
                    table_lines = []
                    continue
                # Detect table end
                if in_table and (table_end_pattern.match(line) or not line):
                    in_table = False
                    finalize_element()
                    continue
                # Collect table data
                if in_table:
                    cleaned_line = re.sub(r'\s{2,}', '|', line)  # Convert whitespace to delimiter
                    table_lines.append(cleaned_line.split('|'))

    def process_page_images(page):
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Preprocess image
            img = Image.open(io.BytesIO(image_bytes))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            # OCR processing
            ocr_text = pytesseract.image_to_string(thresh, config='--psm 6')
            process_text_content(ocr_text)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # First try text extraction
        text_content = page.get_text()
        process_text_content(text_content)
        # Then process images with OCR
        process_page_images(page)

    # Finalize last element if needed
    finalize_element()
    return pd.DataFrame(element_blocks)

# Usage
pdf_path = "RHF_Bunge.pdf"
result_df = extract_combined_tables(pdf_path)
result_df.to_csv("combined_tables.csv", index=False)
print("Combined text and image tables extracted successfully!")