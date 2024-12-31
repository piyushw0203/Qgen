########################
# LOAD LIBRARIES
########################
import spacy
import warnings
import logging
import os
from langchain.llms import Ollama
from pdf2image import convert_from_path
import easyocr

warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_sm")

#######################################
# CONFIGURATION FOR THE LANGUAGE MODEL
#######################################
llm = Ollama(model="llama3", temperature=0.1)

#######################################
# EASYOCR READER
#######################################
reader = easyocr.Reader(['en'])  # Language: English

def pdf_to_images(file_path, start_page, end_page):
    """
    Convert PDF pages to images using pdf2image.
    """
    print("Converting PDF pages to images...")
    images = convert_from_path(file_path, first_page=start_page, last_page=end_page, fmt='jpeg')
    return images

def extract_text_from_images(images):
    """
    Extract text from a list of images using EasyOCR.
    """
    extracted_text = ""
    for idx, img in enumerate(images):
        print(f"Extracting text from page {idx + 1}...")
        results = reader.readtext(img)
        for result in results:
            extracted_text += result[1] + " "
    return extracted_text

def main():
    """
    Main function to extract text from a PDF, convert pages to images, 
    and generate MCQs using EasyOCR and a language model.
    """
    print("\n=== MCQ Generator from PDF (Image-based Text Extraction) ===\n")

    # User input for file and page range
    file_path = "E:\Ojal Didi Project\Radar Handbook.pdf"
    if not os.path.exists(file_path):
        print("Error: File not found. Please check the path and try again.")
        return

    start_page = int(input("Enter the starting page number: ").strip())
    end_page = int(input("Enter the ending page number: ").strip())

    try:
        # Step 1: Convert PDF to Images
        images = pdf_to_images(file_path, start_page, end_page)
        
        # Step 2: Extract Text using EasyOCR
        print("\nExtracting text from images...\n")
        extracted_text = extract_text_from_images(images)

        if not extracted_text.strip():
            print("Error: No text extracted. Ensure the pages contain readable content.")
            return

        print("Text extracted successfully! Preview:")
        print("-" * 50)
        print(extracted_text[:1000])  # Show a preview of the text (first 1000 characters)
        print("-" * 50)

        # Step 3: Generate MCQs using the language model
        print("\nGenerating MCQs...\n")
        test_prompt = f"Generate 5 MCQs based on the following text: {extracted_text}"
        test_output = llm.invoke(test_prompt)

        print("MCQs Generated:")
        print("-" * 50)
        print(test_output)
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
