import spacy
import warnings
import logging
import os
from langchain.llms import Ollama
from pdf2image import convert_from_path
import easyocr
import gradio as gr
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore")

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Load LLM
llm = Ollama(model="llama3", temperature=0.2)

# EasyOCR Reader
reader = easyocr.Reader(['en'])

def pdf_to_images(file_path, start_page, end_page,dpi=150):
    """
    Convert PDF pages to images using pdf2image.
    """
    images = convert_from_path(file_path, first_page=start_page, last_page=end_page, fmt='jpeg',dpi=dpi)
    return images

def extract_text_from_images(images):
    """
    Extract text from images using EasyOCR.
    """
    extracted_text = ""
    for idx, img in enumerate(images):
        results = reader.readtext(img)
        for result in results:
            extracted_text += result[1] + " "
    return extracted_text

def generate_questions(extracted_text, question_counts):
    """
    Generate questions for each type with respective counts using the LLM.
    """
    generated_output = ""
    for qtype, count in question_counts.items():
        if count > 0:  # Only generate if count is > 0
            prompt = (
                f"Generate {count} {qtype} questions. Be very strict about the type and the number of questions generated."
                "Generate them based on the following text: "
                f"{extracted_text}"
            )
            questions = llm.invoke(prompt)
            generated_output += f"--- {qtype} Questions ({count}) ---\n{questions}\n\n"
    return generated_output

def save_questions_to_pdf(output_text, file_name="generated_questions.pdf"):
    """
    Save questions to a PDF using ReportLab.
    """
    c = canvas.Canvas(file_name)
    c.setFont("Helvetica", 12)
    y = 800  # Starting position

    for line in output_text.split("\n"):
        if y < 50:  # Start a new page if space runs out
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 800
        c.drawString(50, y, line)
        y -= 20

    c.save()
    return file_name

def process_pdf(file, start_page, end_page, question_counts):
    """
    Complete pipeline: Convert PDF to text and generate questions.
    """
    if not file:
        return "No file uploaded. Please upload a PDF.", ""

    try:
        # Save uploaded file
        file_path = file.name
        
        # Step 1: Convert PDF pages to images
        images = pdf_to_images(file_path, start_page, end_page)
        
        # Step 2: Extract text from images
        extracted_text = extract_text_from_images(images)

        if not extracted_text.strip():
            return "No text found. Ensure pages contain readable text.", ""

        # Step 3: Generate Questions
        generated_questions = generate_questions(extracted_text, question_counts)
        
        return "Text extraction and question generation successful!", generated_questions

    except Exception as e:
        logging.error(f"Error: {e}")
        return f"An error occurred: {e}", ""

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📝 PDF to Question Generator")
    gr.Markdown("Upload a PDF, select page range, question types, and specify the count for each type.")

    with gr.Row():
        file_input = gr.File(label="Upload PDF", type="filepath")
        start_page_input = gr.Number(label="Start Page", value=1, precision=0)
        end_page_input = gr.Number(label="End Page", value=1, precision=0)

    question_types = ["MCQs", "Fill in the Blanks", "True/False", "Long Answer", "Short Answer"]
    question_count_inputs = {}

    gr.Markdown("### Select Question Types and Specify Counts")
    with gr.Column():
        for qtype in question_types:
            question_count_inputs[qtype] = gr.Number(label=f"Number of {qtype}", value=0, precision=0)

    submit_button = gr.Button("Generate Questions")
    output_status = gr.Textbox(label="Status", interactive=False)
    output_questions = gr.Textbox(label="Generated Questions", lines=15)
    download_button = gr.Button("Download PDF")
    download_file = gr.File(label="Download Questions PDF", interactive=False)

    generated_text = gr.State()

    def save_to_pdf_for_download(questions_text):
        pdf_file_path = save_questions_to_pdf(questions_text)
        return pdf_file_path

    def handle_generation(file, start_page, end_page, *args):
        """
        Handles the generation of questions with dynamic counts.
        """
        question_counts = {}
        for idx, qtype in enumerate(question_types):
            question_counts[qtype] = args[idx]
        
        status, questions = process_pdf(file, start_page, end_page, question_counts)
        
        # Return questions as the third output for 'generated_text' state
        return status, questions, questions


    submit_button.click(
        fn=handle_generation,
        inputs=[file_input, start_page_input, end_page_input] + list(question_count_inputs.values()),
        outputs=[output_status, output_questions, generated_text]
    )

    download_button.click(
        fn=save_to_pdf_for_download,
        inputs=[generated_text],
        outputs=[download_file]
    )

demo.launch()
