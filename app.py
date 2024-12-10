########################
# LOAD LIBRARIES
########################
import spacy
import warnings
import logging
import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from pdfminer.high_level import extract_text
from tqdm import tqdm
from PyPDF2 import PdfReader

warnings.filterwarnings("ignore")
nlp = spacy.load("en_core_web_sm")

#####################################
# FUNCTION DEFINITIONS
#####################################
def nest_sentences(document, max_length=4096):
    """
    Break down a document into manageable chunks of sentences where each chunk is under a specified length.

    Parameters:
    - document (str): The input text document to be processed.
    - max_length (int): The maximum character length for each chunk.

    Returns:
    - list: A list where each element is a group of sentences that together are less than max_length characters.
    """
    nested = []  # List to hold all chunks of sentences
    sent = []    # Temporary list to hold sentences for a current chunk
    length = 0   # Counter to keep track of the character length of the current chunk
    doc = nlp(document)  # Process the document using Spacy to tokenize into sentences

    for sentence in doc.sents:
        length += len(sentence.text)
        if length < max_length:
            sent.append(sentence.text)
        else:
            nested.append(' '.join(sent))  # Join sentences in the chunk and add to the nested list
            sent = [sentence.text]  # Start a new chunk with the current sentence
            length = len(sentence.text)  # Reset the length counter to the length of the current sentence

    if sent:  # Don't forget to add the last chunk if it's not empty
        nested.append(' '.join(sent))

    return nested



def generate_questions(text, llm, max_length=8192):
    """
    Generate multiple-choice questions for provided text using the specified LLM.

    Parameters:
    - text (str): Text to process.
    - llm (LLMChain): The large language model to use for generating questions.
    - max_length (int): The maximum character length for each summary chunk.

    Returns:
    - str: The generated questions in a formatted string.
    """
    sentences = nest_sentences(text, max_length)
    questions = []  # List to hold generated questions
    seen_questions = set()  # Set to track unique questions

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Generate diverse multiple-choice questions/answer are one sentence long based on the context here: {text}. "
                 "Ensure each question is unique and not repetitive. "
                 "Format:\nQuestion: Question?\n- A) Option A.\n- B) Option B.\n- C) Option C.\n- D) Option D.\nAnswer: Answer\n***\n"
    )

    for chunk in tqdm(sentences, desc="Processing Text"):
        prompt = prompt_template.format(text=chunk)
        result = llm.invoke(prompt)
        result_lines = result.strip().split("\n")

        for line in result_lines:
            if line.startswith("Question:"):
                question = line.strip()
                if question not in seen_questions:
                    questions.append(question)
                    seen_questions.add(question)
                    answer = next((l.strip() for l in result_lines if l.startswith("Answer:")), "")
                    questions.append(answer)

    return "\n".join(questions)

#######################################
# CONFIGURATION FOR THE LANGUAGE MODEL
#######################################
llm = Ollama(model="llama3", temperature=0.9)

#############################
# STREAMLIT USER INTERFACE
#############################
st.title("MCQ Generator from PDF")
st.sidebar.header("Configuration")
max_length = st.sidebar.slider("Maximum Chunk Length", min_value=1024, max_value=8192, value=4096, step=512)
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

# Streamlit PDF Page Range Input
def extract_pages_from_stream(file_stream, start_page, end_page):
    """
    Extracts text from specific pages of a PDF file uploaded as a file-like stream.

    Parameters:
    - file_stream (UploadedFile): The uploaded PDF file.
    - start_page (int): Starting page number (1-indexed).
    - end_page (int): Ending page number (1-indexed).

    Returns:
    - str: Extracted text from the specified page range.
    """
    text = ""
    pdf_reader = PdfReader(file_stream)
    total_pages = len(pdf_reader.pages)

    # Ensure the page range is valid
    start_page = max(1, start_page)  # Ensure starting page is at least 1
    end_page = min(total_pages, end_page)  # Ensure ending page does not exceed total pages

    for i in range(start_page - 1, end_page):  # Pages are 0-indexed in PyPDF2
        text += pdf_reader.pages[i].extract_text()

    return text

# Streamlit Code
uploaded_file = st.file_uploader("Upload a PDF File", type="pdf")

if uploaded_file is not None:
    # Page range and file upload handling
    start_page = st.number_input("Start Page", min_value=1, value=1, step=1)
    end_page = st.number_input("End Page", min_value=1, value=1, step=1)

    if st.button("Extract and Generate MCQs"):
        with st.spinner("Extracting text and generating MCQs..."):
            try:
                text = extract_pages_from_stream(uploaded_file, start_page, end_page)

                if not text.strip():
                    st.error("No text extracted. Ensure the pages contain readable content.")
                    st.stop()

                st.text_area("Extracted Text (Preview)", text[:1000], height=300)
                
                test_prompt = "Generate a single MCQ based on this text: The Earth revolves around the Sun."
                test_output = llm.invoke(test_prompt)
                st.text_area("Test LLM Output", test_output)

                questions = generate_questions(text, llm, max_length=4096)

                if not questions:
                    st.error("No MCQs generated. Check the extracted text or LLM configuration.")
                else:
                    st.success("MCQs generated successfully!")
                    st.text_area("Generated MCQs", questions, height=400)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.error(f"Error: {e}")


