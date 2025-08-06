import os
import PIL.Image
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import asyncio
from fpdf import FPDF
from io import BytesIO
import datetime # Import for date in PDF
import re # Import for regular expressions to handle Markdown

# --- Configure Google Generative AI (once) ---
# This configuration will now happen only once in app_fastapi.py's startup_event
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY')) # REMOVED from here

# --- SentenceTransformer Model (for symptoms) ---
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an active internet connection or the model is downloaded locally.")
    sbert_model = None

# --- Wound Analysis Agent ---
def woundAnalysis(img, prompt):
    """
    Analyzes an image of a wound and provides a medical assessment.
    """
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    base_prompt = (
        "You are a medical specialist. Analyze the provided image of a wound "
        "and the user's prompt. Provide a detailed medical assessment of the wound, "
        "including its type, severity, potential complications, and recommended immediate actions. "
        "Do not provide prescriptions or treatment advice, but suggest consulting a doctor."
    )
    response = model.generate_content([base_prompt, img, prompt])
    return response.text

# --- X-ray Analysis Agent ---
def xrayAnalysis(img, prompt):
    """
    Analyzes an X-ray image and provides a medical assessment.
    """
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    base_prompt = (
        "You are a medical specialist. Analyze the provided X-ray image "
        "and the user's prompt. Provide a detailed medical assessment based on the X-ray, "
        "identifying any abnormalities, potential diagnoses, and suggesting further investigations if needed. "
        "Do not provide prescriptions or treatment advice, but suggest consulting a doctor."
    )
    response = model.generate_content([base_prompt, img, prompt])
    return response.text

# --- Image Query Agent ---
def imgQuery(img, prompt):
    """
    Answers a medical query based on a provided image and text prompt.
    """
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    base_prompt = (
        "You are a medical specialist and your task is to answer the query based on the given image "
        "and the prompt. If it is not a medical thing then just answer 'It is not a valid medical input.'"
    )

    response = model.generate_content([base_prompt, img, prompt])
    return response.text

# --- Query Analysis Agent ---
def queryAnalysis(prompt):
    """
    Analyzes a medical query and provides a detailed answer with suggested tests.
    """
    llm = GoogleGenerativeAI(
        model='gemini-2.0-flash',
        temperature=0,
        api_key=os.getenv('GOOGLE_API_KEY'),
        max_tokens=None,
        timeout=30,
        max_retries=2
    )
    input_prompt = ChatPromptTemplate.from_messages([
        (
            'system', "You are talking to a doctor. You're a medical specialist and your task is to provide a detailed answer for the given query. Also suggest the tests that need to be taken. The answer should be in depth and approximately 300 words."
        ),
        ('user', "{input}")
    ])
    chain = input_prompt | llm
    response = chain.invoke({'input':prompt})
    return response.content if hasattr(response, 'content') else str(response)

# --- OCR Agent ---
def OCR(img, prompt):
    """
    Performs OCR on an image (e.g., prescription) and combines the extracted text with the prompt
    for further analysis.
    """
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    base_prompt = "Extract all the text from the image in a clear and well defined manner."
    response = model.generate_content([base_prompt, img])
    output = queryAnalysis(response.text + " " + prompt)
    return output

# --- Symptoms Agent (RAG) ---
def load_vector_db(index_path="faiss_index.idx", chunks_file="text_chunks.pkl"):
    """
    Loads the FAISS index and text chunks from disk.
    """
    if not os.path.exists(index_path) or not os.path.exists(chunks_file):
        raise FileNotFoundError(
            f"Medical knowledge base files not found. Please ensure '{index_path}' and '{chunks_file}' exist in the same directory as this script."
        )
    index = faiss.read_index(index_path)
    with open(chunks_file, "rb") as f:
        text_chunks = pickle.load(f)
    return index, text_chunks

def get_text_embeddings(text):
    """
    Generates embeddings for the given text using the global sbert_model.
    """
    if sbert_model is None:
        raise RuntimeError("SentenceTransformer model not loaded. Cannot generate embeddings.")
    if not text.strip():
        return np.zeros(sbert_model.get_sentence_embedding_dimension())
    embeddings = sbert_model.encode([text])
    return embeddings[0]

def answer_generation_symptoms(input_text):
    """
    Generates an answer based on the provided input text using a medical specialist AI.
    """
    llm = GoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        api_key=os.getenv('GOOGLE_API_KEY'),
        max_tokens=None,
        timeout=30,
        max_retries=2
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''Role: You are a Medical Diagnosis Specialist AI with deep expertise in diseases, symptoms, and medical conditions. Your task is to analyze user symptoms and identify the most probable diagnosis.
Instructions: You are talking to a doctor.
Diagnosis: If symptoms clearly indicate a disease, provide a concise yet detailed explanation.
Also based on recommended diseases, suggest the particular tests to be taken.
Clarification: If multiple conditions match, Give the report on top 2 relevant options and strictly don't ask any questions.
Uncertainty: If data is insufficient, respond with:
"I can't make a definitive diagnosis based on the given data. Please provide more details."
Guidance: Offer medical insights but do not provide prescriptions or treatment advice—recommend consulting a doctor when necessary.
Keep responses accurate, structured, and professional while maintaining an empathetic tone.'''),
        ("human", "{Question}")
    ])
    chain = prompt | llm
    response = chain.invoke({"Question": input_text})
    return response.content if hasattr(response, 'content') else str(response)

def query_vector_db_with_rag(query_text, index, text_chunks, k=3):
    """
    Queries the FAISS vector database with RAG to retrieve relevant text chunks.
    """
    query_embedding = np.array(get_text_embeddings(query_text)).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)
    return context

def retrieve_and_answer(query_text, index_path="faiss_index.idx", chunks_file="text_chunks.pkl", k=4):
    """
    Retrieves relevant information from the vector database and generates an answer.
    """
    try:
        index, stored_chunks = load_vector_db(index_path, chunks_file)
        context = query_vector_db_with_rag(query_text, index, stored_chunks, k)
        response = answer_generation_symptoms(f"Context: {context}\nQuestion: {query_text}")
        return response
    except FileNotFoundError as e:
        raise RuntimeError(f"Medical knowledge base files missing: {e}. Please ensure FAISS index and chunks are correctly placed.")
    except Exception as e:
        raise RuntimeError(f"An error occurred during retrieval and answering: {e}")

# --- Structured Output Agent (currently unused in mainAgent, but included for completeness) ---
def structAgent(prompt, output):
    """
    Generates a structured output based on a given prompt and its answer.
    """
    llm = GoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        api_key=os.getenv('GOOGLE_API_KEY'),
        max_tokens=None,
        timeout=30,
        max_retries=2
    )
    input_prompt = ChatPromptTemplate.from_messages([
        (
            'system', "You're an output structure generator, your task is to generate a structured output based on the given prompt and its answer. Make sure that the output should properly answer the prompt in a simple and easy to understand manner. Here's the answer: {answer}"
        ),
        MessagesPlaceholder("chat_history"),
        ('user', "{input}")
    ])

    chain = input_prompt | llm
    response = chain.invoke({'input':prompt, 'answer':output})
    return response.content if hasattr(response, 'content') else str(response)

# --- ICULogAnalysisAgent ---
def ICULogAnalysisAgent(icu_log_data: str) -> str:
    """
    Analyzes ICU logs or patient information and generates a detailed report.
    """
    llm = GoogleGenerativeAI(
        model='gemini-2.0-flash',
        temperature=0.2,
        api_key=os.getenv('GOOGLE_API_KEY'),
        max_tokens=1500,
        timeout=60,
        max_retries=3
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a highly skilled medical specialist. Your task is to analyze the provided ICU logs and patient information. "
         "Generate a comprehensive and detailed report based on the data. The report should summarize the patient's current status, "
         "highlight any significant changes or critical events, identify trends in vital signs or other metrics, and provide a professional assessment. "
         "Conclude the report with a clear and prominent disclaimer stating that this is a system-generated analysis and should not replace professional medical judgment. "
         "Always recommend consulting a qualified physician or medical team."
        ),
        ("user", "{icu_log_data}")
    ])

    chain = prompt_template | llm
    response = chain.invoke({'icu_log_data': icu_log_data})
    return response.content if hasattr(response, 'content') else str(response)

# --- PDFGeneratorAgent ---
def PDFGeneratorAgent(report_string: str) -> BytesIO:
    """
    Generates a PDF file from a given detailed medical report string, with improved aesthetics.
    """
    pdf = FPDF()
    pdf.add_page()

    # Set margins
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.set_top_margin(20)
    pdf.set_auto_page_break(auto=True, margin=15)

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ICU Patient Log Analysis Report", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Date: {datetime.date.today().strftime('%Y-%m-%d')}", 0, 1, "C")
    pdf.ln(10) # Line break

    # Content
    pdf.set_font("Arial", "", 12)
    # Process the report string line by line to handle Markdown formatting
    lines = report_string.split('\n')
    for line in lines:
        line = line.strip()
        if not line: # Skip empty lines, add a small line break for paragraph separation
            pdf.ln(5)
            continue

        # Handle ## Headings
        if line.startswith('## '):
            pdf.set_font("Arial", "B", 14) # Larger bold font for headings
            pdf.multi_cell(0, 10, line[3:].encode('latin-1', 'replace').decode('latin-1')) # Remove '## '
            pdf.set_font("Arial", "", 12) # Reset font to normal
            pdf.ln(2) # Small break after heading
            continue

        # Handle **bold text** and *list items*
        processed_line = line
        
        # Process bold text: find **text** and replace with bolded version
        # Use a regex to find bolded sections
        def replace_bold(match):
            text_content = match.group(1)
            # FPDF doesn't directly support inline bolding in multi_cell like HTML.
            # We'll print bolded segments separately.
            return f"__BOLD_START__{text_content}__BOLD_END__"
        
        processed_line = re.sub(r'\*\*(.*?)\*\*', replace_bold, processed_line)

        # Process list items (simple simulation with indentation)
        if processed_line.startswith('* '):
            pdf.set_x(pdf.get_x() + 5) # Indent for list item
            processed_line = processed_line[2:] # Remove '* '

        # Print the line, handling bold segments
        parts = processed_line.split('__BOLD_START__')
        for i, part in enumerate(parts):
            if '__BOLD_END__' in part:
                bold_text, remaining_text = part.split('__BOLD_END__', 1)
                pdf.set_font("Arial", "B", 12) # Set bold font
                pdf.write(8, bold_text.encode('latin-1', 'replace').decode('latin-1'))
                pdf.set_font("Arial", "", 12) # Reset to normal font
                pdf.write(8, remaining_text.encode('latin-1', 'replace').decode('latin-1'))
            else:
                pdf.write(8, part.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(8) # New line after processing parts of a line
        
    pdf.ln(10) # Add space before footer

    # Footer
    pdf.set_y(-15) # Position 1.5 cm from bottom
    pdf.set_font("Arial", "I", 8) # Italic font
    pdf.cell(0, 10, f"Page {pdf.page_no()}/{{nb}}", 0, 0, "C") # Page number

    # Get the PDF content as bytes directly from pdf.output()
    pdf_content_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output = BytesIO(pdf_content_bytes)
    pdf_output.seek(0) # Rewind to the beginning of the stream
    return pdf_output

# --- Image Classifier Agent ---
def imgClassifier(img, prompt):
    """
    Classifies the type of image and routes it to the appropriate image analysis agent.
    """
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    base_prompt = (
        "You are an image classifier agent, you have to classify the image into 3 categories: "
        "if the image is an xray, respond 'xray'. "
        "If the image is a prescription or medicine image, respond 'ocr'. "
        "If it is some kind of wound or accident image, respond 'wound'. "
        "If it's anything other than this, respond 'other'. "
        "Always reply with only one of the given options in all lowercase."
    )
    
    response = model.generate_content([base_prompt, img])
    
    if 'wound' in response.text:
        output = woundAnalysis(img, prompt)
    elif 'xray' in response.text:
        output = xrayAnalysis(img, prompt)
    elif 'ocr' in response.text:
        output = OCR(img, prompt)
    else:
        output = imgQuery(img, prompt)
        
    return output

# --- Router Agent (Simplified for explicit API endpoint calls) ---
def routerAgent(img, prompt):
    """
    Routes the input (image or text prompt) to the appropriate agent for general medical queries.
    This simplified router is intended for use when FastAPI endpoints explicitly handle
    ICU log analysis and image processing.
    """
    if img:
        imgAnalysis = imgClassifier(img, prompt)
        return imgAnalysis
    
    if 'symptom' in prompt.lower():
        result = retrieve_and_answer(prompt)
        return result
    else:
        queryOutput = queryAnalysis(prompt)
        return queryOutput

# --- Main Agent Function ---
def mainAgent(prompt, img=None):
    """
    Main agent function to route the medical query based on input type.
    This function is primarily for general medical queries (image, symptoms, query).
    ICU log analysis is handled by a dedicated FastAPI endpoint.
    """
    output = routerAgent(img, prompt)
    return output

