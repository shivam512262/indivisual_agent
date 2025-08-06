import os
import shutil
import tempfile
import base64
from io import BytesIO
import asyncio
from typing import Optional
from pydantic import BaseModel
import datetime
from PIL import Image

# Import dotenv to load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file as early as possible
# This ensures that os.getenv() will work for variables defined in .env
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response

# Import all necessary agents from the consolidated main.py
# We will NOT import genai directly here, as its configuration needs to happen after dotenv.
from main import mainAgent, ICULogAnalysisAgent, PDFGeneratorAgent, imgClassifier, queryAnalysis, retrieve_and_answer
import google.generativeai as genai # Import genai directly here for configuration

app = FastAPI(
    title="Cureify Medical Decision Support API",
    description="API for medical diagnosis and query answering with image/PDF support, and ICU log analysis.",
    version="1.0.0",
)

# --- FastAPI Startup Event ---
# This event runs once when the application starts up, AFTER dotenv is loaded.
@app.on_event("startup")
async def startup_event():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("CRITICAL ERROR: GOOGLE_API_KEY environment variable is not set. API calls will fail.")
        # You might want to raise an exception here to prevent the app from starting
        # raise ValueError("GOOGLE_API_KEY is not set!")
    else:
        print(f"DEBUG: GOOGLE_API_KEY loaded successfully (first 5 chars): {api_key[:5]}*****")
        genai.configure(api_key=api_key)
        print("DEBUG: Google Generative AI configured.")

# Pydantic model for ICU Log Analysis request
class ICULogRequest(BaseModel):
    icu_log_data: str

# Pydantic model for Image Processing request (not directly used by endpoint, but good for clarity)
class ImageProcessRequest(BaseModel):
    prompt: Optional[str] = None
    image: str # Base64 encoded image string is expected here

# Pydantic model for Text Report Generation request
class TextReportRequest(BaseModel):
    prompt: str

@app.post("/analyze_icu_log", summary="Analyze ICU logs and generate a PDF report")
async def analyze_icu_log(request: ICULogRequest):
    """
    Analyzes provided ICU logs and generates a detailed medical report in PDF format.

    - **icu_log_data**: A string containing ICU logs or patient information.
    """
    try:
        # Step 1: Call ICULogAnalysisAgent to get the detailed report string
        detailed_report = await asyncio.to_thread(ICULogAnalysisAgent, request.icu_log_data)
        
        # Step 2: Call PDFGeneratorAgent to create the PDF from the report string
        pdf_bytes_io = await asyncio.to_thread(PDFGeneratorAgent, detailed_report)
        
        # Generate a unique filename for the PDF
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"icu_log_report_{timestamp}.pdf"

        # Step 3: Return the PDF file to the client with Content-Disposition header
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        return Response(content=pdf_bytes_io.getvalue(), media_type="application/pdf", headers=headers)
    except Exception as e:
        # Catch potential RuntimeError from ICULogAnalysisAgent if API key is still an issue
        if "No API_KEY" in str(e) or "authentication" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"API Key issue during ICU log analysis: {e}. Please check your GOOGLE_API_KEY.")
        raise HTTPException(status_code=500, detail=f"An error occurred during ICU log analysis or PDF generation: {e}")

@app.post("/process_image", summary="Process an image with an optional medical query")
async def process_image(
    prompt: Optional[str] = Form(None, description="An optional medical query related to the image."),
    file: UploadFile = File(..., description="The image file (PNG, JPG) to process.")
):
    """
    Processes an uploaded image to perform disease identification, OCR, or general image query.

    - **prompt**: An optional text query related to the image.
    - **file**: The image file (PNG, JPG).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload an image (PNG, JPG).")

    img_data = None
    try:
        file_content = await file.read()
        img_data = Image.open(BytesIO(file_content))
        if img_data.mode in ("RGBA", "P"):
            img_data = img_data.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image file: {e}")

    try:
        result = await asyncio.to_thread(imgClassifier, img_data, prompt if prompt else "")
        return JSONResponse(content={"response_text": result})
    except Exception as e:
        # Catch potential RuntimeError from imgClassifier if API key is still an issue
        if "No API_KEY" in str(e) or "authentication" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"API Key issue during image processing: {e}. Please check your GOOGLE_API_KEY.")
        raise HTTPException(status_code=500, detail=f"An error occurred during image processing: {e}")

@app.post("/generate_text_report", summary="Generate a text-based medical report from a prompt")
async def generate_text_report(request: TextReportRequest):
    """
    Generates a text-based medical report based on the provided symptoms or general medical query.

    - **prompt**: Your medical symptoms or general medical query.
    """
    try:
        if 'symptom' in request.prompt.lower():
            result = await asyncio.to_thread(retrieve_and_answer, request.prompt)
        else:
            result = await asyncio.to_thread(queryAnalysis, request.prompt)
            
        return JSONResponse(content={"response_text": result})
    except Exception as e:
        # Catch potential RuntimeError from agents if API key or FAISS files are missing
        if "No API_KEY" in str(e) or "authentication" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"API Key issue during text report generation: {e}. Please check your GOOGLE_API_KEY.")
        elif isinstance(e, RuntimeError) and "Medical knowledge base files missing" in str(e):
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during text report generation: {e}")

