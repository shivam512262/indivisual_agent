# Use a slim Python base image for smaller size
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install build essentials and other dependencies required by faiss-cpu and PyMuPDF
# apt-get update: Updates the list of available packages
# apt-get install -y --no-install-recommends: Installs packages without recommended dependencies
#   build-essential: Provides essential tools for compiling software (gcc, g++, make, libc-dev etc.)
#   pkg-config: Helps find libraries for compilation
#   libopenblas-dev: OpenBLAS development files, often a dependency for optimized numerical numerical libraries like FAISS
#   liblapack-dev: LAPACK development files, also common for numerical libraries
#   libjpeg-dev: Required by Pillow for JPEG image support
#   zlib1g-dev: Required by Pillow for PNG/Zlib support
#   freetyped-dev: Required by Pillow for font rendering
#   liblcms2-dev: Required by Pillow for Little CMS support
#   libwebp-dev: Required by Pillow for WebP image support
#   tcl-dev, tk-dev: Sometimes needed for tkinter, though not directly in your requirements, good for general Python dev
#   git: Needed by sentence-transformers for downloading models from Hugging Face if not cached
#   poppler-utils: Often needed for PDF processing tools, though PyMuPDF might not strictly require it, it's good practice
#   (Note: PyMuPDF often comes with its own binaries, but these general libs help)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    freetyped-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl-dev \
    tk-dev \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file first to leverage Docker caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# This step will now have the necessary build tools available for faiss-cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app/

# Run the FAISS database generation script
# This script will now execute after all dependencies are installed
RUN python generate_faiss_db.py

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run the FastAPI application
# --host 0.0.0.0 is crucial for Docker containers to be accessible externally
# --port $PORT is used by Railway to inject the correct port
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
# Note: Railway often overrides the port with its own $PORT env var,
# but providing 8000 here as a default is good practice.
