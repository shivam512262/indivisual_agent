import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

def generate_medical_knowledge_base(output_dir="."):
    """
    Generates a FAISS index and text chunks from a sample medical knowledge base.
    In a real application, you would replace the sample_medical_text with
    a much larger and more comprehensive dataset of medical information.

    Args:
        output_dir (str): Directory where the FAISS index and text chunks will be saved.
    """
    print("Starting FAISS database generation process...")
    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory '{output_dir}' exists.")
    except Exception as e:
        print(f"ERROR: Could not create output directory: {e}")
        raise

    index_path = os.path.join(output_dir, "faiss_index.idx")
    chunks_file = os.path.join(output_dir, "text_chunks.pkl")

    # --- Step 1: Define your medical knowledge base ---
    print("Defining sample medical text...")
    sample_medical_text = """
    **Common Cold (Acute Viral Rhinopharyngitis)**
    Symptoms: Runny nose, sneezing, sore throat, cough, congestion, mild headache, low-grade fever.
    Cause: Viral infection, most commonly rhinovirus.
    Duration: Typically 7-10 days.
    Treatment: Rest, fluids, over-the-counter medications for symptom relief (e.g., pain relievers, decongestants). Antibiotics are ineffective.

    **Influenza (Flu)**
    Symptoms: Fever, body aches, chills, fatigue, cough, sore throat, runny or stuffy nose, headache. More severe than a cold.
    Cause: Influenza virus.
    Duration: 1-2 weeks, but fatigue can last longer.
    Treatment: Antiviral drugs (if started early), rest, fluids, symptom relief. Vaccination is highly recommended for prevention.

    **Streptococcal Pharyngitis (Strep Throat)**
    Symptoms: Sudden sore throat, pain when swallowing, fever, red and swollen tonsils (sometimes with white patches or streaks of pus), tiny red spots on the roof of the mouth (petechiae), headache, stomach ache, nausea, vomiting. Usually no cough or runny nose.
    Cause: Streptococcus pyogenes bacteria.
    Diagnosis: Rapid strep test or throat culture.
    Treatment: Antibiotics (e.g., penicillin, amoxicillin) to prevent complications like rheumatic fever.

    **Pneumonia**
    Symptoms: Cough (often with phlegm), fever, chills, shortness of breath, chest pain when breathing or coughing, fatigue, nausea, vomiting, diarrhea.
    Cause: Bacteria, viruses, or fungi.
    Diagnosis: Chest X-ray, blood tests.
    Treatment: Antibiotics (for bacterial), antiviral drugs (for viral), antifungal drugs (for fungal), oxygen therapy, rest, fluids.

    **Migraine**
    Symptoms: Severe throbbing headache, usually on one side of the head, sensitivity to light and sound, nausea, vomiting, aura (visual disturbances, numbness, speech changes) before or during the headache.
    Cause: Complex neurological disorder, often triggered by stress, certain foods, hormonal changes.
    Treatment: Pain relievers (NSAIDs), triptans, CGRP inhibitors, preventive medications (beta-blockers, antidepressants, anti-seizure drugs). Lifestyle changes.

    **Type 2 Diabetes**
    Symptoms: Increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, frequent infections, numbness or tingling in the hands or feet.
    Cause: Insulin resistance or insufficient insulin production.
    Diagnosis: Blood tests (HbA1c, fasting blood sugar, oral glucose tolerance test).
    Treatment: Diet and exercise, oral medications, insulin injections.

    **Hypertension (High Blood Pressure)**
    Symptoms: Often no symptoms (silent killer). Headaches, shortness of breath, nosebleeds (in severe cases).
    Cause: Primary (essential) hypertension with no identifiable cause, or secondary hypertension due to underlying conditions (kidney disease, thyroid problems).
    Diagnosis: Blood pressure measurement.
    Treatment: Lifestyle changes (diet, exercise, sodium reduction), medications (diuretics, ACE inhibitors, ARBs, beta-blockers, calcium channel blockers).

    **Appendicitis**
    Symptoms: Sudden pain that begins on the right side of the lower abdomen, sudden pain that begins around your navel and often shifts to your lower right abdomen, pain that worsens if you cough, walk or make other jarring movements, nausea and vomiting, loss of appetite, low-grade fever that may worsen as the illness progresses, constipation or diarrhea, abdominal bloating.
    Cause: Blockage in the lining of the appendix, resulting in infection.
    Diagnosis: Physical exam, blood test, urine test, imaging tests (ultrasound, CT scan, MRI).
    Treatment: Appendectomy (surgical removal of the appendix).

    **Gastroenteritis (Stomach Flu)**
    Symptoms: Diarrhea, vomiting, abdominal cramps, nausea, fever, headache, muscle aches.
    Cause: Viral (most common), bacterial, or parasitic infection.
    Duration: 1-3 days for viral, longer for bacterial/parasitic.
    Treatment: Rest, fluids to prevent dehydration. Antibiotics are generally not used for viral gastroenteritis.

    **Urinary Tract Infection (UTI)**
    Symptoms: Strong, persistent urge to urinate, a burning sensation when urinating, passing frequent, small amounts of urine, cloudy urine, strong-smelling urine, pelvic pain in women, rectal pain in men.
    Cause: Bacteria entering the urinary tract.
    Diagnosis: Urine test.
    Treatment: Antibiotics.
    """

    # --- Step 2: Chunk the text ---
    print("Chunking text...")
    text_chunks = [chunk.strip() for chunk in sample_medical_text.split('\n\n') if chunk.strip()]
    print(f"Generated {len(text_chunks)} text chunks.")

    # --- Step 3: Generate embeddings for the chunks ---
    try:
        print("Attempting to load SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load SentenceTransformer model: {e}")
        print("Please ensure you have an active internet connection or the model is downloaded locally.")
        raise # Re-raise the exception to fail the build process

    print("Generating embeddings...")
    try:
        chunk_embeddings = model.encode(text_chunks, show_progress_bar=True)
        dimension = chunk_embeddings.shape[1]
        print(f"Embeddings generated with dimension: {dimension}")
    except Exception as e:
        print(f"ERROR: Could not generate embeddings: {e}")
        raise

    # --- Step 4: Build a FAISS index ---
    print("Building FAISS index...")
    try:
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings).astype('float32'))
        print(f"FAISS index built with {index.ntotal} vectors.")
    except Exception as e:
        print(f"ERROR: Could not build FAISS index: {e}")
        raise

    # --- Step 5: Save the FAISS index and text chunks ---
    print(f"Saving FAISS index to: {index_path}")
    print(f"Saving text chunks to: {chunks_file}")
    try:
        faiss.write_index(index, index_path)
        with open(chunks_file, "wb") as f:
            pickle.dump(text_chunks, f)
        print("FAISS index and text chunks saved successfully!")
    except Exception as e:
        print(f"ERROR: Could not save FAISS files: {e}")
        raise

    print("\nFAISS database generation process completed.")

if __name__ == "__main__":
    try:
        generate_medical_knowledge_base()
    except Exception as e:
        print(f"FATAL ERROR during FAISS database generation: {e}")
        exit(1) # Ensure the script exits with a non-zero code on failure
