import chromadb
import fitz  # PyMuPDF for PDF processing
import docx
from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uuid
import os
import json

# Initialize FastAPI Router
router = APIRouter()

# Initialize ChromaDB (Persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="B2B_KM")

# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pydantic model for query request
class QueryRequest(BaseModel):
    query_text: str
    n_results: int = 3

# Function to extract text from different file types
def extract_text_from_file(file_path: str, file_type: str):
    text = ""

    if file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    elif file_type == "pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"

    elif file_type == "docx":
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    
    elif file_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            text = json.dumps(json_data, indent=2)  # Convert JSON to formatted text


    return text.strip()

# Route: Upload file and store document embeddings
@router.post("/add_document/")
async def add_document(file: UploadFile = File(...)):
    try:
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["txt", "pdf", "docx","json"]:
            raise HTTPException(status_code=400, detail="Only .txt, .pdf, json, and .docx files are supported")

        # Save the uploaded file temporarily
        temp_file_path = f"temp_{uuid.uuid4()}.{file_extension}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Extract text from the file
        text = extract_text_from_file(temp_file_path, file_extension)

        # Delete the temporary file
        os.remove(temp_file_path)

        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from the file")

        # Generate unique ID for the document
        doc_id = str(uuid.uuid4())

        # Generate embedding for the extracted text
        embedding = embedding_model.encode(text).tolist()

        # Store document and embedding in ChromaDB
        collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding]
        )

        return {"message": "Document added successfully", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route: Query similar documents
@router.post("/query/")
def query_document(request: QueryRequest):
    try:
        query_embedding = embedding_model.encode(request.query_text).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],  # Fixed parameter name
            n_results=request.n_results
        )

        if not results["documents"]:
            return {"message": "No matching documents found"}

        return {"matches": results["documents"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
