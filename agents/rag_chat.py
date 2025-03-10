import asyncio
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import tiktoken  # For token counting

# Create an API Router
router = APIRouter()

# Initialize ChromaDB (Persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")  
collection = chroma_client.get_or_create_collection(name="B2B_KM")

# Load Sentence Transformer for embedding generation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# External LLM API Endpoint
LLM_API_URL = "http://127.0.0.1:8000/api/sahabat/chat"

# Define token limits
MAX_TOKENS = 7000  # LLM API token limit
MAX_RETRIEVAL_TOKENS = 5000  # Limit document retrieval to 5000 tokens
TOKEN_BUFFER = 500  # Buffer space to prevent overflow

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI tokenizer


class ChatRequest(BaseModel):
    user_input: str
    n_results: int = 6  # Retrieve more documents for better accuracy


def count_tokens(text: str) -> int:
    """Returns the token count of a given text."""
    return len(tokenizer.encode(text))


def truncate_context(context_list: list) -> str:
    """
    Trims retrieved documents to ensure total tokens do not exceed MAX_RETRIEVAL_TOKENS.
    Prioritizes keeping the latest content while fitting within the token limit.
    """
    total_tokens = 0
    truncated_context = []

    for doc in context_list:
        doc_tokens = count_tokens(doc)
        if total_tokens + doc_tokens > MAX_RETRIEVAL_TOKENS:
            break  # Stop adding documents if exceeding retrieval limit
        truncated_context.append(doc)
        total_tokens += doc_tokens

    return "\n\n---\n\n".join(truncated_context)


async def stream_llm_response(context: str, user_input: str):
    """Streams the response from the LLM API while enforcing token limits."""
    
    system_prompt = (
        "You are a highly knowledgeable AI assistant with access to a knowledge base.\n"
        "Use ONLY the provided context to answer the user's question.\n"
        "If the information is missing from the context, say 'I don't have enough information on this topic.'\n"
        "Do NOT use any external knowledge.\n\n"
        "Respond using markdown formatting: use **bold** for important words, *italic* for emphasis, and "
        "color code response using <span style='color:red;'>Red text</span>."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"### CONTEXT ###\n{context}\n\n### QUESTION ###\n{user_input}"}
    ]

    # Adjust max_tokens to fit within API constraints
    input_token_count = count_tokens(context) + count_tokens(user_input)
    max_tokens_allowed = max(1024, min(2048, MAX_TOKENS - input_token_count - TOKEN_BUFFER))

    payload = {
        "user_id": str(uuid.uuid4()),
        "messages": messages,
        "max_tokens": max_tokens_allowed,
        "temperature": 0.7
    }

    print("🚀 Sending Payload to LLM API:", payload)  # Debugging log

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", LLM_API_URL, json=payload, timeout=200) as response:
                print("🔹 LLM API Response Status:", response.status_code)  # ✅ Debugging

                if response.status_code != 200:
                    yield f"Error: LLM API returned status {response.status_code}\n"
                    return

                empty_response = True  # ✅ Flag to check if anything was received

                async for chunk in response.aiter_bytes():
                    decoded_chunk = chunk.decode("utf-8").strip()
                    print("🔹 Received Chunk:", decoded_chunk)  # ✅ Debugging log
                    if decoded_chunk:
                        yield decoded_chunk
                        empty_response = False

                if empty_response:
                    yield "Error: No response received from LLM API\n"

        except httpx.HTTPError as e:
            yield f"Error: Unable to connect to LLM API: {str(e)}\n"
            return
        except Exception as e:
            yield f"Unexpected Error: {str(e)}\n"
            return


import json

@router.post("/rag_chat/")
async def rag_chat(request: ChatRequest):
    try:
        print("🚀 Received Request at /v2/rag_chat/:", request.dict())  # ✅ Debugging log

        # Generate embedding for user input
        query_embedding = embedding_model.encode(request.user_input).tolist()

        # Count total available documents in ChromaDB
        total_docs = collection.count()
        print("📊 Total documents in ChromaDB:", total_docs)  # ✅ Debugging log

        if total_docs == 0:
            return {"message": "No documents available in the knowledge base."}

        # Retrieve relevant documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, total_docs)
        )

        # Extract document content
        relevant_docs = results.get("documents", [])
        flat_docs = [doc[0] if isinstance(doc, list) and len(doc) > 0 else doc for doc in relevant_docs if doc]

        # Ensure retrieved context does not exceed token limits
        context = truncate_context(flat_docs) if flat_docs else "I don't have enough information on this topic."

        print("📌 Final context sent to LLM API:", context)  # ✅ Debugging log

        # Send request to LLM API
        payload = {
            "user_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": context}]
        }

        print("🔹 Sending Request to LLM API:", json.dumps(payload, indent=2))  # ✅ Debugging log

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://127.0.0.1:8000/api/sahabat/chat",
                json=payload,
                timeout=200
            )

        print("🔹 LLM API Response Status:", response.status_code)  # ✅ Debugging log
        print("🔹 LLM API Response Text:", response.text)  # ✅ Debugging log

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"LLM API Error: {response.text}")

        # Parse AI response
        response_data = response.json()
        ai_response_text = response_data.get("assistant", "Error: No response from AI.")

        return {"ai_response": ai_response_text}

    except Exception as e:
        print("🚨 ERROR in /v2/rag_chat/:", str(e))  # ✅ Debugging log
        return PlainTextResponse(content=f"Internal Server Error: {str(e)}", status_code=500)
