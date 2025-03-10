import requests
from langchain.memory import ConversationBufferMemory
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from uuid import UUID

router = APIRouter()

# Define API endpoint
LLM_API_URL = "http://10.34.161.90:11001/v1/chat/completions"

# Dictionary to store user conversation memory
user_memory: Dict[str, ConversationBufferMemory] = {}

class Message(BaseModel):
    role: str  # Can be "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    user_id: UUID
    messages: List[Message]

def get_user_memory(user_id: str):
    """Retrieve or create conversation memory for the user."""
    if user_id not in user_memory:
        user_memory[user_id] = ConversationBufferMemory(return_messages=True)

    if len(user_memory[user_id].chat_memory.messages) > 1:
        user_memory[user_id].chat_memory.messages = user_memory[user_id].chat_memory.messages[-1:]
    return user_memory[user_id]

@router.post("/sahabat/chat")
async def sahabat_chat(chat_request: ChatRequest):
    try:
        user_id = str(chat_request.user_id)
        memory = get_user_memory(user_id)

        if not chat_request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_message = chat_request.messages[-1].content  # Latest user message

        # Build the conversation history for API request
        messages_payload = [
            {"role": msg.role, "content": msg.content}
            for msg in chat_request.messages  # Preserve original roles ("user" & "assistant")
        ]

        # Add system instructions first
        messages_payload.insert(0, {
            "role": "system",
            "content": "Kamu adalah Sahabat, AI asisten yang menjawab dengan singkat dan jelas"
        })

        # Append the latest user message (ensures it's last in sequence)
        messages_payload.append({"role": "user", "content": user_message})

        # Construct request payload
        payload = {
            "model": "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
            "messages": messages_payload,
            "max_tokens": 8000,
            "temperature": 0.7
        }

        # Send request to the LLM API
        response = requests.post(LLM_API_URL, json=payload, headers={"Content-Type": "application/json"})

        # Check response status
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        # Extract AI response
        ai_response_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        # Store conversation in memory
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(ai_response_text)

        return {"assistant": ai_response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Sahabat API error: {e}")
