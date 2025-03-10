from fastapi import APIRouter, HTTPException, Depends
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from typing import List, Dict
from pydantic import BaseModel
from uuid import UUID

router = APIRouter()

try:
    gemini_api_key = config('GEMINI_API_KEY')
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)
except Exception as e:
    raise RuntimeError(f"Error initializing Gemini: {e}")

# Dictionary to store conversation memory per user
user_memory: Dict[str, ConversationBufferMemory] = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: UUID  # Unique user identifier
    messages: List[Message]

def get_user_memory(user_id: str):
    """Retrieve or create conversation memory for a specific user."""
    if user_id not in user_memory:
        user_memory[user_id] = ConversationBufferMemory(return_messages=True)
    return user_memory[user_id]

@router.post("/gemini/chat")
async def gemini_chat(chat_request: ChatRequest):
    try:
        user_id = str(chat_request.user_id)  # Convert UUID to string
        memory = get_user_memory(user_id)  # Retrieve user-specific memory
        
        # Extract latest user message
        if not chat_request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        user_message = chat_request.messages[-1].content  # Last message from user

        # Retrieve chat history
        history = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_request.messages[:-1]])

        # LangChain Prompt Template
        prompt = PromptTemplate(
            input_variables=["history", "message"], 
            template="{history}\nUser: {message}\nAI:"
        )

        # Initialize LLMChain with memory
        llm_chain = LLMChain(
            llm=gemini_llm,
            prompt=prompt,
            memory=memory
        )

        # Generate response
        response = llm_chain.predict(history=history, message=user_message)

        # Store conversation in memory
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(response)

        return {"response": response, "history": memory.load_memory_variables({})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
