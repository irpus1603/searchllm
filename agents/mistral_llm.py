from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache
from typing import List
import time

router = APIRouter()

try:
    model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    prompt_cache = make_prompt_cache(model)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
except Exception as e:
    raise RuntimeError(f"Error initializing Mistral {e}")

class Message(BaseModel):
    role:str
    content:str

class ChatRequest(BaseModel):
    messages: List[Message]

@router.post("/mistral/chat")
async def mistral_chat(request: ChatRequest):
    try:
        print("🚀 Received Payload:", request.dict())  # Debugging log

        # ✅ Load conversation history
        chat_history = memory.load_memory_variables({})["history"]

        messages = []
        for msg in chat_history:
            if isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})

        # ✅ Extract last user message
        if request.messages:
            latest_user_message = request.messages[-1].content  # Fix
        else:
            raise HTTPException(status_code=400, detail="No user message found")

        # ✅ Avoid duplicate user messages in history
        if not messages or messages[-1]["content"] != latest_user_message:
            messages.append({"role": "user", "content": latest_user_message})

        # ✅ Apply Chat Template
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # ✅ Generate LLM Response
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
            verbose=False,
            prompt_cache=prompt_cache,
            max_tokens=20000
        )

        # ✅ Save Context WITHOUT Repeating Messages
        memory.save_context({"input": latest_user_message}, {"output": response})

        # ✅ Format Response
        markdown_response = f"### Response:\n\n{response}\n\n"

        return {"response": markdown_response}

    except Exception as e:
        print("❌ ERROR in mistral_chat():", str(e))  # Debugging log
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
