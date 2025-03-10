from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from .qwenllm import QwenLLM
from decouple import config
import traceback
import sys

router = APIRouter()

qwen_model_path = config("QWEN_MODEL_PATH")

# Initialize Qwen LLM with LangChain
try:
    print("🟢 Initializing Qwen LLM...")

    qwen_llm = QwenLLM(
        model_name=qwen_model_path,
        tokenizer_config={"eos_token": "<|im_end|>"},
        generate_kwargs={
            #"temperature": 0.7,
            #"repetition_penalty": 1.05,
            #"max_tokens": 512,
        }
    )

    print("✅ Qwen LLM initialized successfully!")

    # Memory for context tracking
    memory = ConversationBufferMemory(return_messages=True)

    print("✅ Memory system initialized!")

except Exception as e:
    print("❌ Error initializing Qwen LLM at line", sys.exc_info()[-1].tb_lineno, ":", e)
    print(traceback.format_exc())
    raise RuntimeError(f"Error initializing Qwen LLM: {e}")

@router.post("/qwen/chat")
async def qwen_chat(session_id: str, message: str):
    """
    Chat with Qwen2-7B using LangChain.
    """
    try:
        print("\n🔹 Received request: session_id =", session_id, "| message =", message)

        if not message.strip():
            print("❌ Error: Empty message received!")
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Load history messages
        history_data = memory.load_memory_variables({}).get("history", [])
        print("\n📜 Loaded history:", history_data)

        # Convert history to proper format (list of HumanMessage/AIMessage)
        formatted_history = []
        for msg in history_data:
            if isinstance(msg, dict):  # Handle if stored as dict
                if msg.get("type") == "human":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                else:
                    formatted_history.append(AIMessage(content=msg["content"]))
            elif isinstance(msg, str):  # Handle if stored as raw string
                formatted_history.append(AIMessage(content=msg))
            else:
                formatted_history.append(msg)  # Already in correct format

        print("\n✅ Formatted history:", formatted_history)

        # Prepare input (Only pass messages, avoid dict errors)
        messages = formatted_history + [HumanMessage(content=message)]

        # Pass messages list directly to `invoke()`
        print("\n🚀 Sending data to chatbot...")
        response = qwen_llm.invoke(messages)  # Directly call the model

        print("\n✅ Response from Qwen:", response)

        # Save conversation history
        memory.save_context({"input": message}, {"output": response})

        return {"response": response}

    except Exception as e:
        line_number = sys.exc_info()[-1].tb_lineno
        print(f"\n❌ Error occurred on line {line_number}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Qwen chatbot error (line {line_number}): {e}")
