import requests
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)

# Replace with your Telegram Bot Token
TELEGRAM_BOT_TOKEN = "7950271563:AAFh-ZK04L929sRyBvarIfe6ruFR97SVFXQ"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
LLM_API_URL = "http://127.0.0.1:8000/v2/rag_chat/"

router = APIRouter()

class TelegramMessage(BaseModel):
    update_id: int
    message: dict

@router.post("/webhook")
async def telegram_webhook(update: TelegramMessage):
    """
    Handle Telegram webhook updates.
    """
    chat_id = update.message["chat"]["id"]
    user_text = update.message.get("text", "")

    logging.info(f"📩 Received message from {chat_id}: {user_text}")

    if not user_text:
        logging.warning("⚠️ No text found in message.")
        return {"status": "ignored"}

    try:
        # Send user input to LLM API
        response = requests.post(LLM_API_URL, json={"user_input": user_text, "n_results": 3})
        response.raise_for_status()  # Raise error if response code is not 200
        llm_response = response.json().get("response", "Sorry, no response from AI.")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Error contacting LLM API: {e}")
        raise HTTPException(status_code=500, detail="LLM API is unavailable.")

    # Send response back to Telegram
    success = send_message(chat_id, llm_response)
    
    return {"status": "ok" if success else "failed"}

def send_message(chat_id: int, text: str) -> bool:
    """
    Sends a message to the user via Telegram.
    """
    payload = {"chat_id": chat_id, "text": text}

    try:
        response = requests.post(TELEGRAM_API_URL, json=payload)
        response.raise_for_status()
        logging.info(f"✅ Message sent to {chat_id}: {text}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to send message to {chat_id}: {e}")
        return False
