import gradio as gr
import requests

def chat_with_mistral(user_input, history):
    try:
        # Make a streaming request to the FastAPI endpoint
        with requests.post("http://127.0.0.1:8000/api/mistral/chat/", json={"user_input": user_input}, stream=True) as response:
            if response.status_code == 200:
                generated_text = ""
                for chunk in response.iter_content(chunk_size=1024):  # Read chunks from the stream
                    if chunk:
                        generated_text += chunk.decode("utf-8")  # Decode text from bytes

                return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": generated_text}]
            else:
                return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": f"Error: {response.status_code}, {response.text}"}]
    except Exception as e:
        return history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": f"Error: {str(e)}"}]

chat_interface = gr.ChatInterface(
    fn=chat_with_mistral, 
    chatbot=gr.Chatbot(type="messages", format="markdown"), 
    title="B2B Knowledge Management",
    
)
if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=7860)
