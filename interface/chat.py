import gradio as gr
import requests
import json  # Ensure we handle JSON properly

def chat_with_mistral(user_input, history):
    try:
        # Start a streaming request to the FastAPI endpoint
        with requests.post("http://127.0.0.1:8000/v2/rag_chat/", json={"user_input": user_input}, stream=True) as response:
            print("🔹 Gradio Received HTTP Status:", response.status_code)  # ✅ Debugging

            if response.status_code == 200:
                generated_text = ""

                # Read the streaming response line by line
                for line in response.iter_lines(decode_unicode=True):
                    print("🔹 Received Chunk from FastAPI:", line)  # ✅ Debugging log

                    if line:
                        try:
                            response_json = json.loads(line.strip())  # ✅ Ensure JSON parsing
                            chunk = response_json.get("response", "")  # ✅ Extract correct response
                        except json.JSONDecodeError:
                            chunk = line.strip()  # Fallback to raw text if JSON parsing fails

                        generated_text += chunk + "\n"

                        # Update chat history live
                        yield history + [{"role": "user", "content": f"**You:** {user_input}"},
                                         {"role": "assistant", "content": generated_text.strip()}]

                return history + [{"role": "user", "content": f"**You:** {user_input}"},
                                  {"role": "assistant", "content": generated_text.strip()}]
            else:
                print("🔴 Gradio Received Error Response:", response.text)  # ✅ Debugging log
                return history + [{"role": "user", "content": user_input}, 
                                  {"role": "assistant", "content": f"⚠️ Error: {response.status_code}, {response.text}"}]
    except Exception as e:
        print("🔴 Gradio Encountered Exception:", str(e))  # ✅ Debugging log
        return history + [{"role": "user", "content": user_input}, 
                          {"role": "assistant", "content": f"❌ Error: {str(e)}"}]

# Create the Gradio Chat Interface with Markdown support
chat_interface = gr.ChatInterface(
    fn=chat_with_mistral,
    type="messages",  # ✅ Enables streaming
    title="📚 B2B Knowledge Management",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    autofocus=True,
    autoscroll=True,
    stop_btn=True
)

if __name__ == "__main__":
    chat_interface.launch(server_name="0.0.0.0", server_port=7860)
