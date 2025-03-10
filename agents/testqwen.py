from qwen import QwenLLM
from decouple import config

qwen_llm = config("QWEN_MODEL_PATH")

# Initialize the model
qwen_llm = QwenLLM(model_name=qwen_llm)

# Print available attributes and methods
print("Supported Attributes:", dir(qwen_llm))
print("Available generate_step parameters:", qwen_llm.generate.__doc__)  # Check expected arguments
