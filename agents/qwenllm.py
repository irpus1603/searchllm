from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult
from typing import List, Optional, Any
from pydantic import PrivateAttr
from mlx_lm import load, generate
import mlx.core as mx
from decouple import config

qwen_model_path = config("QWEN_MODEL_PATH")

class QwenLLM(BaseLLM):
    model_name: str = qwen_model_path
    tokenizer_config: Optional[dict] = {"eos_token": "<|im_end|>"}

    # Define default generation parameters
    generate_kwargs: dict = {
        #"max_tokens": 512,  # Keep max tokens
        #"temperature": 0.7,  # Add temperature
        #"top_p": 0.8,  # Add top_p for nucleus sampling
    }

    # Exclude model and tokenizer from validation
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model and tokenizer using mlx_lm's latest API
        self._model, self._tokenizer = load(self.model_name, tokenizer_config=self.tokenizer_config)

    def _sampling_function(self, logits: mx.array) -> mx.array:
        """
        Implements temperature scaling and top-p (nucleus) sampling manually.
        """
        top_p = self.generate_kwargs.get("top_p", 0.8)
        temperature = self.generate_kwargs.get("temperature", 0.7)

        # Step 1: Apply temperature scaling
        logits = logits / temperature  # Lower temperature → confident predictions, Higher → diverse output

        # Step 2: Convert logits to probabilities
        probs = mx.softmax(logits, axis=-1)

        # Step 3: Sort probabilities (fix for new mlx_lm version)
        sorted_indices = mx.argsort(probs, axis=-1)  # Get indices of sorted elements
        sorted_probs = mx.take_along_axis(probs, sorted_indices[:, ::-1], axis=-1)  # Reverse order manually

        # Step 4: Compute cumulative probability for nucleus (top-p) sampling
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Step 5: Mask tokens that exceed the top-p threshold
        mask = cumulative_probs <= top_p
        sorted_probs = sorted_probs * mask

        # Step 6: Renormalize probabilities
        return mx.softmax(sorted_probs, axis=-1)

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """
        Generate responses for a list of prompts.
        """
        responses = []
        for prompt in prompts:
            # Prepare the input prompt for Qwen LLM
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Call generate() with temperature control through sampler
            response = generate(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=text,
                max_tokens=self.generate_kwargs["max_tokens"],  # Explicitly pass max_tokens
                sampler=self._sampling_function,  # Apply custom temperature and top_p handling
            )
            responses.append(response)

        # Return responses formatted for LangChain
        return LLMResult(generations=[[{"text": response}] for response in responses])

    @property
    def _llm_type(self) -> str:
        return "QwenLLM"
