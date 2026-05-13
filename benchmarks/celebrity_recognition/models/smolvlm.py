import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

from .base import VLMBase


class SmolVLM(VLMBase):
    def __init__(self, model_id: str, dtype: str = "bfloat16"):
        self.model_id = model_id
        self.dtype = getattr(torch, dtype)
        self.model = None
        self.processor = None

    @staticmethod
    def _best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        device = self._best_device()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(device).eval()
        print(f"  device: {device}")

    def predict(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        # strip the prompt echo that some versions include
        if "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1]
        return decoded.strip()
