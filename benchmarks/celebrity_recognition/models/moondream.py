import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import VLMBase


class Moondream(VLMBase):
    def __init__(self, model_id: str, revision: str, dtype: str = "float32"):
        self.model_id = model_id
        self.revision = revision
        self.dtype = getattr(torch, dtype)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            trust_remote_code=True,
            dtype=self.dtype,
        ).eval()

    def predict(self, image: Image.Image, prompt: str) -> str:
        return self.model.query(image, prompt)["answer"].strip()
