# models.py
from transformers import pipeline
from utils import logger, timed

class BaseHFModel:
    task = None
    hf_model_id = None
    friendly_name = "Unnamed"
    description = ""
    input_type = "Unknown"
    output_type = "Unknown"
    license = "Varies"

    def __init__(self):
        self._pipe = None
        if not self.task or not self.hf_model_id:
            raise ValueError("Subclass must set 'task' and 'hf_model_id'.")

    def _ensure_loaded(self):
        if self._pipe is None:
            self._pipe = pipeline(self.task, model=self.hf_model_id)  # or device=-1

    @logger
    @timed
    def run(self, data, **kwargs):
        self._ensure_loaded()
        return self._pipe(data, **kwargs)

    def get_info(self) -> dict:
        """Used by the Model Info tab in the GUI."""
        return {
            "Name": self.friendly_name,
            "Hugging Face ID": self.hf_model_id,
            "Task": self.task,
            "Input": self.input_type,
            "Output": self.output_type,
            "Description": self.description,
            "License": self.license,
        }

# Sentiment model (must INHERIT BaseHFModel) 
class HuggingFaceModel1(BaseHFModel):
    # Binary sentiment (POSITIVE / NEGATIVE)
    task = "sentiment-analysis"
    hf_model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    friendly_name = "DistilBERT Sentiment (POS/NEG)"
    description = "Binary sentiment on SST-2; returns POSITIVE or NEGATIVE."
    input_type = "Text (string)"
    output_type = "Label + score"

    @logger
    @timed
    def run(self, input_text):
        """Return a clean string like 'NEGATIVE (98.3%)'."""
        self._ensure_loaded()

        text = (str(input_text) if input_text is not None else "").strip()
        if not text:
            return "Error: empty text input."

        raw = self._pipe(text, truncation=True, return_all_scores=True)
        # raw -> [[{'label':'NEGATIVE','score':...}, {'label':'POSITIVE','score':...}]]

        scores = raw[0] if isinstance(raw, list) else raw
        best = max(scores, key=lambda x: float(x["score"]))
        label = best["label"].upper()                  # POSITIVE / NEGATIVE
        conf  = float(best["score"])

        return f"{label} ({conf*100:.1f}%)"
    
# Image model (must INHERIT BaseHFModel) 
class HuggingFaceModel2(BaseHFModel):
    task = "image-classification"
    hf_model_id = "google/vit-base-patch16-224"
    friendly_name = "ViT Base Patch16-224 (ImageNet)"
    description = "Vision Transformer classifier."
    input_type = "Image path or PIL Image"
    output_type = "Top-k labels + scores"

    @logger
    @timed
    def run(self, image_path: str, top_k: int = 5):
        self._ensure_loaded()
        preds = self._pipe(image_path, top_k=top_k)
        for p in preds:
            p["label"] = p["label"].replace("_", " ").title()
            p["score"] = float(p["score"])
        return preds
