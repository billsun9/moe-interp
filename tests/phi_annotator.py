# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="microsoft/Phi-4-mini-reasoning")
messages = [
    {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": "What's the best way to make flan for cheap"}
]
pipe(messages)!