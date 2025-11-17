from lab.models.openai_backend import OpenAIBackend
b = OpenAIBackend({"model_name":"gpt-4o-mini","temperature":0.0,"max_tokens":8})
text, usage, secs = b.generate("Say ok only.")
print("OUT:", text, "| usage:", usage, "| secs:", round(secs,3))
