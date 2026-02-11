import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ors_api_key")

if not api_key:
    raise Exception("API key not found")

print("API key loaded successfully")
print("Key starts with:", api_key[:6], "…")
