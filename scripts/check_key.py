'''
Docstring for scripts.check_key
Checks if the API key is loaded correctly from the .env file.

'''


import os
from dotenv import load_dotenv

load_dotenv()

# get the key from environment variable

api_key = os.getenv("ors_api_key")

# return an error if the key is not found, otherwise print the first few characters of the key to confirm it loaded correctly
if not api_key:
    raise Exception("API key not found")

print("API key loaded successfully")
print("Key starts with:", api_key[:6], "…")
