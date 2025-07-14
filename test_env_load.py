import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

print(f"Loaded .env file from: {env_path.resolve()}")
serpapi_key = os.getenv('SERPAPI_KEY')
if serpapi_key:
    print(f"Loaded SERPAPI_KEY: {serpapi_key[:4]}****")
else:
    print("SERPAPI_KEY is not set or could not be loaded.")
