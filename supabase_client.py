"""
Centralized Supabase client.

Import `supabase` from this module anywhere in the project.
All Supabase REST operations go through this single client instance.

Note: This client uses the SUPABASE_KEY (service role key) which bypasses
Row Level Security. Never expose this key to a browser/frontend.
For future FastAPI endpoints, keep this import server-side only.
"""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_url = os.getenv("SUPABASE_URL")
_key = os.getenv("SUPABASE_KEY")

if not _url:
    raise ValueError(
        "SUPABASE_URL is missing from environment variables. "
        "Add it to your .env file."
    )
if not _key:
    raise ValueError(
        "SUPABASE_KEY is missing from environment variables. "
        "Add it to your .env file."
    )

supabase: Client = create_client(_url, _key)

print("✅ Supabase client initialized")
