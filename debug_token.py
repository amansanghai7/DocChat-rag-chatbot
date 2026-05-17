"""
Temporary diagnostic script — run this to verify JWT authentication.

Usage:
    python debug_token.py <your_bearer_token>

Copy the token from the curl sign-in response (the full access_token value).
"""

import json
import os
import sys

import jwt as pyjwt
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
token = sys.argv[1] if len(sys.argv) > 1 else ""

if not token:
    print("Usage: python debug_token.py <token>")
    sys.exit(1)

print(f"\n── Token ───────────────────────────────")
print(f"  Length : {len(token)}")

print(f"\n── Unverified header ───────────────────")
try:
    header = pyjwt.get_unverified_header(token)
    print(json.dumps(header, indent=2))
    alg = header.get("alg", "unknown")
    kid = header.get("kid", "none")
    print(f"\n  Algorithm : {alg}")
    print(f"  Key ID    : {kid}")
except Exception as e:
    print(f"  FAILED: {e}")
    print("  → Token is structurally malformed. Check you pasted the full token.")
    sys.exit(1)

print(f"\n── Unverified claims ───────────────────")
try:
    claims = pyjwt.decode(token, options={"verify_signature": False})
    print(json.dumps(claims, indent=2, default=str))
except Exception as e:
    print(f"  FAILED: {e}")

print(f"\n── JWKS ────────────────────────────────")
jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
print(f"  URL: {jwks_url}")
try:
    resp = requests.get(jwks_url, timeout=10)
    resp.raise_for_status()
    jwks = resp.json()
    keys = jwks.get("keys", [])
    print(f"  Keys found: {len(keys)}")
    for k in keys:
        print(f"    kid={k.get('kid')!r}  kty={k.get('kty')}  alg={k.get('alg')}")
except Exception as e:
    print(f"  FAILED to fetch JWKS: {e}")
    sys.exit(1)

print(f"\n── Signature verification ──────────────")
kid = header.get("kid")
candidates = [k for k in keys if k.get("kid") == kid] if kid else keys
if not candidates:
    print(f"  ❌ No key matching kid={kid!r}")
    sys.exit(1)

key_data = candidates[0]
kty = key_data.get("kty", "EC")

try:
    if kty == "EC":
        from jwt.algorithms import ECAlgorithm
        public_key = ECAlgorithm.from_jwk(json.dumps(key_data))
    elif kty == "RSA":
        from jwt.algorithms import RSAAlgorithm
        public_key = RSAAlgorithm.from_jwk(json.dumps(key_data))
    else:
        print(f"  ❌ Unsupported key type: {kty}")
        sys.exit(1)

    payload = pyjwt.decode(
        token, public_key,
        algorithms=[alg],
        options={"verify_aud": False},
    )
    print(f"  ✅ Signature VALID")
    print(f"     user_id : {payload.get('sub')}")
    print(f"     email   : {payload.get('email')}")
    print(f"     aud     : {payload.get('aud')!r}")
    print(f"\n  Authentication is working correctly.")
except pyjwt.ExpiredSignatureError:
    print("  ❌ Token has EXPIRED — get a fresh token from Supabase")
except pyjwt.InvalidTokenError as e:
    print(f"  ❌ Signature INVALID: {e}")

print()
