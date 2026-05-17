"""
JWT authentication for FastAPI.

Supabase now issues ES256 tokens (ECDSA asymmetric keys).
Verification uses the public key fetched from Supabase's JWKS endpoint —
no shared secret required or used.

Flow:
  1. Decode the JWT header (without verifying) to read the key ID (kid).
  2. Fetch Supabase's public keys from the JWKS endpoint (cached after first fetch).
  3. Match the key by kid and reconstruct the public key object.
  4. Verify the full token signature with that public key.
  5. Extract user_id from the verified 'sub' claim.
"""

import json
import logging
from typing import Optional

import jwt as pyjwt
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# Module-level JWKS cache.
# Supabase public keys are stable per project — one HTTP fetch per process.
# The lifespan handler warms this up at startup so the first request is instant.
_jwks_cache: Optional[dict] = None


def _get_jwks() -> dict:
    """
    Fetch and cache Supabase's JWKS (JSON Web Key Set).

    JWKS contains the public keys Supabase uses to sign JWTs.
    The endpoint is always: <SUPABASE_URL>/auth/v1/.well-known/jwks.json
    """
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    url = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    logger.info("Fetching Supabase JWKS from %s", url)

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    _jwks_cache = response.json()
    logger.info("Supabase JWKS cached: %d key(s)", len(_jwks_cache.get("keys", [])))
    return _jwks_cache


def _resolve_public_key(token: str):
    """
    Read the JWT header to find the key ID (kid), look it up in the JWKS,
    and return a public key object that PyJWT can use for signature verification.

    Supports EC keys (ES256/ES384) and RSA keys (RS256).
    Returns (public_key, algorithm_string).
    """
    try:
        header = pyjwt.get_unverified_header(token)
    except pyjwt.exceptions.DecodeError as exc:
        # Token is structurally malformed — not even a valid JWT
        raise pyjwt.exceptions.DecodeError(str(exc))

    kid = header.get("kid")
    alg = header.get("alg", "ES256")

    jwks = _get_jwks()
    all_keys = jwks.get("keys", [])

    # Prefer the key that matches kid; fall back to all keys if kid is absent
    candidates = [k for k in all_keys if k.get("kid") == kid] if kid else all_keys

    if not candidates:
        logger.warning("No JWKS key found for kid=%r (available kids: %s)",
                       kid, [k.get("kid") for k in all_keys])
        raise ValueError(f"No matching public key for kid={kid!r}")

    key_data = candidates[0]
    kty = key_data.get("kty", "EC")

    if kty == "EC":
        from jwt.algorithms import ECAlgorithm
        public_key = ECAlgorithm.from_jwk(json.dumps(key_data))
    elif kty == "RSA":
        from jwt.algorithms import RSAAlgorithm
        public_key = RSAAlgorithm.from_jwk(json.dumps(key_data))
    else:
        raise ValueError(f"Unsupported key type in JWKS: {kty!r}")

    return public_key, alg


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency — inject into any route to require authentication.

    Usage:
        @router.get("/example")
        def example(current_user: dict = Depends(get_current_user)):
            user_id = current_user["user_id"]

    Returns:
        {"user_id": str, "email": str}

    Raises HTTPException 401 for missing/invalid/expired tokens.
    Raises HTTPException 503 if the JWKS endpoint is unreachable.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token. Include 'Authorization: Bearer <token>' in your request.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # ── Step 1: Resolve the public key ───────────────────────────────────────
    try:
        public_key, alg = _resolve_public_key(token)
    except pyjwt.exceptions.DecodeError:
        # Structurally malformed token (not a valid JWT at all)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        # JWKS fetch failed or key not found
        logger.error("Public key resolution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable.",
        )

    # ── Step 2: Verify signature and standard claims ──────────────────────────
    try:
        payload = pyjwt.decode(
            token,
            public_key,
            algorithms=[alg],
            options={"verify_aud": False},  # audience verified manually below
        )
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except pyjwt.InvalidTokenError as exc:
        logger.warning("JWT verification failed (%s): %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Step 3: Audience check ────────────────────────────────────────────────
    aud = payload.get("aud")
    if aud is not None:
        aud_list = aud if isinstance(aud, list) else [aud]
        if "authenticated" not in aud_list:
            logger.warning("JWT audience rejected: %r", aud)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token audience.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── Step 4: Extract user identity ─────────────────────────────────────────
    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing required user information.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": user_id,
        "email": payload.get("email", ""),
    }
