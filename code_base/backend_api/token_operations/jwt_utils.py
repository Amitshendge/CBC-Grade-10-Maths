import httpx
import jwt
from fastapi import HTTPException, Request
from datetime import datetime, timedelta
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from token_operations.access_token_utils import get_session_id
from config import ACCESS_TOKEN_EXPIRATION, JWT_SECRET, JWT_EXPIRATION, REFRESH_TOKEN_EXPIRATION, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
from typing import TypedDict
from databases_framework.redis_utils import RedisUtils

class UserInfo(TypedDict):
    sub: int
    email: str
    name: str
    role: str
    exp: int    # expiration time in seconds

async def get_jwt_token(request: Request) -> str | None:
    """Extract JWT token from request headers or cookies."""
    return request.cookies.get("Authorization")

async def create_jwt(user_info: UserInfo) -> str:
    """Create JWT for authenticated user"""
    payload = {
        "sub": user_info.get("sub") or user_info.get("oid") or user_info.get("id"),       # unique MS user id
        "email": user_info.get("email") or user_info.get("preferred_username") or user_info.get("upn"),
        "name": user_info.get("name"),
        "role": user_info.get("role", "user"),
        "exp": datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token

async def decode_jwt(token: str | None) -> UserInfo:
    """Decode JWT and return user info if valid"""
    if not token:
        raise HTTPException(status_code=401, detail="JWT Token missing")
    payload = jwt.decode(token.split(" ")[1], JWT_SECRET, algorithms=["HS256"])
    return UserInfo(
        sub=payload.get("sub"),
        email=payload.get("email"),
        name=payload.get("name"),
        role=payload.get("role", "user"),
        exp=payload.get("exp"),
    )

async def refresh_access_token(refresh_token: str):
    """
    Exchange Google refresh_token for a new access_token,
    fetch userinfo, then issue a new JWT for your app.
    """
    try:
        # Step 1: call Google token endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        google_tokens = response.json()
        access_token = google_tokens.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Failed to refresh access token")

        # Step 2: fetch updated userinfo from Google
        async with httpx.AsyncClient() as client:
            userinfo_resp = await client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )

        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch user info")

        user_info = userinfo_resp.json()

        # Step 3: issue a new JWT for your app
        new_jwt = await create_jwt(user_info)

        return {
            "access_token": new_jwt,
            "refresh_token": refresh_token,     # Google's refresh token
            "token_type": "bearer",
            "expires_in": 3600
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


async def get_new_jwt(request: Request) -> str:
    """Renew JWT if it's about to expire"""
    session_id = get_session_id(request)
    print(f"Session ID: {session_id}")
    access_token = RedisUtils().get(f"auth:{session_id}:access_token")
    print(f"Access Token from Redis: {access_token}")
    if access_token:
        print("Using existing access token to fetch user info")
        async with httpx.AsyncClient() as client:
        # Microsoft OIDC userinfo endpoint (Graph)
            ui_resp = await client.get(
                "https://graph.microsoft.com/oidc/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15,
            )
        if ui_resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch user info")
        user_info = ui_resp.json()
        jwt_token = await create_jwt(user_info)
        return jwt_token
    else:
        print("Access token missing or expired, attempting to refresh")
        refresh_token = RedisUtils().get(f"auth:{session_id}:refresh_token")
        print(f"Refresh Token from Redis: {refresh_token}")
        if refresh_token:
            response = await refresh_access_token(refresh_token)
            RedisUtils().set(f"auth:{session_id}:access_token", response['access_token'], ttl=ACCESS_TOKEN_EXPIRATION)  # store in Redis with 1 hour TTL
            RedisUtils().set(f"auth:{session_id}:refresh_token", response['refresh_token'], ttl=REFRESH_TOKEN_EXPIRATION)  # store in Redis with 1 hour TTL
            return await get_new_jwt(request)
        else:
            raise HTTPException(status_code=401, detail="Session expired, please log in again")