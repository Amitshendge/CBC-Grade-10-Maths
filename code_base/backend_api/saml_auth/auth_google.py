import uuid
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from databases_framework.redis_utils import RedisUtils
from token_operations.jwt_utils import create_jwt
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from authlib.integrations.starlette_client import OAuth
from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, ACCESS_TOKEN_EXPIRATION, REFRESH_TOKEN_EXPIRATION, HOST_URL
import httpx

router = APIRouter()
oauth = OAuth()

oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

FRONTEND_HOST_URL = ""

@router.get("/google/login")
async def login(request: Request, next: str = "/"):
    print("Initiating Google OAuth login")
    global FRONTEND_HOST_URL
    if not FRONTEND_HOST_URL:
        FRONTEND_HOST_URL = request.headers.get('referer').rstrip('/') # pyright: ignore[reportOptionalMemberAccess]
    redirect_uri = str(request.url_for("auth"))
    redirect_uri = redirect_uri.replace("http://", "https://", 1) if not redirect_uri.startswith("http://localhost") else redirect_uri
    return await oauth.google.authorize_redirect( # type: ignore # pyright: ignore[reportOptionalMemberAccess]
        request,
        redirect_uri,
        access_type="offline",   # ✅ ask Google for refresh token
        prompt="consent",         # ensures refresh token is issued every time
        state=next                # preserve "next" URL through the OAuth flow
    )


@router.get("/google/auth")
async def auth(request: Request):
    global FRONTEND_HOST_URL
    token = await oauth.google.authorize_access_token(request) # pyright: ignore[reportOptionalMemberAccess] # type: ignore
    user_info = token.get("userinfo")
    if not user_info:
        return JSONResponse({"error": "Failed to fetch user info"}, status_code=400)
    session_id = uuid.uuid4().hex
    jwt_token = await create_jwt(user_info)

    RedisUtils().set(f"auth:{session_id}:access_token", token.get("access_token"), ttl=ACCESS_TOKEN_EXPIRATION)  # store in Redis with 1 hour TTL
    RedisUtils().set(f"auth:{session_id}:refresh_token", token.get("refresh_token"), ttl=REFRESH_TOKEN_EXPIRATION)  # store in Redis with 1 hour TTL
    next_url = request.query_params.get("state") or "/"
    response = RedirectResponse(url=f"{FRONTEND_HOST_URL}{next_url}")
    response.set_cookie(
            key="Authorization",
            value=f"Bearer {jwt_token}",
            httponly=True,
            secure=True,      # only send over HTTPS (enable in production)
            samesite="none",  # or "lax" if your app spans subdomains
            max_age=None       # JWT expiry (5 minutes in this example)
        )
    response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
                secure=True,      # only send over HTTPS (enable in production)
                samesite="none",  # or "lax" if your app spans subdomains
                max_age=None       # JWT expiry (5 minutes in this example)
            )
    return response

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

@router.post("/google/auth/logout")
def logout(response: Response):
    # Clear the refresh token cookie
    response.delete_cookie(key="refresh_token")
    return {"message": "Logged out successfully"}

