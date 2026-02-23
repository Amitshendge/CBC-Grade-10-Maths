import httpx
from fastapi import Request
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from databases_framework.redis_utils import RedisUtils

def get_session_id(request: Request) -> str | None:
    """Extract session ID from request headers or cookies."""
    return request.cookies.get("session_id")

async def get_access_token(request: Request) -> str | None:
    """Extract access token from request headers or cookies."""
    session_id = get_session_id(request)
    if session_id:
        access_token = RedisUtils().get(f"auth:{session_id}:access_token")
        if access_token:
            return access_token
        else:
            refresh_url = request.url_for("ms_refresh")
            async with httpx.AsyncClient() as client:
                await client.post(str(refresh_url), cookies=request.cookies)
                return RedisUtils().get(f"auth:{session_id}:access_token")
    return None

