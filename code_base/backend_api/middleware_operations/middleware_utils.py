import uuid
from fastapi import HTTPException, Request
from token_operations.access_token_utils import get_session_id
from token_operations.jwt_utils import get_jwt_token
import json
from fastapi.responses import JSONResponse

middleware_mapping_file = open('middleware_operations/middleware_mapping.json', 'r')
midleware_mapping = json.load(middleware_mapping_file)
api_routes = midleware_mapping.keys()

@staticmethod
async def route_exists(request: Request) -> bool:
    return request.url.path in api_routes

@staticmethod
async def is_public_route(request: Request) -> bool:
    if midleware_mapping[request.url.path][request.method]["public_route"]:
        return True
    return False

@staticmethod
async def is_logged_in(request: Request) -> bool:
    jwt_tocken = await get_jwt_token(request)
    if not jwt_tocken:
        return False
    return True

@staticmethod
async def has_role_based_access(request: Request) -> bool:
    if midleware_mapping[request.url.path][request.method]["roles_required"]:
        if not request.state.user['role'] in midleware_mapping[request.url.path][request.method]["roles_required"]:
            return False
    return True

@staticmethod
async def middleware_logic(request: Request):
    if request.method == "OPTIONS":
        return request
    if not await route_exists(request):
        raise HTTPException(status_code=404, detail="Route not found")
    if await is_public_route(request):
        return request
    if not await is_logged_in(request):
        raise HTTPException(status_code=401, detail="Not logged in")
    if not await has_role_based_access(request):
        raise HTTPException(status_code=403, detail="Access Denied")
    return request


