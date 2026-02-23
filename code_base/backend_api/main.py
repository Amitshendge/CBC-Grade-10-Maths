import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import jwt
from starlette.middleware.sessions import SessionMiddleware
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from token_operations.jwt_utils import get_new_jwt
from config import MIDDLEWARE_SESSION_SECRET
from saml_auth.auth_google import router as google_router
from api_routes.basic_router import router as basic_router
from api_routes.lesson_planner_router import router as lesson_planner_router
from middleware_operations.middleware_utils import middleware_logic
from token_operations.jwt_utils import decode_jwt
from token_operations.jwt_utils import get_jwt_token
app = FastAPI()

# Add session middleware (needed for OAuth)
app.add_middleware(SessionMiddleware, secret_key=MIDDLEWARE_SESSION_SECRET)

# CORS settings
origins = [
    "*",  # React app running locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,    # Disable cookies
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Register routes
app.include_router(basic_router, prefix="/api", tags=["basic"])
app.include_router(google_router, prefix="/api", tags=["google"])
app.include_router(lesson_planner_router, prefix="/api", tags=["lesson-planner"])

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    return await call_next(request)
    if request.url.path.startswith("/google/login") or request.url.path.startswith("/google/auth"):
        return await call_next(request)
    if request.cookies.get("Authorization"):
        try:
            decoded_jwt = await decode_jwt(await get_jwt_token(request))
            new_jwt = None
        except jwt.ExpiredSignatureError:
            try:
                new_jwt = await get_new_jwt(request)
                decoded_jwt = await decode_jwt(new_jwt)
            except Exception as e:
                raise HTTPException(status_code=401, detail="Not logged in")
        except Exception as e:
            raise HTTPException(status_code=401, detail="Not logged in")
        request.state.user = decoded_jwt if decoded_jwt else None
    else:
        new_jwt = None
        request.state.user = None
    await middleware_logic(request)
    response = await call_next(request)
    if new_jwt:
        response.set_cookie(
            key="Authorization",
            value=f"Bearer {new_jwt}",
            httponly=True,
            secure=True,      # only send over HTTPS (enable in production)
            samesite="none",  # or "lax" if your app spans subdomains
            max_age=None       # JWT expiry (5 minutes in this example)
        )
    return response

@app.get("/")
async def root():
    return {"message": "Backend API is running."}
