import sys
import pathlib
import os
import json
import time
from fastapi import APIRouter, BackgroundTasks, File, Query, Request, Response, HTTPException, UploadFile, Form
sys.path.append(str(pathlib.Path(__file__).parent.parent))




router = APIRouter()
@router.get("/health_check")
async def health_check():
    return {"status": "ok"}




