from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import datetime
import uuid
import json

from app.ocr_nlp import extract_from_pdf
from app.change_detect import detect_changes
from app.risk_score import compute_risk_score

app = FastAPI(title="FRA AI Inference API")

# OCR/NLP Endpoint
@app.post("/api/v1/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        extracted = extract_from_pdf(contents)
        return JSONResponse(content=extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Change Detection Endpoint
class ChangeRequest(BaseModel):
    bbox: List[float]  # [minx, miny, maxx, maxy]
    time_from: str     # YYYY-MM-DD
    time_to: str       # YYYY-MM-DD
    image_path: str    # Path to satellite image/tile (local or S3 later)

@app.post("/api/v1/infer/change")
async def infer_change(req: ChangeRequest):
    try:
        events = detect_changes(req.image_path, req.bbox, req.time_from, req.time_to)
        response = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "events": events
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Risk Scoring Endpoint
class ScoreRequest(BaseModel):
    claim_id: int
    overlap_area: float
    change_area: float
    ocr_confidence: float
    prior_claims_count: int

@app.post("/api/v1/score")
async def score_claim(req: ScoreRequest):
    try:
        score, reasons = compute_risk_score(
            overlap_area=req.overlap_area,
            change_area=req.change_area,
            ocr_confidence=req.ocr_confidence,
            prior_claims_count=req.prior_claims_count
        )
        return {"claim_id": req.claim_id, "score": score, "reasons": reasons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))