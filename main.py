# ==============================================================================
# NEUROSCAN AI - COVID-19 CHEST X-RAY CLASSIFICATION API
# Enterprise FastAPI Backend with PDF Report Generation
# ==============================================================================

import os
import io
import time
import uuid
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf

from fpdf import FPDF
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NeuroScanAPI")

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
TARGET_SIZE = (224, 224)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# ==============================================================================
# GLOBAL IN-MEMORY STATE
# ==============================================================================

app_state = {
    "model": None,
    "labels": [],
    "history": [],
    "image_cache": {},   # Stores recent images for PDF generation
    "report_cache": {},  # Stores generated PDF bytes
    "stats": {
        "total_predictions": 0,
        "class_counts": {},
        "average_confidence": 0.0,
        "average_latency_ms": 0.0
    }
}

# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================

class ReportRequest(BaseModel):
    prediction_id: str
    patient_name: str
    patient_age: str
    patient_gender: str
    doctor_name: str

# ==============================================================================
# LIFESPAN & INITIALIZATION
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing NeuroScan AI Backend...")
    try:
        if os.path.exists(MODEL_PATH):
            app_state["model"] = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("TensorFlow Model loaded successfully.")
        else:
            logger.warning(f"Model {MODEL_PATH} not found! Ensure it is in the root directory.")

        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                app_state["labels"] = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]
            for label in app_state["labels"]:
                app_state["stats"]["class_counts"][label] = 0
            logger.info(f"Labels loaded: {app_state['labels']}")
        else:
            logger.warning(f"Labels {LABELS_PATH} not found!")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield
    logger.info("Shutting down NeuroScan AI Backend...")
    app_state["model"] = None
    app_state["image_cache"].clear()
    app_state["report_cache"].clear()

app = FastAPI(
    title="NeuroScan AI Medical API",
    description="Enterprise API for COVID-19 Chest X-Ray inference and diagnostic report generation.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def update_statistics(prediction: str, confidence: float, latency: float):
    stats = app_state["stats"]
    t = stats["total_predictions"]
    stats["average_confidence"] = ((stats["average_confidence"] * t) + confidence) / (t + 1)
    stats["average_latency_ms"] = ((stats["average_latency_ms"] * t) + latency) / (t + 1)
    stats["total_predictions"] += 1
    stats["class_counts"][prediction] = stats["class_counts"].get(prediction, 0) + 1

class MedicalReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(29, 78, 216) # Blue
        self.cell(0, 10, 'NEUROSCAN AI CLINICAL DIAGNOSTIC REPORT', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, 'Automated Chest Radiograph Analysis', 0, 1, 'C')
        self.line(10, 28, 200, 28)
        self.ln(10)

    def footer(self):
        self.set_y(-25)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.multi_cell(0, 4, "DISCLAIMER: This report is generated by an AI model for preliminary screening only. It does not replace a professional clinical diagnosis by a certified physician.", 0, 'C')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(req_data: dict, img_bytes: bytes) -> bytes:
    pdf = MedicalReportPDF()
    pdf.add_page()
    
    # Patient Info (Clean & Simple Layout)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(100, 8, f"Patient Name: {req_data['patient_name']}", 0, 0)
    pdf.cell(90, 8, f"Report ID: {req_data['prediction_id'][:8].upper()}", 0, 1, 'R')
    pdf.set_font('Arial', '', 11)
    pdf.cell(100, 6, f"Age / Gender: {req_data['patient_age']} / {req_data['patient_gender']}", 0, 0)
    pdf.cell(90, 6, f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", 0, 1, 'R')
    pdf.cell(100, 6, f"Referring Doctor: {req_data['doctor_name']}", 0, 1)
    pdf.ln(5)

    # Results Section
    is_covid = "covid" in req_data['prediction'].lower()
    color = (220, 38, 38) if is_covid else (16, 185, 129)
    
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(0, 10, ' AI INFERENCE RESULTS', 0, 1, 'L', 1)
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 8, 'Primary Finding:', 0, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, req_data['prediction'].upper(), 0, 1)
    
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 8, 'Confidence Score:', 0, 0)
    pdf.cell(0, 8, f"{req_data['confidence']}%", 0, 1)
    
    risk_level = "HIGH RISK" if is_covid else "LOW RISK"
    pdf.cell(50, 8, 'Risk Level:', 0, 0)
    pdf.set_text_color(*color)
    pdf.cell(0, 8, risk_level, 0, 1)
    
    # Save Image to temp file and insert
    pdf.ln(5)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        tmp_img.write(img_bytes)
        tmp_img_path = tmp_img.name
    
    pdf.image(tmp_img_path, x=10, w=80)
    os.remove(tmp_img_path)

    # Chart Generation (Keeping it clean)
    probs = req_data['probabilities']
    labels = list(probs.keys())
    values = list(probs.values())
    
    plt.figure(figsize=(6, 4))
    bars = plt.barh(labels, values, color=['#ef4444' if 'covid' in l.lower() else '#3b82f6' for l in labels])
    plt.xlim(0, 100)
    plt.title('Class Probabilities (%)')
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart:
        plt.savefig(tmp_chart.name, format='png', dpi=150)
        tmp_chart_path = tmp_chart.name
    plt.close()
    
    pdf.image(tmp_chart_path, x=100, y=pdf.get_y()-80, w=100)
    os.remove(tmp_chart_path)
    
    # Clinical Notes
    pdf.set_y(150)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, ' CLINICAL OBSERVATIONS & RECOMMENDATIONS', 0, 1, 'L', 1)
    pdf.set_font('Arial', '', 11)
    pdf.ln(3)
    
    if is_covid:
        obs = "The AI model detected patterns highly consistent with COVID-19."
        rec = "Immediate clinical correlation required. Recommend RT-PCR confirmation and isolation protocols."
    else:
        obs = "The AI model did not detect significant radiological patterns consistent with COVID-19."
        rec = "Routine clinical assessment advised. If patient is symptomatic, consider follow-up imaging."
        
    pdf.multi_cell(0, 6, f"Observation: {obs}")
    pdf.ln(2)
    pdf.multi_cell(0, 6, f"Recommendation: {rec}")

    # Return as raw bytes
    return bytes(pdf.output())

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    return {"status": "Online", "service": "NeuroScan AI Engine"}

@app.get("/health")
async def health():
    return {
        "status": "Healthy",
        "model_loaded": app_state["model"] is not None,
        "labels": len(app_state["labels"]),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/version")
async def version():
    return {"api_version": "2.0.0", "engine": f"TensorFlow {tf.__version__}"}

@app.get("/model-info")
async def model_info():
    return {
        "classes": app_state["labels"],
        "input_shape": TARGET_SIZE,
        "type": "Deep Convolutional Neural Network"
    }

@app.get("/statistics")
async def get_stats():
    return app_state["stats"]

@app.get("/history")
async def get_history():
    return {"count": len(app_state["history"]), "data": app_state["history"][::-1]}

@app.delete("/history")
async def reset_history():
    app_state["history"].clear()
    app_state["image_cache"].clear()
    app_state["report_cache"].clear()
    return {"message": "System memory cleared."}

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="AI Engine Offline. Check if keras_model.h5 is loaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid payload. Image required.")

    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File exceeds 5MB limit.")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(TARGET_SIZE)
        img_array = np.asarray(image, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = app_state["model"].predict(img_array, verbose=0)[0]
        
        pred_idx = int(np.argmax(preds))
        pred_class = app_state["labels"][pred_idx] if app_state["labels"] else f"Class {pred_idx}"
        confidence = round(float(preds[pred_idx]) * 100, 2)

        probs = {
            (app_state["labels"][i] if app_state["labels"] else f"Class {i}"): round(float(p) * 100, 2)
            for i, p in enumerate(preds)
        }

        latency = round((time.time() - start_time) * 1000, 2)
        req_id = str(uuid.uuid4())

        result = {
            "id": req_id,
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": probs,
            "latency_ms": latency,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Cache state for history and PDF generation
        app_state["history"].append(result)
        app_state["image_cache"][req_id] = contents
        background_tasks.add_task(update_statistics, pred_class, confidence, latency)

        return result

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted image file.")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal Neural Network Error.")

@app.post("/generate-report")
async def generate_report(req: ReportRequest):
    req_id = req.prediction_id
    
    if req_id not in app_state["image_cache"]:
        raise HTTPException(status_code=404, detail="Prediction ID expired or invalid.")
        
    # Find prediction data
    pred_data = next((item for item in app_state["history"] if item["id"] == req_id), None)
    if not pred_data:
        raise HTTPException(status_code=404, detail="Prediction data not found.")

    try:
        report_data = {**pred_data, **req.dict()}
        pdf_bytes = generate_pdf_report(report_data, app_state["image_cache"][req_id])
        
        report_id = f"REP-{str(uuid.uuid4())[:8].upper()}"
        app_state["report_cache"][report_id] = pdf_bytes
        
        return {"report_id": report_id, "message": "Report generated successfully."}
    except Exception as e:
        logger.error(f"PDF Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compile PDF report: {str(e)}")

@app.get("/download-report/{report_id}")
async def download_report(report_id: str):
    if report_id not in app_state["report_cache"]:
        raise HTTPException(status_code=404, detail="Report not found or expired.")
        
    pdf_bytes = app_state["report_cache"][report_id]
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=NeuroScan_Report_{report_id}.pdf"}
    )