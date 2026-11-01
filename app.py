# Agri backend: Disease (Plant_AI) + Crop (Random Forest) + Soil (GEE SMAP/MODIS)
# One backend, three APIs: POST /api/predict, POST /api/crop-predict, GET /api/soil/quality

from __future__ import division, print_function
import os
import json
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

import ee
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from model import predict_image, is_likely_leaf
from crop_predictor import predict_crop, predict_crop_top_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)


# Allow mobile app and Flutter web to call this API from any origin
@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# ---------------------------------------------------------------------------
# Earth Engine initialization
# Mirrors the WeatherNow (FastAPI) app:
#   - reads EE_PROJECT_ID / EE_SERVICE_ACCOUNT_FILE (and legacy vars as fallback)
#   - resolves the service-account JSON path
#   - validates the key file
#   - initializes ee with ServiceAccountCredentials
#   - verifies with a real computation
#   - retries in a background thread until success
# ---------------------------------------------------------------------------

ee_initialized = False
initialization_error = None

EE_RETRY_INTERVAL = 60  # seconds


def _resolve_project_id():
    return (
        os.environ.get("EE_PROJECT_ID", "").strip()
        or os.environ.get("EE_PROJECT", "").strip()
        or os.environ.get("GEE_PROJECT", "").strip()
    )


def _resolve_key_path():
    key_file = os.environ.get("EE_SERVICE_ACCOUNT_FILE", "").strip()
    if not key_file:
        return None
    key_path = Path(key_file).expanduser()
    if not key_path.is_absolute():
        key_path = Path(__file__).resolve().parent / key_path
    return key_path.resolve()


def initialize_earth_engine():
    """Same initialization contract as the WeatherNow FastAPI app."""
    global ee_initialized, initialization_error

    try:
        project_id = _resolve_project_id()
        if not project_id:
            raise RuntimeError("EE_PROJECT_ID is missing from environment")

        key_path = _resolve_key_path()
        if not key_path:
            raise RuntimeError(
                "EE_SERVICE_ACCOUNT_FILE is missing from environment"
            )

        if not key_path.is_file():
            raise FileNotFoundError(
                f"Service-account JSON not found: {key_path}"
            )

        with key_path.open("r", encoding="utf-8") as file:
            key_data = json.load(file)

        if key_data.get("type") != "service_account":
            raise ValueError("The JSON file is not a service-account key")

        email = key_data.get("client_email")
        if not email:
            raise ValueError("Missing client_email in service-account JSON")

        credentials = ee.ServiceAccountCredentials(email, str(key_path))
        ee.Initialize(credentials, project=project_id)

        # Confirm authentication with a real EE computation.
        result = ee.Number(1).add(1).getInfo()
        if result != 2:
            raise RuntimeError("Earth Engine authentication test failed")

        ee_initialized = True
        initialization_error = None
        logger.info(
            "Earth Engine initialized successfully using service account."
        )

    except Exception as exc:
        ee_initialized = False
        initialization_error = str(exc)
        logger.exception("Earth Engine initialization failed.")


def earth_engine_retry_worker():
    """Background retry loop — runs until EE is initialized (like weather app)."""
    while not ee_initialized:
        logger.info("Attempting to initialize Google Earth Engine...")
        initialize_earth_engine()

        if ee_initialized:
            logger.info("Earth Engine is ready.")
            return

        logger.warning(
            "Earth Engine is unavailable. Retrying in %s seconds. "
            "Backend remains running.",
            EE_RETRY_INTERVAL,
        )
        # Use time.sleep here (thread, not async).
        import time
        time.sleep(EE_RETRY_INTERVAL)


def _ensure_gee():
    """Kept for compatibility with existing soil/water endpoints."""
    if ee_initialized:
        return True, None
    if initialization_error:
        return False, initialization_error
    # Trigger a synchronous init attempt on first use.
    initialize_earth_engine()
    if ee_initialized:
        return True, None
    return False, (
        initialization_error
        or "Earth Engine not initialized. Check EE_PROJECT_ID and "
        "EE_SERVICE_ACCOUNT_FILE environment variables."
    )


def parse_class(class_name):
    parts = class_name.split("___", 1)
    crop = parts[0].strip() if parts else "Unknown"
    disease = parts[1].strip().replace("_", " ") if len(parts) > 1 else "Unknown"
    return crop, disease


# ---------------------------------------------------------------------------
# Health / reconnect endpoints (parity with WeatherNow)
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "earth_engine_initialized": ee_initialized,
        "initialization_error": initialization_error,
    })


@app.route("/api/earth/reconnect", methods=["POST"])
def reconnect_earth_engine():
    global ee_initialized, initialization_error

    if ee_initialized:
        return jsonify({
            "success": True,
            "connected": True,
            "message": "Earth Engine is already connected.",
        })

    initialize_earth_engine()

    if ee_initialized:
        logger.info("Earth Engine reconnected through API.")
        return jsonify({
            "success": True,
            "connected": True,
            "message": "Earth Engine connected successfully.",
        })

    logger.warning(
        "Manual Earth Engine reconnection failed: %s", initialization_error
    )
    return jsonify({
        "success": False,
        "connected": False,
        "message": "Earth Engine connection failed.",
        "error": initialization_error,
    }), 503


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No file provided"}), 400
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(file_path)
        with open(file_path, "rb") as fp:
            img_bytes = fp.read()
        if not is_likely_leaf(img_bytes):
            return jsonify({"error": "Please upload a close-up leaf image.", "not_leaf": True}), 400
        class_name, confidence = predict_image(img_bytes)
        crop, disease = parse_class(class_name)
        result = f"Predicted Crop:{crop}  Predicted Disease:{disease}"
        return result
    return ""


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint for Next.js. Expects multipart form 'file' or 'image'."""
    f = request.files.get("file") or request.files.get("image")
    if not f:
        return jsonify({"error": "No file provided. Send multipart field 'file' or 'image'."}), 400
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, "uploads", secure_filename(f.filename or "upload.jpg"))
    f.save(file_path)
    try:
        with open(file_path, "rb") as fp:
            img_bytes = fp.read()
        if not is_likely_leaf(img_bytes):
            return jsonify({
                "not_leaf": True,
                "message": "Please upload a close-up leaf image for accurate disease detection.",
            }), 400
        class_name, confidence = predict_image(img_bytes)
        crop, disease = parse_class(class_name)
        return jsonify({
            "crop": crop,
            "disease": disease,
            "confidence": confidence,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/crop-predict", methods=["POST"])
def api_crop_predict():
    """Crop recommendation (ben041 Random Forest). Expects JSON: { N, P, K, temperature, humidity, ph, rainfall }."""
    data = request.get_json(silent=True) or {}
    try:
        N = float(data.get("N", 0))
        P = float(data.get("P", 0))
        K = float(data.get("K", 0))
        temperature = float(data.get("temperature", 25))
        humidity = float(data.get("humidity", 60))
        ph = float(data.get("ph", 6.5))
        rainfall = float(data.get("rainfall", 100))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input. Send N, P, K, temperature, humidity, ph, rainfall as numbers."}), 400

    try:
        top_k = predict_crop_top_k(N, P, K, temperature, humidity, ph, rainfall, k=5)
        if not top_k:
            return jsonify({"error": "Model returned no crops.", "crop": None, "crops": []}), 503
        top_crop = top_k[0][0]
        crops = [{"crop": name, "confidence": conf} for name, conf in top_k]
        return jsonify({"crop": top_crop, "crops": crops})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/soil/quality", methods=["GET"])
def api_soil_quality():
    """Soil quality from GEE (SMAP + MODIS). GET ?lat=&lng=&buffer_km=2&start_date=&end_date= (YYYY-MM-DD)."""
    ok, err = _ensure_gee()
    if not ok:
        return jsonify({"error": err}), 503

    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    buffer_km = request.args.get("buffer_km", default=2, type=float)
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng are required"}), 400
    if not start_date or not end_date:
        return jsonify({"error": "start_date and end_date are required (YYYY-MM-DD)"}), 400

    buffer_km = max(0.5, min(20, buffer_km))

    try:
        from gee_soil import get_soil_quality_for_area
        result = get_soil_quality_for_area(lat, lng, buffer_km, start_date, end_date)
    except Exception as e:
        print("GEE soil/quality error:", e)
        return jsonify({"error": str(e)}), 500

    missing = [k for k, v in result.items() if v is None]

    if len(missing) == 4:
        return jsonify({
            "error": "No soil data for this area/date range. Try a different location or a wider date window.",
            "result": result,
        }), 503

    response = {}
    if result.get("soil_moisture") is not None:
        response["soil_moisture"] = round(result["soil_moisture"], 4)
    if result.get("ndvi") is not None:
        response["ndvi"] = round(result["ndvi"], 4)
    if result.get("ndwi") is not None:
        response["ndwi"] = round(result["ndwi"], 4)
    if result.get("lst") is not None:
        response["lst"] = round(result["lst"], 2)
    if missing:
        response["missingBands"] = missing

    return jsonify(response), 200


@app.route("/api/water/availability", methods=["GET"])
def api_water_availability():
    """Water availability from GEE (GLDAS). GET ?lat=&lng=&start_date=&end_date= (YYYY-MM-DD)."""
    ok, err = _ensure_gee()
    if not ok:
        return jsonify({"error": err}), 503

    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()

    if lat is None or lng is None:
        return jsonify({"error": "lat and lng are required"}), 400
    if not start_date or not end_date:
        return jsonify({"error": "start_date and end_date are required (YYYY-MM-DD)"}), 400

    try:
        from gee_water import get_water_availability_timeseries
        data = get_water_availability_timeseries(lat, lng, start_date, end_date)
    except Exception as e:
        print("GEE water/availability error:", e)
        return jsonify({"error": str(e)}), 500

    if not data:
        return jsonify({"error": "No GLDAS data for this date/area.", "data": []}), 503

    return jsonify({"data": data})


if __name__ == "__main__":
    import threading

    # Start the same background retry worker as the weather app.
    threading.Thread(target=earth_engine_retry_worker, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)