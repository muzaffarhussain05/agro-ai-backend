# Agri backend: Disease (Plant_AI) + Crop (Random Forest) + Soil (GEE SMAP/MODIS)
# One backend, three APIs: POST /api/predict, POST /api/crop-predict, GET /api/soil/quality

from __future__ import division, print_function
import os

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from model import predict_image, is_likely_leaf
from crop_predictor import predict_crop, predict_crop_top_k

app = Flask(__name__)

# Allow mobile app and Flutter web to call this API from any origin
@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


_gee_initialized = None
_gee_error = None


def _ensure_gee():
    """Return (True, None) on success, (False, error_message) on failure."""
    global _gee_initialized, _gee_error
    if _gee_initialized is True:
        return True, None
    if _gee_initialized is False:
        return False, (_gee_error or "Earth Engine not initialized. Run: earthengine authenticate")
    try:
        import ee
        project = os.environ.get("EE_PROJECT") or os.environ.get("GEE_PROJECT")
        if not project:
            _gee_initialized = False
            _gee_error = "EE_PROJECT env var not set. Set it to your Google Cloud project ID."
            return False, _gee_error
        ee.Initialize(project=project)
        _gee_initialized = True
        return True, None
    except Exception as e:
        _gee_initialized = False
        _gee_error = str(e).strip() or type(e).__name__
        if "no project found" in _gee_error.lower():
            _gee_error = (
                "Earth Engine needs a project. Set env EE_PROJECT to your Google Cloud project ID "
                "(e.g. my-project-123). See https://developers.google.com/earth-engine/guides/python_install#authenticate."
            )
        print("GEE not initialized:", _gee_error)
        return False, _gee_error


def parse_class(class_name):
    parts = class_name.split("___", 1)
    crop = parts[0].strip() if parts else "Unknown"
    disease = parts[1].strip().replace("_", " ") if len(parts) > 1 else "Unknown"
    return crop, disease


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


def _warmup_gee():
    """Warm up GEE by running a tiny compute on startup (eliminates 16s first-call penalty)."""
    ok, err = _ensure_gee()
    if not ok:
        print(f"[warmup] GEE not ready: {err}")
        return
    try:
        import ee
        ee.Image(1).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry.Point([0, 0]),
            scale=1000,
        ).getInfo()
        print("[warmup] GEE warmed up successfully")
    except Exception as e:
        print(f"[warmup] GEE warmup failed (non-fatal): {e}")


if __name__ == "__main__":
    import threading
    threading.Thread(target=_warmup_gee, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)