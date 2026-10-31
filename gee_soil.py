"""
Google Earth Engine: clip SMAP (surface soil moisture 0-5 cm) and MODIS (NDVI, NDWI, LST),
compute mean over the area around the marker (buffer). No ML — statistics only.
Run: earthengine authenticate
"""
from __future__ import annotations
import time
from functools import wraps
import ee
from concurrent.futures import ThreadPoolExecutor

_soil_cache = {}

def cached_soil(ttl_seconds=21600):  # 6 hours — soil changes slower than weather
    def decorator(fn):
        @wraps(fn)
        def wrapper(lat, lng, buffer_km, start_date, end_date):
            key = (round(lat, 2), round(lng, 2), buffer_km, start_date, end_date)
            now = time.time()
            if key in _soil_cache:
                result, expires = _soil_cache[key]
                if now < expires:
                    return result
            result = fn(lat, lng, buffer_km, start_date, end_date)
            _soil_cache[key] = (result, now + ttl_seconds)
            return result
        return wrapper
    return decorator
def get_geometry(lat: float, lng: float, buffer_km: float) -> ee.Geometry:
    """Build geometry: point + buffer (radius in km → meters)."""
    point = ee.Geometry.Point([lng, lat])
    return point.buffer(buffer_km * 1000)


def get_smap_mean(region: ee.Geometry, start_date: str, end_date: str) -> float | None:
    try:
        col = (
            ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
            .filterDate(start_date, end_date)
            .filterBounds(region)                            # ← add this
            .select(["soil_moisture_am"])
        )
        img = col.mean()
        info = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=9000,
            maxPixels=1e9,
        ).getInfo()
        val = info.get("soil_moisture_am") if info else None

        if val is None:
            col_005 = (
                ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005")
                .filterDate(start_date, end_date)
                .filterBounds(region)                        # ← add this too
                .select(["soil_moisture_am"])
            )
            img_005 = col_005.mean()
            info_005 = img_005.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=9000,
                maxPixels=1e9,
            ).getInfo()
            val = info_005.get("soil_moisture_am") if info_005 else None

        return float(val) if val is not None else None
    except Exception:
        return None


def get_modis_ndvi_mean(region: ee.Geometry, start_date: str, end_date: str) -> float | None:
    try:
        col = (
            ee.ImageCollection("MODIS/061/MOD09GA")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .select(["sur_refl_b01", "sur_refl_b02"])
        )
        img = col.mean()
        red = img.select("sur_refl_b01").multiply(0.0001)
        nir = img.select("sur_refl_b02").multiply(0.0001)
        ndvi_img = nir.subtract(red).divide(nir.add(red)).rename("ndvi")

        info = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9,
        ).getInfo()

        if info and info.get("ndvi") is not None:
            return max(-1.0, min(1.0, float(info["ndvi"])))
    except Exception as e:
        print(f"MODIS NDVI error: {e}")
    return None

def get_modis_ndwi_mean(region: ee.Geometry, start_date: str, end_date: str) -> float | None:
    try:
        col = (
            ee.ImageCollection("MODIS/061/MOD09GA")
            .filterDate(start_date, end_date)
            .filterBounds(region)
            .select(["sur_refl_b02", "sur_refl_b06"])
        )
        img = col.mean()
        nir = img.select("sur_refl_b02").multiply(0.0001)
        swir = img.select("sur_refl_b06").multiply(0.0001)
        ndwi_img = nir.subtract(swir).divide(nir.add(swir)).rename("ndwi")

        info = ndwi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9,
        ).getInfo()

        if info and info.get("ndwi") is not None:
            return max(-1.0, min(1.0, float(info["ndwi"])))
    except Exception as e:
        print(f"MODIS NDWI error: {e}")
    return None



def get_modis_lst_mean(region, start_date, end_date):
    try:
        collection = (
            ee.ImageCollection("MODIS/061/MOD21A1D")
            .filterDate(start_date, end_date)
            .filterBounds(region)              # ← add filterBounds, big speedup
            .select(["LST_1KM", "QC"])
        )

        def prepare_lst(image):
            qc = image.select("QC")
            mandatory_qa = qc.bitwiseAnd(3).lte(1)
            data_quality = qc.rightShift(2).bitwiseAnd(3).lte(1)
            cloud_quality = qc.rightShift(4).bitwiseAnd(3).lte(1)
            quality_mask = mandatory_qa.And(data_quality).And(cloud_quality)
            return (
                image.select("LST_1KM")
                .updateMask(quality_mask)
                .subtract(273.15)
                .rename("lst_c")
            )

        temporal_mean = collection.map(prepare_lst).mean()

        # ONE getInfo call
        stats = temporal_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        if not stats or stats.get("lst_c") is None:
            return None

        return round(float(stats["lst_c"]), 2)

    except Exception as e:
        print(f"MODIS LST error: {e}")
        return None



@cached_soil(ttl_seconds=21600)
def get_soil_quality_for_area(
    lat: float,
    lng: float,
    buffer_km: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    For the area (marker + buffer): clip SMAP and MODIS, compute mean values.
    Runs the 4 GEE queries in parallel to reduce total latency.
    """
    region = get_geometry(lat, lng, buffer_km)

    # Run all four GEE queries concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        smap_future = executor.submit(get_smap_mean,       region, start_date, end_date)
        ndvi_future = executor.submit(get_modis_ndvi_mean, region, start_date, end_date)
        ndwi_future = executor.submit(get_modis_ndwi_mean, region, start_date, end_date)
        lst_future  = executor.submit(get_modis_lst_mean,  region, start_date, end_date)

        soil_moisture = smap_future.result()
        ndvi          = ndvi_future.result()
        ndwi          = ndwi_future.result()
        lst           = lst_future.result()

    def clamp_sm(v):
        if v is None: return None
        return max(0.0, min(0.5, float(v)))

    def clamp_nd(v):
        if v is None: return None
        return max(-1.0, min(1.0, float(v)))

    def clamp_lst(v):
        if v is None: return None
        return float(v)

    result = {
        "soil_moisture": clamp_sm(soil_moisture),
        "ndvi": clamp_nd(ndvi),
        "ndwi": clamp_nd(ndwi),
        "lst": clamp_lst(lst),
    }

    # Log what's missing (helps debugging)
    missing = [k for k, v in result.items() if v is None]
    if missing:
        print(f"[soil] WARNING missing bands: {missing} for {lat},{lng} ({start_date}→{end_date})")

    return result