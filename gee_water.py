"""
Google Earth Engine: GLDAS (precipitation, evapotranspiration, soil moisture) for water availability.
Returns daily time series for a point. No ML — statistics only.
Run: earthengine authenticate
"""
from __future__ import annotations
import time
from functools import wraps
import ee
from datetime import datetime, timedelta


_water_cache = {}
_WATER_CACHE_TTL = 3600  # 1 hour

def cached_water(ttl_seconds=3600):
    def decorator(fn):
        @wraps(fn)
        def wrapper(lat, lng, start_date, end_date):
            key = (round(lat, 2), round(lng, 2), start_date, end_date)
            now = time.time()
            if key in _water_cache:
                result, expires = _water_cache[key]
                if now < expires:
                    return result
            result = fn(lat, lng, start_date, end_date)
            _water_cache[key] = (result, now + ttl_seconds)
            return result
        return wrapper
    return decorator
def get_geometry(lat: float, lng: float) -> ee.Geometry:
    """Point geometry for GLDAS (27 km resolution)."""
    return ee.Geometry.Point([lng, lat])


def get_gldas_daily_for_date(
    region: ee.Geometry,
    date_str: str,
) -> dict | None:
    """
    Get GLDAS-2.1 3-hourly means for one day; convert to daily precip (mm), ET (mm/day), soil moisture (m³/m³).
    Bands: Rainf_f_tavg (kg/m²/s), Evap_tavg (kg/m²/s), SoilMoi0_10cm_inst (kg/m²).
    """
    try:
        end = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        end_str = end.strftime("%Y-%m-%d")
        col = (
            ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
            .filterDate(date_str, end_str)
            .select(["Rainf_f_tavg", "Evap_tavg", "SoilMoi0_10cm_inst"])
        )
        img = col.mean()
        reduced = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=27830,
            maxPixels=1e9,
        )
        info = reduced.getInfo()
        if not info:
            return None
        rain = info.get("Rainf_f_tavg")
        evap = info.get("Evap_tavg")
        sm = info.get("SoilMoi0_10cm_inst")
        if rain is None and evap is None and sm is None:
            return None
        # kg/m²/s -> mm/day: multiply by 86400 s/day (1 kg/m² = 1 mm)
        precip_mm = (float(rain) * 86400) if rain is not None else 0.0
        et_mm = (float(evap) * 86400) if evap is not None else 0.0
        # SoilMoi0_10cm_inst kg/m² over 0–10 cm -> approx m³/m³: / 100 (1000 kg/m³ * 0.1 m)
        sm_vol = (float(sm) / 100.0) if sm is not None else None
        if sm_vol is not None:
            sm_vol = max(0.0, min(0.5, sm_vol))
        return {
            "date": date_str,
            "soilMoisture": sm_vol,
            "precipitation": round(precip_mm, 2),
            "evapotranspiration": round(et_mm, 2),
        }
    except Exception:
        return None

@cached_water(ttl_seconds=3600)
def get_water_availability_timeseries(
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Daily time series: GLDAS precipitation, evapotranspiration, soil moisture for the point.
    Uses getRegion() — single GEE round-trip for the entire date range (30x faster than per-day getInfo).
    """
    region = get_geometry(lat, lng)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if (end - start).days > 31:
        end = start + timedelta(days=31)
    
    # GLDAS collection covers the range; one getRegion returns all timesteps
    end_inclusive = end + timedelta(days=1)
    collection = (
        ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
        .filterDate(start.strftime("%Y-%m-%d"), end_inclusive.strftime("%Y-%m-%d"))
        .filterBounds(region)
        .select(["Rainf_f_tavg", "Evap_tavg", "SoilMoi0_10cm_inst"])
    )
    
    # Single network round-trip — returns rows like:
    #   [lon, lat, time, Rainf_f_tavg, Evap_tavg, SoilMoi0_10cm_inst]
    # time is Unix epoch in milliseconds
    region_data = collection.getRegion(
        geometry=region,
        scale=27830,           # GLDAS native resolution
    ).getInfo()
    
    if not region_data or len(region_data) < 2:
        # Fallback: return a full range with defaults if GEE returned nothing
        out = []
        current = start
        while current <= end:
            out.append({
                "date": current.strftime("%Y-%m-%d"),
                "soilMoisture": 0.2,
                "precipitation": 0.0,
                "evapotranspiration": 3.0,
            })
            current += timedelta(days=1)
        return out
    
    header = region_data[0]
    rows = region_data[1:]
    
    # Column indexes — getRegion header includes 'lon', 'lat', 'time', then band names
    idx_time = header.index("time")
    idx_rain = header.index("Rainf_f_tavg")
    idx_evap = header.index("Evap_tavg")
    idx_sm = header.index("SoilMoi0_10cm_inst")
    
    # Aggregate GLDAS' 3-hourly records into daily values
    daily: dict[str, dict] = {}
    for row in rows:
        if row[idx_time] is None:
            continue
        date_obj = datetime.utcfromtimestamp(row[idx_time] / 1000.0)
        date_str = date_obj.strftime("%Y-%m-%d")
        
        if date_str not in daily:
            daily[date_str] = {
                "rain_sum": 0.0,
                "rain_n": 0,
                "evap_sum": 0.0,
                "evap_n": 0,
                "sm_sum": 0.0,
                "sm_n": 0,
            }
        
        # GLDAS is 3-hourly: rain/evap are rates (kg/m²/s), so each sample covers 3 hours = 10800s
        # Summing 8 samples/day gives daily total. We average rate then multiply by 86400 below.
        if row[idx_rain] is not None:
            daily[date_str]["rain_sum"] += float(row[idx_rain])
            daily[date_str]["rain_n"] += 1
        if row[idx_evap] is not None:
            daily[date_str]["evap_sum"] += float(row[idx_evap])
            daily[date_str]["evap_n"] += 1
        if row[idx_sm] is not None:
            daily[date_str]["sm_sum"] += float(row[idx_sm])
            daily[date_str]["sm_n"] += 1
    
    # Build final output
    out = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        if date_str in daily:
            d = daily[date_str]
            # Average the 3-hourly rates, then convert to daily mm
            rain_rate = d["rain_sum"] / d["rain_n"] if d["rain_n"] else 0.0
            evap_rate = d["evap_sum"] / d["evap_n"] if d["evap_n"] else 0.0
            sm_mean = d["sm_sum"] / d["sm_n"] if d["sm_n"] else None
            
            precip_mm = rain_rate * 86400.0       # kg/m²/s -> mm/day
            et_mm = evap_rate * 86400.0
            sm_vol = (sm_mean / 100.0) if sm_mean is not None else 0.2
            sm_vol = max(0.0, min(0.5, sm_vol))
            
            out.append({
                "date": date_str,
                "soilMoisture": sm_vol,
                "precipitation": round(precip_mm, 2),
                "evapotranspiration": round(et_mm, 2),
            })
        else:
            # Missing day — use fallback
            out.append({
                "date": date_str,
                "soilMoisture": 0.2,
                "precipitation": 0.0,
                "evapotranspiration": 3.0,
            })
        current += timedelta(days=1)
    
    return out