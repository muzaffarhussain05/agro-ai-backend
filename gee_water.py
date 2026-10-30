"""
Google Earth Engine: GLDAS (precipitation, evapotranspiration, soil moisture) for water availability.
Returns daily time series for a point. No ML — statistics only.
Run: earthengine authenticate
"""
from __future__ import annotations

import ee
from datetime import datetime, timedelta


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


def get_water_availability_timeseries(
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """
    Daily time series: GLDAS precipitation, evapotranspiration, soil moisture for the point.
    Returns list of { date, soilMoisture, precipitation, evapotranspiration }.
    Uses a fixed range (e.g. 30 days) to avoid too many GEE calls; caps at 31 days.
    """
    region = get_geometry(lat, lng)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    if (end - start).days > 31:
        end = start + timedelta(days=31)
    out = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        row = get_gldas_daily_for_date(region, date_str)
        if row:
            if row.get("soilMoisture") is None:
                row["soilMoisture"] = 0.2  # fallback if band missing
            out.append(row)
        else:
            out.append({
                "date": date_str,
                "soilMoisture": 0.2,
                "precipitation": 0.0,
                "evapotranspiration": 3.0,
            })
        current += timedelta(days=1)
    return out
