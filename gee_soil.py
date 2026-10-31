"""
Google Earth Engine: clip SMAP (surface soil moisture 0-5 cm) and MODIS (NDVI, NDWI, LST),
compute mean over the area around the marker (buffer). No ML — statistics only.
Run: earthengine authenticate
"""
from __future__ import annotations

import ee


def get_geometry(lat: float, lng: float, buffer_km: float) -> ee.Geometry:
    """Build geometry: point + buffer (radius in km → meters)."""
    point = ee.Geometry.Point([lng, lat])
    return point.buffer(buffer_km * 1000)


def get_smap_mean(region: ee.Geometry, start_date: str, end_date: str) -> float | None:
    """
    SMAP L3 9 km: surface soil moisture (0-5 cm). Band soil_moisture_am, volume fraction ~0-0.5.
    """
    try:
        col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006").filterDate(start_date, end_date)
        col = col.select(["soil_moisture_am"])
        img = col.mean()
        reduced = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=9000,
            maxPixels=1e9,
        )
        info = reduced.getInfo()
        val = info.get("soil_moisture_am") if info else None
        if val is None:
            col_005 = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/005").filterDate(start_date, end_date)
            col_005 = col_005.select(["soil_moisture_am"])
            img_005 = col_005.mean()
            reduced_005 = img_005.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=9000,
                maxPixels=1e9,
            )
            info_005 = reduced_005.getInfo()
            val = info_005.get("soil_moisture_am") if info_005 else None
        return float(val) if val is not None else None
    except Exception:
        return None


def get_modis_ndvi_mean(region: ee.Geometry, start_date: str, end_date: str) -> float | None:
    """MODIS Terra Daily NDVI. Tries 006 pre-computed; fallback: compute from MOD09GA 061 (b01=red, b02=NIR)."""
    try:
        col = ee.ImageCollection("MODIS/MOD09GA_006_NDVI").filterDate(start_date, end_date)
        img = col.mean()
        reduced = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9,
        )
        info = reduced.getInfo()
        if info:
            val = info.get("NDVI") or info.get("ndvi")
            if val is not None:
                v = float(val)
                if v > 1 or v < -1:
                    v = v * 0.0001
                return v
    except Exception:
        pass
    # Fallback: MOD09GA_006_NDVI ends 2023-02-17; use MOD09GA 061 and compute NDVI (2000-02-24 to 2024-06-14)
    try:
        col = ee.ImageCollection("MODIS/061/MOD09GA").filterDate(start_date, end_date).select(["sur_refl_b01", "sur_refl_b02"])
        img = col.mean()
        scale = 0.0001
        red = img.select("sur_refl_b01").multiply(scale)
        nir = img.select("sur_refl_b02").multiply(scale)
        ndvi_img = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
        reduced = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9,
        )
        info = reduced.getInfo()
        if info:
            val = info.get("ndvi")
            if val is not None:
                return max(-1.0, min(1.0, float(val)))
    except Exception:
        pass
    return None


def get_modis_ndwi_mean(region: ee.Geometry, start_date: str, end_date: str) -> float | None:
    """MODIS Terra Daily NDWI. Tries 006 pre-computed; fallback: compute from MOD09GA 061 (b02=NIR, b06=SWIR)."""
    try:
        col = ee.ImageCollection("MODIS/MOD09GA_006_NDWI").filterDate(start_date, end_date)
        img = col.mean()
        reduced = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9,
        )
        info = reduced.getInfo()
        if info:
            val = info.get("NDWI") or info.get("ndwi")
            if val is not None:
                v = float(val)
                if v > 1 or v < -1:
                    v = v * 0.0001
                return v
    except Exception:
        pass
    # Fallback: compute NDWI from MOD09GA 061 (NIR - SWIR)/(NIR + SWIR)
    try:
        col = ee.ImageCollection("MODIS/061/MOD09GA").filterDate(start_date, end_date).select(["sur_refl_b02", "sur_refl_b06"])
        img = col.mean()
        scale = 0.0001
        nir = img.select("sur_refl_b02").multiply(scale)
        swir = img.select("sur_refl_b06").multiply(scale)
        ndwi_img = nir.subtract(swir).divide(nir.add(swir)).rename("ndwi")
        reduced = ndwi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9,
        )
        info = reduced.getInfo()
        if info:
            val = info.get("ndwi")
            if val is not None:
                return max(-1.0, min(1.0, float(val)))
    except Exception:
        pass
    return None




def get_modis_lst_mean(region, start_date, end_date):
    try:
        collection = (
            ee.ImageCollection("MODIS/061/MOD21A1D")
            .filterDate(start_date, end_date)
            .select(["LST_1KM", "QC"])
        )

        image_count = collection.size().getInfo()
        print(f"MODIS LST images: {image_count}")

        if image_count == 0:
            print("No MODIS LST images found for this date range.")
            return None

        def prepare_lst(image):
            qc = image.select("QC")

            # Bits 0-1: Mandatory QA (0 = good, 1 = nominal)
            mandatory_qa = qc.bitwiseAnd(3).lte(1)

            # Bits 2-3: Data quality (0 = good)
            data_quality = qc.rightShift(2).bitwiseAnd(3).eq(0)

            # Bits 4-5: Cloud flag (0 = clear, 1 = thin cirrus)
            cloud_quality = qc.rightShift(4).bitwiseAnd(3).lte(1)

            quality_mask = (
                mandatory_qa
                .And(data_quality)
                .And(cloud_quality)
            )

            # Earth Engine has already applied the scale factor.
            lst_celsius = (
                image.select("LST_1KM")
                .updateMask(quality_mask)
                .subtract(273.15)
                .rename("lst_c")
            )

            return lst_celsius

        lst_collection = collection.map(prepare_lst)
        temporal_mean = lst_collection.mean()

        stats = temporal_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            maxPixels=1e9,
            bestEffort=True
        )

        pixel_stats = temporal_mean.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=1000,
            maxPixels=1e9,
            bestEffort=True
        )

        stats_info = stats.getInfo()
        pixel_info = pixel_stats.getInfo()

        value = stats_info.get("lst_c")
        valid_pixels = pixel_info.get("lst_c", 0)

        print(f"MODIS LST result: {value}")
        print(f"MODIS LST valid pixels: {valid_pixels}")

        if value is None or valid_pixels == 0:
            print("No valid LST pixels after quality masking.")
            return None

        return round(float(value), 2)

    except Exception as e:
        print(f"MODIS LST error: {e}")
        return None




def get_soil_quality_for_area(
    lat: float,
    lng: float,
    buffer_km: float,
    start_date: str,
    end_date: str,
) -> dict:
    """
    For the area (marker + buffer): clip SMAP and MODIS, compute mean values.
    Returns dict: soil_moisture (0-0.5), ndvi, ndwi, lst (°C). Missing values as None.
    """
    region = get_geometry(lat, lng, buffer_km)

    soil_moisture = get_smap_mean(region, start_date, end_date)
    ndvi = get_modis_ndvi_mean(region, start_date, end_date)
    ndwi = get_modis_ndwi_mean(region, start_date, end_date)
    lst = get_modis_lst_mean(region, start_date, end_date)
    print("\n========== SOIL REQUEST ==========")
    print(f"Latitude: {lat}")
    print(f"Longitude: {lng}")
    print(f"Buffer: {buffer_km} km")
    print(f"Dates: {start_date} - {end_date}")
    print("==================================")

    def clamp_sm(v):
        if v is None:
            return None
        return max(0.0, min(0.5, float(v)))

    def clamp_nd(v):
        if v is None:
            return None
        return max(-1.0, min(1.0, float(v)))

    def clamp_lst(v):
        if v is None:
            return None
        return float(v)

    return {
        "soil_moisture": clamp_sm(soil_moisture),
        "ndvi": clamp_nd(ndvi),
        "ndwi": clamp_nd(ndwi),
        "lst": clamp_lst(lst),
    }
