import ee
from PIL import Image
import requests
import io

try:
    # IMPORTANT: 'your-google-cloud-project-id'-ku badhila ungaloda GEE / Google Cloud Project ID-ah podunga
    ee.Initialize(project='earthengine-512413')
except:
    ee.Authenticate()
    ee.Initialize(project='earthengine-512413')

def fetch_live_sentinel_tile(lat, lon, start_date='2025-01-01', end_date='2026-10-01', buffer_km=3):
    """
    Fetches real Sentinel-2 satellite imagery directly from Google Earth Engine servers.
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_km * 1000).bounds()

    s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(region)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                     .median())

    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.4
    }

    url = s2_collection.getThumbURL({
        'region': region,
        'dimensions': 512,
        'format': 'png',
        **vis_params
    })

    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    return img