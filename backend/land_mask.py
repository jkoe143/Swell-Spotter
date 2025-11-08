import json
import pickle
import numpy as np
from pathlib import Path
from shapely.geometry import Point, shape

# splits into 259,200 grid cells
GRID_WIDTH = 720
GRID_HEIGHT = 360


_LAND_MASK_CACHE = None

def load_land_cells():
    # https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_land.geojson
    geojson_path = Path(__file__).parent / "data" / "ne_50m_land.geojson"
    # https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_10m_minor_islands.geojson
    islands_path = Path(__file__).parent / "data" / "ne_10m_minor_islands.geojson" 
    
    cells = []
    
    # load main land geojson cells
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    for feature in geojson_data.get('features', []):
        try:
            poly = shape(feature['geometry'])
            cells.append(poly)
        except Exception:
            continue
    
    # load minor islands geojson cells
    if islands_path.exists():
        with open(islands_path, 'r', encoding='utf-8') as f:
            islands_data = json.load(f)
        
        for feature in islands_data.get('features', []):
            try:
                poly = shape(feature['geometry'])
                cells.append(poly)
            except Exception:
                continue
    
    return cells

def generate_land_mask(width=GRID_WIDTH, height=GRID_HEIGHT):
    mask = np.zeros((height, width), dtype=np.uint8)
    land_cells = load_land_cells()
    
    for row in range(height):
        lat = 90 - (row + 0.5) * (180 / height)
        
        for col in range(width):
            lon = -180 + (col + 0.5) * (360 / width)
            point = Point(lon, lat)
            
            for poly in land_cells:
                if poly.contains(point):
                    mask[row, col] = 1
                    break
    return mask

def get_land_mask():
    global _LAND_MASK_CACHE
    
    if _LAND_MASK_CACHE is None:
        mask_path = Path(__file__).parent / "data" / "land_mask.pkl"
        
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Land mask file not found at {mask_path}. "
                "Run generate_land_mask.py in the backend directory to create it."
            )
        
        with open(mask_path, 'rb') as f:
            _LAND_MASK_CACHE = pickle.load(f)
    return _LAND_MASK_CACHE

def latlon_to_grid(lat, lon, width=GRID_WIDTH, height=GRID_HEIGHT):
    row = int((90 - lat) * height / 180)
    row = max(0, min(height - 1, row))
    
    col = int((lon + 180) * width / 360)
    col = max(0, min(width - 1, col))
    
    return (row, col)

def grid_to_latlon(row, col, width=GRID_WIDTH, height=GRID_HEIGHT):
    lat = 90 - (row + 0.5) * (180 / height)
    lon = -180 + (col + 0.5) * (360 / width)
    return (lat, lon)