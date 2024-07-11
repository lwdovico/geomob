import geopandas
import requests
import zipfile
import tempfile

LANDCOVER_URL = 'https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip'

from .preprocess import create_geometry, gpd_fromlist

def _read_remote_zipped_geofile(url, geofile_path = '', bbox = None):
    """
    Reads a geofile from a remote zipped file.
    
    Args:
        url (str): The URL of the zipped file.
        geofile_path (str): The path of the file inside the zipped file.
        bbox (tuple, optional): The bounding box to filter the data.
        
    Returns:
        geopandas.GeoDataFrame: The GeoDataFrame containing the data.
    """
    
    file_downloaded = requests.get(url)
        
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', prefix='landcover', suffix='.zip') as temp_file:
        temp_file.write(file_downloaded.content)
        zip_path = temp_file.name
        
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            return geopandas.read_file(f'{temp_dir}/{geofile_path}', bbox=bbox).dissolve()[['geometry']]
    
    
def retrieve_osm(query, user_agent = 'MyApp/1.0 (mymail@gmail.com)'):
    """
    Retrieves OpenStreetMap data for a given query.

    Args:
        query (str): The search query for the desired location.
        user_agent (str, optional): The user agent string to be used in the request headers.
            Defaults to 'MyApp/1.0 (mymail@gmail.com)'.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the retrieved OpenStreetMap geometry.
        
    """
    
    endpoint = 'nominatim.openstreetmap.org'
    nominatim_url = f"https://{endpoint}/search.php?q={query}&polygon_geojson=1&format=json"

    response = requests.get(nominatim_url, 
                            headers = { 'User-Agent': user_agent }).json()
    
    features = [{
                    'type': 'Feature',
                    'geometry': create_geometry(res['geojson']),
                    'properties': {'place_name': res['display_name']}
                        
                } for res in response
                ]

    return geopandas.GeoDataFrame.from_features(features, crs='EPSG:4326')

def landcover_from_bbox(bbox):
    """
    Retrieves landcover data for a given bounding box.
    
    Args:
        bbox (tuple): The bounding box to filter the data.
    """
    return _read_remote_zipped_geofile(LANDCOVER_URL, 
                                      geofile_path='land-polygons-split-4326/land_polygons.shp', 
                                      bbox = bbox)
    
def clip_shape_to_landcover(shape, landcover = None):
    """
    Clips a shape to the landcover data.
    
    Args:
        shape (geopandas.GeoDataFrame): The GeoDataFrame containing the shape to be clipped.
        landcover (geopandas.GeoDataFrame, optional): The GeoDataFrame containing the landcover data.
                                                      If None, the landcover data will be retrieved.
    """
    shape_bounds = tuple(gpd_fromlist(shape['geometry']).dissolve().bounds.values[0])
    
    if landcover is None:
        land_area = landcover_from_bbox(bbox = shape_bounds).dissolve()[['geometry']]
    else:
        land_area = landcover.dissolve()[['geometry']]
    
    clipped_shape = shape.clip(land_area)
    
    return clipped_shape