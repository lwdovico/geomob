import geopandas
import requests

from .preprocess import create_geometry

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