import geopandas
import shapely

def create_geometry(geom):
    """
    Create a Shapely geometry from a GeoJSON, WKT or WKB representation.
    
    Args:
        geom (dict, str, bytes): The GeoJSON, WKT or WKB representation of the geometry.
        
    Returns:
        shapely.geometry: The Shapely geometry.
    """
    
    if isinstance(geom, dict): 
        return shapely.from_geojson(str(geom).replace("'", '"'))
    elif isinstance(geom, str):
        return shapely.wkt.loads(geom)
    elif isinstance(geom, bytes):
        return shapely.wkb.loads(geom)
    else:
        raise ValueError("Invalid geometry representation.")

def gpd_fromlist(geometries, crs = 'EPSG:4326'):
    """
    Create a GeoDataFrame from a list of geometries.

    Args:
        geometries (list): A list of Shapely geometries.
        crs (str, optional): The coordinate reference system. Default is 'EPSG:4326'.

    Returns:
        geopandas.GeoDataFrame: The GeoDataFrame created from the list of geometries.
    """
    
    gdf = geopandas.GeoDataFrame(geometry = geometries, crs = crs)
    
    return gdf