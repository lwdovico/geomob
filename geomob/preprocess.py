import pandas
import geopandas
import shapely
import haversine

UNIVERSAL_CRS = 'EPSG:3857'

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

def stop_detection(llt_df, stop_radius, stop_seconds, no_data_seconds):
    """
    Detect stops in a trajectory based on given parameters.

    Args:
        llt_df (pandas.DataFrame): DataFrame containing lat, lng, and timestamp columns.
        stop_radius (float): Radius in kilometers to define a stop.
        stop_seconds (int): Minimum duration in seconds to consider a stop.
        no_data_seconds (int): Maximum duration in seconds without data to consider a stop.

    Returns:
        pandas.DataFrame: DataFrame with additional stop_id, stop_lat, stop_lng, arrival_time, and leaving_time columns.
    """
    
    df = llt_df.sort_values('timestamp').reset_index(drop=True)
    df['lat'] = df['lat'].astype(float)
    df['lng'] = df['lng'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    
    df['next_lat'] = df['lat'].shift(-1)
    df['next_lng'] = df['lng'].shift(-1)
    df['next_timestamp'] = df['timestamp'].shift(-1)

    df['delta_space'] = df.apply(lambda r: haversine.haversine((r['lat'], r['lng']), 
                                                               (r['next_lat'], r['next_lng'])), axis=1)
    
    df['delta_time'] = (df['next_timestamp'] - df['timestamp'])
    
    df['speed'] = (df['delta_space'] / df['delta_time'] * 3600).replace(float('inf'), 0) # in km/h, in case of no delta_time it returns 0 km/h (assuming no movement)

    stop_ids = [0]
    waiting_time = 1
    
    for i in range(1, len(df)): 
        lat, lng, t = df.iloc[i][['lat', 'lng', 'timestamp']].values
        lat_stop, lng_stop, t_stop = df.iloc[stop_ids[-1]][['lat', 'lng', 'timestamp']].values
        
        if (t - t_stop) > no_data_seconds:
            stop_ids.extend(range(stop_ids[-1] + 1, stop_ids[-1]  + waiting_time + 1))
            waiting_time = 1
            continue
        
        space_condition = haversine.haversine([lat, lng], [lat_stop, lng_stop]) < stop_radius
        time_condition = (t - t_stop) > stop_seconds
        
        if space_condition and time_condition:
            stop_ids.extend([stop_ids[-1]]*waiting_time)
            waiting_time = 1
                
        elif space_condition:
            waiting_time += 1
        
        else:
            stop_ids.extend(range(stop_ids[-1] + 1, stop_ids[-1] + waiting_time + 1))
            waiting_time = 1
            
    stop_ids.extend([stop_ids[-1]]*(waiting_time - 1))
        
    df['stop_id'] = stop_ids
    
    df = df.set_index('stop_id').join(df.groupby('stop_id').agg(stop_lat = ('lat', 'mean'), 
                                                                stop_lng = ('lng', 'mean'), 
                                                                arrival_time = ('timestamp', 'first'),
                                                                leaving_time = ('next_timestamp', 'last'))).reset_index()
    
    return df

def timestamp_to_timezone(timestamp, pytz_timezone):
    return pandas.Timestamp(timestamp, unit='s', tz=pytz_timezone)

def home_catchement(stop_df, timezone, start_night = 22, end_night = 7, home_radius = None):
    stop_df = stop_df.sort_values('arrival_time').reset_index(drop=True)
    return