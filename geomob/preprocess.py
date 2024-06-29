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

def stop_detection(llt_df, stop_radius, stop_seconds, no_data_seconds, max_speed = None):
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
    
    # in km/h, in case of no delta_time it returns 0 km/h (assuming no movement)
    df['speed'] = (df['delta_space'] / df['delta_time'] * 3600).replace(float('inf'), 0)

    stop_ids = [0]
    waiting_time = 1
    latlngt = df[['lat', 'lng', 'timestamp']].values
    
    for i in range(1, len(df)): 
        lat, lng, t = latlngt[i]
        lat_stop, lng_stop, t_stop = latlngt[stop_ids[-1]]
        
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
    
    agg_by_stop = df.assign(pings = 1).groupby('stop_id').agg({'delta_time' : 'sum', 'pings' : 'count'})
    df['stop_duration'] = df['stop_id'].map(agg_by_stop['delta_time'].to_dict())
    df['pings'] = df['stop_id'].map(agg_by_stop['pings'].to_dict())
    if max_speed is not None:
        df['is_stop'] = (df['stop_duration'] > stop_seconds) & (df['speed'] < max_speed)
    else:
        df['is_stop'] = df['stop_duration'] > stop_seconds
    df.loc[~df['is_stop'], 'pings'] = 0
    
    return df

def location_ranking(stop_df, timezone, start_window = '22:00', end_window = '07:00', method = 'most_frequent', radius = None):
    """
    Calculate the location area based on stop data.

    Args:
        stop_df (pandas.DataFrame): DataFrame containing stop data.
        timezone (str): Timezone of the stop data.
        start_window (str, optional): Start time of the window period. Defaults to '22:00'.
        end_window (str, optional): End time of the window period. Defaults to '07:00'.
        method (str, optional): Method for ranking locations. 
                                Options are 'most_frequent', 'most_certain', and 'longest'. Defaults to 'most_frequent'.
        radius (float, optional): Radius for creating location cluster. 
                                  Keep it low, since it joins stops based on overlapping areas starting from the radius. 
                                  Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the ranking of locations based on the specified method.
            Columns:
                - stop_lat: Latitude of the location.
                - stop_lng: Longitude of the location.
                - most_frequent: Number of unique stop IDs in the location.
                - most_certain: Total number of stop IDs in the location.
                - longest: Sum of stop durations in the location.
    """
    
    stops = stop_df[stop_df['is_stop']].dropna(subset = 'timestamp').sort_values(by='timestamp')

    if radius is not None:
        points = stops.apply(lambda r: shapely.geometry.Point(r['stop_lng'], r['stop_lat']), axis=1)
        stops = geopandas.GeoDataFrame(stops, geometry = points, crs = 'EPSG:4326')
        
        stop_clusters = stops.to_crs(UNIVERSAL_CRS).buffer(radius).to_crs('EPSG:4326').reset_index(name = 'geometry')\
                             .dissolve().reset_index()[['geometry']].explode(index_parts = False).reset_index(drop = True)
        stops = stops.sjoin(stop_clusters, how = 'left', predicate = 'within').rename(columns = {'index_right' : 'cluster_id'})
        stops = stops.set_index('cluster_id').drop(['stop_lat', 'stop_lng'], axis = 1)\
                     .join(stops.groupby('cluster_id').agg({'stop_lat' : 'mean', 'stop_lng' : 'mean'})).reset_index()
                     
    window_stops = stops['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='s', tz=timezone))
    
    if len(window_stops) == 0:
        return pandas.DataFrame(columns = ['stop_lat', 'stop_lng', 'most_frequent', 'most_certain', 'longest'])
    
    window_stop_id = stops.set_index(pandas.DatetimeIndex(window_stops)).between_time(start_window, end_window)['stop_id'].values
        
    window_visits = stops[stops['stop_id'].isin(window_stop_id)]
    
    loc_ranking = window_visits .groupby(['stop_lat', 'stop_lng'])\
                                .agg(most_frequent  = ('stop_id', 'nunique'), 
                                     most_certain   = ('stop_id', 'count'), 
                                     longest        = ('stop_duration', 'sum'))\
                                .sort_values(by = method, ascending = False)\
                                .reset_index()

    return loc_ranking