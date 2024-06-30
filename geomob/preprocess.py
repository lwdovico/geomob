import numpy
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

def trajectory_detection(llt_df, stop_radius, stop_seconds, no_data_seconds):
    """
    Enrich the events of a user with trajectory stats based on a given set of stop parameters.

    Args:
        llt_df (pandas.DataFrame): DataFrame containing lat, lng, and timestamp columns.
        stop_radius (float): Radius in kilometers to define a stop.
        stop_seconds (int): Minimum duration in seconds to consider a stop.
        no_data_seconds (int): Maximum duration in seconds without data to consider it the same trajectory.

    Returns:
        pandas.DataFrame: DataFrame with additional columns: 
        - prev_lat (float): Latitude of the previous point.
        - prev_lng (float): Longitude of the previous point.
        - prev_timestamp (float): Timestamp of the previous point.
        - next_lat (float): Latitude of the next point.
        - next_lng (float): Longitude of the next point.
        - next_timestamp (float): Timestamp of the next point.
        - delta_space (float): Distance in kilometers between the current point and the next point.
        - delta_time (float): Time difference in seconds between the current point and the next point.
        - speed (float): Speed in km/h between the current point and the next point.
        - traj_id (int): Identifier of the trajectory to which the current point belongs.
        - orig_lat (float): Latitude of the origin point of the current trajectory.
        - orig_lng (float): Longitude of the origin point of the current trajectory.
        - dest_lat (float): Latitude of the destination point of the current trajectory.
        - dest_lng (float): Longitude of the destination point of the current trajectory.
        - mean_lat (float): Mean latitude of the trajectory.
        - mean_lng (float): Mean longitude of the trajectory.
        - from_timestamp (int): Timestamp of the point of the trajectory from which the current point comes.
        - start_time (int): Timestamp of the first point of the trajectory.
        - end_time (int): Timestamp of the next point after the trajectory.
        - total_duration (int): Total duration in seconds of the trajectory.
        - pings_in_traj (int): Number of points in the trajectory.
        - is_stop (bool): Indicates whether the trajectory is considered a stop or not on the set conditions.
    """
    
    df = llt_df.sort_values('timestamp').reset_index(drop=True)
    df['lat'] = df['lat'].astype(float)
    df['lng'] = df['lng'].astype(float)
    df['timestamp'] = df['timestamp'].astype(int)
    
    df['prev_lat'] = df['lat'].shift(1)
    df['prev_lng'] = df['lng'].shift(1)
    df['prev_timestamp'] = df['timestamp'].shift(1)
    
    df['next_lat'] = df['lat'].shift(-1)
    df['next_lng'] = df['lng'].shift(-1)
    df['next_timestamp'] = df['timestamp'].shift(-1)

    df['delta_space'] = df.apply(lambda r: haversine.haversine((r['lat'], r['lng']), 
                                                               (r['next_lat'], r['next_lng'])), axis=1)
    
    df['delta_time'] = (df['next_timestamp'] - df['timestamp'])
    
    # in km/h, in case of no delta_time it returns 0 km/h (assuming no movement)
    df['speed'] = (df['delta_space'] / df['delta_time'] * 3600).replace(float('inf'), 0)

    traj_ids = [0]
    waiting_time = 1
    latlngt = df[['lat', 'lng', 'timestamp']].values
    
    for i in range(1, len(df)): 
        lat, lng, t = latlngt[i]
        lat_stop, lng_stop, t_stop = latlngt[traj_ids[-1]]
        
        if (t - t_stop) > no_data_seconds:
            traj_ids.extend(range(traj_ids[-1] + 1, traj_ids[-1]  + waiting_time + 1))
            waiting_time = 1
            continue
        
        space_condition = haversine.haversine([lat, lng], [lat_stop, lng_stop]) < stop_radius
        time_condition = (t - t_stop) > stop_seconds
        
        if space_condition and time_condition:
            traj_ids.extend([traj_ids[-1]]*waiting_time)
            waiting_time = 1
                
        elif space_condition:
            waiting_time += 1
        
        else:
            traj_ids.extend(range(traj_ids[-1] + 1, traj_ids[-1] + waiting_time + 1))
            waiting_time = 1
            
    traj_ids.extend([traj_ids[-1]]*(waiting_time - 1))
        
    df['traj_id'] = traj_ids
    
    agg_by_traj = df.assign(pings = 1).groupby('traj_id').agg(orig_lat = ('prev_lat', 'first'),
                                                              orig_lng = ('prev_lng', 'first'),
                                                              dest_lat = ('next_lat', 'last'),
                                                              dest_lng = ('next_lng', 'last'),
                                                              mean_lat = ('lat', 'mean'), 
                                                              mean_lng = ('lng', 'mean'), 
                                                              from_timestamp = ('prev_timestamp', 'first'),
                                                              start_time = ('timestamp', 'first'),
                                                              end_time = ('next_timestamp', 'last'),
                                                              total_distance = ('delta_space', 'sum'),
                                                              total_duration = ('delta_time', 'sum'),
                                                              traj_max_speed = ('speed', 'max'),
                                                              pings_in_traj = ('pings', 'sum'))
    
    df = df.set_index('traj_id').join(agg_by_traj).reset_index()

    df['is_stop'] = df['total_duration'] > stop_seconds
    
    return df

def filter_blocks(df, filter_id, **kwargs):
    """
    Helper function to filter out strange blocks of data.
    """
    min_radius = kwargs.get('min_radius', None)
    min_radius_condition = df.assign(filtering = True)['filtering']
    
    if min_radius is not None:
        min_radius_apply = lambda x: numpy.max(haversine.haversine_vector(x[['lat', 'lng']].values, 
                                                                          x[['lat', 'lng']].values, comb=True)) > min_radius
        
        min_radius_condition = df[filter_id].map(df.groupby(filter_id).apply(min_radius_apply, include_groups=False).to_dict())
    
    filtering = min_radius_condition
    
    return filtering

def stop_detection(df, max_speed = None):
    """
    Detect stops in a DataFrame based on the 'is_stop' column and an optional maximum speed threshold.

    Args:
        df (pandas.DataFrame): DataFrame containing trajectory data.
        max_speed (float, optional): Maximum speed threshold in km/h. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the detected stops.
            Columns:
                - traj_id: Identifier of the trajectory.
                - stop_id: Identifier of the stop.
                - mean_lat: Mean latitude of the stop.
                - mean_lng: Mean longitude of the stop.
                - arrival_time: Timestamp of the first point of the stop.
                - leaving_time: Timestamp of the next point after the stop.
                - total_duration: Total duration in seconds of the stop.
                - pings_in_stop: Number of points in the stop.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    stop_cols = ['traj_id', 'mean_lat', 'mean_lng', 
                 'arrival_time', 'leaving_time', 'total_duration', 'pings_in_stop']
    
    df['consider_stop'] = df['is_stop']
    
    if max_speed is not None:
        df['consider_stop'] = df['consider_stop'] & (df['traj_max_speed'] < max_speed)
    
    stop_df = df[df['consider_stop']].rename(columns = {'start_time' : 'arrival_time', 
                                                        'end_time' : 'leaving_time',
                                                        'pings_in_traj' : 'pings_in_stop'})[stop_cols]\
                                     .drop_duplicates()\
                                     .reset_index(drop = True)\
                                     .reset_index(names = 'stop_id')
    
    return stop_df

def sequence_integrity(df, index = 'trip_id', integrity_speed = 250):
    """
    Clean sequence of points based on the integrity speed.
    
    Args:
        df (pandas.DataFrame): DataFrame containing data points with timestamp, lat and lng.
        integrity_speed (float): The minimum speed to consider an OD as valid.
    
    Returns:
        pandas.Index: Index of the valid ODs.
    
    """
    seq_aggregated = df.groupby(index).agg( depart_time = ('timestamp', 'first'),
                                            dest_time = ('timestamp', 'last'),
                                            depart_lat = ('lat', 'first'),
                                            dest_lat = ('lat', 'last'),
                                            depart_lng = ('lng', 'first'),
                                            dest_lng = ('lng', 'last'))
    
    seq_aggregated['integrity_speed'] = False
    
    # remove sequentially blocks of points that do not meet the integrity speed
    
    while ~(seq_aggregated['integrity_speed'].all()):
        seq_aggregated['prev_dest_lat'] = seq_aggregated['dest_lat'].shift(1)
        seq_aggregated['prev_dest_lng'] = seq_aggregated['dest_lng'].shift(1)
        
        seq_aggregated['distance_from_previous_seq'] = seq_aggregated.apply(lambda r: haversine.haversine((r['depart_lat'], r['depart_lng']),
                                                                                                          (r['prev_dest_lat'], r['prev_dest_lng'])), axis=1)
        
        seq_aggregated['time_from_previous_seq'] = seq_aggregated['depart_time'] - seq_aggregated['dest_time'].shift(1)
        
        seq_aggregated['speed_from_previous_seq'] = (seq_aggregated['distance_from_previous_seq'] / seq_aggregated['time_from_previous_seq'] * 3600).fillna(0)
        seq_aggregated['integrity_speed'] = seq_aggregated['speed_from_previous_seq'] < integrity_speed
        
        problematic_od = seq_aggregated[~seq_aggregated['integrity_speed']].index
        
        if len(problematic_od) > 0:
            seq_aggregated.drop(problematic_od[0], inplace = True)
        
    return seq_aggregated.index
    
def trip_detection(df, integrity_speed = 250, 
                       integrity_space = 100,
                       integrity_time = 60*60,
                       min_max_distance = None,
                       min_pings_in_trip = None,
                       return_one_exec = False):
    """
    Detect trips in a DataFrame based on the 'is_stop' column and the minimum and maximum duration thresholds.

    Args:
        df (pandas.DataFrame): DataFrame containing trajectory data.
        min_max_distance(float, optional): The minimum maximum distance of a pair of points to define a trip, in kilometers. Defaults to None.
        
    """
    df = df[['lat', 'lng', 'timestamp', 'delta_space', 'delta_time', 'speed', 'is_stop', 'pings_in_traj']].sort_values('timestamp').reset_index(drop=True)
    
    df['break_point'] = (df['speed'] > integrity_speed) | (df['delta_space'] > integrity_space) | (df['delta_time'] > integrity_time)
    
    df['consider_trip'] = ~df['is_stop'] & ~df['break_point']
    
    df.insert(0, 'trip_id', (df['consider_trip'] & (~df['consider_trip'].shift(fill_value=False))).cumsum())
    df['trip_id'] += df['break_point'].cumsum()
    
    trip_df = df[df['consider_trip']].copy(deep = True)
    
    if min_max_distance is not None:
        trip_df['consider_trip_distance'] = filter_blocks(trip_df, filter_id = 'trip_id', min_radius = min_max_distance)
        trip_df = trip_df[trip_df['consider_trip_distance']].copy(deep = True)
    
    trip_df['pings_in_traj'] = trip_df['trip_id'].map(trip_df.groupby('trip_id')['pings_in_traj'].sum().to_dict())
    
    if min_pings_in_trip is not None:
        trip_df['consider_trip_pings'] = trip_df['pings_in_traj'] > min_pings_in_trip
        trip_df = trip_df[trip_df['consider_trip_pings']].copy(deep = True)
    
    integ_trips = sequence_integrity(trip_df, index = 'trip_id', integrity_speed = integrity_speed)
    trip_df_wattr = trip_df.set_index('trip_id').loc[integ_trips].reset_index()
    
    trip_df_wattr['next_lat'] = trip_df_wattr['lat'].shift(-1)
    trip_df_wattr['next_lng'] = trip_df_wattr['lng'].shift(-1)
    trip_df_wattr['delta_space'] = trip_df_wattr.apply(lambda r: haversine.haversine((r['lat'], r['lng']),
                                                                                     (r['next_lat'], r['next_lng'])), axis=1)
    trip_df_wattr['next_timestamp'] = trip_df_wattr['timestamp'].shift(-1)
    trip_df_wattr['delta_time'] = trip_df_wattr['next_timestamp'] - trip_df_wattr['timestamp']
    trip_df_wattr['speed'] = trip_df_wattr['delta_space'] / trip_df_wattr['delta_time'] * 3600
    trip_df_wattr['is_stop'] = False
    
    trip_cols = ['trip_id', 'lat', 'lng', 'timestamp',  'delta_space', 'delta_time', 'speed', 'is_stop', 'pings_in_traj']
    
    trip_res = trip_df_wattr[trip_cols]
    
    if return_one_exec:
        return trip_res
    
    return trip_detection(trip_res, integrity_speed = integrity_speed, integrity_space = integrity_space, integrity_time = integrity_time, return_one_exec = True)

def cluster_stops(stop_df, method = 'radius', radius = 0.150):
    """
    Helper function to cluster stops.
    
    Args:
        stop_df (pandas.DataFrame): DataFrame containing stop data.
        method (str, optional): Method for clustering stops. Options are 'radius'. Defaults to 'radius'.
        radius (float, optional): Radius in kilometers to cluster stops. Defaults to 0.150.
    """
    
    stops = stop_df.copy(deep = True)
    
    if method == 'radius':
        points = stops.apply(lambda r: shapely.geometry.Point(r['mean_lng'], r['mean_lat']), axis=1)
        stops = geopandas.GeoDataFrame(stops, geometry = points, crs = 'EPSG:4326')
        
        stop_clusters = stops.to_crs(UNIVERSAL_CRS).buffer(radius).to_crs('EPSG:4326').reset_index(name = 'geometry')\
                                .dissolve().reset_index()[['geometry']].explode(index_parts = False).reset_index(drop = True)
        stops = stops.sjoin(stop_clusters, how = 'left', predicate = 'within').rename(columns = {'index_right' : 'cluster_id'})
        stops = stops.set_index('cluster_id').drop(['mean_lat', 'mean_lng'], axis = 1)\
                        .join(stops.groupby('cluster_id').agg({'mean_lat' : 'mean', 'mean_lng' : 'mean'})).reset_index()
                        
    return stops
    
def location_ranking(stop_df, timezone, start_window = '19:00', end_window = '07:00', method = 'most_frequent'):
    """
    Calculate the ranking of location based on stop data.
    The best approach would be to cluster the stops before ranking them.

    Args:
        stop_df (pandas.DataFrame): DataFrame containing stop data.
        timezone (str): Timezone of the stop data.
        start_window (str, optional): Start time of the window period. Defaults to '22:00'.
        end_window (str, optional): End time of the window period. Defaults to '07:00'.
        method (str, optional): Method for ranking locations. 
                                Options are 'most_frequent', 'most_certain', and 'longest'. Defaults to 'most_frequent'.

    Returns:
        pandas.DataFrame: DataFrame containing the ranking of locations based on the specified method.
            Columns:
                - ranking: Position in the ranking according to the criterion.
                - mean_lat: Latitude of the location.
                - mean_lng: Longitude of the location.
                - most_frequent: Number of unique stops in the location.
                - most_certain: Total number of pings in the location.
                - longest: Sum of stop durations in the location.
    """
    methods = ['most_frequent', 'most_certain', 'longest']
    
    stops = stop_df.dropna(subset = 'timestamp').sort_values(by='timestamp')
                     
    window_stops = stops['timestamp'].apply(lambda t: pandas.Timestamp(t, unit='s', tz=timezone))
    
    if len(window_stops) == 0:
        return pandas.DataFrame(columns = ['mean_lat', 'mean_lng', 'most_frequent', 'most_certain', 'longest'])
    
    window_traj_id = stops.set_index(pandas.DatetimeIndex(window_stops)).between_time(start_window, end_window)['traj_id'].values
        
    window_visits = stops[stops['traj_id'].isin(window_traj_id)]
    
    loc_ranking = window_visits .groupby(['mean_lat', 'mean_lng'])\
                                .agg(most_frequent  = ('traj_id', 'nunique'), 
                                     most_certain   = ('traj_id', 'count'), 
                                     longest        = ('delta_time', 'sum'))\
                                .sort_values(by = [method]+[x for x in methods if x != method], 
                                             ascending = False)\
                                .reset_index()

    return loc_ranking.reset_index(names = 'ranking')