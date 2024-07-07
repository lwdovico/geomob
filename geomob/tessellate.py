import geopandas
import shapely
import pandas
import numpy
import json

import pygeohash
import pytess
import s2cell
import h3

from .preprocess import gpd_fromlist, create_geometry

UNIVERSAL_CRS = 'EPSG:3857'

def sq_tessellate(base_shape, meters, project_on_crs = None, within = False):
    """
    Function to tessellate a base shape into square polygons.

    Parameters:
    - base_shape: geopandas.GeoDataFrame.
        The base shape to be tessellated.
    - meters: float
        The size of each square polygon in meters.

    Returns:
    - geopandas.GeoDataFrame
        The tessellated polygons as a GeoDataFrame.
    """
    
    if project_on_crs is None:
        project_on_crs = UNIVERSAL_CRS
        
    shape = base_shape.to_crs(project_on_crs).unary_union

    min_x, min_y, max_x, max_y = shape.bounds

    # Find number of square for each side
    x_squares = int(numpy.ceil(numpy.fabs(max_x - min_x) / meters))
    y_squares = int(numpy.ceil(numpy.fabs(min_y - max_y) / meters))

    # Placeholder for the polygon
    polygons = []
    
    for i in range(x_squares):
        x1, x2 = min_x + meters * i, min_x + meters * (i + 1)
        
        for j in range(y_squares):
            y1, y2 = min_y + meters * j, min_y + meters * (j + 1)
            polygon = shapely.geometry.Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            
            if within and polygon.within(shape):
                polygons.append({"geometry": polygon})
            elif not within and polygon.intersects(shape):
                polygons.append({"geometry": polygon})

    squared_tess = geopandas.GeoDataFrame(polygons, crs=project_on_crs).to_crs(base_shape.crs)
        
    return squared_tess

def h3_tessellate(base_shape, resolution, within = False):
    """
    Tessellates a base shape using H3 hexagons.

    Args:
        base_shape (geopandas.GeoDataFrame): The base shape to tessellate in 4326.
        resolution (int): The H3 resolution level.
        within (bool, optional): If True, only H3 hexagons fully contained within the base shape will be returned.
            If False, H3 hexagons intersecting with the base shape will be returned. Defaults to False.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the tessellated H3 hexagons.

    """
    
    shape = base_shape.unary_union
    
    if isinstance(shape, shapely.geometry.Polygon):
        shape = shapely.geometry.MultiPolygon([shape])
    
    h3_indexes = set()
    
    for x in shape.geoms:
        
        if not within:
            boundaries = x.boundary
            
            if isinstance(boundaries, shapely.geometry.LineString):
                boundaries = shapely.geometry.MultiLineString([boundaries])
                
            for sub_line in boundaries.geoms:
                for lon, lat in sub_line.coords:
                    h3_indexes.add(h3.geo_to_h3(lat, lon, resolution))
            
        h3_indexes = h3_indexes.union(h3.polyfill_geojson(json.loads(shapely.to_geojson(x)), resolution))
        
    if len(h3_indexes) == 0:
        return geopandas.GeoDataFrame(columns = ['geometry', 'h3-id'], crs = base_shape.crs)
    
    polygons = [{"geometry" : shapely.geometry.Polygon(h3.h3_to_geo_boundary(h3_index, geo_json=True)),
                 'h3-id' : 'h3_'+str(h3_index)} for h3_index in sorted(h3_indexes)]
    
    return geopandas.GeoDataFrame(polygons, crs=base_shape.crs)

def tri_tessellate(base_shape, project_on_crs = None):
    """
    Tessellates a base shape into triangles. 
    Triangles may be used to process irregular polygons using properties of regular polygons.

    Parameters:
    - base_shape: geopandas.GeoDataFrame.
        The base shape to be tessellated.

    Returns:
    - geopandas.GeoDataFrame
        The tessellated triangles as a GeoDataFrame.
    """
    if project_on_crs is None:
        project_on_crs = UNIVERSAL_CRS
        
    shape = base_shape.to_crs(project_on_crs).unary_union
    triangles = shapely.ops.triangulate(shape)
    triangles_gdf = gpd_fromlist(triangles, crs = project_on_crs)
    
    centroid_gdf = geopandas.GeoDataFrame(geometry = triangles_gdf.centroid, crs = project_on_crs)\
                            .sjoin(       gpd_fromlist([shape], crs = project_on_crs), 
                                          how = 'inner', predicate = 'within'
                                    )
    
    triangles_within = triangles_gdf.loc[centroid_gdf.index]
    triangles_within['area'] = triangles_within.geometry.area
    
    return triangles_within.sort_values('area', ascending = False).reset_index(drop = True).to_crs(base_shape.crs)

def _random_point_in_triangle(triangle):
    """
    Generates a random point within a triangle's vertices.

    Parameters:
    - triangle: shapely.geometry.Polygon
        The triangle to generate a random point in.

    Returns:
    - shapely.geometry.Point
        The randomly generated point within the triangle.
    """
    
    v1, v2, v3, _ = numpy.array(list(zip(*triangle.exterior.coords.xy)))
    r1, r2 = numpy.random.random(), numpy.random.random()
    
    pt = v1 * (1.0 - (r1**0.5)) + v2 * (1.0 - r2) * (r1**0.5) + v3 * r2 * (r1**0.5)
    
    return shapely.geometry.Point(pt)

def random_points_in_polygon(base_shape, n_points):
    """
    Generates a set of random points inside a base shape.

    Parameters:
        base_shape (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The base shape to tessellate.
        n_points (int): the number of points to generate.

    Returns:
    - geopandas.GeoDataFrame
        The GeoDataFrame containing n_points random points within the base shape.
    """
    
    triangles = tri_tessellate(base_shape)
    triangles['probability'] = (triangles['area'] / triangles['area'].sum())
    triangles_selected = numpy.random.choice(triangles['geometry'].values, size = n_points, p = triangles['probability'])
    
    return gpd_fromlist([_random_point_in_triangle(triangle) for triangle in triangles_selected], crs = base_shape.crs)

def vor_tessellate(base_shape, points):
    """
    Perform Voronoi tessellation on a base shape using a set of points.
    
    Args:
        base_shape (geopandas.GeoDataFrame): The base shape to tessellate.
        points (geopandas.GeoDataFrame or int or list or numpy.ndarray): The points to use for tessellation.
            If a GeoDataFrame, it should be in the same coordinate reference system (CRS) as the base shape.
            If an integer, it represents the number of random points to generate within the base shape.
            If a list or numpy.ndarray, it should contain either shapely.geometry.Point objects or pairs of floats
            representing the coordinates of the points.
    
    Returns:
        geopandas.GeoDataFrame: The resulting Voronoi tessellation as a GeoDataFrame.
    """
    
    shape = base_shape.unary_union
    
    if isinstance(points, geopandas.GeoDataFrame):
        points = points.to_crs(base_shape.crs)
    
    elif isinstance(points, int):
        points = random_points_in_polygon(base_shape, points)
    
    elif isinstance(points, list) or isinstance(points, numpy.ndarray):
        if all(isinstance(item, shapely.geometry.Point) for item in points):
            points = gpd_fromlist(points, crs = base_shape.crs)
            
        elif all(len(item) == 2 for item in points) and \
             all(all([isinstance(num, float) for num in item]) for item in points):
                 
            points = gpd_fromlist([shapely.geometry.Point(item) for item in points], crs = base_shape.crs)
    
    earth_boundaries = gpd_fromlist([shapely.geometry.Point(-18000, -9000), 
                                     shapely.geometry.Point(-18000,  9000), 
                                     shapely.geometry.Point( 18000,  9000), 
                                     shapely.geometry.Point( 18000, -9000)], crs = points.crs)
    
    earth_boundaries.index = ['earth_boundaries'] * 4
    
    points = pandas.concat([points, earth_boundaries])
    
    vor = pytess.voronoi(points.geometry.apply(lambda x: (x.x, x.y)).values)
    poly_vor = gpd_fromlist([shapely.geometry.Polygon(polygon) for _, polygon in vor if len(polygon) > 2])
    
    points['point_geometry'] = points.geometry
    
    vor_gdf = poly_vor  .sjoin(points.drop('earth_boundaries'), 
                            how = 'left', 
                            predicate = 'contains')\
                        .rename(columns = {'index_right': 'pt-id'})\
                        .dropna()\
                        .clip(shape)\
                        .reset_index(drop = True)
                        
    vor_gdf['pt-id'] = vor_gdf['pt-id'].astype(int)
    vor_gdf['point_geometry'] = vor_gdf['point_geometry'].apply(lambda x: x.wkt)
    
    return vor_gdf

def _get_s2_cell_corners(s2_id):
    """
    Returns the corners of an S2 cell.
    
    Args:
        s2_id (int): The S2 cell identifier.
        
    Returns:
        numpy.ndarray: An array of tuples containing the corners of the S2 cell.
    """
    
    face, i, j = s2cell.s2_cell_id_to_face_ij(s2_id)
    level = s2cell.cell_id_to_level(s2_id)
    size = 1 << (30 - level)
    
    face_to_lonlat = lambda f, i, j, l : s2cell.cell_id_to_lat_lon(
                                         s2cell.s2cell._s2_face_ij_to_wrapped_cell_id(f, i, j, l))
    
    get_offset = lambda s : [ (s, -s), (s, s), (-s, s), (-s, -s) ]
    
    corners_theoric = numpy.array([tuple(reversed(face_to_lonlat(face, x + i, y + j, level))) for x, y in get_offset(size)])
    corners = numpy.array([tuple(reversed(face_to_lonlat(face, x + i, y + j, level))) for x, y in get_offset(size // 2)])
    
    corners_adjusted = corners - corners.mean(axis = 0) + corners_theoric.mean(axis = 0)
    
    return corners_adjusted

def _s2_polyfill(geojson, level, within = False):
    """
    Polyfills a GeoJSON geometry with S2 cells.
    
    Args:
        geojson (dict): The GeoJSON geometry to polyfill.
        level (int): The S2 level (1 - 30).
        within (bool, optional): If True, only S2 cells fully contained within the base shape will be returned.
            If False, S2 cells intersecting with the base shape will be returned. Defaults to False.
            
    Returns:
        set: A set containing the S2 indexes of the polyfilled geometry.
    """
    
    shape = create_geometry(geojson)
    lon, lat = shape.centroid.coords[0]
    s2_indexes = {s2cell.lat_lon_to_cell_id(lat, lon, level)}
    
    to_check = set(s2_indexes)
    checked = set()

    while to_check:
        current = to_check.pop()
        checked.add(current)

        neighbors = s2cell.cell_id_to_neighbor_cell_ids(current)
        for neighbor in neighbors:
            if neighbor not in checked and neighbor not in to_check:
                neighbor_shape = shapely.geometry.Polygon(_get_s2_cell_corners(neighbor))
            
                joins_with = neighbor_shape.within(shape) if within else neighbor_shape.intersects(shape)
        
                if joins_with:
                    s2_indexes.add(neighbor)
                    to_check.add(neighbor)
                    
    return s2_indexes

def s2_tessellate(base_shape, level, within = False):
    """
    Tessellates a base shape using S2 cells.

    Args:
        base_shape (geopandas.GeoDataFrame): The base shape to tessellate in 4326.
        level (int): The S2 level (1 - 30).
        within (bool, optional): If True, only S2 cells fully contained within the base shape will be returned.
            If False, S2 cells intersecting with the base shape will be returned. Defaults to False.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the tessellated S2 cells.

    """
    
    shape = base_shape.unary_union
    
    if isinstance(shape, shapely.geometry.Polygon):
        shape = shapely.geometry.MultiPolygon([shape])
    
    s2_indexes = set()
    
    for x in shape.geoms:
        for s2_cell in set(_s2_polyfill(json.loads(shapely.to_geojson(x)), level, within)):
            s2_indexes.add(s2_cell)
        
    if len(s2_indexes) == 0:
        return geopandas.GeoDataFrame(columns = ['geometry', 's2-id'], crs = base_shape.crs)
    
    polygons = [{"geometry" : shapely.geometry.Polygon(_get_s2_cell_corners(s2_index)),
                 's2-id' : 's2_'+str(s2cell.cell_id_to_token(s2_index))} for s2_index in sorted(s2_indexes)]
    
    return geopandas.GeoDataFrame(polygons, crs=base_shape.crs)

def _get_geohash_corners(geohash):
    """ Get the corners of a geohash."""
    # Decode the geohash to get the bounding box
    lat, lon, lat_err, lon_err = pygeohash.decode_exactly(geohash)

    # Calculate min and max latitudes and longitudes
    min_lat = lat - lat_err
    max_lat = lat + lat_err
    min_lon = lon - lon_err
    max_lon = lon + lon_err

    # Define the corners
    corners = [(min_lon, min_lat),
               (max_lon, min_lat), 
               (max_lon, max_lat),
               (min_lon, max_lat)]

    return corners

def _get_adjacent_geohashes(geohash, direct):
        
        _base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

        neighboring_hashes = { "right"  : { "even" : "bc01fg45238967deuvhjyznpkmstqrwx",
                                            "odd"  : "p0r21436x8zb9dcf5h7kjnmqesgutwvy" },
                               "left"   : { "even" : "238967debc01fg45kmstqrwxuvhjyznp",
                                            "odd"  : "14365h7k9dcfesgujnmqp0r2twvyx8zb" },
                               "top"    : { "even" : "p0r21436x8zb9dcf5h7kjnmqesgutwvy",
                                            "odd"  : "bc01fg45238967deuvhjyznpkmstqrwx" },
                               "bottom" : { "even" : "14365h7k9dcfesgujnmqp0r2twvyx8zb",
                                            "odd"  : "238967debc01fg45kmstqrwxuvhjyznp" } }

        # Used change of parent tile
        borders   = { "right"  : {  "even" : "bcfguvyz", "odd"  : "prxz" },
                      "left"   : {  "even" : "0145hjnp", "odd"  : "028b"  },
                      "top"    : {  "even" : "prxz",     "odd"  : "bcfguvyz" },
                      "bottom" : {  "even" : "028b",     "odd"  : "0145hjnp" } }
        
        if (len(geohash) == 0):
            raise ValueError("The geohash length cannot be 0. Possible when close to poles")
        srcHash = geohash.lower()
        lastChr = srcHash[-1]
        base = srcHash[:-1]

        splitDirection = ['even', 'odd'][len(srcHash)%2]

        if lastChr in borders[direct][splitDirection]:
            base = _get_adjacent_geohashes(base, direct)

        return base + _base32[neighboring_hashes[direct][splitDirection].index(lastChr)]
    
def _get_cornering_geohashes(geohash):
    """Get the geohashes that share a corner with the input geohash."""
    
    # Get the corners of the geohash
    directions = ['right', 'left', 'top', 'bottom']
    cornering_geohashes = {_get_adjacent_geohashes(geohash, direct) for direct in directions}
    
    return cornering_geohashes

def _gh_polyfill(geojson, precision, within = False):
    """Get a list of geohashes within a polygon."""
    
    shape = create_geometry(geojson)
    
    lon, lat = shape.centroid.coords[0]
    gh_indexes = {pygeohash.encode(lat, lon, precision)}
    
    to_check = set(gh_indexes)
    checked = set()

    while to_check:
        current = to_check.pop()
        checked.add(current)

        neighbors = _get_cornering_geohashes(current)
        for neighbor in neighbors:
            if neighbor not in checked and neighbor not in to_check:
                neighbor_shape = shapely.geometry.Polygon(_get_geohash_corners(neighbor))
            
                joins_with = neighbor_shape.within(shape) if within else neighbor_shape.intersects(shape)
        
                if joins_with:
                    gh_indexes.add(neighbor)
                    to_check.add(neighbor)
                    
    return list(gh_indexes)

def gh_tessellate(base_shape, precision, within = False):
    """
    Tessellates a base shape using geohashes.

    Args:
        base_shape (geopandas.GeoDataFrame): The base shape to tessellate in 4326.
        precision (int): The geohash precision (avoid numbers beyond 8, they require too much time).
        within (bool, optional): If True, only geohash cells fully contained within the base shape will be returned.
            If False, geohash cells intersecting with the base shape will be returned. Defaults to False.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the tessellated S2 cells.

    """
    
    shape = base_shape.unary_union
    
    if isinstance(shape, shapely.geometry.Polygon):
        shape = shapely.geometry.MultiPolygon([shape])
    
    gh_indexes = set()
    
    for x in shape.geoms:
        for gh_cell in _gh_polyfill(json.loads(shapely.to_geojson(x)), precision, within):
            gh_indexes.add(gh_cell)
        
    if len(gh_indexes) == 0:
        return geopandas.GeoDataFrame(columns = ['geometry', 'gh-id'], crs = base_shape.crs)
    
    polygons = [{"geometry" : shapely.geometry.Polygon(_get_geohash_corners(gh_index)),
                 'gh-id' : 'gh_'+str(gh_index)} for gh_index in sorted(gh_indexes)]
    
    return geopandas.GeoDataFrame(polygons, crs=base_shape.crs)