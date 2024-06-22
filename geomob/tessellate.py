import geopandas
import shapely
import pandas
import numpy
import json

import pytess
import h3

from .preprocess import gpd_fromlist

UNIVERSAL_CRS = 'EPSG:3857'

def sq_tessellate(base_shape, meters, project_on_crs = None):
    """
    Function to tessellate a base shape into square polygons.

    Parameters:
    - base_shape: geopandas.GeoDataFrame or shapely.geometry.Polygon
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
            
            if shape.intersects(polygon):
                polygons.append({"geometry": polygon})

    return geopandas.GeoDataFrame(polygons, crs=project_on_crs).to_crs(base_shape.crs)

def h3_tessellate(base_shape, resolution, within = False):
    """
    Tessellates a base shape using H3 hexagons.

    Args:
        base_shape (shapely.geometry.Polygon or shapely.geometry.MultiPolygon): The base shape to tessellate.
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
            for lon, lat in x.boundary.coords:
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
    - base_shape: geopandas.GeoDataFrame or shapely.geometry.Polygon
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

def random_point_in_triangle(triangle):
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
    
    return gpd_fromlist([random_point_in_triangle(triangle) for triangle in triangles_selected], crs = base_shape.crs)

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