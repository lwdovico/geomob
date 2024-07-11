import geopandas
import shapely.geometry
from .preprocess import gpd_fromlist, UNIVERSAL_CRS

def get_isochrones(points, max_weight, igraph_graph, graph_weights, concavity = 0.7):
    points = points
    max_weight = max_weight
    igraph_graph = igraph_graph
    graph_weights = graph_weights
    concavity = concavity

    nodes = gpd_fromlist(igraph_graph.vs['geometry'])    
    nearest_nodes = geopandas.sjoin_nearest(points.to_crs(UNIVERSAL_CRS), 
                                            nodes .to_crs(UNIVERSAL_CRS)).to_dict()['index_right']
        
    isochrones = []
    
    for pti, nni in nearest_nodes.items():
        path_tree = igraph_graph.get_shortest_paths(nni, 
                                                    weights = graph_weights, 
                                                    mode='out', 
                                                    output='epath')

        path_tree_weights = [(i, sum(igraph_graph.es[path][graph_weights])) for i, path in enumerate(path_tree)]
        edges = {edge for i, weight in path_tree_weights for edge in path_tree[i] if weight <= max_weight}

        isochrone_edges = shapely.geometry.MultiLineString(igraph_graph.es[list(edges)]['geometry'])
        
        isochrones.append({'point_index': pti, 
                            'point_geometry' : points.loc[pti, 'geometry'], 
                            'nearest_node_index' : nni, 
                            'node_geometry' : nodes.loc[nni, 'geometry'], 
                            'edge_geometry': isochrone_edges,
                            'geometry' : shapely.concave_hull(isochrone_edges, ratio = 1 - concavity)})
                
    return geopandas.GeoDataFrame(isochrones, crs='EPSG:4326', geometry='geometry')
                