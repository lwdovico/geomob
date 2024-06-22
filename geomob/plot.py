import keplergl

def plot_gdf(gdf, name = 'plot_gdf'): 
    return keplergl.KeplerGl(height = 800, data = {name: gdf})