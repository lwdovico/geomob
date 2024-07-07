# GeoMob

Welcome to **GeoMob**, a lightweight library designed to facilitate working with geospatial data. At the moment this library provides essential tools for preprocessing and tessellating geospatial mobility data, making it an excellent choice for developers and researchers in the field of geographic information systems (GIS). I am planning to add many more features. Stay tuned!

## Features

### Current functionalities

1. **Preprocess Module**
   - Convert input GeoJSON formats into Shapely Geometries.
   - A lightweight improved version of a stop detection algorithm (2004, Ramaswamy, H., Toyama, K.).
   - A custom made algorithm to detect clean user's trips.
   - A location function to rank users locations during a time window based on different criterions.
   - A function to compress (into an OD table) user's trips.

2. **Tessellate Module**
   - Module to discretize the space of a shape.
   - Support for many tessellation algorithms (e.g., Squared, S2, GeoHash, H3, Voronoi, etc.)
   - Function to simplify complex polygons into a complex of triangles
   - Efficient function to compute a random set of points in a polygon (useful for synthetic data generation, and for the randomized version of the Voronoi tessellation when points are not provided etc.)

3. **Routing Module**
   - It supports only iGraph at the moment
   - A simple python isochrone algorithm (not very efficient with large graphs)
   
4. **Retrieve Module**
   - A simple function to query OSM for geometries

5. **Plot Module**
   - At the moment it contains just a function to plot dataframes on KeplerGl

## Installation

To install GEOMOB, you can use pip:

```sh
pip install geomob
```

## Contributing

I welcome contributions from the community. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push your branch to your fork.
4. Submit a pull request to our main repository.

Please ensure all contributions adhere to our coding standards and include appropriate tests.

## License

GEOMOB is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

If you have any questions or need further assistance, feel free to open an issue on GitHub or contact me at lwdovico@protonmail.com

Happy Mapping!