# odc-loader

**Tools for constructing xarray objects from parsed metadata.**

`odc-loader` is a Python library that provides a flexible and efficient way to load geospatial raster data into `xarray` DataArrays and Datasets. It is designed to work with metadata that describes data sources, their locations, and their properties, allowing for on-demand loading, reprojection, and resampling. This library is a core component of the Open Data Cube ecosystem.

## Key Features

- **Metadata-driven loading**: Define what data you want, and `odc-loader` figures out how to load it.
- **Xarray integration**: Loads data directly into `xarray` objects, ready for analysis.
- **Reprojection and Resampling**: Handles coordinate reference system (CRS) transformations and resolution changes on the fly.
- **Dask for Parallelism**: Leverages Dask to parallelize data loading and processing for large datasets.
- **Extensible Driver System**: Supports different data formats and storage backends through a reader driver mechanism (e.g., GeoTIFF, Zarr).
- **Chunking Support**: Works with chunked data for efficient I/O and memory management.

## Installation

You can install `odc-loader` using pip:

```bash
pip install odc-loader
```

To include support for specific backends like AWS S3 or Zarr, you can install optional dependencies:

```bash
pip install odc-loader[botocore]  # For AWS S3 access
pip install odc-loader[zarr]      # For Zarr support
```

## Basic Usage (Conceptual)

The library is typically used by higher-level applications that parse user requests and data source metadata. A simplified conceptual workflow looks like this:

1.  **Define a Load Request**: Specify the measurements (bands), spatial extent, resolution, and CRS for the desired output.
2.  **Discover Data Sources**: Identify relevant data source files or objects based on the request.
3.  **Parse Metadata**: Extract metadata from these sources (e.g., file paths, band information, georeferencing).
4.  **Load Data**: Use `odc-loader` to construct an `xarray.Dataset` based on the request and parsed metadata.

```python
# Conceptual example:
# (Actual usage involves more setup with metadata and data sources)

from odc.loader import load
from odc.loader.types import ParsedLoadRequest, BandInfo, DataSource  # Fictional simplified types for illustration

# 1. Define data sources (simplified)
source1 = DataSource(uri="path/to/band1.tif", band_info=BandInfo(name="red"))
source2 = DataSource(uri="path/to/band2.tif", band_info=BandInfo(name="green"))

# 2. Define a load request (simplified)
request = ParsedLoadRequest(
    measurements=["red", "green"],
    sources=[source1, source2],
    # ... other parameters like crs, resolution, geopolygon
)

# 3. Load data
# dataset = load(request) # Actual API might differ
# print(dataset)
```

For more detailed examples, please refer to the full documentation and usage within Open Data Cube applications.

## Documentation

Full documentation, including API references and usage guides, is built with Sphinx.

**To build the documentation locally:**

1.  Ensure Sphinx and the theme are installed. If you have a development environment set up (e.g., by cloning the repository and installing `docs` dependencies from `pyproject.toml`):
    ```bash
    pip install .[docs]
    ```
2.  Navigate to the `docs` directory:
    ```bash
    cd docs
    ```
3.  Build the HTML documentation:
    ```bash
    make html
    ```
4.  Open `docs/_build/html/index.html` in your web browser.

A hosted version of the documentation will be available at [TODO: Add link to hosted documentation, e.g., ReadTheDocs or GitHub Pages].

## Contributing

Contributions are welcome! Please see the main Open Data Cube [contributing guidelines](https://opendatacube.readthedocs.io/en/latest/contribute.html) (if applicable, or add project-specific guidelines).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
