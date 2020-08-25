
from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export



class ArchiveAccess(CubeQueryTask):
    """
        This process will just extract a product and return it.
    """

    display_name = "Archive"
    description = """
    Archive: Access some of the routine products from the datacube.
    """

    parameters = [
        Parameter("aoi", "AOI", DType.WKT, "Area of interest."),
        Parameter(
            "output_projection",
            "Output Projection",
            DType.STRING,
            "Projection to generate the output in.",
            ["EPSG:3460"]
        ),
        Parameter("year", "Year", DType.STRIGN, "The year you are looking for."),
        Parameter(
            "platform",
            "Satellite",
            DType.STRING,
            "Satellite to use.",
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "res",
            "Resolution in meters",
            DType.INT,
            "Pixel resution in meters.",
            [0, 500],
        ),
        Parameter("aoi_crs", "AIO CRS", DType.STRING, "CRS of the Area of Interest.", ["EPSG:4326"]),
    ]

    CubeQueryTask.cal_significant_kwargs(parameters)

    def generate_product(
        self,
        dc,
        path_prefix,
        aoi,
        output_projection,
        year,
        res,
        aoi_crs,
        **kwargs,
    ):
  
    dask_chunks = dict(time=10, x=600, y=600)

    query = create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks)
    
    start_time = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
    end_time = datetime.strptime(f"{year}-12-31", "%Y-%m-%d")

    data = dc.load(time=(start_time, end_time), product="ls8_geomedian_annual", **query)

    data = data.rename({"x": "longitude", "y": "latitude"})
    data = data.mean(dim="time") # Should be safe as there should only ever be one entry and we just need to get rid of the dim
    file_name = path.join(path_prefix, f"archive-{year}.tiff")

    import_export.export_xarray_to_geotiff(
        data,
        file_name,
        crs=output_projection,
        x_coord="longitude",
        y_coord="latitude",
    )

    return [file_name]
