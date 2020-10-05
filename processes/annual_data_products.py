from datetime import datetime
from os import path
from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.query import (
    create_base_query,
    create_product_measurement,
    is_dataset_empty,
)



class AnnualDataProducts(CubeQueryTask):
    """
        This process will just extract a product and return it.
    """

    display_name = "Annual Data Products"
    description = """
    The geomedian is a multi-dimensional median of surface reflectance for each of the spectral measurements or bands over a time period, it can be used to obtain a cloud-free image
    """
    img_url = "https://arcgis01.satapps.org/portal//sharing/rest/content/items/a499849ccd1f4c7fb0403b4c719f9dc1/resources/Geomedian.png?v=1601652955166"
    info_url = "https://arcgis01.satapps.org/portal/apps/sites/?fromEdit=true#/data/pages/data-cube"

    parameters = [
        Parameter("aoi", "Area Of Interest", DType.WKT, "Area of interest."),
        Parameter(
            "output_projection",
            "Output Projection",
            DType.STRING,
            "Projection to generate the output in.",
            ["EPSG:3460"]
        ),
        Parameter("year", "Year", DType.INT, "The year you are looking for.", [1970,2017]),
        Parameter(
            "platform",
            "Satellite",
            DType.STRING,
            "Satellite to use.",
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "product",
            "Product",
            DType.STRING,
            "Which product to select",
            ["geomedian"],
        ),
        Parameter(
            "res",
            "Resolution in meters",
            DType.INT,
            "Pixel resolution in meters.",
            [10, 500],
        ),
        Parameter("aoi_crs", "Area Of Interest CRS", DType.STRING, "CRS of the Area of Interest.", ["EPSG:4326"]),
    ]

    CubeQueryTask.cal_significant_kwargs(parameters)

    def generate_product(
        self,
        dc,
        path_prefix,
        aoi,
        output_projection,
        year,
        platform,
        product,
        res,
        aoi_crs,
        **kwargs,
    ):
  
        dask_chunks = dict(time=10, x=600, y=600)

        query = create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks)
        
        start_time = datetime.strptime(f"{year}-01-01", "%Y-%m-%d")
        end_time = datetime.strptime(f"{year}-12-31", "%Y-%m-%d")

        product_name = f"{map_satellite(platform)}_{product}_annual"

        data = dc.load(time=(start_time, end_time), product=product_name, **query)
        if is_dataset_empty(data):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Dataset!"
            )
        data = data.rename({"x": "longitude", "y": "latitude"})
        data = data.mean(dim="time") # Should be safe as there should only ever be one entry and we just need to get rid of the dim
        file_name = path.join(path_prefix, f"archive_{year}.tiff")
        
        import_export.export_xarray_to_geotiff(
            data,
            file_name,
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )

        return [file_name]


def map_satellite(platform):
    mapping = {
        "LANDSAT_4" : "ls4", 
        "LANDSAT_5" : "ls5", 
        "LANDSAT_7" : "ls7", 
        "LANDSAT_8" : "ls8"
    }

    return mapping[platform]
