from os import path

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.query import (
    create_base_query,
    create_product_measurement,
    is_dataset_empty,
)


class WaterPermanency(CubeQueryTask):
    """
    Task for calculation of water permanency.
    """

    display_name = "Water Permanency"
    description = "Represents how often an area is identified as being water over a period of time, to find regions of fluctuating water cover as a result of flooding or drying."
    img_url = "https://arcgis01.satapps.org/portal//sharing/rest/content/items/a499849ccd1f4c7fb0403b4c719f9dc1/resources/Water%20permanency.png?v=1601648787323"
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
        Parameter("start_date", "Start Date", DType.DATE, "Start date of the period."),
        Parameter("end_date", "End Date", DType.DATE, "End date of the period."),
        Parameter(
            "platform",
            "Satellite",
            DType.STRING,
            "Satellite to use.",
            ["SENTINEL_2", "LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
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
        start_date,
        end_date,
        platform,
        res,
        aoi_crs,
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=1, x=2000, y=2000)

        query = create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks)

        all_measurements = ["green", "red", "blue", "nir", "swir1", "swir2"]
        _product, _measurement, water_product = create_product_measurement(
            platform, all_measurements
        )

        time = (start_date, end_date)

        ## Create dask graph

        ds = dc.load(
            time=time,
            platform=platform,
            product=water_product,
            group_by="solar_day",
            measurements=["water"],
            **query,
        )

        if is_dataset_empty(ds):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Baseline Dataset!"
            )

        water = ds.where(ds != -9999)
        water_composite_mean = water.water.mean(dim="time")

        ## Compute

        water_composite_mean_output = water_composite_mean.compute()

        ## Write files

        file_name = path.join(path_prefix, "water.tiff")
        import_export.export_xarray_to_geotiff(
            water_composite_mean_output,
            file_name,
            bands=["water"],
            crs=output_projection,
            x_coord="x",
            y_coord="y",
        )

        return [file_name]
