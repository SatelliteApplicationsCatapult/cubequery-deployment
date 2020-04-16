import xarray as xr
import utils
from os import path

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export


class WaterPermanency(CubeQueryTask):
    """
    Task for calculation of water permanency.
    """

    display_name = "Water Permanency"
    description = "Water Permanency."

    parameters = [
        Parameter("aoi", "AOI", DType.WKT, "Area of interest."),
        Parameter(
            "output_projection",
            "Output Projection",
            DType.STRING,
            "Projection to generate the output in.",
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
            "Pixel resution in meters.",
            [0, 500],
        ),
        Parameter("aoi_crs", "AIO CRS", DType.STRING, "CRS of the Area of Interest."),
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
        platform_analysis,
        res,
        aoi_crs,
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=1, x=2000, y=2000)

        query = utils.create_base_query(
            aoi, res, output_projection, aoi_crs, dask_chunks
        )

        all_measurements = ["green", "red", "blue", "nir", "swir1", "swir2"]
        _product, _measurement, water_product = utils.create_product_measurement(
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

        if utils.is_dataset_empty(ds):
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
        ds = xr.DataArray.to_dataset(
            water_composite_mean_output, dim=None, name="water"
        )
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=["water"],
            crs=output_projection,
            x_coord="x",
            y_coord="y",
        )

        return [file_name]
