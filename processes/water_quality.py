import xarray as xr
import dask
from os import path

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.masking import mask_good_quality
from datacube_utilities.dc_water_quality import tsm
from datacube_utilities.query import (
    create_base_query,
    create_product_measurement,
    is_dataset_empty,
)


class WaterQuality(CubeQueryTask):
    """
    Task for calculation of Total Suspended Matter(TSM) in water bodies. Uses inputs of
    Landsat-4, Landsat-5, Landsat-7, Landsat-8 and Sentinel-2.

    TSM stands for "Total Suspended Matter" - also called TSS which stands for "Total Suspended
    Solids". It is the dry-weight of particles suspended (not dissolved) in a body of water. It is a
    proxy of water quality.

    This task is based from work by ceos-seo at the following link,
    https://github.com/ceos-seo/data_cube_notebooks/blob/master/TSM_Demo_Notebook.ipynb, where the
    TSM equation originates from.

    Academic papers include Lymburner, L., Botha, E., Hestir, E., Anstee, J., Sagar, S., Dekker, A.
    and Malthus, T., 2016. Landsat 8: providing continuity and increased precision for measuring
    multi-decadal time series of total suspended matter. Remote Sensing of Environment, 185,
    pp.108-118.

    This paper demonstrates continuity between the Landsat sensors for TSM assessment. Calibration
    for S2 use has not been carried out.
    """

    display_name = "Water Quality"
    description = "Indicates where areas of highest total suspended matter (a measure of water quality) are within a user defined area. "
    img_url = "https://arcgis01.satapps.org/portal//sharing/rest/content/items/a499849ccd1f4c7fb0403b4c719f9dc1/resources/Water%20quality.png?v=1601648787339"
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
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
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

        dask_chunks = dict(time=10, x=1000, y=1000)

        query = create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks)

        all_measurements = ["green", "red", "blue", "nir", "swir1", "swir2"]
        product, measurement, water_product = create_product_measurement(
            platform, all_measurements
        )

        time = (start_date, end_date)

        ## Create dask graph

        ds = dc.load(
            time=time,
            platform=platform,
            product=product,
            measurements=measurement,
            **query,
        )

        if is_dataset_empty(ds):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Baseline Dataset!"
            )

        water_scenes = dc.load(
            product=water_product,
            measurements=["water_classification"],
            time=time,
            **query,
        )

        # Set land to no_data
        water_dataset = water_scenes.where(water_scenes > 0)

        good_quality = mask_good_quality(ds, product)
        ds_clear = ds.where(good_quality)
        ds_clear_land = ds_clear.where(water_dataset.water_classification > 0)
        tsm_dataset = xr.map_blocks(tsm, ds_clear_land)

        mean_tsm = tsm_dataset.mean(dim=["time"])
        max_tsm = tsm_dataset.max(dim=["time"])
        min_tsm = tsm_dataset.min(dim=["time"])

        ## Compute

        mean_tsm, max_tsm, min_tsm = dask.compute(mean_tsm, max_tsm, min_tsm)

        ## Write files

        result = []

        file_name = path.join(path_prefix, "mean_tsm.tiff")
        import_export.export_xarray_to_geotiff(
            mean_tsm, file_name, crs=output_projection, x_coord="x", y_coord="y",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "min_tsm.tiff")
        import_export.export_xarray_to_geotiff(
            min_tsm, file_name, crs=output_projection, x_coord="x", y_coord="y",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "max_tsm.tiff")
        import_export.export_xarray_to_geotiff(
            max_tsm, file_name, crs=output_projection, x_coord="x", y_coord="y",
        )
        result.append(file_name)

        return result
