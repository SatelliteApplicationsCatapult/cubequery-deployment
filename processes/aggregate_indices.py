import xarray as xr
import dask
from os import path



from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.masking import mask_good_quality
from datacube_utilities.createindices import NDVI, NDWI, EVI
from datacube_utilities.dc_mosaic import (
    create_max_ndvi_mosaic,
    create_median_mosaic,
    create_mean_mosaic,
    create_min_ndvi_mosaic,
)
from datacube_utilities.query import (
    create_base_query,
    create_product_measurement,
    is_dataset_empty,
)


class AggregateIndices(CubeQueryTask):
    """
    Task to allow generation of mosaics using Indices.

    Indicies include Normalise Difference Vegetation Index (NDVI), Enhanced Vegetation Index (EVI)
    and Normalised Difference Water Index (NDWI).
    """

    display_name = "Aggregate Indices"
    description = "Provides access to indices used by data cube products for further analysis. These include NDVI, NDWI, and EVI. "
    img_url = "https://arcgis01.satapps.org/portal//sharing/rest/content/items/a499849ccd1f4c7fb0403b4c719f9dc1/resources/Indices%20-%20composite.png?v=1601648787371"
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
        Parameter(
            "mosaic_type",
            "Mosaic Type",
            DType.STRING,
            "The cloud-free mosaic type.",
            ["min", "max", "mean", "median"],
        ),
        Parameter(
            "indices",
            "Indices type",
            DType.STRING,
            "The indices to calculate.",
            ["EVI", "NDVI", "NDWI"],
        ),
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
        mosaic_type,
        indices,
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=1, x=2000, y=2000)

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

        clean_mask = mask_good_quality(ds, product)

        # Perform mosaic

        mosaic_function = {
            "median": create_median_mosaic,
            "max": create_max_ndvi_mosaic,
            "mean": create_mean_mosaic,
            "min": create_min_ndvi_mosaic,
        }
        mosaic_compositor = mosaic_function[mosaic_type]
        mosaiced_composite = dask.delayed(mosaic_compositor)(ds, clean_mask=clean_mask)

        # Calculate Indices

        indices_function = {"NDVI": NDVI, "NDWI": NDWI, "EVI": EVI}
        indices_compositor = indices_function[indices]
        indices_composite = indices_compositor(mosaiced_composite)

        ## Compute

        indices_composite = indices_composite.compute()

        ## Write files

        file_name = path.join(path_prefix, "indices_composite.tiff")
        ds = xr.DataArray.to_dataset(indices_composite, dim=None, name=indices)
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=[indices],
            crs=output_projection,
            x_coord="x",
            y_coord="y",
        )

        return [file_name]
