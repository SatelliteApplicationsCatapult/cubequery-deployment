import numpy as np
import xarray as xr
from os import path

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.dc_fractional_coverage_classifier import frac_coverage_classify
from datacube_utilities.geomedian import geomedian
from datacube_utilities.query import (
    create_base_query,
    create_product_measurement,
    is_dataset_empty,
)


class FractionalCover(CubeQueryTask):
    """
    This task provides a fractional cover product which can be used to represent land cover.

    The basis for this task below and the following explanation are from Digital Earth Australia
    http://geoscienceaustralia.github.io/digitalearthau/notebooks/02_DEA_datasets/Introduction_to_Fractional_Cover.html.
    What is Fractional Cover

    Fractional Cover represents the proportion of the land surface that is bare (BS), covered by
    photosynthetic vegetation (PV), or non-photosynethic vegetation (NPV).

    The Fractional Cover product was generated using the spectral unmixing algorithm developed by
    the Joint Remote Sensing Research Program (JRSRP) which used the spectral signature for each
    pixel to break it up into three fractions, based on field work that determined the spectral
    characteristics of these fractions. The fractions were retrieved by inverting multiple linear
    regression estimates and using synthetic endmembers in a constrained non-negative least squares
    unmixing model.

    The green (PV) fraction includes leaves and grass, the non-photosynthetic fraction (NPV)
    includes branches, dry grass and dead leaf litter, and the bare soil (BS) fraction includes bare
    soil or rock.  Fractional Cover Bands

    Bare Soil (bare ground, rock, disturbed) (BS): - Bare Ground (bare soil, rock) percentage;
    Digital Number 10000 = 100%

    Photosythetic Vegetation. (green grass, trees, etc.) (PV): - Photosynthetic Vegetation: Green
    Vegetation percentage;Digital Number 10000 = 100%

    Non-Photosythetic vegetation (litter, dead leaf and branches) (NPV): - Non-Photosynthetic
    Vegetation (litter, dead leaves andbranches) percentage; Digital Number 10000 = 100%

    Unmixing Error (UE): - Unmixing Error. The residual error, defined as the Euclidean Norm of the
    Residual Vector. High values express less confidence in the fractional components.

    This task builds on the single L8 image with cloud and water mask provided by Digital Earth
    Austrlia to provide a median product for a time range. This reduces the influence of cloud and
    variability associated with water identification.

    The product has also been developed to function for Landsat-7, Landsat-5, Landsat-4 and
    Sentinel-2.
    """

    display_name = "Fractional Cover"
    description = """
    Fractional cover: the proportion of land surface that is bare (BS), covered by photosynthetic
    vegetation (PV), or non-photosynthic vegetation (NPV).
    """

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
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
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
        res,
        aoi_crs,
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=10, x=600, y=600)

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
        water_scenes = water_scenes.where(water_scenes >= 0)

        water_composite_mean = water_scenes.water_classification.mean(dim="time")
        water_composite_mean = water_composite_mean.rename(
            {"x": "longitude", "y": "latitude"}
        )

        land_composite = geomedian(ds, product, all_measurements)
        land_composite.rename({"x": "longitude", "y": "latitude"})

        # Fractional Cover Classification

        frac_classes = xr.map_blocks(
            frac_coverage_classify, land_composite, kwargs={"no_data": np.nan}
        )

        # Mask to remove clounds, cloud shadow, and water.
        frac_cov_masked = frac_classes.where(
            (frac_classes != np.nan) & (water_composite_mean <= 0.4)
        )

        ## Compute

        fractional_cover_output = frac_cov_masked.compute()

        ## Write file

        file_name = path.join(path_prefix, "fractional_cover.tiff")
        ds = xr.DataArray.to_dataset(
            fractional_cover_output, dim=None, name="fractional_cover"
        )
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=["fractional_cover"],
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )

        return [file_name]
