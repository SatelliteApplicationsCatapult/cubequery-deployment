import numpy as np
import xarray as xr
import dask
from os import path

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.createindices import NDVI
from datacube_utilities.masking import mask_good_quality
from datacube_utilities.dc_mosaic import (
    create_median_mosaic,
    create_mean_mosaic,
    create_max_ndvi_mosaic,
)
from datacube_utilities.query import (
    create_base_query,
    create_product_measurement,
    is_dataset_empty,
)


class NDVIAnomaly(CubeQueryTask):
    """
    generate NDVI anomaly, showing changes in NDVI between two time periods. To show actual change two time period of
    similar seasonality should be used. The output is an NDVI anomaly and also a threshold product, the thresholds
    assocaited with the product can be adjusted depending on what you changes the user is interested in.

    Adapted from https://github.com/ceos-seo/data_cube_notebooks for us in Satellite Applications Catapult Common Sensing DataCube

    This notebook compares NDVI between two time periods to detect land change. In the case of deforestation,
    the NDVI values will reduce from (0.6 to 0.9 ... typical for forests) to lower values (<0.6). This change can be
    detected and used to investigate deforestation or monitor the extent of the land change.
    """

    display_name = "NDVI Anomaly"
    description = "NDVI anomaly, showing changes in NDVI between two time periods."

    parameters = [
        Parameter("aoi", "AOI", DType.WKT, "Area of interest"),
        Parameter(
            "output_projection",
            "Output Projection",
            DType.STRING,
            "Projection to generate the output in.",
        ),
        Parameter(
            "baseline_start_date",
            "Baseline Start Date",
            DType.DATE,
            "Start date of the period to use for the baseline",
        ),
        Parameter(
            "baseline_end_date",
            "Baseline End Date",
            DType.DATE,
            "End date of the period to use for the baseline",
        ),
        Parameter(
            "analysis_start_date",
            "Analysis Start Date",
            DType.DATE,
            "Start date of the period to use for the analysis",
        ),
        Parameter(
            "analysis_end_date",
            "Analysis End Date",
            DType.DATE,
            "End date of the period to use for the analysis",
        ),
        Parameter(
            "platform_base",
            "Baseline Satellite",
            DType.STRING,
            "Satellite to use for the baseline",
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "platform_analysis",
            "Analysis Satellite",
            DType.STRING,
            "Satellite to use for the analysis",
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "res",
            "resolution in meters",
            DType.INT,
            "Pixel resution in meters",
            [0, 500],
        ),
        Parameter("aoi_crs", "AIO CRS", DType.STRING, "CRS of the Area of Interest"),
        Parameter(
            "mosaic_type",
            "Mosaic Type",
            DType.STRING,
            "Mosaic type to use for the analysis",
            ["median", "mean", "max_ndvi"],
        ),
    ]

    CubeQueryTask.cal_significant_kwargs(parameters)

    def generate_product(
        self,
        dc,
        path_prefix,
        aoi,
        output_projection,
        baseline_start_date,
        baseline_end_date,
        analysis_start_date,
        analysis_end_date,
        platform_base,
        platform_analysis,
        res,
        aoi_crs,
        mosaic_type,
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=40, x=2000, y=2000)

        query = create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks)

        all_measurements = ["green", "red", "blue", "nir", "swir1", "swir2"]
        (
            baseline_product,
            baseline_measurement,
            baseline_water_product,
        ) = create_product_measurement(platform_base, all_measurements)
        (
            analysis_product,
            analysis_measurement,
            analysis_water_product,
        ) = create_product_measurement(platform_analysis, all_measurements)

        baseline_time_period = (baseline_start_date, baseline_end_date)
        analysis_time_period = (analysis_start_date, analysis_end_date)

        ## Create dask graph

        baseline_ds = dc.load(
            time=baseline_time_period,
            platform=platform_base,
            product=baseline_product,
            measurements=baseline_measurement,
            **query,
        )

        analysis_ds = dc.load(
            time=analysis_time_period,
            platform=platform_analysis,
            product=analysis_product,
            measurements=analysis_measurement,
            **query,
        )

        if is_dataset_empty(baseline_ds):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Baseline Dataset!"
            )

        if is_dataset_empty(analysis_ds):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Analysis Dataset!"
            )

        water_scenes_baseline = dc.load(
            product=baseline_water_product,
            measurements=["water_classification"],
            time=baseline_time_period,
            **query,
        )
        water_scenes_analysis = dc.load(
            product=analysis_water_product,
            measurements=["water_classification"],
            time=analysis_time_period,
            **query,
        )

        b_good_quality = mask_good_quality(baseline_ds, baseline_product)
        a_good_quality = mask_good_quality(analysis_ds, analysis_product)

        baseline_ds = baseline_ds.where(b_good_quality)
        analysis_ds = analysis_ds.where(a_good_quality)

        mosaic_function = {
            "median": create_median_mosaic,
            "mean": create_mean_mosaic,
            "max_ndvi": create_max_ndvi_mosaic,
        }

        new_compositor = mosaic_function[mosaic_type]

        if mosaic_type == "median" or mosaic_type == "mean":
            # the mean and medan functions work automatically with dask without using `dask.delayed`
            # because they exclusively use xarray functions which already support dask.
            # this gives us a ~20% time saving on small datasets
            baseline_composite = new_compositor(baseline_ds, clean_mask=b_good_quality)
            analysis_composite = new_compositor(analysis_ds, clean_mask=a_good_quality)
        else:
            baseline_composite = dask.delayed(new_compositor)(
                baseline_ds, clean_mask=b_good_quality
            )
            analysis_composite = dask.delayed(new_compositor)(
                analysis_ds, clean_mask=a_good_quality
            )

        water_classes_base = water_scenes_baseline.where(water_scenes_baseline >= 0)
        water_classes_analysis = water_scenes_analysis.where(water_scenes_analysis >= 0)

        water_composite_base = water_classes_base.water_classification.mean(dim="time")
        water_composite_analysis = water_classes_analysis.water_classification.mean(
            dim="time"
        )

        baseline_composite = baseline_composite.where(
            (baseline_composite != np.nan) & (water_composite_base == 0)
        )
        analysis_composite = analysis_composite.where(
            (analysis_composite != np.nan) & (water_composite_analysis == 0)
        )

        ndvi_baseline_composite = NDVI(baseline_composite)
        ndvi_analysis_composite = NDVI(analysis_composite)

        ndvi_anomaly = ndvi_analysis_composite - ndvi_baseline_composite

        ## Compute

        ndvi_anomaly = ndvi_anomaly.compute()

        ## Write file

        file_name = path.join(path_prefix, "ndvi_anomaly.tiff")
        ndvi_anomaly_export = xr.DataArray.to_dataset(
            ndvi_anomaly, dim=None, name="ndvi_anomaly"
        )
        import_export.export_xarray_to_geotiff(
            ndvi_anomaly_export,
            file_name,
            bands=["ndvi_anomaly"],
            crs=output_projection,
            x_coord="x",
            y_coord="y",
        )

        return [file_name]
