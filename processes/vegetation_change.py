import dask
import numpy as np
import xarray as xr
from os import path
import odc.algo

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.createindices import NDVI, EVI
from datacube_utilities.masking import mask_good_quality
from datacube_utilities.dc_fractional_coverage_classifier import frac_coverage_classify
from datacube_utilities.geomedian import geomedian
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


def createparametercomposite(indices, ds):
    """Calculate the chosen indicies.
    """
    if indices == "NDVI":
        parameter_composite = NDVI(ds)
    elif indices == "EVI":
        parameter_composite = EVI(ds)
    elif indices == "FC":
        parameter_composite_all = frac_coverage_classify(ds, no_data=np.nan)
        parameter_composite = parameter_composite_all.pv.where(
            np.logical_not(np.isnan(ds.red.values))
        )
    return parameter_composite


class VegetationChange(CubeQueryTask):
    """
    This task uses changes in NDVI, EVI or Fractional Cover to identify vegetation change.

    The algorithm identifies a "baseline" and "analysis" time period and then compares the spectral
    parameters in each of those time periods. Significant reductions in vegetation are coincident
    with land change. In some cases these changes could be deforestation.

    Users of this algorithm should not accept the accuracy of the results but should conduct ground
    validation testing to assess accuracy. In most cases, these algorithms can be used to identify
    clusters of pixels that have experienced change and allow targeted investigation of those areas
    by local or regional governments.
    """

    display_name = "Vegetation Change"
    description = "Vegetation Change derived from changes in NDVI, EVI, or Fractional Cover cover between two time periods."

    parameters = [
        Parameter("aoi", "AOI", DType.WKT, "Area of interest."),
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
            "Start date of the period to use for the baseline.",
        ),
        Parameter(
            "baseline_end_date",
            "Baseline End Date",
            DType.DATE,
            "End date of the period to use for the baseline.",
        ),
        Parameter(
            "analysis_start_date",
            "Analysis Start Date",
            DType.DATE,
            "Start date of the period to use for the analysis.",
        ),
        Parameter(
            "analysis_end_date",
            "Analysis End Date",
            DType.DATE,
            "End date of the period to use for the analysis.",
        ),
        Parameter(
            "platform_base",
            "Baseline Satellite",
            DType.STRING,
            "Satellite to use for the baseline.",
            ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "platform_analysis",
            "Analysis Satellite",
            DType.STRING,
            "Satellite to use for the analysis.",
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
        Parameter(
            "mosaic_type",
            "Mosaic Type",
            DType.STRING,
            "Mosaic type to use for the analysis.",
            ["max", "median", "mean", "geomedian"],
        ),
        Parameter(
            "indices",
            "Indices",
            DType.STRING,
            "Indices to use for the analysis.",
            ["EVI", "NDVI", "FC"],
        ),
        Parameter(
            "minC",
            "Indices Minimum Threshold",
            DType.FLOAT,
            "Typical Values: NDVI=-0.7, EVI=-1.75, FC=-70",
            [-100, 100],
        ),
        Parameter(
            "maxC",
            "Indices Maximum Threshold",
            DType.FLOAT,
            "Typical Values: NDVI=-0.2, EVI=-0.5, FC=-20",
            [-100, 100],
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
        indices,
        minC,
        maxC,
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=1, x=1000, y=1000)

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

        baseline_clean_mask = mask_good_quality(baseline_ds, baseline_product)
        analysis_clean_mask = mask_good_quality(analysis_ds, analysis_product)

        xx_data_b = baseline_ds[all_measurements]
        xx_data_a = analysis_ds[all_measurements]

        baseline_ds_masked = odc.algo.keep_good_only(
            xx_data_b, where=baseline_clean_mask
        )
        analysis_ds_masked = odc.algo.keep_good_only(
            xx_data_a, where=analysis_clean_mask
        )

        if mosaic_type == "geomedian":
            baseline_composite = geomedian(
                baseline_ds_masked, baseline_product, all_measurements
            )
            analysis_composite = geomedian(
                analysis_ds_masked, analysis_product, all_measurements
            )
        else:
            mosaic_function = {
                "median": create_median_mosaic,
                "max": create_max_ndvi_mosaic,
                "mean": create_mean_mosaic,
            }
            new_compositor = mosaic_function[mosaic_type]
            baseline_composite = dask.delayed(new_compositor)(
                baseline_ds_masked, clean_mask=baseline_clean_mask, no_data=np.nan
            )
            analysis_composite = dask.delayed(new_compositor)(
                analysis_ds_masked, clean_mask=analysis_clean_mask, no_data=np.nan
            )

        water_scenes_baseline = dc.load(
            product=baseline_water_product,
            measurements=["water_classification"],
            time=baseline_time_period,
            **query,
        )
        water_scenes_baseline = water_scenes_baseline.where(water_scenes_baseline >= 0)
        water_scenes_analysis = dc.load(
            product=analysis_water_product,
            measurements=["water_classification"],
            time=analysis_time_period,
            **query,
        )
        water_scenes_analysis = water_scenes_analysis.where(water_scenes_analysis >= 0)

        baseline_composite = baseline_composite.rename(
            {"y": "latitude", "x": "longitude"}
        )
        analysis_composite = analysis_composite.rename(
            {"y": "latitude", "x": "longitude"}
        )

        # Spectral Parameter

        parameter_baseline_composite = createparametercomposite(
            indices, baseline_composite
        )
        parameter_analysis_composite = createparametercomposite(
            indices, analysis_composite
        )

        # Generate water mask

        water_composite_base = dask.delayed(
            water_scenes_baseline.water_classification.mean(dim="time")
        )
        water_composite_analysis = dask.delayed(
            water_scenes_analysis.water_classification.mean(dim="time")
        )

        # Apply water mask

        vegetation_baseline = parameter_baseline_composite.where(
            water_composite_base.values <= 0.4
        ).where(parameter_baseline_composite != -9999)
        vegetation_analysis = parameter_analysis_composite.where(
            water_composite_analysis.values <= 0.4
        ).where(parameter_analysis_composite != -9999)

        parameter_anomaly = vegetation_analysis - vegetation_baseline

        ## Compute

        parameter_anomaly_output = parameter_anomaly.compute()

        ## Anomaly Threshold Product

        no_data_mask = np.isnan(parameter_anomaly_output)
        a = parameter_anomaly_output
        b = a.where((a < maxC) | (no_data_mask == True), 200)
        c = b.where((b > minC) | (no_data_mask == True), 300)
        d = c.where(((c >= maxC) | (c <= minC)) | (no_data_mask == True), 100)
        param_thres_output = xr.DataArray.to_dataset(d, dim=None, name="param_thres")

        ## Write files

        result = []

        file_name = path.join(path_prefix, "veg_change.tiff")
        import_export.export_xarray_to_geotiff(
            parameter_anomaly_output,
            file_name,
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "param_thres.tiff")
        import_export.export_xarray_to_geotiff(
            param_thres_output,
            file_name,
            bands=["param_thres"],
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        return result
