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


class LandChange(CubeQueryTask):
    """
    This task uses changes in Fractional Cover to identify land change. The algorithm identifies
    a "baseline" and "analysis" time period and then compares the spectral parameters in each of
    those time periods.

    Fractional Cover represents the proportion of the land surface which is bare (BS), covered by
    photosynthetic vegetation (PV), or non-photosynthetic vegetation(NPV).

    The Fractional Cover product was generated using the spectral unmixing algorithm developed by
    the Joint Remote Sensing Research Program (JRSRP) which used the spectral signature for each
    pixel to break it up into three fractions, based on field work that determined the spectral
    characteristics of these fractions. The fractions were retrieved by inverting multiple linear
    regression estimates and using synthetic endmembers in a constrained non-negative least squares
    unmixing model.

    The green (PV) fraction includes leaves and grass, the non-photosynthetic fraction (NPV)
    includes branches, dry grass and dead leaf litter, and the bare soil (BS) fraction includes bare
    soil or rock.

    Changes in each fraction are conincident with land change.

    In some cases these changes could be deforestation. Users of this algorithm should not accept
    the accuracy of the results but should conduct ground validation testing to assess accuracy. In
    most cases, these algorithms can be used to identify clusters of pixels that have experienced
    change and allow targeted investigation of those areas by local or regional governments.

    This output of this task is a raster product for each of the fractional cover bands - where
    positive changes represents gain in that band, and negative change represents loss.
    """

    display_name = "Land Change"
    description = (
        "Land Change, showing changes in fractional cover between two time periods."
    )

    parameters = [
        Parameter("aoi", "AOI", DType.WKT, "Area of interest."),
        Parameter(
            "output_projection",
            "Output Projection",
            DType.STRING,
            "Projection to generate the output in.",
            ["EPSG:3460"]
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
        Parameter("aoi_crs", "AIO CRS", DType.STRING, "CRS of the Area of Interest.", ["EPSG:4326"]),
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
        **kwargs,
    ):

        ## Create datacube query

        dask_chunks = dict(time=10, x=500, y=500)

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
        water_scenes_baseline = water_scenes_baseline.where(water_scenes_baseline >= 0)
        water_scenes_analysis = dc.load(
            product=analysis_water_product,
            measurements=["water_classification"],
            time=analysis_time_period,
            **query,
        )
        water_scenes_analysis = water_scenes_analysis.where(water_scenes_analysis >= 0)

        baseline_composite = geomedian(baseline_ds, baseline_product, all_measurements)
        analysis_composite = geomedian(analysis_ds, analysis_product, all_measurements)

        water_classes_base = water_scenes_baseline.where(water_scenes_baseline >= 0)
        water_classes_analysis = water_scenes_analysis.where(water_scenes_analysis >= 0)

        water_composite_base = water_classes_base.water_classification.mean(dim="time")
        water_composite_analysis = water_classes_analysis.water_classification.mean(
            dim="time"
        )

        baseline_composite = baseline_composite.rename(
            {"y": "latitude", "x": "longitude"}
        )
        water_composite_base = water_composite_base.rename(
            {"y": "latitude", "x": "longitude"}
        )
        analysis_composite = analysis_composite.rename(
            {"y": "latitude", "x": "longitude"}
        )
        water_composite_analysis = water_composite_analysis.rename(
            {"y": "latitude", "x": "longitude"}
        )

        # Spectral Parameter Anomaly

        parameter_baseline_composite = xr.map_blocks(
            frac_coverage_classify, baseline_composite, kwargs={"no_data": np.nan}
        )
        parameter_analysis_composite = xr.map_blocks(
            frac_coverage_classify, analysis_composite, kwargs={"no_data": np.nan}
        )

        frac_cov_baseline = parameter_baseline_composite.where(
            (water_composite_base <= 0.4) & (parameter_baseline_composite != -9999)
        )

        frac_cov_analysis = parameter_analysis_composite.where(
            (water_composite_analysis <= 0.4) & (parameter_analysis_composite != -9999)
        )
        parameter_anomaly = frac_cov_analysis - frac_cov_baseline

        ## Compute

        parameter_anomaly_output = parameter_anomaly.compute()

        ## Export products

        bs_output = parameter_anomaly_output.bs
        pv_output = parameter_anomaly_output.pv
        npv_output = parameter_anomaly_output.npv

        ## Write files

        result = []

        file_name = path.join(path_prefix, "land_change.tiff")
        import_export.export_xarray_to_geotiff(
            parameter_anomaly_output,
            file_name,
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "bs_change.tiff")
        import_export.export_xarray_to_geotiff(
            bs_output,
            file_name,
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "pv_change.tiff")
        import_export.export_xarray_to_geotiff(
            pv_output,
            file_name,
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "npv_change.tiff")
        import_export.export_xarray_to_geotiff(
            npv_output,
            file_name,
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        return result
