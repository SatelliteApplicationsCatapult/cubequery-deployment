import numpy as np
import xarray as xr
import utils
from os import path

from cubequery.tasks import CubeQueryTask, Parameter, DType
from datacube_utilities import import_export
from datacube_utilities.masking import mask_good_quality
from datacube_utilities.dc_fractional_coverage_classifier import frac_coverage_classify

import odc.algo
from odc.algo import to_f32, from_float, xr_geomedian


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
            ["SENTINEL_2", "LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "platform_analysis",
            "Analysis Satellite",
            DType.STRING,
            "Satellite to use for the analysis",
            ["SENTINEL_2", "LANDSAT_4", "LANDSAT_5", "LANDSAT_7", "LANDSAT_8"],
        ),
        Parameter(
            "res",
            "resolution in meters",
            DType.INT,
            "Pixel resution in meters",
            [0, 500],
        ),
        Parameter("aoi_crs", "AIO CRS", DType.STRING, "CRS of the Area of Interest"),
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

        query = utils.create_base_query(
            aoi, res, output_projection, aoi_crs, dask_chunks
        )

        all_measurements = ["green", "red", "blue", "nir", "swir1", "swir2"]
        baseline_product, baseline_measurement, baseline_water_product = utils.create_product_measurement(
            platform_base, all_measurements
        )
        analysis_product, analysis_measurement, analysis_water_product = utils.create_product_measurement(
            platform_analysis, all_measurements
        )

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

        if utils.is_dataset_empty(baseline_ds):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Baseline Dataset!"
            )

        if utils.is_dataset_empty(analysis_ds):
            raise Exception(
                "DataCube Load returned an empty Dataset."
                + "Please check load parameters for Analysis Dataset!"
            )

        if platform_base in ["LANDSAT_8", "LANDSAT_7", "LANDSAT_5", "LANDSAT_4"]:
            water_scenes_baseline = dc.load(
                product=baseline_water_product,
                measurements=["water_classification"],
                time=baseline_time_period,
                **query,
            )
            water_scenes_baseline = water_scenes_baseline.where(
                water_scenes_baseline >= 0
            )
            water_scenes_analysis = dc.load(
                product=analysis_water_product,
                measurements=["water_classification"],
                time=analysis_time_period,
                **query,
            )
            water_scenes_analysis = water_scenes_analysis.where(
                water_scenes_analysis >= 0
            )
        else:
            raise Exception("S2 does not yet have daskable water classification")

        baseline_clean_mask = mask_good_quality(baseline_ds, baseline_product)
        analysis_clean_mask = mask_good_quality(analysis_ds, analysis_product)

        xx_data_b = baseline_ds[all_measurements]
        xx_data_a = analysis_ds[all_measurements]

        xx_clean_b = odc.algo.keep_good_only(xx_data_b, where=baseline_clean_mask)
        xx_clean_a = odc.algo.keep_good_only(xx_data_a, where=analysis_clean_mask)

        scale, offset = (
            1 / 10_000,
            0,
        )  # differs per product, aim for 0-1 values in float32

        xx_clean_b_32 = to_f32(xx_clean_b, scale=scale, offset=offset)
        yy_b = xr_geomedian(
            xx_clean_b_32,
            num_threads=1,  # disable internal threading, dask will run several concurrently
            eps=0.2 * scale,  # 1/5 pixel value resolution
            nocheck=True,
        )  # disable some checks inside geomedian library that use too much ram

        baseline_composite = from_float(
            yy_b, dtype="int16", nodata=-9999, scale=1 / scale, offset=-offset / scale
        )

        xx_clean_a_32 = to_f32(xx_clean_a, scale=scale, offset=offset)
        yy_a = xr_geomedian(
            xx_clean_a_32,
            num_threads=1,  # disable internal threading, dask will run several concurrently
            eps=0.2 * scale,  # 1/5 pixel value resolution
            nocheck=True,
        )  # disable some checks inside geomedian library that use too much ram

        analysis_composite = from_float(
            yy_a, dtype="int16", nodata=-9999, scale=1 / scale, offset=-offset / scale
        )

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
        ds = xr.DataArray.to_dataset(
            parameter_anomaly_output, dim=None, name="land_change"
        )
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=["land_change"],
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "bs_change.tiff")
        ds = xr.DataArray.to_dataset(bs_output, dim=None, name="bs_change")
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=["bs_change"],
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "pv_change.tiff")
        ds = xr.DataArray.to_dataset(pv_output, dim=None, name="bs_change")
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=["pv_change"],
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        file_name = path.join(path_prefix, "npv_change.tiff")
        ds = xr.DataArray.to_dataset(npv_output, dim=None, name="npv_change")
        import_export.export_xarray_to_geotiff(
            ds,
            file_name,
            bands=["npv_change"],
            crs=output_projection,
            x_coord="longitude",
            y_coord="latitude",
        )
        result.append(file_name)

        return result
