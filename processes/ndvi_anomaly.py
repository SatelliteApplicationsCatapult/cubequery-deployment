import numpy as np
import xarray as xr
import re

from cubequery.tasks import CubeQueryTask, Parameter, DType
from utils_dcal.data_cube_utilities.dc_utilities import write_geotiff_from_xr
from utils_dcal.data_cube_utilities.dc_water_classifier import wofs_classify
from sac_utils.clean_mask import landsat_qa_clean_mask
from sac_utils.createAOI import create_lat_lon
from sac_utils.dc_mosaic import create_median_mosaic


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
        Parameter("aoi", DType.WKT, "Area of interest"),
        Parameter("projection", DType.STRING, "projection to generate the output in."),
        Parameter("baseline_start_date", DType.DATE, "Start date of the period to use for the baseline"),
        Parameter("baseline_end_date", DType.DATE, "End date of the period to use for the baseline"),
        Parameter("analysis_start_date", DType.DATE, "Start date of the period to use for the analysis"),
        Parameter("analysis_end_date", DType.DATE, "End date of the period to use for the analysis"),
        Parameter("platform_base", DType.STRING, "Satellite to use for the baseline"),
        Parameter("platform_analysis", DType.STRING, "Satellite to use for the analysis"),
        Parameter("res", DType.INT, "Pixel resution in meters"),
    ]

    CubeQueryTask.cal_significant_kwargs(parameters)

    def generate_product(self,
                         dc,
                         aoi,
                         projection,
                         baseline_start_date,
                         baseline_end_date,
                         analysis_start_date,
                         analysis_end_date,
                         platform_base,
                         platform_analysis,
                         res,
                         **kwargs
                         ):
        lat_extents, lon_extents = create_lat_lon(aoi)

        all_measurements = ["green", "red", "blue", "nir", "swir1", "swir2"]
        baseline_product, baseline_measurement = create_product_measurement(platform_base, all_measurements)
        analysis_product, analysis_measurement = create_product_measurement(platform_analysis, all_measurements)

        baseline_time_period = (baseline_start_date, baseline_end_date)
        analysis_time_period = (analysis_start_date, analysis_end_date)

        query = {
            'y': lat_extents,
            'x': lon_extents,
            'output_crs': projection,
            'resolution': res
        }

        baseline_ds = dc.load(
            time=baseline_time_period,
            platform=platform_base,
            product=baseline_product,
            measurements=baseline_measurement,
            **query
        )

        analysis_ds = dc.load(
            time=analysis_time_period,
            platform=platform_analysis,
            product=analysis_product,
            measurements=analysis_measurement,
            **query
        )

        if is_dataset_empty(baseline_ds):
            raise Exception("DataCube Load returned an empty Dataset." +
                            "Please check load parameters for Baseline Dataset!")

        if is_dataset_empty(analysis_ds):
            raise Exception("DataCube Load returned an empty Dataset." +
                            "Please check load parameters for Analysis Dataset!")

        baseline_clean_mask = landsat_qa_clean_mask(baseline_ds, platform_base)
        analysis_clean_mask = landsat_qa_clean_mask(analysis_ds, platform_analysis)

        baseline_ds = baseline_ds.where(baseline_clean_mask)
        analysis_ds = analysis_ds.where(analysis_clean_mask)

        baseline_composite = create_median_mosaic(baseline_ds, clean_mask=baseline_clean_mask)
        analysis_composite = create_median_mosaic(analysis_ds, clean_mask=analysis_clean_mask)

        water_class_base = wofs_classify(baseline_composite, mosaic=True, x_coord='x', y_coord='y').wofs
        baseline_composite = baseline_composite.copy(deep=True).where(
            (baseline_composite != np.nan) & (water_class_base == 0))
        water_class_analy = wofs_classify(analysis_composite, mosaic=True, x_coord='x', y_coord='y').wofs
        analysis_composite = analysis_composite.copy(deep=True).where(
            (analysis_composite != np.nan) & (water_class_analy == 0))

        ndvi_baseline_composite = NDVI(baseline_composite)
        ndvi_analysis_composite = NDVI(analysis_composite)

        ndvi_baseline_export = xr.DataArray.to_dataset(ndvi_baseline_composite, dim=None, name='ndvi_baseline')
        write_geotiff_from_xr('ndvi_baseline.tiff', ndvi_baseline_export, ["ndvi_baseline"], crs=projection,
                              x_coord='x', y_coord='y')

        ndvi_analysis_export = xr.DataArray.to_dataset(ndvi_analysis_composite, dim=None, name='ndvi_analysis')
        write_geotiff_from_xr('ndvi_analysis.tiff', ndvi_analysis_export, ["ndvi_analysis"], crs=projection,
                              x_coord='x', y_coord='y')

        ndvi_anomaly = ndvi_analysis_composite - ndvi_baseline_composite

        ndvi_anomaly_export = xr.DataArray.to_dataset(ndvi_anomaly, dim=None, name='ndvi_anomaly')
        write_geotiff_from_xr('ndvi_anomaly.tiff', ndvi_anomaly_export, ["ndvi_anomaly"], crs=projection,
                              x_coord='x', y_coord='y')


def create_product_measurement(platform, all_measurements):
    

    if platform in ["SENTINEL_2"]:
        product = 's2_esa_sr_granule'
        measurements = all_measurements + ["coastal_aerosol", "scene_classification"]
    else :
        product_match = re.search("LANDSAT_(\d)", platform)
        if product_match:
            product = f"ls{product_match.group(1)}_usgs_sr_scene"
            measurements = all_measurements + ["pixel_qa"]
        else:
            raise Exception(f"invalid platform_name {platform}")

    return product, measurements


def is_dataset_empty(ds: xr.Dataset) -> bool:
    checks_for_empty = [
        lambda x: len(x.dims) == 0,  # Dataset has no dimensions
        lambda x: len(x.data_vars) == 0  # Dataset no variables
    ]
    for f in checks_for_empty:
        if f(ds):
            return True
    return False
