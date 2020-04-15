import re
import xarray as xr
import odc.algo
from pyproj import Proj, transform
from datacube_utilities.createAOI import create_lat_lon
from datacube_utilities.masking import mask_good_quality


def create_base_query(aoi, res, output_projection, aoi_crs, dask_chunks):
    lat_extents, lon_extents = create_lat_lon(aoi)
    inProj = Proj("+init=EPSG:4326")
    outProj = Proj("+init=EPSG:3460")

    min_lat, max_lat = lat_extents
    min_lon, max_lon = lon_extents

    x_A, y_A = transform(inProj, outProj, min_lon, min_lat)
    x_B, y_B = transform(inProj, outProj, max_lon, max_lat)

    lat_range = (y_A, y_B)
    lon_range = (x_A, x_B)

    resolution = (-res, res)

    query = {
        "y": lat_range,
        "x": lon_range,
        "output_crs": output_projection,
        "resolution": resolution,
        "dask_chunks": dask_chunks,
        "crs": aoi_crs,
    }
    return query


def create_product_measurement(platform, all_measurements):

    if platform in ["SENTINEL_2"]:
        product = "s2_esa_sr_granule"
        measurements = all_measurements + ["coastal_aerosol", "scene_classification"]
        # Change with S2 WOFS ready
        water_product = "SENTINEL_2_PRODUCT DEFS"
    else:
        product_match = re.search("LANDSAT_(\d)", platform)
        if product_match:
            product = f"ls{product_match.group(1)}_usgs_sr_scene"
            measurements = all_measurements + ["pixel_qa"]
            water_product = f"ls{product_match.group(1)}_water_classification"
        else:
            raise Exception(f"invalid platform_name {platform}")

    return product, measurements, water_product


def is_dataset_empty(ds: xr.Dataset) -> bool:
    checks_for_empty = [
        lambda x: len(x.dims) == 0,  # Dataset has no dimensions
        lambda x: len(x.data_vars) == 0,  # Dataset no variables
    ]
    for f in checks_for_empty:
        if f(ds):
            return True
    return False


def geomedian(ds, product, all_measurements):
    good_quality = mask_good_quality(ds, product)

    xx_data = ds[all_measurements]
    xx_clean = odc.algo.keep_good_only(xx_data, where=good_quality)

    scale, offset = (
        1 / 10_000,
        0,
    )  # differs per product, aim for 0-1 values in float32

    xx_clean = odc.algo.to_f32(xx_clean, scale=scale, offset=offset)
    yy = odc.algo.xr_geomedian(
        xx_clean,
        num_threads=1,  # disable internal threading, dask will run several concurrently
        eps=0.2 * scale,  # 1/5 pixel value resolution
        nocheck=True,
    )  # disable some checks inside geomedian library that use too much ram

    yy = odc.algo.from_float(
        yy, dtype="int16", nodata=-9999, scale=1 / scale, offset=-offset / scale
    )
    return yy
