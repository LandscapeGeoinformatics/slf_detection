# define gdal polygonization function
import os
import glob
import numpy as np
from osgeo import gdal, ogr, osr

# Threshold
threshold = 500
raster_folder = "/landscape_elements/working/postprocessing/mosaic/tile_masked_sieved"
output_gpkg = "/landscape_elements/working/postprocessing/polygonize/predicted_output.gpkg"

# define function
def raster_to_polygons_gdal(
    raster_folder,
    output_gpkg,
    threshold=500
):
    """
    Polygonizes all rasters in a folder to a single GeoPackage,
    including only pixels >= threshold using GDAL.

    Parameters
    ----------
    raster_folder : str
        Folder containing .tif raster files.
    output_gpkg : str
        Path to output GeoPackage.
    threshold : int or float, optional
        Threshold for binary mask (default=500).
    """


    # Collect all TIFF files
    raster_files = glob.glob(os.path.join(raster_folder, "*prob.tif"))
    if not raster_files:
        raise FileNotFoundError(f"No .tif files found in folder: {raster_folder}")

    # Prepare output driver
    driver = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_gpkg):
        driver.DeleteDataSource(output_gpkg)
    out_ds = driver.CreateDataSource(output_gpkg)

    out_layer = None  # will create first time inside loop

    for i, raster_path in enumerate(raster_files, 1):
        print(f"[{i}/{len(raster_files)}] Processing {raster_path}...")

        src_ds = gdal.Open(raster_path)
        if src_ds is None:
            print(f"Skipping unreadable file: {raster_path}")
            continue

        src_band = src_ds.GetRasterBand(1)
        arr = src_band.ReadAsArray()
        if arr is None:
            print(f"Empty or invalid raster: {raster_path}")
            continue

        # Apply threshold
        mask = (arr >= threshold).astype(np.uint8)

        # Create in-memory raster for mask
        mem_driver = gdal.GetDriverByName("MEM")
        mask_ds = mem_driver.Create("", src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Byte)
        mask_ds.SetGeoTransform(src_ds.GetGeoTransform())
        mask_ds.SetProjection(src_ds.GetProjection())
        mask_ds.GetRasterBand(1).WriteArray(mask)

        # Get CRS
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())

        # Create layer if not created yet
        if out_layer is None:
            out_layer = out_ds.CreateLayer("polygons", srs=srs, geom_type=ogr.wkbPolygon)
            #out_layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))
            out_layer.CreateField(ogr.FieldDefn("source", ogr.OFTString))

        # Polygonize
        tmp_layer_name = os.path.splitext(os.path.basename(raster_path))[0]
        gdal.Polygonize(
            mask_ds.GetRasterBand(1),
            mask_ds.GetRasterBand(1),
            out_layer,
            0,  # field index for "value"
            [],
            callback=None
        )

        # Update the "source" field with raster name
        for feature in out_layer:
            feature.SetField("source", tmp_layer_name)
            out_layer.SetFeature(feature)

        # Cleanup
        src_ds = None
        mask_ds = None

    out_ds = None  # flush to disk
    print(f"All rasters processed and saved to {output_gpkg}")

# apply function
raster_to_polygons_gdal(
    raster_folder=raster_folder,
    output_gpkg=output_gpkg,
    threshold=threshold
)