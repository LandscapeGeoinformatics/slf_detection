import os
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from shapelysmooth import taubin_smooth

# smooth with buffer
def smooth_shapely(
    input_file,
    output_file,
    smooth_distance=10, # CRS unit (m)
    segments=5,
    cap_style=1,    # 1=round
    join_style=1,   # 1=round
    mitre_limit=2.0,
    simplify_tolerance_pre=None,
    simplify_tolerance_post=0.1,
    preserve_topology=True
):
    """
    Smooth polygons with Shapely buffer (QGIS-style), with optional simplify before/after.
    """

    print(f"Smoothing (QGIS-style buffer): {os.path.basename(input_file)}")

    gdf = gpd.read_file(input_file)
    geom_union = unary_union(gdf.geometry)

    # Simplify before buffer
    if simplify_tolerance_pre is not None:
        geom_union = geom_union.simplify(
            simplify_tolerance_pre, preserve_topology=preserve_topology
        )

    # Apply buffer-based smoothing
    smoothed = geom_union.buffer(
        smooth_distance,
        resolution=segments,
        cap_style=cap_style,
        join_style=join_style,
        mitre_limit=mitre_limit,
    ).buffer(
        -smooth_distance-0.5,
        resolution=segments,
        cap_style=cap_style,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )

    # Simplify after buffer (optional)
    if simplify_tolerance_post is not None:
        smoothed = smoothed.simplify(
            simplify_tolerance_post, preserve_topology=preserve_topology
        )

    # Build output GeoDataFrame
    if isinstance(smoothed, MultiPolygon):
        out_gdf = gpd.GeoDataFrame(geometry=list(smoothed.geoms), crs=gdf.crs)
    else:
        out_gdf = gpd.GeoDataFrame(geometry=[smoothed], crs=gdf.crs)

    out_gdf.to_file(output_file, driver="GPKG")
    print(f"Saved smoothed polygons to {output_file}")
    return out_gdf

# smooth with Taubin
def smooth_taubin(
    input_file,
    output_file,
    factor=0.35,
    mu=-0.34,
    steps=3,
    simplify_tolerance_pre=None,
    simplify_tolerance_post=0.1,
    preserve_topology=True
):
    """
    Pipeline: simplify first → Taubin smoothing → simplify again.
    """
    gdf = gpd.read_file(input_file)
    smoothed = []

    for geom in gdf.geometry:
        try:
            # Simplify first
            if simplify_tolerance_pre is not None:
                geom = geom.simplify(
                    simplify_tolerance_pre, preserve_topology=preserve_topology
                )

            # Apply Taubin smoothing
            smoothed_geom = taubin_smooth(geom, factor=factor, mu=mu, steps=steps)

            # Simplify again after smoothing
            if simplify_tolerance_post is not None:
                smoothed_geom = smoothed_geom.simplify(
                    simplify_tolerance_post, preserve_topology=preserve_topology
                )

            smoothed.append(smoothed_geom)

        except Exception as e:
            print(f"Taubin smoothing failed on geometry: {e}")
            smoothed.append(geom)

    # Build output GeoDataFrame
    out_gdf = gpd.GeoDataFrame(geometry=smoothed, crs=gdf.crs)
    out_gdf.to_file(output_file, driver="GPKG")

    return out_gdf

# define
input = "predicted_output.gpkg"
output_buffer = "predicted_output_smoothed_buffer.gpkg"
output_taubin = "predicted_output_smoothed_taubin.gpkg"

# run
smooth_shapely(input, output_buffer)
#smooth_taubin(input, output_taubin)
