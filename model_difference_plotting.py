#!/usr/bin/env python3
"""
This script generates a variety of rainfall accumulation and difference plots for two trial model outputs. 
It compares hourly and 6-hourly precipitation accumulations between two trials, producing thresholded, raw, and difference plots, as well as index-based visualizations. 
The script is designed for batch processing and saving of images, with options to control which plot types are generated. 
It relies on the internal oemplotlib library for utility functions and expects input data in the form of Iris cubes. 
Command-line arguments specify the datetime and trial names to process.

Usage:
    python model_diff_stretch.py --datetime <YYYYMMDD_HH>Z --trial1 <trial1_name> --trial2 <trial2_name>

Requirements:
    - Python3
    - Iris (>= v2.4.0)
    - Matplotlib
    - Cartopy
    - oemplotlib (local lib)

"""
import oemplotlib
import matplotlib
matplotlib.use('Agg')
import iris
import sys
import iris.plot as iplt
import iris.quickplot as qplt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator
import matplotlib.patches as mpatches
import logging
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import argparse


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# When set to True, when plotting is run the images will just be shown and not saved
DEVMODE = False

# Switches for which plotting functions to run
THRESHOLD_PLOTS = True
RAW_MODEL_OUTPUT = True
RAW_MODEL_DIFF = True
THRESHOLD_DIFFS = True
THRESHOLD_DIFFS_BOTH = True
INDEX_DIFFS = True
THRESHOLD_INDEX_DIFFS = True

IMAGE_PATH = "/data/users/tevans/PS47/PS47_thresholdplot_images"

# RAINFALL THRESHOLDS (mm/hr)
RAINFALL_THRESHOLDS = {
    "VERY_LIGHT_RAIN": [0.01, 0.25],
    "LIGHT_RAIN": [0.25, 1.0],
    "MODERATE_RAIN": [1.0, 4.0],
    "HEAVY_RAIN": [4.0, 10],
    "VERY_HEAVY_RAIN": [10]
}

DRY_THRESHOLD = [0.000001, 0.01]

BIN_INDEXES = {
    "DRY": 0.5,
    "VERY_LIGHT_RAIN": 1.5,
    "LIGHT_RAIN": 2.5,
    "MODERATE_RAIN": 3.5,
    "HEAVY_RAIN": 4.5,
    "VERY_HEAVY_RAIN": 5.5
}

INDEXED_COLORBAR_VALS = {
    "VERY_LIGHT_RAIN": [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5],
    "LIGHT_RAIN": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    "MODERATE_RAIN": [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5],
    "HEAVY_RAIN": [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
    "VERY_HEAVY_RAIN": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
}

BOUNDS = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
RGBs = [[158, 5, 252],
        [15, 109, 249],
        [50, 167, 255],
        [51, 222, 255],
        [136, 250, 250],
        [255, 255, 255],  # white
        [255, 245, 51],
        [255, 215, 51],
        [255, 181, 51],
        [255, 119, 51],
        [255, 0, 0]]

RGBs_2 = [[0, 0, 55],
          [3, 17, 168],
          [0, 0, 250],
          [111, 168, 252],
          [122, 228, 255],
          [5, 228, 252],
          [255, 255, 255],  # white
          [255, 255, 255],  # white
          [255, 255, 0],
          [255, 195, 0],
          [255, 188, 33],
          [245, 124, 49],
          [255, 73, 23],
          [250, 0, 0],
          [190, 2, 2]]

INDEXED_COLORBAR_RGBs = {
    "VERY_LIGHT_RAIN": [
        [123, 10, 204],
        [255, 93, 251],
        [236, 50, 27],
        [255, 255, 255],
        [51, 204, 255],
    ],
    "LIGHT_RAIN": [
        [123, 10, 204],
        [255, 93, 251],
        [255, 255, 255],
        [236, 50, 27],
        [51, 204, 255],
    ],
    "MODERATE_RAIN": [
        [123, 10, 204],
        [255, 255, 255],
        [255, 93, 251],
        [236, 50, 27],
        [51, 204, 255],
    ],
    "HEAVY_RAIN": [
        [255, 255, 255],
        [123, 10, 204],
        [255, 93, 251],
        [236, 50, 27],
        [51, 204, 255],
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", required=True)
    parser.add_argument("--trial1", required=True)
    parser.add_argument("--trial2", required=True)
    args = parser.parse_args()
    return args


def _fraction_covered(binary_cube_timeslice):
    x = binary_cube_timeslice.data.shape
    count = np.count_nonzero(binary_cube_timeslice.data)
    number_of_cells = x[0] * x[1]
    fraction = round((count / number_of_cells) * 100, 2)
    return round(100 - fraction, 2)


def _mean_value(data):
    x = np.mean(abs(data))
    return round(float(x), 2)


def _get_minimal_cube(cube, i):
    time_constraint = iris.Constraint(forecast_period=i)
    cube_at_lt = cube.extract(time_constraint)
    mask = DRY_THRESHOLD[0] > cube_at_lt.data
    cube_at_lt.data = np.ma.array(cube_at_lt.data, mask=mask)
    masked_cube = cube_at_lt.copy(data=cube_at_lt.data)
    return masked_cube


def _get_binary_cube(cube, i, rain_bin):
    time_constraint = iris.Constraint(forecast_period=i)
    cube_at_lt = cube.extract(time_constraint)
    if rain_bin == "VERY_HEAVY_RAIN":
        mask = RAINFALL_THRESHOLDS[rain_bin][0] < cube_at_lt.data
        cube_at_lt.data = np.where(mask, 1, 0)
        binary_cube = cube_at_lt.copy(data=cube_at_lt.data)
    else:
        mask = (RAINFALL_THRESHOLDS[rain_bin][0] < cube_at_lt.data) & (
                cube_at_lt.data <= RAINFALL_THRESHOLDS[rain_bin][1]
        )
        cube_at_lt.data = np.where(mask, 1, 0)
        binary_cube = cube_at_lt.copy(data=cube_at_lt.data)
    return binary_cube


def _put_data_in_index_bins(cube, i):
    """
    this function will assign the appropriate index for the values depending on the threshold input
    VERY LIGHT RAIN: 1, LIGHT RAIN: 2, MODERATE RAIN: 3, HEAVY RAIN: 4
    the indexing above is the same for all plots
    """
    time_constraint = iris.Constraint(forecast_period=i)
    cube_at_lt = cube.extract(time_constraint)
    bin_names = [key for key in BIN_INDEXES]
    # assign masks
    mask1 = (DRY_THRESHOLD[0] < cube_at_lt.data) & (
            cube_at_lt.data <= DRY_THRESHOLD[1]
    )
    mask2 = (RAINFALL_THRESHOLDS[bin_names[1]][0] < cube_at_lt.data) & (
            cube_at_lt.data <= RAINFALL_THRESHOLDS[bin_names[1]][1]
    )
    mask3 = (RAINFALL_THRESHOLDS[bin_names[2]][0] < cube_at_lt.data) & (
            cube_at_lt.data <= RAINFALL_THRESHOLDS[bin_names[2]][1]
    )
    mask4 = (RAINFALL_THRESHOLDS[bin_names[3]][0] < cube_at_lt.data) & (
            cube_at_lt.data <= RAINFALL_THRESHOLDS[bin_names[3]][1]
    )
    mask5 = (RAINFALL_THRESHOLDS[bin_names[4]][0] < cube_at_lt.data) & (
            cube_at_lt.data <= RAINFALL_THRESHOLDS[bin_names[4]][1]
    )
    mask6 = RAINFALL_THRESHOLDS[bin_names[5]][0] < cube_at_lt.data
    # apply masks to cube data
    cube_at_lt.data = np.where(mask1, BIN_INDEXES[bin_names[0]], cube_at_lt.data)
    cube_at_lt.data = np.where(mask2, BIN_INDEXES[bin_names[1]], cube_at_lt.data)
    cube_at_lt.data = np.where(mask3, BIN_INDEXES[bin_names[2]], cube_at_lt.data)
    cube_at_lt.data = np.where(mask4, BIN_INDEXES[bin_names[3]], cube_at_lt.data)
    cube_at_lt.data = np.where(mask5, BIN_INDEXES[bin_names[4]], cube_at_lt.data)
    cube_at_lt.data = np.where(mask6, BIN_INDEXES[bin_names[5]], cube_at_lt.data)

    return cube_at_lt


def _mask_values_below_minimal(cube, i):
    time_constraint = iris.Constraint(forecast_period=i)
    cube_at_lt = cube.extract(time_constraint)

    mask = DRY_THRESHOLD[0] > cube_at_lt.data
    cube_at_lt.data = np.ma.array(cube_at_lt.data, mask=mask)
    masked_cube = cube_at_lt.copy(data=cube_at_lt.data)

    return masked_cube


def _round_down_mask(cube):
    mask1 = (-0.5 == cube.data)
    mask2 = (-1.5 == cube.data)
    mask3 = (-2.5 == cube.data)
    mask4 = (-3.5 == cube.data)
    cube.data = np.where(mask1, -1, cube.data)
    cube.data = np.where(mask2, -2, cube.data)
    cube.data = np.where(mask3, -3, cube.data)
    cube.data = np.where(mask4, -4, cube.data)
    masked_cube = cube.copy(data=cube.data)
    return masked_cube


def _mask_differences(cube1, cube2, i, threshold):
    """
    This function plots a white colour where the difference (trial1-trial2) is between -0.5 and 0.5, blue where it's 1.5, red where it's -1.5,
    green where it's 2.5 etc
    Maybe mix of greens getting darker where trial1 dominates and reds getting darker in trial2 direction
    """
    indexed_control_cube = _put_data_in_index_bins(cube1, i, threshold)
    LOGGER.info(f"indexed_control: {indexed_control_cube.data[0]}")
    indexed_experiment_cube = _put_data_in_index_bins(cube2, i, threshold)
    LOGGER.info(f"indexed_experiment: {indexed_experiment_cube.data[0]}")
    diff_data = indexed_control_cube.data - indexed_experiment_cube.data
    index_diff_cube = indexed_control_cube.copy(data=diff_data)

    mask = (-1.5 < index_diff_cube.data) & (index_diff_cube.data <= -0.5)
    mask1 = (-0.5 < index_diff_cube.data) & (index_diff_cube.data <= 0.5)
    mask2 = (0.5 < index_diff_cube.data) & (index_diff_cube.data <= 1.5)
    mask3 = (1.5 < index_diff_cube.data) & (index_diff_cube.data <= 2.5)
    mask4 = (2.5 < index_diff_cube.data) & (index_diff_cube.data <= 3.5)
    mask5 = (3.5 < index_diff_cube.data) & (index_diff_cube.data <= 4.5)
    index_diff_cube.data = np.where(mask, index_diff_cube.data, -1)
    index_diff_cube.data = np.where(mask1, index_diff_cube.data, 0)
    index_diff_cube.data = np.where(mask2, index_diff_cube.data, 1)
    index_diff_cube.data = np.where(mask3, index_diff_cube.data, 2)
    index_diff_cube.data = np.where(mask4, index_diff_cube.data, 3)
    index_diff_cube.data = np.where(mask5, index_diff_cube.data, 4)
    binary_cube = index_diff_cube.copy(data=index_diff_cube.data)
    cmap = INDEXED_COLORBAR_RGBs[threshold]
    # Create the plot
    plt.figure(figsize=(5, 6))
    plt.title(f"Difference gradient at {threshold}")
    iplt.pcolormesh(binary_cube, cmap=cmap, alpha=0.5)
    ax = plt.gca()
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
    ax.gridlines()

    if DEVMODE:
        plt.show()
        sys.exit(1)


def plot_masked_index_diffs(cube1, cube2, i, fp, cube_dt, title):
    # step 1: get sum of cubes (masked over minimum threshold)
    a = _get_minimal_cube(cube1, i)
    b = _get_minimal_cube(cube2, i)
    cube_of_intersection = a + b

    # step 2: put remaining values into bins
    indexed_masked_cube1 = _put_data_in_index_bins(cube1, i)
    indexed_masked_cube2 = _put_data_in_index_bins(cube2, i)

    # step 3: take away these values from eachother as in plot_absolute_diffs
    diff_data = indexed_masked_cube1.data - indexed_masked_cube2.data
    index_diff_cube = indexed_masked_cube1.copy(data=diff_data)

    # step 4: mask values of -0.5 to -1, -1.5 to -2, -2.5 to -3, -3.5 to -4 (because of how colorbar assigns tick values)
    index_diff_cube = _round_down_mask(index_diff_cube)

    # step 5: mask over intersection of cube (so not plotting any values that don't intersect)
    intersecting_data = np.ma.masked_array(index_diff_cube.data, mask=cube_of_intersection.data.mask)
    intersected_and_indexed_cube = index_diff_cube.copy(data=intersecting_data)

    # get mean score (to put in title)
    mean = _mean_value(intersecting_data)

    # step 6: plot
    # INDEXING PLOTTING
    rgb = np.array(RGBs) / 255
    cmap = ListedColormap(rgb)
    norm = BoundaryNorm(BOUNDS, cmap.N, clip=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"Index differences {title} \n mean of absolute values = {mean}")
    iplt.pcolormesh(intersected_and_indexed_cube, cmap=cmap, norm=norm)

    # Calculate midpoints between consecutive bounds
    midpoints = [
        ((BOUNDS[i] + BOUNDS[i + 1]) / 2 - 0.5) for i in range(len(BOUNDS) - 1)
    ]
    midpoints_offset = [val + 0.5 * (BOUNDS[1] - BOUNDS[0]) for val in midpoints]

    plt.colorbar(ticks=midpoints_offset)
    ax = plt.gca()
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
    ax.gridlines()

    if DEVMODE:
        plt.show()
        sys.exit(1)

    plt.savefig(
        f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/Index_Diffs_DT_{cube_dt}_{fp}Hrly_Accum_{title}_T+{i}H.png",
        format="png",
    )
    plt.close()
    LOGGER.info(f"image saved for index diff {title} at T+{i}H")


def plot_threshold_index_diffs(cube1, cube2, i, fp, cube_dt, rain_bin, title):
    # step 1: get sum of cubes (masked over minimum threshold)
    a = _get_minimal_cube(cube1, i)
    b = _get_minimal_cube(cube2, i)
    cube_of_intersection = a + b

    # step 2: put remaining values into bins
    indexed_masked_cube1 = _put_data_in_index_bins(cube1, i)
    indexed_masked_cube2 = _put_data_in_index_bins(cube2, i)

    # step 3: take away these values from eachother as in plot_absolute_diffs
    diff_data = indexed_masked_cube1.data - indexed_masked_cube2.data
    index_diff_cube = indexed_masked_cube1.copy(data=diff_data)

    # step 4: mask values of -0.5 to -1, -1.5 to -2, -2.5 to -3, -3.5 to -4, -4.5 to -5 (because of how colorbar assigns tick values)
    index_diff_cube = _round_down_mask(index_diff_cube)

    # step 5: mask over intersection of cube (so not plotting any values that don't intersect)
    intersecting_data = np.ma.masked_array(index_diff_cube.data, mask=cube_of_intersection.data.mask)

    diffcube1 = _get_binary_cube(cube1, i, rain_bin)
    diffcube2 = _get_binary_cube(cube2, i, rain_bin)

    cube1_copy = diffcube1.copy()
    cube2_copy = diffcube2.copy()

    mask = (cube1_copy.data == 1) & (cube2_copy.data == 0)
    cube1_copy.data = np.where(mask, 0, 1)
    difference_binary_cube = cube1_copy.copy(data=cube1_copy.data)

    ######################## RAIN BIN THRESHOLD STEPS #####################################################
    # # TODO: put this step in its own function so cubes can be arguments
    # time_constraint = iris.Constraint(forecast_period=i)
    # cube1_at_lt = cube1.extract(time_constraint)
    # cube2_at_lt = cube2.extract(time_constraint)
    #
    # if rain_bin == "HEAVY_RAIN":
    #     mask = RAINFALL_THRESHOLDS[rain_bin][0] > cube1_at_lt.data
    #     cube1_at_lt.data = np.ma.array(cube1_at_lt.data, mask=mask)
    #     cube2_at_lt.data = np.ma.array(cube2_at_lt.data, mask=mask)
    #     masked_cube1 = cube1_at_lt.copy(data=cube1_at_lt.data)
    #     masked_cube2 = cube2_at_lt.copy(data=cube2_at_lt.data)
    # else:
    #     mask1 = (RAINFALL_THRESHOLDS[rain_bin][0] > cube1_at_lt.data)
    #     mask2 = (cube1_at_lt.data >= RAINFALL_THRESHOLDS[rain_bin][1])
    #     mask3 = (RAINFALL_THRESHOLDS[rain_bin][0] > cube2_at_lt.data)
    #     mask4 = (cube2_at_lt.data >= RAINFALL_THRESHOLDS[rain_bin][1])
    #     cube1_at_lt.data = np.ma.array(cube1_at_lt.data, mask=mask1)
    #     cube1_at_lt.data = np.ma.array(cube1_at_lt.data, mask=mask2)
    #     cube2_at_lt.data = np.ma.array(cube2_at_lt.data, mask=mask3)
    #     cube2_at_lt.data = np.ma.array(cube2_at_lt.data, mask=mask4)
    #     masked_cube1 = cube1_at_lt.copy(data=cube1_at_lt.data)
    #     masked_cube2 = cube2_at_lt.copy(data=cube2_at_lt.data)
    #     print(masked_cube1.data[0])
    #     print("**************")
    #     print(masked_cube2.data[0])
    #     print("**************")
    #
    # threshold_intersection_cube = masked_cube1 + masked_cube2
    # LOGGER.info(f"masked_cube1.data[0]: {masked_cube1.data[0]}")
    # LOGGER.info(f"masked_cube2.data[0]: {masked_cube2.data[0]}")
    # LOGGER.info(f"threshold_mask_cube.data[0]: {threshold_intersection_cube.data[0]}")
    # LOGGER.info(f"threshold_mask_cube.data[0].mask: {threshold_intersection_cube.data[0].mask}")

    threshold_masked_data = np.ma.masked_array(intersecting_data, mask=difference_binary_cube.data)
    # LOGGER.info(f"threshold_masked_data[0]: {threshold_masked_data[0]}")
    intersected_and_indexed_threshold_cube = index_diff_cube.copy(data=threshold_masked_data)

    ####################################################################################################

    # step 6: plot
    # INDEXING PLOTTING
    rgb = np.array(RGBs) / 255
    cmap = ListedColormap(rgb)
    norm = BoundaryNorm(BOUNDS, cmap.N, clip=True)
    plt.figure(figsize=(6, 6))
    plt.title(f"{rain_bin} index difference {title}")
    iplt.pcolormesh(intersected_and_indexed_threshold_cube, cmap=cmap, norm=norm)

    #contour_levels = [0.5]
    #iplt.contour(intersected_and_indexed_cube, levels=contour_levels, colors='k', linewidths=0.3)

    # Calculate midpoints between consecutive bounds
    midpoints = [
        ((BOUNDS[i] + BOUNDS[i + 1]) / 2 - 0.5) for i in range(len(BOUNDS) - 1)
    ]
    midpoints_offset = [val + 0.5 * (BOUNDS[1] - BOUNDS[0]) for val in midpoints]

    plt.colorbar(ticks=midpoints_offset)
    ax = plt.gca()
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
    ax.gridlines()

    if DEVMODE:
        plt.show()
        sys.exit(1)

    plt.savefig(
        f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/{rain_bin}_index_Diffs_DT_{cube_dt}_{fp}Hrly_Accum_{title}_T+{i}H.png",
        format="png",
    )
    plt.close()
    LOGGER.info(f"image saved for index diff {title} at T+{i}H")


def plot_model_output(cube, fp, i, cube_dt, model):
    if os.path.exists(
            f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/DT_{cube_dt}_{fp}Hrly_{model}_raw_output_T+{i}H.png"):
        return LOGGER.info(f"DT_{cube_dt}_{fp}Hrly_{model}_raw_output_T+{i}H.png already exists. Skipping..")
    else:
        time_constraint = iris.Constraint(forecast_period=i)
        extracted_cube = cube.extract(time_constraint)
        extracted_cube.long_name = f"{fp}Hrly {model} values for 20211207 06Z at T+{i}H"
        extracted_cube.units = "mm"

        # MesoPrecip
        # note that in OVELAYS anything below 0.01 is masked and is white
        rgb_colormap = [
            [255, 255, 255],
            [121, 178, 233],
            [81, 147, 212],
            [40, 104, 189],
            [31, 201, 27],
            [255, 240, 58],
            [255, 155, 0],
            [236, 50, 27],
            [255, 93, 251],
            [123, 10, 204],
        ]
        rgb = np.array(rgb_colormap) / 255
        bounds = [0.01, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
        cmap = matplotlib.colors.ListedColormap(rgb)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, clip=True)

        # Create the plot
        extracted_cube.long_name = f"{model} raw output"
        plt.figure(figsize=(6, 6))
        plt.title(extracted_cube.long_name)
        iplt.pcolormesh(extracted_cube, cmap=cmap, norm=norm)
        ax = plt.gca()
        cbar = plt.colorbar(ticks=bounds)
        cbar.set_ticks(bounds)
        cbar.ax.set_aspect(10)

        # map features
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
        ax.gridlines()
        if DEVMODE:
            plt.show()
            sys.exit(1)
        plt.savefig(
            f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/DT_{cube_dt}_{fp}Hrly_{model}_raw_output_T+{i}H.png",
            format="png",
        )
        LOGGER.info(f"image saved for {extracted_cube.long_name}")


def plot_overall_diff(cube1, cube2, fp, cube_dt, i, trial1, trial2):
    """
    Args:
        cube1: cube with trial1 data
        cube2: cube with trial2 data
        fp: forecast period
        i: lead time

    Returns: plot of overall difference between the two cubes
    """
    # Calculate the difference between the two cubes
    difference = cube1 - cube2

    time_constraint = iris.Constraint(forecast_period=i)
    diff = difference.extract(time_constraint)
    diff.units = "mm"
    bounds = [-5, -4, -3, -2, -1, -0.01, 0, 0.01, 1, 2, 3, 4, 5]

    # INDEXING PLOTTING
    rgb = np.array(RGBs_2) / 255
    cmap = ListedColormap(rgb)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    # Create a contour plot of the difference
    plt.figure(figsize=(5, 6))
    plt.title(f"raw output {trial1}-{trial2}")
    iplt.pcolormesh(diff, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ticks=bounds)
    cbar.set_ticks(bounds)
    ax = plt.gca()
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
    ax.gridlines()
    if DEVMODE:
        plt.show()
        sys.exit(1)
    plt.savefig(
        f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/DT_{cube_dt}_{fp}Hrly_raw_diff_{trial1}-{trial2}_T+{i}H.png",
        format="png",
    )
    plt.close()
    LOGGER.info(f"image saved for raw differences T+{i}H")


def plot_binary_diff_cube(input_cube1, input_cube2, fp, i, cube_dt, name, trial1, trial2, rain_bin):
    cube1 = _get_binary_cube(input_cube1, i, rain_bin)
    cube2 = _get_binary_cube(input_cube2, i, rain_bin)

    cube1_copy = cube1.copy()
    cube2_copy = cube2.copy()

    mask = (cube1_copy.data == 1) & (cube2_copy.data == 0)
    cube1_copy.data = np.where(mask, 0, 1)
    difference_binary_cube = cube1_copy.copy(data=cube1_copy.data)

    if name == f"{trial2} - {trial1}":
        difference_binary_cube.long_name = f"{trial2}-{trial1}"
        cmap = plt.cm.colors.ListedColormap(["khaki", "white"])
    else:
        difference_binary_cube.long_name = f"{trial1}-{trial2}"
        cmap = plt.cm.colors.ListedColormap(["cornflowerblue", "white"])

    # Create the plot
    plt.figure(figsize=(5, 6))
    plt.title(difference_binary_cube.long_name)
    iplt.pcolormesh(difference_binary_cube, cmap=cmap)
    ax = plt.gca()
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
    ax.gridlines()
    if DEVMODE:
        plt.show()
        sys.exit(1)
    plt.savefig(
        f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/Diff_DT_{cube_dt}_{fp}Hrly_Accum_{rain_bin}_{difference_binary_cube.long_name}_T+{i}H.png",
        format="png",
    )
    plt.close()
    LOGGER.info(f"image saved for {difference_binary_cube.long_name}")


def plot_both_binary_diff_cube(input_cube1, input_cube2, fp, i, cube_dt, rain_bin, trial1, trial2):
        
        cube1 = _get_binary_cube(input_cube1, i, rain_bin)
        cube2 = _get_binary_cube(input_cube2, i, rain_bin)

        cube1_copy = cube1.copy()
        cube2_copy = cube2.copy()

        mask = (cube1_copy.data == 1) & (cube2_copy.data == 0)
        cube1_copy.data = np.where(mask, 1, 0)
        mask2 = cube1_copy.data == 0
        masked_data = np.ma.masked_where(mask2, cube1_copy.data, copy=True)
        difference_binary_cube1 = cube1_copy.copy(data=masked_data)

        cube1_second_copy = cube1.copy()
        cube2_second_copy = cube2.copy()

        mask_r = (cube2_second_copy.data == 1) & (cube1_second_copy.data == 0)
        cube2_second_copy.data = np.where(mask_r, 1, 0)
        mask3 = cube2_second_copy.data == 0
        masked_data1 = np.ma.masked_where(mask3, cube2_second_copy.data, copy=True)
        difference_binary_cube2 = cube2_second_copy.copy(data=masked_data1)

        # Apply the mask to cube1 to get the difference
        cmap = ListedColormap(["cornflowerblue", "white"])
        cmap1 = ListedColormap(["khaki", "white"])

        # Create the plot
        plt.figure(figsize=(5, 6))
        plt.title(f"{trial2} Diff & {trial1} Diff")
        iplt.pcolormesh(difference_binary_cube1, cmap=cmap, alpha=0.5)
        iplt.pcolormesh(difference_binary_cube2, cmap=cmap1, alpha=0.5)
        ax = plt.gca()
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
        ax.gridlines()

        if DEVMODE:
            plt.show()
            sys.exit(1)
        plt.savefig(
            f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/Dual_Diffs_DT_{cube_dt}_{fp}Hrly_Accum_{rain_bin}_{trial1}_{trial2}_T+{i}H.png",
            format="png",
        )
        plt.close()
        LOGGER.info(f"image saved for {trial2} Diffs vs {trial1} Diffs at T+{i}H")


def plot_threshold_cube(cube, fp, i, cube_dt, rain_bin, model):
    """
    Args:
        cube: cube to be plotted
        fp: forecast period
        i: lead time
        rain_bin (str): string which is assigned to a particular threshold in rainfall_thresholds dict

    Returns: plot of cube within or over certain threshold decided by rain_bin
    """
    if os.path.exists(
            f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/{model}_DT_{cube_dt}_{fp}Hrly_Accum_{rain_bin}_T+{i}H.png"):
        return LOGGER.info(f"{model}_DT_{cube_dt}_{fp}Hrly_Accum_{rain_bin}_T+{i}H.png already exists. Skipping..")
    else:
        time_constraint = iris.Constraint(forecast_period=i)
        cube_at_lt = cube.extract(time_constraint)

        if rain_bin == "VERY_HEAVY_RAIN":
            mask = RAINFALL_THRESHOLDS[rain_bin][0] < cube_at_lt.data
            cube_at_lt.data = np.where(mask, 0, 1)
            binary_cube = cube_at_lt.copy(data=cube_at_lt.data)
        else:
            mask = (RAINFALL_THRESHOLDS[rain_bin][0] < cube_at_lt.data) & (
                    cube_at_lt.data <= RAINFALL_THRESHOLDS[rain_bin][1]
            )
            cube_at_lt.data = np.where(mask, 0, 1)
            binary_cube = cube_at_lt.copy(data=cube_at_lt.data)

        # get fraction covered to include in plot title
        frac = _fraction_covered(binary_cube)

        # assign plot color depending on rain_bin value:
        if rain_bin == "VERY_HEAVY_RAIN":
            cmap = plt.cm.colors.ListedColormap(["maroon", "white"])
        elif rain_bin == "HEAVY_RAIN":
            cmap = plt.cm.colors.ListedColormap(["darksalmon", "white"])
        elif rain_bin == "MODERATE_RAIN":
            cmap = plt.cm.colors.ListedColormap(["lightgreen", "white"])
        elif rain_bin == "LIGHT_RAIN":
            cmap = plt.cm.colors.ListedColormap(["skyblue", "white"])
        elif rain_bin == "VERY_LIGHT_RAIN":
            cmap = plt.cm.colors.ListedColormap(["orchid", "white"])

        if frac == 0.0:
            cmap = plt.cm.colors.ListedColormap(["white", "white"])

        # Create the plot
        binary_cube.long_name = f"{model} (coverage {frac}%)"
        plt.figure(figsize=(5, 6))
        plt.title(binary_cube.long_name)
        iplt.pcolormesh(binary_cube, cmap=cmap)
        ax = plt.gca()
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
        ax.gridlines()
        if DEVMODE:
            plt.show()
            sys.exit(1)

        plt.savefig(
            f"{IMAGE_PATH}/{cube_dt}/{fp}hrly_accums/{model}_DT_{cube_dt}_{fp}Hrly_Accum_{rain_bin}_T+{i}H.png",
            format="png",
        )
        plt.close()
        LOGGER.info(f"image saved for {binary_cube.long_name}")


def get_hrly_rain_snow_amnt(cubes, rain_stash, snow_stash):
    def _loader(cubelist, stash_constraint):
        try:
            constraint = iris.Constraint(cube_func=lambda cube: cube.cell_methods)
            outcube = cubelist.extract(
                iris.AttributeConstraint(STASH=stash_constraint) & constraint,
                strict=True,
            )
        except iris.exceptions.ConstraintMismatchError as err:
            outcube = cubelist.extract(iris.AttributeConstraint(STASH=stash_constraint))
            msg = ""
            for cube in outcube:
                msg += f"{cube}\n"
            LOGGER.exception(
                "error loading for stash %s, found cubes:\n%s",
                stash_constraint,
                msg,
            )
            raise err
        return outcube.copy()

    stratiform_rain = _loader(cubes, rain_stash)
    try:
        stratiform_snow = _loader(cubes, snow_stash)
        rainsnow = stratiform_rain + stratiform_snow
    except iris.exceptions.ConstraintMismatchError:
        LOGGER.warning(
            "plot_total_precip_rate couldn't load convective precip rates, "
            "plotting will resume assuming this is a convection permitting model"
        )
        rainsnow = stratiform_rain
    # first handle ensemble possibly having multidimensional time coordinates
    rainsnow_separated = oemplotlib.cube_utils.separate_realization_time(rainsnow)
    # only want hourly output so extract times that are on the hour
    rainsnow_filtered = iris.cube.CubeList()
    if isinstance(rainsnow_separated, iris.cube.CubeList):
        for p in rainsnow_separated:
            try:
                p = oemplotlib.cube_utils.snap_to_time(
                    p,
                    minutes_past_hour=0,
                    max_window_minutes=5,
                )
                rainsnow_filtered.append(p)
            except ValueError:
                # probably a sub-hourly off hour cube
                pass
    else:
        rainsnow_filtered.append(
            oemplotlib.cube_utils.snap_to_time(
                rainsnow_separated,
                minutes_past_hour=0,
                max_window_minutes=5,
            )
        )
    if len(rainsnow_filtered) < 1:
        raise ValueError("Unable to separate/snap rainsnow cubes")
    elif len(rainsnow_filtered) == 1:
        rainsnow = rainsnow_filtered[0]
    else:
        rainsnow = rainsnow_filtered.merge()
    rainsnow = oemplotlib.cube_utils.fix_running_cube_time(rainsnow)
    if isinstance(rainsnow, iris.cube.CubeList):
        [p.attributes.update(stratiform_rain.attributes) for p in rainsnow]
        # kg m-2 is equivalent to mm as density of water is approx 1000 kg m-3
        [p.convert_units("kg m-2") for p in rainsnow]
        for method in stratiform_rain.cell_methods:
            [p.add_cell_method(method) for p in rainsnow]
    else:
        rainsnow.attributes.update(stratiform_rain.attributes)
        rainsnow.convert_units("kg m-2")
        for method in stratiform_rain.cell_methods:
            rainsnow.add_cell_method(method)
    return rainsnow


class Precip_accumulations:
    # class level variable
    # rain stash, snow stash
    rainfall_stash = ["m01s04i201", "m01s04i202"]

    def __init__(self, filepath):
        self.filepath = filepath

    def get_cube_accumulations(self, period_hrs):
        cubes = iris.load(self.filepath)

        def _loader(cubelist, stash_constraint):
            try:
                constraint = iris.Constraint(cube_func=lambda cube: cube.cell_methods)
                outcube = cubelist.extract(
                    iris.AttributeConstraint(STASH=stash_constraint) & constraint,
                    strict=True,
                )
            except iris.exceptions.ConstraintMismatchError as err:
                outcube = cubelist.extract(
                    iris.AttributeConstraint(STASH=stash_constraint)
                )
                msg = ""
                for cube in outcube:
                    msg += f"{cube}\n"
                LOGGER.exception(
                    "error loading for stash %s, found cubes:\n%s",
                    stash_constraint,
                    msg,
                )
                raise err
            return outcube.copy()

        stratiform_rain = _loader(cubes, Precip_accumulations.rainfall_stash[0])
        try:
            stratiform_snow = _loader(cubes, Precip_accumulations.rainfall_stash[1])
            rainsnow = stratiform_rain + stratiform_snow
        except iris.exceptions.ConstraintMismatchError:
            LOGGER.warning(
                "plot_total_precip_rate couldn't load stratiform snow rates, "
                "plotting will resume assuming this is a startiform rain-only model"
            )
            rainsnow = stratiform_rain
        # first handle ensemble possibly having multidimensional time coordinates
        rainsnow_separated = oemplotlib.cube_utils.separate_realization_time(rainsnow)
        # only want hourly output so extract times that are on the hour
        rainsnow_filtered = iris.cube.CubeList()
        if isinstance(rainsnow_separated, iris.cube.CubeList):
            for p in rainsnow_separated:
                try:
                    p = oemplotlib.cube_utils.snap_to_time(
                        p,
                        minutes_past_hour=0,
                        max_window_minutes=5,
                    )
                    rainsnow_filtered.append(p)
                except ValueError:
                    # probably a sub-hourly off hour cube
                    pass
        else:
            rainsnow_filtered.append(
                oemplotlib.cube_utils.snap_to_time(
                    rainsnow_separated,
                    minutes_past_hour=0,
                    max_window_minutes=5,
                )
            )
        if len(rainsnow_filtered) < 1:
            raise ValueError("Unable to separate/snap rainsnow cubes")
        elif len(rainsnow_filtered) == 1:
            rainsnow = rainsnow_filtered[0]
        else:
            rainsnow = rainsnow_filtered.merge()
        rainsnow = oemplotlib.cube_utils.fix_running_cube_time(rainsnow)
        if isinstance(rainsnow, iris.cube.CubeList):
            [p.attributes.update(stratiform_rain.attributes) for p in rainsnow]
            # kg m-2 is equivalent to mm as density of water is approx 1000 kg m-3
            [p.convert_units("kg m-2") for p in rainsnow]
            for method in stratiform_rain.cell_methods:
                [p.add_cell_method(method) for p in rainsnow]
        else:
            rainsnow.attributes.update(stratiform_rain.attributes)
            rainsnow.convert_units("kg m-2")
            for method in stratiform_rain.cell_methods:
                rainsnow.add_cell_method(method)

        try:
            accum_rain = oemplotlib.cube_utils.running_accum_to_period(
                rainsnow,
                period_minutes=period_hrs * 60,
                out_cube_name=f"Rain Accumulation {period_hrs} hrly",
            )
        except Exception:
            LOGGER.exception(
                "plot_rain_amnt: Error accumulating rain for %s period, skipping",
                period_hrs,
            )

        return accum_rain


def main():
    args = parse_args()
    cube_dt = args.datetime
    trial_name1 = args.trial1
    trial_name2 = args.trial2
    LOGGER.info(f"iris version: {iris.__version__}")
    LOGGER.info(f"DEVMODE set to {DEVMODE}")
    LOGGER.info(f"DATETIME: {cube_dt}")

    # ensure directory structure is correct
    if not os.path.exists(f"{IMAGE_PATH}/{cube_dt}"):
        (os.mkdir(f"{IMAGE_PATH}/{cube_dt}"), os.mkdir(f"{IMAGE_PATH}/{cube_dt}/1hrly_accums"),
         os.mkdir(f"{IMAGE_PATH}/{cube_dt}/6hrly_accums"))

    accum_periods = [1, 6]

    file1 = f"/data/users/tevans/PS47/PS47_thresholdplot_data/{cube_dt}_{trial_name1}.pp"
    file2 = f"/data/users/tevans/PS47/PS47_thresholdplot_data/{cube_dt}_{trial_name2}.pp"

    rain1 = Precip_accumulations(file1)
    rain2 = Precip_accumulations(file2)

    for accum in accum_periods:
        LOGGER.info(f"accumulation period: {accum}")
        LOGGER.info("making accumulations..")
        accumulations_rain1 = rain1.get_cube_accumulations(accum)
        accumulations_rain2 = rain2.get_cube_accumulations(accum)
        LOGGER.info("accumulations done!")

        fp_length = len(accumulations_rain1.coord("forecast_period").points)
        LOGGER.info(f"fp_length: {fp_length}")
        for lead_time in range(1, fp_length + 1):
            
            if THRESHOLD_PLOTS:
                plot_threshold_cube(accumulations_rain1, accum, lead_time, cube_dt, "VERY_LIGHT_RAIN", trial_name1)
                plot_threshold_cube(accumulations_rain1, accum, lead_time, cube_dt, "LIGHT_RAIN", trial_name1)
                plot_threshold_cube(accumulations_rain1, accum, lead_time, cube_dt, "MODERATE_RAIN", trial_name1)
                plot_threshold_cube(accumulations_rain1, accum, lead_time, cube_dt, "HEAVY_RAIN", trial_name1)
                plot_threshold_cube(accumulations_rain1, accum, lead_time, cube_dt, "VERY_HEAVY_RAIN", trial_name1)
                LOGGER.info(f"plotted for {lead_time} in {trial_name1}")
            
                plot_threshold_cube(accumulations_rain2, accum, lead_time, cube_dt, "VERY_LIGHT_RAIN", trial_name2)
                plot_threshold_cube(accumulations_rain2, accum, lead_time, cube_dt, "LIGHT_RAIN", trial_name2)
                plot_threshold_cube(accumulations_rain2, accum, lead_time, cube_dt, "MODERATE_RAIN", trial_name2)
                plot_threshold_cube(accumulations_rain2, accum, lead_time, cube_dt, "HEAVY_RAIN", trial_name2)
                plot_threshold_cube(accumulations_rain2, accum, lead_time, cube_dt, "VERY_HEAVY_RAIN",  trial_name2)
                LOGGER.info(f"plotted for {lead_time} in {trial_name2}")
            
            if RAW_MODEL_OUTPUT:
                plot_model_output(accumulations_rain1, accum, lead_time, cube_dt, trial_name1)
                LOGGER.info(f"plotted raw output for {lead_time} in {trial_name1}")
                plot_model_output(accumulations_rain2, accum, lead_time, cube_dt, trial_name2)
                LOGGER.info(f"plotted raw output for {lead_time} in {trial_name2}")
            
            if RAW_MODEL_DIFF:
                plot_overall_diff(accumulations_rain1, accumulations_rain2, accum, cube_dt, lead_time, trial_name1, trial_name2)
            
            if THRESHOLD_DIFFS:
                plot_binary_diff_cube(accumulations_rain2, accumulations_rain1, accum, lead_time, cube_dt,
                                      f"{trial_name2}-{trial_name1}", trial_name1, trial_name2,
                                      "VERY_LIGHT_RAIN")
                plot_binary_diff_cube(accumulations_rain2, accumulations_rain1, accum, lead_time, cube_dt,
                                      f"{trial_name2}-{trial_name1}", trial_name1, trial_name2,
                                      "LIGHT_RAIN")
                plot_binary_diff_cube(accumulations_rain2, accumulations_rain1, accum, lead_time, cube_dt,
                                      f"{trial_name2}-{trial_name1}", trial_name1, trial_name2,
                                      "MODERATE_RAIN")
                plot_binary_diff_cube(accumulations_rain2, accumulations_rain1, accum, lead_time, cube_dt,
                                      f"{trial_name2}-{trial_name1}", trial_name1, trial_name2,
                                      "HEAVY_RAIN")
                plot_binary_diff_cube(accumulations_rain2, accumulations_rain1, accum, lead_time, cube_dt,
                                      f"{trial_name2}-{trial_name1}", trial_name1, trial_name2,
                                      "VERY_HEAVY_RAIN")
                LOGGER.info(f"plotted for {lead_time} in {trial_name2} - {trial_name1} diffs")
            
                plot_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                      f"{trial_name1}-{trial_name2}", trial_name1, trial_name2,
                                      "VERY_LIGHT_RAIN")
                plot_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                      f"{trial_name1}-{trial_name2}", trial_name1, trial_name2,
                                      "LIGHT_RAIN")
                plot_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                      f"{trial_name1}-{trial_name2}", trial_name1, trial_name2,
                                      "MODERATE_RAIN")
                plot_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                      f"{trial_name1}-{trial_name2}", trial_name1, trial_name2,
                                      "HEAVY_RAIN")
                plot_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                      f"{trial_name1}-{trial_name2}", trial_name1, trial_name2,
                                      "VERY_HEAVY_RAIN")
                LOGGER.info(f"plotted for {lead_time} in {trial_name1} - {trial_name2} diffs")
            
            if THRESHOLD_DIFFS_BOTH:
                plot_both_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                           "VERY_LIGHT_RAIN", trial_name1, trial_name2)
                plot_both_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                           "LIGHT_RAIN", trial_name1, trial_name2)
                plot_both_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                           "MODERATE_RAIN", trial_name1, trial_name2)
                plot_both_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                           "HEAVY_RAIN", trial_name1, trial_name2)
                plot_both_binary_diff_cube(accumulations_rain1, accumulations_rain2, accum, lead_time, cube_dt,
                                           "VERY_HEAVY_RAIN", trial_name1, trial_name2)
                LOGGER.info(f"plotted for {lead_time} in dual diffs")

            if INDEX_DIFFS:
                plot_masked_index_diffs(accumulations_rain1, accumulations_rain2, lead_time, accum, cube_dt,
                                        f"{trial_name1}-{trial_name2}")

            if THRESHOLD_INDEX_DIFFS:
                plot_threshold_index_diffs(accumulations_rain1, accumulations_rain2, lead_time, accum, cube_dt,
                                           "VERY_LIGHT_RAIN", f"{trial_name1}-{trial_name2}")
                plot_threshold_index_diffs(accumulations_rain1, accumulations_rain2, lead_time, accum, cube_dt,
                                           "LIGHT_RAIN", f"{trial_name1}-{trial_name2}")
                plot_threshold_index_diffs(accumulations_rain1, accumulations_rain2, lead_time, accum, cube_dt,
                                           "MODERATE_RAIN", f"{trial_name1}-{trial_name2}")
                plot_threshold_index_diffs(accumulations_rain1, accumulations_rain2, lead_time, accum, cube_dt,
                                           "HEAVY_RAIN", f"{trial_name1}-{trial_name2}")
                plot_threshold_index_diffs(accumulations_rain1, accumulations_rain2, lead_time, accum, cube_dt,
                                           "VERY_HEAVY_RAIN", f"{trial_name1}-{trial_name2}")

        LOGGER.info("main() finished running!")


if __name__ == "__main__":
    main()
