#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Visualization for Reactor Yield Optimization

This script contains functions to generate various plots used in the analysis of reactor yield and partial current density
optimization experiments. The plots include search space visualization, reactor yield vs. experiment
number, CO partial current density vs. experiment number, and others.

Each function is self-contained and reads data from CSV files located in the same directory as this script.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches


def plot_search_space():
    """
    Plot the search space as a function of the number of variables.
    """
    size_1 = 15
    size_2 = 18

    # Define the number of variables and corresponding deltas and precisions
    x = [1, 2, 3, 4, 5, 6]
    delta = [300, 160, 24, 55, 2.95, 2.55]
    percision = [60, 32, 5, 11, 0.6, 0.5]
    result = [round(d / p) for d, p in zip(delta, percision)]

    # Calculate the cumulative product to represent the search space
    result_new = []
    product = 1
    for value in result:
        product *= value
        result_new.append(product)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(x, result_new, color='black', s=200)
    plt.plot(x, result_new, color='black', linewidth=2)
    plt.yscale('log')

    # Labels and formatting
    plt.xlabel("Number of variables", fontsize=size_2)
    plt.ylabel("Number of points (search space)", fontsize=size_2)
    plt.xlim([0, x[-1]*1.05])
    plt.ylim([0, result_new[-1]*10])

    # Style adjustments
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=size_1)
    ax.tick_params(axis='y', labelsize=size_1)

    # Save and display the plot
    plt.savefig('Search_space.png', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_reactor_yield_vs_experiment_number():
    """
    Plot reactor yield versus experiment number, highlighting different experimental types.
    """
    size_1 = 20
    size_2 = 25

    # Read the data
    data = pd.read_csv('Data_analysis_RY.csv', delimiter=',')

    # Extract data
    x = data['Expmt #']
    y = data['Reactor yield']
    types = data['Type']

    # Mapping for labels
    _labels = {
        'Initialization': 'Random',
        'RY_Optimization': 'BO',
        'RY_Space_filling': 'SF',
        'RY_Human_in_the_loop': 'HIL'
    }

    # Plotting
    plt.figure(figsize=(11, 8))
    plt.scatter(x, y, alpha=1, s=200, zorder=3)

    # Labels and formatting
    plt.xlabel("Experiment number", fontsize=size_2)
    plt.ylabel("Reactor yield (mA cm$^{-2}$)", fontsize=size_2)
    plt.xlim([0, x.max() * 1.1])
    plt.ylim([0, y.max() * 1.1])

    # Style adjustments
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=size_1)
    ax.tick_params(axis='y', labelsize=size_1)
    ax.set_yticks(np.arange(0, 50, 10))

    # Annotate regions and draw lines
    start_idx = 0
    for i in range(1, len(types)):
        if types.iloc[i] != types.iloc[i-1]:
            midpoint_x = (x.iloc[start_idx] + x.iloc[i-1]) / 2
            plt.annotate(_labels[types.iloc[i-1]], (midpoint_x, y.max() * 1.08),
                         textcoords="offset points", xytext=(0, 10), ha='center',
                         fontsize=size_1, color='black', backgroundcolor='white')
            plt.axvline(x=x.iloc[i] - 0.5, color='gray', linestyle='--', linewidth=1)
            start_idx = i
    midpoint_x = (x.iloc[start_idx] + x.iloc[-1] + 5) / 2
    plt.annotate(_labels[types.iloc[-1]], (midpoint_x, y.max() * 1.08),
                 textcoords="offset points", xytext=(0, 10), ha='center',
                 fontsize=size_1, color='black', backgroundcolor='white')

    # Connect specific points with lines
    points_to_connect = [1, 2, 6, 7, 13, 36]
    for i in range(len(points_to_connect) - 1):
        start = points_to_connect[i]
        end = points_to_connect[i + 1]
        y_start = y[x == start].values[0]
        y_end = y[x == end].values[0]
        plt.plot([start, end], [y_start, y_start], color='orange', linestyle='-', linewidth=4.5)
        if y_start != y_end:
            plt.plot([end, end], [y_start, y_end], color='orange', linestyle='-', linewidth=4.5)
    last_point = points_to_connect[-1]
    y_last = y[x == last_point].values[0]
    plt.plot([last_point, x.max()], [y_last, y_last], color='orange', linestyle='-', linewidth=4.5)

    # Standard condition line
    plt.axhline(y=11.1, color='gray', linestyle='-')
    plt.text(45, 9, "Standard condition", rotation=0, fontsize=size_1, color="black")

    # Highlight the highest reactor yield point
    max_yield = y.max()
    max_yield_exp = x[y.idxmax()]
    plt.scatter(max_yield_exp, max_yield, color='red', marker='*', s=600, zorder=4)
    plt.annotate(f"Max RY: {max_yield:.2f}", (max_yield_exp, max_yield),
                 textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=size_1, color='red', fontweight='bold')

    # Save and display the plot
    plt.savefig('RY_plot.png', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_co_partial_current_density_vs_experiment_number():
    """
    Plot CO partial current density versus experiment number, highlighting reactor yield.
    """
    size_1 = 18
    size_2 = 24

    # Read the data
    data = pd.read_csv('Data_analysis_JCO.csv', delimiter=',')

    # Extract data
    x = data['Expmt #']
    y = data['CO partial current density']
    color_var = data['Reactor yield']
    types = data['Type']

    # Mapping for labels
    _labels = {
        'Initialization': 'RY Optimization (prior knowledge)',
        'Optimization': 'Bayesian Optimization (new campaign)'
    }

    # Plotting
    plt.figure(figsize=(14, 8))
    sc = plt.scatter(x, y, c=color_var, alpha=0.8, cmap='viridis', s=200)

    # Labels and formatting
    plt.xlabel("Experiment number", fontsize=size_2)
    plt.ylabel("J$_{CO}$ (mA cm$^{-2}$)", fontsize=size_2)
    plt.xlim([0, x.max() * 1.1])
    plt.ylim([0, y.max() * 1.1])

    # Colorbar
    cbar = plt.colorbar(sc, ticks=np.arange(0, 40, 10))
    cbar.outline.set_visible(False)
    cbar.set_label('Reactor yield (mA cm$^{-2}$)', fontsize=size_2)
    cbar.ax.tick_params(labelsize=size_2 - 2)

    # Style adjustments
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=size_2 - 2)
    ax.tick_params(axis='y', labelsize=size_2 - 2)
    ax.set_yticks(np.arange(0, 155, 25))

    # Annotate regions and draw lines
    start_idx = 0
    for i in range(1, len(types)):
        if types.iloc[i] != types.iloc[i - 1]:
            midpoint_x = (x.iloc[start_idx] + x.iloc[i - 1]) / 2
            plt.annotate(_labels[types.iloc[i - 1]], (midpoint_x, y.max() * 1.08),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=size_1 - 3, color='black', backgroundcolor='white')
            plt.axvline(x=x.iloc[i] - 0.5, color='gray', linestyle='--', linewidth=1)
            start_idx = i
    midpoint_x = (x.iloc[start_idx] + x.iloc[-1] + 9) / 2
    plt.annotate(_labels[types.iloc[-1]], (midpoint_x, y.max() * 1.08),
                 textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=size_1 - 3, color='black', backgroundcolor='white')

    # Connect specific points with lines
    points_to_connect = [1, 2, 5, 6, 8, 23, 55, 56, 58]
    for i in range(len(points_to_connect) - 1):
        start = points_to_connect[i]
        end = points_to_connect[i + 1]
        y_start = y[x == start].values[0]
        y_end = y[x == end].values[0]
        plt.plot([start, end], [y_start, y_start], color='orange', linestyle='-', linewidth=3.5)
        if y_start != y_end:
            plt.plot([end, end], [y_start, y_end], color='orange', linestyle='-', linewidth=3.5)
    last_point = points_to_connect[-1]
    y_last = y[x == last_point].values[0]
    plt.plot([last_point, x.max()], [y_last, y_last], color='orange', linestyle='-', linewidth=3.5)

    # Save and display the plot
    plt.savefig('JCO_plot.png', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_FECO_vs_input_current_density():
    """
    Plot Faradaic efficiency for CO (FECO) versus input current density.
    """
    # Read the data
    data = pd.read_csv('Data_analysis_JCO.csv', delimiter=',')

    # Extract data
    x = data['Input Current density']
    y = data['FECO']
    types = data['Type']

    # Mapping for labels
    short_type_labels = {
        'Initialization': 'Initialization',
        'Optimization': 'Optimization: data obtained while optimizing for J$_{CO}$'
    }
    type_colors = {
        'Initialization': 'white',
        'Optimization': 'orange'
    }
    colors = types.map(type_colors)

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y, c=colors, alpha=1, s=300, edgecolor='black', linewidth=2)

    # Labels and formatting
    plt.xlabel("Input Current Density (mA cm$^{-2}$)", fontsize=20)
    plt.ylabel("FE$_{CO}$ (%)", fontsize=20)
    plt.xlim([0, x.max() * 1.1])
    plt.ylim([0, y.max() * 1.1])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    # Style adjustments
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=color, markersize=15)
               for color in type_colors.values()]
    labels = short_type_labels.values()
    plt.legend(handles, labels, frameon=False, loc='lower left', fontsize=16)

    # Ellipse and shaded area
    circle = patches.Ellipse((334, 36), width=46, height=18, edgecolor='black',
                             facecolor='none', linestyle='-', linewidth=2)
    ax.add_patch(circle)
    x1_start, y1_start = 30, 35
    x1_end, y1_end = 350, 3
    slope = (y1_end - y1_start) / (x1_end - x1_start)
    x_values = np.linspace(x1_start, x1_end, 100)
    y1_values = slope * (x_values - x1_start) + y1_start
    y_shift = 22
    y2_values = slope * (x_values - x1_start) + (y1_start + y_shift)
    ax.fill_between(x_values, y1_values, y2_values, color='lightgrey', alpha=0.25)

    # Save and display the plot
    plt.savefig('FE_vs_Input_Current.png', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_co2_utilization_vs_jco_contour():
    """
    Plot CO2 utilization versus CO partial current density with contour lines for reactor yield.
    """
    size_1 = 15
    size_2 = 18

    # Read the data
    data = pd.read_csv('Data_analysis_RY.csv', delimiter=',')

    x = data['CO partial current density']
    y = data['CO2 util']
    types = data['Type']

    # Generate mesh grid for contour
    x_mesh, y_mesh = np.meshgrid(np.linspace(1, 200, 201), np.linspace(1, 100, 101))
    z_mesh = x_mesh * y_mesh

    # Plotting
    plt.figure(figsize=(10, 10))
    contour_levels = np.arange(0, 120, 10)
    _contour = plt.contour(x_mesh, y_mesh, z_mesh / 100, levels=contour_levels,
                           colors='black', alpha=0.4)

    # Custom formatter for contour labels
    def custom_fmt(value):
        if value == contour_levels[8]:
            return 'Reactor yield (mA cm$^{-2}$)'
        if value == contour_levels[0]:
            return None
        return f'{value:.0f}'

    plt.clabel(_contour, inline=True, fontsize=size_2, fmt=custom_fmt)

    # Scatter plot with different types
    for t in np.unique(types):
        idx = types == t
        if t == 'RY_Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='orange', edgecolors='black', linewidth=2,
                        label='RY Optimization', zorder=3)
        else:
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='none', edgecolors='black', linewidth=2, zorder=3)

    # Annotations and labels
    plt.legend(frameon=False, fontsize=size_1, loc='upper right')
    plt.step(x, y, color='black', linestyle='-', linewidth=4)
    plt.xlim([0, 200])
    plt.ylim([0, 100])
    plt.xlabel("J$_{CO}$ (mA cm$^{-2}$)", fontsize=size_2)
    plt.ylabel("CO$_2$ utilization (%)", fontsize=size_2)
    plt.tick_params(axis='both', which='major', labelsize=size_1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save and display the plot
    plt.savefig('CO2_utilization_vs_JCO.png', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_co2_utilization_vs_jco_with_frontier(data_file, save_filename):
    """
    Plot CO2 utilization versus CO partial current density with frontier and different optimization types.
    """
    size_1 = 15
    size_2 = 18

    # Read the data
    data = pd.read_csv(data_file, delimiter=',')

    x = data['CO partial current density']
    y = data['CO2 util']
    types = data['Type']

    # Identify frontier points
    points = np.vstack((x, y)).T
    points = points[points[:, 0].argsort()]
    frontier = [points[0]]
    for point in points[1:]:
        while frontier and frontier[-1][1] < point[1]:
            frontier.pop()
        frontier.append(point)
    frontier_x = [p[0] for p in frontier]
    frontier_y = [p[1] for p in frontier]

    # Generate mesh grid for contour
    x_mesh, y_mesh = np.meshgrid(np.linspace(1, 200, 201), np.linspace(1, 100, 101))
    z_mesh = x_mesh * y_mesh

    # Plotting
    plt.figure(figsize=(10, 10))
    contour_levels = np.arange(0, 120, 10)
    _contour = plt.contour(x_mesh, y_mesh, z_mesh / 100, levels=contour_levels,
                           colors='black', alpha=0.4)

    # Custom formatter for contour labels
    def custom_fmt(value):
        if value == contour_levels[8]:
            return 'Reactor yield (mA cm$^{-2}$)'
        if value == contour_levels[0]:
            return None
        return f'{value:.0f}'

    plt.clabel(_contour, inline=True, fontsize=size_2, fmt=custom_fmt)

    # Scatter plot with legend
    for t in np.unique(types):
        idx = types == t
        if t == 'Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='purple', edgecolors='black', linewidth=2,
                        label='J$_{CO}$ Optimization', zorder=3)
        elif t == 'RY_Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='orange', edgecolors='black', linewidth=2,
                        label='RY Optimization', zorder=3)
        else:
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='none', edgecolors='black', linewidth=2,
                        label='Others', zorder=3)

    plt.legend(frameon=False, fontsize=size_1, loc='upper right')

    # Plot frontier
    plt.step(frontier_x, frontier_y, color='black', linestyle='-', linewidth=4)
    plt.xlim([0, 200])
    plt.ylim([0, 100])

    # Labels and formatting
    plt.xlabel("J$_{CO}$ (mA cm$^{-2}$)", fontsize=size_2)
    plt.ylabel("CO$_2$ utilization (%)", fontsize=size_2)
    plt.tick_params(axis='both', which='major', labelsize=size_1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save and display the plot
    plt.savefig(save_filename, dpi=1000, bbox_inches='tight')
    plt.show()


def plot_co2_utilization_vs_reactor_yield_with_trendline():
    """
    Plot CO2 utilization versus reactor yield with a trendline.
    """
    size_1 = 15
    size_2 = 18

    # Read the data
    data = pd.read_csv('Data_analysis_JCO_2.csv', delimiter=',')

    x = data['CO2 util']
    y = data['Reactor yield']
    types = data['Type']

    # Plotting
    plt.figure(figsize=(10, 10))
    for t in np.unique(types):
        idx = types == t
        if t == 'Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='purple', edgecolors='black', linewidth=2,
                        label='J$_{CO}$ Optimization', zorder=3)
        elif t == 'RY_Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='orange', edgecolors='black', linewidth=2,
                        label='RY Optimization', zorder=3)
        else:
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='none', edgecolors='black', linewidth=2,
                        label='Others', zorder=3)

    # Trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r-", label='Best fit line')

    plt.legend(frameon=False, fontsize=size_1, loc='upper right')
    plt.xlim([0, 70])
    plt.ylim([0, 50])

    # Labels and formatting
    plt.xlabel("CO$_2$ utilization (%)", fontsize=size_2)
    plt.ylabel("Reactor yield (mA cm$^{-2}$)", fontsize=size_2)
    plt.tick_params(axis='both', which='major', labelsize=size_1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save and display the plot
    plt.savefig('CO2_utilization_vs_Reactor_Yield.png', dpi=1000, bbox_inches='tight')
    plt.show()

    # Correlation
    correlation = data['CO2 util'].corr(data['Reactor yield'])
    print(f"Correlation between CO2 utilization and Reactor yield: {correlation:.4f}")


def plot_co_partial_current_density_vs_reactor_yield_with_trendline():
    """
    Plot CO partial current density versus reactor yield with a trendline.
    """
    size_1 = 15
    size_2 = 18

    # Read the data
    data = pd.read_csv('Data_analysis_JCO_2.csv', delimiter=',')

    x = data['CO partial current density']
    y = data['Reactor yield']
    types = data['Type']

    # Plotting
    plt.figure(figsize=(10, 10))
    for t in np.unique(types):
        idx = types == t
        if t == 'Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='purple', edgecolors='black', linewidth=2,
                        label='J$_{CO}$ Optimization', zorder=3)
        elif t == 'RY_Optimization':
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='orange', edgecolors='black', linewidth=2,
                        label='RY Optimization', zorder=3)
        else:
            plt.scatter(x[idx], y[idx], marker='o', alpha=1, s=200,
                        facecolors='none', edgecolors='black', linewidth=2,
                        label='Others', zorder=3)

    # Trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r-", label='Best fit line')

    plt.legend(frameon=False, fontsize=size_1, loc='upper right')
    plt.xlim([0, 160])
    plt.ylim([0, 50])

    # Labels and formatting
    plt.xlabel("CO partial current density (mA cm$^{-2}$)", fontsize=size_2)
    plt.ylabel("Reactor yield (mA cm$^{-2}$)", fontsize=size_2)
    plt.tick_params(axis='both', which='major', labelsize=size_1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save and display the plot
    plt.savefig('CO_partial_current_density_vs_Reactor_Yield.png', dpi=1000, bbox_inches='tight')
    plt.show()

    # Correlation
    correlation = data['CO partial current density'].corr(data['Reactor yield'])
    print(f"Correlation between CO partial current density and Reactor yield: {correlation:.4f}")


def plot_combined_RY_parameters():
    """
    Plot multiple subplots of various parameters versus experiment number, colored by reactor yield.
    """
    # Read the data
    data = pd.read_csv('Data_analysis_RY.csv', delimiter=',')

    list_param = ['Bicarb conc', 'Input Current density', 'Bicarbonate flow',
                  'KOH conc', 'KOH flow', 'Bicarb temp']
    list_param_full = ['Bicarbonate concentration (M)', 'Current density (mA/cm$^{2}$)',
                       'Bicarbonate flow rate (mL/min)', 'KOH concentration (M)',
                       'KOH flow rate (mL/min)', 'Bicarbonate temperature (℃)']

    _labels = {
        'Initialization': 'Random',
        'RY_Optimization': 'BO',
        'RY_Space_filling': 'SF',
        'RY_Human_in_the_loop': 'HIL'
    }

    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()

    for idx, input in enumerate(list_param):
        ax = axes[idx]
        color_ = 'Reactor yield'
        x = data['Expmt #']
        y = data[input]
        color_var = data[color_]
        types = data['Type']

        sc = ax.scatter(x, y, c=color_var, alpha=0.8, cmap='viridis', s=200)
        ax.text(-0.1, 1.1, panel_letters[idx], transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='right')
        ax.set_xlabel("Experiment number", fontsize=20)
        ax.set_ylabel(list_param_full[idx], fontsize=20)

        if idx % 2 == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(sc, cax=cax, label=color_)
            cbar.outline.set_visible(False)
            cbar.set_label('Reactor yield (mA cm$^{-2}$)', fontsize=20)
            cbar.ax.tick_params(labelsize=20)

        if idx not in [4, 5]:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Experiment number", fontsize=20)

        points_to_connect = [1, 2, 6, 7, 13, 37]
        for i in range(len(points_to_connect) - 1):
            start = points_to_connect[i]
            end = points_to_connect[i + 1]
            y_start = y[x == start].values[0]
            y_end = y[x == end].values[0]
            ax.plot([start, end], [y_start, y_start], color='orange', linestyle='-', linewidth=8)
            if y_start != y_end:
                ax.plot([end, end], [y_start, y_end], color='orange', linestyle='-', linewidth=8)
        last_point = points_to_connect[-1]
        y_last = y[x == last_point].values[0]
        ax.plot([last_point, x.max()], [y_last, y_last], color='orange', linestyle='-', linewidth=8)

        start_idx = 0
        for i in range(1, len(types)):
            if types.iloc[i] != types.iloc[i - 1]:
                midpoint_x = (x.iloc[start_idx] + x.iloc[i - 1]) / 2
                ax.annotate(_labels[types.iloc[i - 1]], (midpoint_x, y.max() * 1.01),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=20, color='black', backgroundcolor='white')
                ax.axvline(x=x.iloc[i] - 0.5, color='gray', linestyle='--', linewidth=1)
                start_idx = i
        midpoint_x = (x.iloc[start_idx] + x.iloc[-1] + 3) / 2
        ax.annotate(_labels[types.iloc[-1]], (midpoint_x, y.max() * 1.01),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=20, color='black', backgroundcolor='white')

        ax.set_xlim([0, x.max() * 1.1])
        ax.set_ylim([0, y.max() * 1.1])
        ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.savefig('Combined_RY_Plots.png', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_combined_JCO_parameters():
    """
    Plot multiple subplots of various parameters versus experiment number, colored by CO partial current density.
    """
    # Read the data
    data = pd.read_csv('Data_analysis_JCO.csv', delimiter=',')

    list_param = ['Input Current density', 'Bicarb conc', 'Bicarbonate flow',
                  'KOH conc', 'Bicarb temp', 'KOH flow']
    list_param_full = ['Current density (mA/cm$^{2}$)', 'Bicarbonate concentration (M)',
                       'Bicarbonate flow rate (mL/min)', 'KOH concentration (M)',
                       'Bicarbonate temperature (℃)', 'KOH flow rate (mL/min)']

    _labels = {
        'Initialization': 'Initialization',
        'Optimization': 'Optimization'
    }

    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()

    for idx, input in enumerate(list_param):
        ax = axes[idx]
        color_ = 'CO partial current density'
        x = data['Expmt #']
        y = data[input]
        color_var = data[color_]
        types = data['Type']

        sc = ax.scatter(x, y, c=color_var, alpha=0.8, cmap='viridis', s=200)
        ax.text(-0.1, 1.1, panel_letters[idx], transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='right')
        ax.set_xlabel("Experiment number", fontsize=20)
        ax.set_ylabel(list_param_full[idx], fontsize=20)

        if idx % 2 == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(sc, cax=cax, label=color_)
            cbar.outline.set_visible(False)
            cbar.set_label('CO partial current density (mA cm$^{-2}$)', fontsize=20)
            cbar.ax.tick_params(labelsize=20)

        if idx not in [4, 5]:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Experiment number", fontsize=20)

        # Annotate regions
        start_idx = 0
        for i in range(1, len(types)):
            if types.iloc[i] != types.iloc[i - 1]:
                midpoint_x = (x.iloc[start_idx] + x.iloc[i - 1]) / 2
                ax.annotate(_labels[types.iloc[i - 1]], (midpoint_x, y.max() * 1.01),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=20, color='black', backgroundcolor='white')
                ax.axvline(x=x.iloc[i] - 0.5, color='gray', linestyle='--', linewidth=1)
                start_idx = i
        midpoint_x = (x.iloc[start_idx] + x.iloc[-1] + 3) / 2
        ax.annotate(_labels[types.iloc[-1]], (midpoint_x, y.max() * 1.01),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=20, color='black', backgroundcolor='white')

        # Connect points
        points_to_connect = [1, 2, 5, 6, 8, 23, 55, 56, 58]
        for i in range(len(points_to_connect) - 1):
            start = points_to_connect[i]
            end = points_to_connect[i + 1]
            y_start = y[x == start].values[0]
            y_end = y[x == end].values[0]
            ax.plot([start, end], [y_start, y_start], color='orange', linestyle='-', linewidth=5)
            if y_start != y_end:
                ax.plot([end, end], [y_start, y_end], color='orange', linestyle='-', linewidth=5)
        last_point = points_to_connect[-1]
        y_last = y[x == last_point].values[0]
        ax.plot([last_point, x.max()], [y_last, y_last], color='orange', linestyle='-', linewidth=5)

        ax.set_xlim([0, x.max() * 1.1])
        ax.set_ylim([0, y.max() * 1.1])
        ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.savefig('Combined_JCO_Plots.png', dpi=1000, bbox_inches='tight')
    plt.show()


def main():
    """
    Main function to execute all plotting functions.
    """
    plot_search_space()
    plot_reactor_yield_vs_experiment_number()
    plot_co_partial_current_density_vs_experiment_number()
    plot_FECO_vs_input_current_density()
    plot_co2_utilization_vs_jco_contour()
    plot_co2_utilization_vs_jco_with_frontier('Data_analysis_JCO_2.csv', 'CO2_utilization_vs_JCO_2.png')
    plot_co2_utilization_vs_reactor_yield_with_trendline()
    plot_co_partial_current_density_vs_reactor_yield_with_trendline()
    plot_combined_RY_parameters()
    plot_combined_JCO_parameters()


if __name__ == "__main__":
    main()
