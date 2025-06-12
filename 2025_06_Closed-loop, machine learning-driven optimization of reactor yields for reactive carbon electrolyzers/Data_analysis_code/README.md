# Reactor Yield Optimization - Data Visualization

This script provides a comprehensive suite of visualization tools to support the analysis and interpretation of data from closed-loop optimization experiments on reactive carbon electrolyzers. It generates publication-quality plots used in the manuscript titled:

**"Closed-loop, Machine Learning-Driven Optimization of Reactor Yields for Reactive Carbon Electrolyzers."**

## Overview

The script includes multiple functions for visualizing:

- Search space complexity by dimensionality.
- Trends in reactor yield and CO partial current density across experiments.
- Relationships between operating parameters, CO₂ utilization, and performance metrics.
- Contour plots and trendline-based analyses to highlight optimization progress.
- Combined subplot panels used in figure generation.

## Key Visualizations

- **Search Space Plot**: Number of optimization points vs. number of variables.
- **Reactor Yield Trends**: Yield evolution over experiments with annotations by optimization method.
- **J<sub>CO</sub> Trends**: Partial CO current density over time and vs. yield/utilization.
- **FECO vs. Input Current Density**: Efficiency profiles under varying operating conditions.
- **Contour Maps**: Reactor yield as a function of J<sub>CO</sub> and CO₂ utilization.
- **Multi-panel Figures**: Input parameters colored by RY or J<sub>CO</sub>.
