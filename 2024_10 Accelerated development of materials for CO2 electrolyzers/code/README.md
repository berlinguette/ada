# XRF Data Processing Scripts

This folder contains utility scripts to process raw XRF files (BCF files) into heatmap images. The scripts work with hyperspectral data from the Bruker Tornado M4, converting it into a format suitable for analysis and visualization.

## Scripts Overview

1. **main_xrf.py**
   - This is the main script that utilizes `HyperPos` from the `hyperspec` module to convert raw BCF files into PNG images.
   - It reads each BCF file, processes it, and generates corresponding heatmap images of elemental distributions.
   
2. **hyperspy_mod.py**
   - This script contains utility functions that are used within `hyperspec.py` to read Bruker BCF files.
   - It handles loading the raw data into the appropriate class structure for further processing.

3. **hyperspec.py**
   - This script processes the hyperspectral data for a single map region (i.e., position) from the Bruker Tornado M4.
   - It is responsible for handling the data once loaded, managing the hyperspectral analysis, and preparing it for conversion into heatmap images.

## How to Use

To generate heatmap images from the XRF data:

1. Ensure that all scripts are located in the same directory.
2. Prepare your raw BCF files in a separate folder.
3. **Update the `dst` variable in `main_xrf.py`** to point to the actual directory path where your raw BCF files are stored. For example:
   ```python
   dst = "/path/to/your/bcf_files/"
   ```
4. Run the `main_xrf.py` script:
   ```bash
   python main_xrf.py
   ```

## Dependencies
   - Python 3.x
   - Required libraries:
      - matplotlib
      - pandas
      - numpy
      - dask

## Output
The output PNG images will display heatmaps representing the distribution of elements such as Ag, Cu, Cl, and S, as extracted from the raw XRF data.
