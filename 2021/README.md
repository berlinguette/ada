## Data organization

The `data/optimizations` directory contains four optimizations (or campaigns). These optimizations are labelled `%Y-%m-%d_%H-%M-%S` (see [ref](https://strftime.org/)), and contain all the raw data collected for each optimization. Each optimization contains many samples, labelled `sample_x`. Each sample directory contains directories for each measurement taken, including `CONDUCTIVITY`, `IMAGES`, and `SENSOR` for oven data. Note that the raw XRF data has not been included due to space limitations. Please communicate with the authors if the data is required.

The processed data for this campaign is in the folder prepended with `_results`.

Additional `.json` and `.log` files contain sample- and optimization-level metadata, such as instrument configuration and experimental parameters.

