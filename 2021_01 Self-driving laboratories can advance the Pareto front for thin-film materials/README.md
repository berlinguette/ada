# Self-driving laboratories can advance the Pareto front for thin-film materials
Benjamin P. MacLeod<sup>1,2,</sup>\*, Fraser G. L. Parlane<sup>1,2,</sup>\*, Kevan E. Dettelbach<sup>1</sup>, Michael S. Elliott<sup>1</sup>, Connor C. Rupnow<sup>1,2,3</sup>, Thomas D. Morrissey<sup>1,2</sup>, Ted H. Haley<sup>1</sup>, Oleksii Proskurin<sup>1</sup>, Michael B. Rooney<sup>1</sup>, Nina Taherimakhsousi<sup>1</sup>, David J. Dvorak<sup>2</sup>, Hsi N. Chiu<sup>1</sup>, Christopher E. B. Waizenegger<sup>1</sup>, Karry Ocean<sup>1</sup>, & Curtis P. Berlinguette<sup>1,2,3,4,†</sup>

<sup>1</sup>Department of Chemistry, The University of British Columbia, 2036 Main Mall, Vancouver, BC V6T 1Z1, Canada. \
<sup>2</sup>Stewart Blusson Quantum Matter Institute, The University of British Columbia, 2355 East Mall, Vancouver, BC V6T 1Z4, Canada. \
<sup>3</sup>Department of Chemical and Biological Engineering, The University of British Columbia, 2360 East Mall, Vancouver, BC V6T 1Z3, Canada. \
<sup>4</sup>Canadian Institute for Advanced Research (CIFAR), MaRS Centre, 661 University Avenue Suite 505, Toronto, ON M5G 1M1, Canada.
  
\*These authors contributed equally to this work. \
<sup>†</sup>Corresponding author.

## Processed data organization
In this work, four individual optimizations were performed on the self-driving laboratory. The experiment inputs and outputs from those optimizations (called campaigns) are available in `processed_data/`:
 * `campaign 2020-12-18_17-38-40.csv`
 * `campaign 2020-12-23_17-06-50.csv`
 * `campaign 2021-01-04_08-37-39.csv`
 * `campaign 2020-01-12_16-26-56.csv`
 
Each campaign is composed of several samples (or ordered rows). Each campaign CSV file has the following columns: 
 * `sample`: The ordered sample number 
 * `x0: fuel to oxidizer ratio`: The ratio of fuel to oxidizer, where 0 is no fuel, and 1 is a 1:1 ratio. 
 * `x1: acac amount`: The glycine to acetylacetone composition, where 0 is pure glycine and 1 is pure acetylacetone.
 * `x2: total concentration`: The total concetration of the precursor solution (g mL<sup>-1</sup>).
 * `x3: temperature`: The temperature at which the film was annealed (Celcius).
 * `conductance - mean`: The conductance per position on the sample - mean (Siemens).
 * `conductance - std`: The conductance per position on the sample - standard deviation (Siemens).
 * `XRF-normalized conductance - mean`: The XRF-normalized conductance - mean (S cps<sup>-1</sup>). 
 * `XRF-normalized conductance - std`: The XRF-normalized conductance - standard deviation (S cps<sup>-1</sup>).
 * `Conductivity - mean`: The conductivity of the film - mean (S m<sup>-1</sup>).

## Raw data organization

The `raw_data/` directory contains four optimizations (or campaigns). These optimizations are labelled `%Y-%m-%d_%H-%M-%S` (see [ref](https://strftime.org/)) and contain all the raw data collected for each optimization. Each optimization contains many samples, labelled `sample_x`. Each sample directory contains directories for each measurement taken, including `CONDUCTIVITY`, `IMAGES`, and `SENSOR` for oven data. Note that the raw XRF data has not been included due to space limitations. Please communicate with the authors if the data is required.

The processed data for this campaign is in the folder prepended with `_results`.

Additional `.json` and `.log` files contain sample- and optimization-level metadata, such as instrument configuration and experimental parameters.

