# Self-driving laboratories can advance the Pareto front for thin-film materials
Connor C. Rupnow<sup>1</sup>, Benjamin P. MacLeod<sup>2,</sup>, Mehrdad Mokhtari<sup>2</sup>, Karry Ocean<sup>2</sup>, Kevan E. Dettelbach<sup>2</sup>,     Daniel Lin<sup>2,</sup>, Fraser G. L. Parlane<sup>2,</sup>\*, Michael B. Rooney<sup>2</sup>, Hsi N. Chiu<sup>2</sup>, Christopher E. B. Waizenegger<sup>2</sup>,  & Curtis P. Berlinguette<sup>1,2,3,4,†</sup>

<sup>1</sup>Department of Chemical and Biological Engineering, The University of British Columbia, 2360 East Mall, Vancouver, BC V6T 1Z3, Canada. \
<sup>2</sup>Department of Chemistry, The University of British Columbia, 2036 Main Mall, Vancouver, BC V6T 1Z1, Canada. \
<sup>3</sup>Stewart Blusson Quantum Matter Institute, The University of British Columbia, 2355 East Mall, Vancouver, BC V6T 1Z4, Canada. \
<sup>4</sup>Canadian Institute for Advanced Research (CIFAR), MaRS Centre, 661 University Avenue Suite 505, Toronto, ON M5G 1M1, Canada.
  
<sup>†</sup>Corresponding author.

## data organization

In this work, an optimization was performed using a self-driving laboratory. Following the optimization, the champion material was scaled up and deposited on a substrate 8x larger. 

The data from the optimization can be found in the optimization campaign data folder. The self-driving lab performed 91 unique experiments in duplicate (except for some samples that failed) and created 179 individual Pd film samples. The self-driving lab was stopped and started five times over the course of the optimization due to various minor issues (e.g. software error, robot arm crashes into something, out of pipettes, etc.). The raw sample data from the opimization can be found in the folder "raw optimization campaign data" in the subfolders:

* `2022-07-11_12-55-37` (samples 0-29, random)
* `2022-07-12_10-41-55` (samples 0-29, random)
* `2022-07-13_09-53-56` (samples 0-29, random)
* `2022-07-13_23-11-10` (samples 0-29, random)
* `2022-07-14_14-54-21` (samples 0-29, random)

The data contained within these folders is shared with you as it is taken directly from the self-driving laboratory. Opening a sample folder within one of these dated folders will provide you with processed and unprocessed conductivity, microscope, XRF, and camera images and data. There is also a sample_log.log that records amounts, timestamps, and other important events that happen to each indiviudal sample. The raw data is porcessed and contained within the 'data_pipeline' folder.

The important inputs, outputs, measurements, and timestamps for each sample have been compiled into a csv contained within the main 'optimization campaign data' folder; called compiled_optimization_data.csv. Each row of the csv file is an individual sample and each sample has the following columns: 
* 'sample': the unique sample identifier (in the order they were made)
* 'concentration_realized': the total concetration of the precursor ink (g/mL)
* 'DMSO_content_realized': the relative amount of DMSO in the precursor ink (v/v)
* 'combustion_temp_realized': the temperature of the hotplate fixture surface as measured by a thermocouple (°C)
* 'air_flow_rate_realized': the relative amount that the airflow valve was open (%)
* 'spray_flow_rate_realized': the flowrate of the ink out of the spray nozzle as determined by the syringe pump (mL/s)
* 'spray_height_realized': the height of the nozzle above the substrate (mm)
* 'num_passes_realized': the number of times the spraycoater would repeat the spraycoating pattern over the substrate
* 'concentration_requested'
* DMSO_content_requested
* combustion_temp_requested
* air_flow_rate_requested
* spray_flow_rate_requested
* spray_height_requested
* num_passes_requested
* Pd_ACN_robot_realized
* acac_ACN_robot_realized
* ACN_robot_realized
* DMSO_robot_realized
* Pd_ACN_robot_requested
* acac_ACN_robot_requested
* ACN_robot_requested
* DMSO_robot_requested
* conductance_mean
* conductance_std
* conductive_fraction
* thickness_avg
* thickness_std
* sheet_conductance_avg
* sheet_conductance_std
* sheet_resistance_avg
* sheet_resistance_std
* conductivity_avg
* conductivity_std
* resistivity_avg
* resistivity_std
* campaign_ID
* exp_num
* running_best_conductivity
* beta
* nozzle_speed
* SAMPLE_START	
* MIX_CHEMICALS_START	
* SPRAY_COAT_START	
* XRF_START	
* MICROSCOPE_START	
* CONDUCTIVITY_START	
* FLIR_CAMERA_START	
* MIX_CHEMICALS_FINISH	
* SPRAY_COAT_FINISH	
* XRF_FINISH	
* MICROSCOPE_FINISH	
* CONDUCTIVITY_FINISH	
* FLIR_CAMERA_FINISH	
* SAMPLE_FINISH

 
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


  
