# A self-driving laboratory optimizes a scalable materials manufacturing process
Connor C. Rupnow<sup>1</sup>, Benjamin P. MacLeod<sup>2,</sup>, Mehrdad Mokhtari<sup>2</sup>, Karry Ocean<sup>2</sup>, Kevan E. Dettelbach<sup>2</sup>,     Daniel Lin<sup>2,</sup>, Fraser G. L. Parlane<sup>2,</sup>, Michael B. Rooney<sup>2</sup>, Hsi N. Chiu<sup>2</sup>, Christopher E. B. Waizenegger<sup>2</sup>,  & Curtis P. Berlinguette<sup>1,2,3,4,†</sup>

<sup>1</sup>Department of Chemical and Biological Engineering, The University of British Columbia, 2360 East Mall, Vancouver, BC V6T 1Z3, Canada. \
<sup>2</sup>Department of Chemistry, The University of British Columbia, 2036 Main Mall, Vancouver, BC V6T 1Z1, Canada. \
<sup>3</sup>Stewart Blusson Quantum Matter Institute, The University of British Columbia, 2355 East Mall, Vancouver, BC V6T 1Z4, Canada. \
<sup>4</sup>Canadian Institute for Advanced Research (CIFAR), MaRS Centre, 661 University Avenue Suite 505, Toronto, ON M5G 1M1, Canada.
  
<sup>†</sup>Corresponding author.

## data organization

In this work, the conductivity of spray-combustion synthesized palladium thin films was optimized using a self-driving laboratory. Following the optimization, the champion material was scaled-up and deposited on a substrate 8x larger using an ultrasonic spray-coater. The data for such experiments can be found here.

### Raw campaign data

The data from the optimization can be found in the optimization campaign data folder. The self-driving lab performed 91 unique experiments in duplicate (except for some samples that failed) and created 179 individual Pd film samples. The self-driving lab was stopped and started five times over the course of the optimization due to various minor issues (e.g. software error, robot arm crashes into something, out of pipettes, etc.). The raw sample data from the opimization can be found in the folder `raw optimization campaign data` in the subfolders:

* `2022-07-11_12-55-37` (samples 0-29, random)
* `2022-07-12_10-41-55` (samples 30-86, bayesian optimizer)
* `2022-07-13_09-53-56` (samples 87-116, bayesian optimizer)
* `2022-07-13_23-11-10` (samples 117-162, bayesian optimizer)
* `2022-07-14_14-54-21` (samples 163-179, bayesian optimizer)

The data contained within these folders is shared with you as it is taken directly from the self-driving laboratory. Opening a sample folder within one of these dated folders will provide you with processed and unprocessed conductivity, microscope, XRF, camera images and data. There is also a sample_log.log that records amounts, timestamps, and other important events that happen to each indiviudal sample. The processed data is contained within the `data_pipeline` folder.

### Compiled campaign data

The important inputs, outputs, measurements, and timestamps for each sample have been compiled into a CSV contained within the main `optimization campaign data` folder; called `compiled_optimization_data.csv`. Each row of the CSV file is an individual sample and each sample has the columns: 
* `sample`: unique sample identifier (in the order they were made)
* The following columns have `_requested` and `_realized` columns corresponding to the value of the parameter requested by the experimental planning algorithm (e.g. `combustion_temp_requested` = 250 °C) and the value of the parameter as measured by the sensors (e.g. `combustion_temp_realized` = 250.465 °C). Some parameters do not have sensors or cannot be measured with a sensor, thus the realized value is equal to the requested value (and are denoted by \*).
  * `concentration`: the total concentration of the precursor ink (g/mL)
  * `DMSO_content`: the relative amount of DMSO in the precursor ink (v/v)
  * `combustion_temp`: the temperature of the hotplate fixture surface as measured by a thermocouple (°C)
  * \*`air_flow_rate`: the relative amount that the airflow valve was open (%)
  * \*`spray_flow_rate`: the flowrate of the ink out of the spray nozzle as determined by the syringe pump (mL/s)
  * \*`spray_height`: the height of the nozzle above the substrate (mm)
  * \*`num_passes`: the number of times the spraycoater would repeat the spraycoating pattern over the substrate
  * `Pd_ACN_robot_realized`: the amount of palladium nitrate in acetonitrile stock solution in the precursor ink (mL)
  * `acac_ACN_robot_realized`: the amount of acetylacetone in acetonitrile stock solution in the precursor ink (mL)
  * `ACN_robot_realized`: the amount of acetonitrile in the precursor ink (mL)
  * `DMSO_robot_realized`: the amount of dimethylsulfoxide in the precursor ink (mL)
* The following measurements have `_avg` and `_std` columns corresponding to the average and standard deviation of the measurements
  * \**`conductance` 
  * `thickness`
  * `sheet_conductance`
  * `sheet_resistance`
  * `conductivity`
  * `resistivity`
  * \** note that for conductance, `_avg` is labelled as `_mean` instead
* `conductive_fraction`: the number of 4-point probe measurements (out of 5) that resulted in conductance greater than zero (expressed as a fraction)
* `campaign_ID`: the optimization campaign ID corresponding to when the sample was created. Each individual sammple can be found in a subfolder of the same name in the folder `raw optimization campaign data`
* `exp_num`: a unique identifier corresponding to duplicates with the same requested experimental conditions
* `running_best_conductivity`: the value of best conductivity at the time the sample was created
* `beta`: the sample-selection mode that the acquisition function was using to determine the experimental conditions for the sample. The first 30 samples (15 experiments) were selected randomly. The following samples were selected using an alternating acquisition mode. The four sampling modes were: upper confidence bound (UCB) beta = 0.2, UCB beta = 20, UCB beta = 400, space-filling (SF) point.
* `nozzle_speed`: the speed at which the nozzle moves while spraying. The nozzle speed is calculated by the equation `spray_flowrate`\*`num_lines`\*`spray_pattern_length`\*`num_passes`\/`spray_volume` in mm/s
* The following columns are timestamps corresponding to the time at which each task started and finished. Each columns has a `_START` and `_FINISH`.
  * `SAMPLE`
  * `MIX_CHEMICALS`
  * `SPRAY_COAT`
  * `XRF_START`
  * `MICROSCOPE`
  * `CONDUCTIVITY`
  * `FLIR_CAMERA`
  
The Bayesian optimizer used the manipulated variables (inputs) `concentration_realized`, `DMSO_content_realized`, `combustion_temp_realized`, `air_flow_rate_realized`, `spray_flow_rate_realized`, `spray_height_realized`, `num_passes_realized` and the responding variable (output) `conductivity_avg` to carry out the optimization.

### Scale-up experiment

CSV files of the `conductance`, `thickness`, and `conductivity` are shared for the 50mm x 25mm film and the 100mm x 100mm Pd films. Each CSV file is a function of position. Details on the measurements and positions can be found in the methods of the paper. 

### Supplementary figS7 dataset

A follow-up experiment was performed using the self-driving laboratory to confirm that DMSO can suppress the Leidenfrost effect. The dataset provided is similar to the `compiled_optimization_data.csv`, containing many of the same columns. This experiment was not an optimization but a grid experiments, showcasing the versatility of a self-driving lab for both hypothesis driven experiments and optimizations.
