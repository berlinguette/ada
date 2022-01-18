# Self-driving laboratory for accelerated discovery of thin-film materials: data repository

B. P. MacLeod<sup>1,3†</sup>, F. G. L. Parlane<sup>1,3†</sup>, T. D. Morrissey<sup>1,3</sup>, F. Häse<sup>4-7</sup>, L. M. Roch<sup>4-7</sup>, K. E. Dettelbach<sup>1</sup>, R. Moreira<sup>1</sup>, L. P. E. Yunker<sup>1</sup>, M. B. Rooney<sup>1</sup>, J. R. Deeth<sup>1</sup>, V. Lai<sup>1</sup>, G. J. Ng<sup>1</sup>, H. Situ<sup>1</sup>, R. H. Zhang<sup>1</sup>, M. S. Elliott<sup>1</sup>, T. H. Haley<sup>1</sup>, D. J. Dvorak<sup>3</sup>, A. Aspuru-Guzik<sup>4-8*</sup>, J. E. Hein<sup>1*</sup>, C. P. Berlinguette<sup>1-3,8*</sup>

<sup>1</sup>Department of Chemistry, The University of British Columbia, Vancouver, British Columbia, Canada  
<sup>2</sup>Department of Chemical & Biological Engineering, The University of British Columbia, Vancouver, British Columbia, Canada  
<sup>3</sup>Stewart Blusson Quantum Matter Institute, The University of British Columbia, Vancouver, British Columbia, Canada  
<sup>4</sup>Department of Chemistry and Chemical Biology, Harvard University, Cambridge, Massachusetts, USA  
<sup>5</sup>Department of Chemistry, University of Toronto, Toronto, Ontario, Canada  
<sup>6</sup>Department of Computer Science, University of Toronto, Toronto, Ontario, Canada  
<sup>7</sup>Vector Institute for Artificial Intelligence, MaRS Centre, Toronto, Ontario, Canada  
<sup>8</sup>Canadian Institute for Advanced Research (CIFAR), MaRS Centre, Toronto, Ontario, Canada  
<sup>†</sup>These authors contributed equally to this work

## Data organization

The data in this repository are organized in the following folder hierarchy:   
<p align='center'><i>run > sample > position</i></p>

**A run** (also referred to here as a *campaign*) is defined here as a series of robotic thin film synthesis experiments, each experiment producing one *sample*. Each of the two run folders included in this repository contains a folder for each sample. Data that is included at the run level includes:
* **campaign_parameters.json**: This file provides meta-data about the run such as chemical names, units, and vial locations.
* **campaign_log.log**: This file contains a time-series documentation of events that had been executed by the main Python run script.

**A sample** is defined here as a single 3" × 1" × 1 mm glass substrate, on which a thin film is deposited at some point during the experiment. Each sample folder contains a variety of experimental data and metadata recorded by the self-driving laboratory:

* **_experiment-parameters.json**: This file provides meta-data about the experiment such as the spin-coating conditions, dispensed solution masses, and mapping offsets (which indicate the positions at which spectroscopy and conductance measurements were made - see *Positions* below).
* **Images**: Images located within the sample folder are of the sample after solution has been deposited to the substrate. The image file name denotes whether the image had been taken before or after annealing of the sample. 
* **Position folders**: Position folders include all of the data that was measured at that indexed position.

A position is defined as a location on the sample where measurements were made. In the attached data, measurements are made at positions 0 to 6 mm from the geometric center of the 3” × 1” substrate along the long axis. Each position folder contains data files for the following measurements made at the associated position on the sample:

* **Ultraviolet–visible spectroscopy**: Reflection and transmission ultraviolet–visible spectroscopy data is measured at each position. To calculate the absorption of the film, we also measure reflection and transmission ultraviolet–visible spectroscopy data of the blank substrate.
* **Conductivity**: The conductivity file contains voltage readings across multiple currents.