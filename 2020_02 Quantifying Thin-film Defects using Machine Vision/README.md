# Quantifying Thin-film Defects using Machine Vision

N. Taherimakhsousi<sup>1</sup>, B. P. MacLeod<sup>1,2</sup>, F. G. L. Parlane<sup>1,2</sup>, T. D. Morrissey<sup>1,2</sup>, N. P. Booker<sup>1</sup>, K. E. Dettelbach<sup>1</sup>, C. P. Berlinguette<sup>1-4</sup>*


<sup>1</sup>Department of Chemistry, The University of British Columbia, Vancouver, British Columbia, Canada  
<sup>2</sup>Stewart Blusson Quantum Matter Institute, The University of British Columbia, Vancouver, British Columbia, Canada  
<sup>3</sup>Department of Chemical & Biological Engineering, The University of British Columbia, Vancouver, British Columbia, Canada    
<sup>4</sup>Canadian Institute for Advanced Research (CIFAR), MaRS Centre, Toronto, Ontario, Canada

## Data organization
The complete experimental conditions in which this data was collected can be found in the methods section of the associated manuscript. The data is divided into two datasets: ​**Darkfield images**​ and ​**Brightfield images**​.

### 1. Darkfield images
Within the `Darkfield images`​folder, the data for the `Monotonic trend`​experiment and the `Organic thin films` can be found. Labels for these files can be found in `Darkfield images - labels.csv`.

#### 1.1 Monotonic trend
This folder is comprised of the unedited images taken from the monotonic dewetting experiment, as described in the manuscript. These images are suffixed with the sequence in which they were collected.

#### 1.2 Organic thin films
These images are categorized by their associated true label: `Cracks`, `Dewetting`, `NoCracks`, and `NoDewetting`. The labeled image segments can be found within these folders.

### 2. Brightfield images
This data was collected using an OLYMPUS LEXT OLS 3100 microscope operating in bright-field reflection mode. All brightfield images are divided into two folders based upon the material being imaged: `Metal oxide thin films`​and `Organic thin films`.

#### 2.1 Metal oxide thin films
The data is then divided by the objective power:​`5×` and `20×`. Within these folders, the data is categorized by their associated true label: `Cracks`, `No-defects`, `Non-uniform`, `Particles`, and `Scratches`.

#### 2.2 Organic thin films
This data contains micrographs collected at a variety of magnifications. Within this folder, the data is categorized by their associated true label: `Cracks`, `Dewetting`, `Particles`, and `Scratches`.

### 3. Labelling app
The app used by experts to label the images can be found in the `Labelling app` directory.

### 4. DeepThin model
The `DeepThin model` directory contains the model used to label the images.

### 5. License and Reuse
This data is licensed Creative Commons BY-NC-SA 4.0.
