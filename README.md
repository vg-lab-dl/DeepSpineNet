# DeepSpineNet
This repository contains the code used for the deep learning part of our work. We developed our models and our training/testing environment using TensorFlow.

## Installation
Requires **Python 3.6.8 or later** and **CUDA 10.1** (to enable the use of GPUs, it can be used without CUDA, on CPU, with longer training/testing times.).
1. Download the current project:
   > git clone https://github.com/vg-lab-dl/DeepSpineNet.git
2. Install dependencies (from the root directory of the project):
   > pip install -r requirements.txt.

## Example of data used during training
You can download an example of data used to train our models from: https://bit.ly/3LSPtlQ

## Usage
For reproducibility purposes, we provide the configuration files ([M1.yaml](config_files/M1.yaml), [M2.yaml](config_files/M2.yaml) and [M3.yaml](config_files/M3.yaml)) used for the models described in our work. These files are in the [config_files](config_files) folder and can be used with the below instructions. 
### Training
To train a model, use the following command:
> python main.py -p=train -cf=config.yaml

### Testing
To test a model, use the following command:
> python main.py -p=test -cf=config.yaml

## Acknowledgments
The authors gratefully acknowledges the computer resources at Artemisa, funded by the European Union ERDF and Comunitat Valenciana as well as the technical support provided by the Instituto de Física Corpuscular, IFIC (CSIC-UV).

## License 
DeepSpineNet is distributed under a Dual License model, depending on its usage. For its non-commercial use, it is released under an open-source license (GPLv3). Please contact us if you are interested in commercial license.
