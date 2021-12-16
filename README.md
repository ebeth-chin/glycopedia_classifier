## README for Glycopedia Classifier   

This pipeline was designed on the USDA SCINet [Atlas cluster](https://www.hpc.msstate.edu/computing/atlas/). The purpose of this work is to optimize pipeline for image classification for eventual use on the Glycopedia Image Database (Glycopedia ImageDB).

This work was supported by the United States Department of Agriculture (USDA)/NSF AI Institute for Next Generation Food Systems (AIFS), USDA award number 2020-67021-32855.

### Installation

This code uses python 3.8 and Tensorflow 2.4.1   

You can build the conda environment using the .yml file:    

`conda env create -f corefoods.yml`

and activate it:

`conda activate corefoods`

### Data

Download the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) to the `data/` folder
Then run `scripts/food101_split.py` to split Food-101 into train/val/test (80/10/10) sets, and mini-datasets with 3 and 20 classes of foods
