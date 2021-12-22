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

Make a folder called `models` if you want to save your models    
Make a folder called `ckpt-backups` to save checkpoints    

File `scripts/run.sh` gives an example of batch script you would use on Atlas.    
Example how to run it below. Replace account with your actual account (usually your project folder)    
`sbatch -A account-name run.sh`    

Here is the accuracy after training the model after adding a new classification layer (5 epochs) and then fine-tuning the last 250 layers (10 epochs), using 1 GPU node (2 GPUs total)

n classes | accuracy | fine-tuned accuracy | total time
----------|----------|---------------------|-----------
20        |0.698     | 0.811 | 11 min
101       |0.579     | 0.678 | 1 h 12 min
