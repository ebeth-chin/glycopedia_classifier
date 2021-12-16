#######
# food101_split.py
# purpose: make train/dev/test (80/10/10) from the food-101 dataset, make smaller datasets (3 and 20 class)
#######

import argparse
import random
import os
import splitfolders
from shutil import copy, copytree, rmtree


def dataset_mini(food_list, src, dataset, dest):
	dest_path = os.path.join(dest, dataset)
	src_path = os.path.join(src, dataset)
	if os.path.exists(dest_path):
		rmtree(dest_path)
	if not os.path.exists(dest_path):
		os.mkdir(dest_path)
	for food_item in food_list:
		print("Copying images into", food_item)
		food_path = os.path.join(dest_path, food_item)
		copytree(os.path.join(src_path, food_item), food_path, dirs_exist_ok = True)

if __name__ == '__main__':
	food101_dir = './data/food-101/images/'
	out_folder = './data/food-101/'

	#split all the food101 data into 80:10:10
	splitfolders.ratio(food101_dir, output=out_folder, seed=1337, ratio=(.8, .1, .1), group_prefix=None)

	#now we will make a small dataset of just 3 food classes
	dest_mini = "./data/food-101/data_3class/"
	os.mkdir(dest_mini)
	food_list = ['samosa', 'omelette', 'pizza']

	dataset_mini(food_list = food_list, src = out_folder, dataset = "test", dest = dest_mini)
	dataset_mini(food_list = food_list, src = out_folder, dataset = "val", dest = dest_mini)
	dataset_mini(food_list = food_list, src = out_folder, dataset = "train", dest = dest_mini)

	#make medium sized dataset of 20 arbitrarily chosen classes
	dest20 = "./data/food-101/data_20class/"
	os.mkdir(dest20)
	food_list20 = ['waffles', 'tuna_tartare', 'tiramisu', 'seaweed_salad', 'samosa', 'ravioli', 'prime_rib', 'peking_duck', 'huevos_rancheros', 'frozen_yogurt', 'fried_rice', 'deviled_eggs', 'carrot_cake', 'cannoli', 'cesar_salad', 'bruschetta', 'beef_tartare', 'beef_carpaccio', 'baklava', 'apple_pie']

	dataset_mini(food_list = food_list20, src = out_folder, dataset = "test", dest = dest20)
	dataset_mini(food_list = food_list20, src = out_folder, dataset = "val", dest = dest20)
	dataset_mini(food_list = food_list20, src = out_folder, dataset = "train", dest = dest20)
