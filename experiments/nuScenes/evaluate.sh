#!/bin/bash

#	--data ../processed/nuScenes_test_mini_full.pkl \
MODEL_NAME=models_02_Mar_2021_22_37_40_int_ee
TAG_NAME=02_Mar_2021_22_37_40_int_ee
python evaluate.py \
	--model models/$MODEL_NAME \
	--data ../processed/nuScenes_test_full.pkl \
	--checkpoint=20 \
	--output_path results \
	--output_tag $TAG_NAME \
	--node_type VEHICLE \
	--prediction_horizon 6
