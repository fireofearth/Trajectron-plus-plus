"""Plot the predictions of Trajectron++
"""

import sys
sys.path.append('../../trajectron')
import os
import numpy as np
import torch
import dill
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
import visualization

AGENT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
NCOLORS = len(AGENT_COLORS)

# Load nuScenes SDK
nuScenes_data_path = "/home/fireofearth/code/robotics/trajectron-plus-plus/experiments/nuScenes/v1.0"
# Data Path to nuScenes data set
nuScenes_devkit_path = './devkit/python-sdk/'
sys.path.append(nuScenes_devkit_path)
from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name='boston-seaport')

# Load dataset
with open('../processed/nuScenes_test_full.pkl', 'rb') as f:
    eval_env = dill.load(f, encoding='latin1')
eval_scenes = eval_env.scenes

# Load model
ph = 6
log_dir = './models'
model_dir = os.path.join(log_dir, 'int_ee_me')
eval_stg, hyp = load_model(
    model_dir, eval_env, ts=12)

ph = 6
num_samples = 100
# ptype = '_all_z_sep'
ptype = ''

def plot_scene_timestep(scene, t):
    timesteps = np.array([t])
    predictions = eval_stg.predict(scene,
            timesteps, ph, num_samples=num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=False,
                all_z_sep=False)

    v_nodes = list(filter(lambda k: 'VEHICLE' in repr(k), predictions[t].keys()))
    # print(v_nodes)

    prediction_dict, histories_dict, futures_dict = \
            prediction_output_to_trajectories(
                predictions, dt=scene.dt, max_h=10, ph=ph, map=None)


    plt.figure(figsize=(8, 8), dpi=80)
    for idx, node in enumerate(v_nodes):
        player_future = futures_dict[t][node]
        player_past = histories_dict[t][node]
        player_predict = prediction_dict[t][node]

        plt.plot(player_future[:,0], player_future[:,1],
                    marker='s', color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1, markersize=8, markerfacecolor='none')
        plt.plot(player_past[:,0], player_past[:,1],
                    marker='d', color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1, markersize=8, markerfacecolor='none')
        for row in player_predict[0]:
            plt.plot(row[:,0], row[:,1],
                    marker='o', color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1, alpha=0.1, markersize=4)
    savepath = 'plots/predict_scene{}_t{}{}.png'.format(scene.name, t, ptype)
    plt.savefig(savepath, dpi=100)
    plt.close('all')

# scenes_to_use = ['329']
# eval_scenes = list(filter(lambda s : s.name in scenes_to_use, eval_scenes))

with torch.no_grad():
    for scene in eval_scenes:
        print(f"Plotting scene {scene.name} ({scene.timesteps} timesteps)")
        for timestep in range(12, scene.timesteps - 6, 6):
            print(f"    timestep {timestep}")
            plot_scene_timestep(scene, timestep)

