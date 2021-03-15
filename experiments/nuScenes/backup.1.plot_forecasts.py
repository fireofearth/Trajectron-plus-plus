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

# Import to generate latents
from model.dataset import *
from model.components import *
from model.model_utils import *

AGENT_COLORS = ['blue', 'green', 'red', 'cyan',
                'magenta', 'yellow', 'orange',
                'dodgerblue', 'coral']
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
ptype = ''
# ptype = '_full_dist'
viewport_hw = 60

def plot_scene_timestep(scene, t):
    global nusc_map
    timesteps = np.array([t])

    # Run Trajectron++ predict
    predictions = eval_stg.predict(scene,
            timesteps, ph, num_samples=num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=False,
                all_z_sep=False)

    # Obtain, past, predict and ground truth predictions
    prediction_dict, histories_dict, futures_dict = \
        prediction_output_to_trajectories(
            predictions, dt=scene.dt, max_h=10, ph=ph, map=None)

    v_nodes = list(filter(lambda k: 'VEHICLE' in repr(k), predictions[t].keys()))
    v_nodes.sort(key=lambda k: 0 if 'ego' in repr(k) else 1)
    node = v_nodes[0]

    minpos = np.array([scene.x_min, scene.y_min])
    ego_lastpos = histories_dict[t][node][-1]
    ego_lastx = ego_lastpos[0]
    ego_lasty = ego_lastpos[1]
    center = np.array([
        scene.ego_initx + ego_lastx,
        scene.ego_inity + ego_lasty])

    center = minpos + ego_lastpos
    my_patch = (center[0] - viewport_hw, center[1] - viewport_hw,
                center[0] + viewport_hw, center[1] + viewport_hw)
    if scene.map_name != nusc_map.map_name:
        nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name=scene.map_name)
    fig, ax = nusc_map.render_map_patch(my_patch, scene.layer_names,
            figsize=(10, 10), alpha=0.2, render_egoposes_range=False)
    
    # fig, ax = plt.subplot
    # plt.figure(figsize=(8, 8), dpi=80)
    for idx, node in enumerate(v_nodes):
        player_future = futures_dict[t][node]
        player_past = histories_dict[t][node]
        player_predict = prediction_dict[t][node]
        player_future += minpos
        player_past += minpos
        player_predict += minpos

        ax.plot(player_future[:,0], player_future[:,1],
                    marker='s', color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1, markersize=8, markerfacecolor='none')
        ax.plot(player_past[:,0], player_past[:,1],
                    marker='d', color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1, markersize=8, markerfacecolor='none')
        for row in player_predict[0]:
            ax.plot(row[:,0], row[:,1],
                    marker='o', color=AGENT_COLORS[idx % NCOLORS],
                    linewidth=1, alpha=0.1, markersize=4)

    savepath = 'plots/predict_scene{}_t{}{}_overhead.png'.format(scene.name, t, ptype)
    plt.savefig(savepath, dpi=250)
    plt.close('all')

def plot_latents_scene_timestep(
            scene, t,
            num_samples = 200,
            z_mode=False, gmm_mode = False, full_dist = False, all_z_sep = False):
    timesteps = np.array([t])
    # Trajectron.predict() arguments
    min_future_timesteps = 0
    min_history_timesteps = 1

    node_type = eval_stg.env.NodeType.VEHICLE
    if node_type not in eval_stg.pred_state:
        raise Exception("fail")

    model = eval_stg.node_models_dict[node_type]

    # Get Input data for node type and given timesteps
    batch = get_timesteps_data(env=eval_stg.env, scene=scene, t=timesteps, node_type=node_type, state=eval_stg.state,
                               pred_state=eval_stg.pred_state, edge_types=model.edge_types,
                               min_ht=min_history_timesteps, max_ht=eval_stg.max_ht, min_ft=min_future_timesteps,
                               max_ft=min_future_timesteps, hyperparams=eval_stg.hyperparams)
    # There are no nodes of type present for timestep
    if batch is None:
        raise Exception("fail")

    (first_history_index,
     x_t, y_t, x_st_t, y_st_t,
     neighbors_data_st,

     neighbors_edge_value,
     robot_traj_st_t,
     map), nodes, timesteps_o = batch

    x = x_t.to(eval_stg.device)
    x_st_t = x_st_t.to(eval_stg.device)
    if robot_traj_st_t is not None:
        robot_traj_st_t = robot_traj_st_t.to(eval_stg.device)
    if type(map) == torch.Tensor:
        map = map.to(eval_stg.device)

    # MultimodalGenerativeCVAE.predict() arguments
    inputs = x
    inputs_st = x_st_t
    first_history_indices = first_history_index
    neighbors = neighbors_data_st
    neighbors_edge_value = neighbors_edge_value
    robot = robot_traj_st_t
    prediction_horizon = ph

    mode = ModeKeys.PREDICT

    x, x_nr_t, _, y_r, _, n_s_t0 = model.obtain_encoded_tensors(mode=mode,
                                                               inputs=inputs,
                                                               inputs_st=inputs_st,
                                                               labels=None,
                                                               labels_st=None,
                                                               first_history_indices=first_history_indices,
                                                               neighbors=neighbors,
                                                               neighbors_edge_value=neighbors_edge_value,
                                                               robot=robot,
                                                               map=map)

    model.latent.p_dist = model.p_z_x(mode, x)
    z, num_samples, num_components = model.latent.sample_p(num_samples,
                                                          mode,
                                                          most_likely_z=z_mode,
                                                          full_dist=full_dist,
                                                          all_z_sep=all_z_sep)
    
    # after getting latents, Trajectron calls model.p_y_xz(). This part is not included here.
    
    # finally, plot the latents
    znp = z.detach().numpy()
    znp_counts = np.sum(znp, axis=0) / num_samples

    fig, ax = plt.subplots()
    ax.set_facecolor("grey")
    for idx, zz in enumerate(znp_counts):
        ax.plot(range(25), zz, c=AGENT_COLORS[idx % NCOLORS])
    n_vehicles = znp_counts.shape[0]
    ax.set_xlabel("z = value")
    ax.set_ylabel("p(z = value)")
    ax.set_title(f"Dist. latent values from {n_vehicles} vehicles ")
    savepath = 'plots/predict_scene{}_t{}{}_latents.png'.format(scene.name, t, ptype)
    plt.savefig(savepath, dpi=80)
    plt.close('all')

## To select specific scenes to plot
# scenes_to_use = ['329']
# eval_scenes = list(filter(lambda s : s.name in scenes_to_use, eval_scenes))

with torch.no_grad():
    for scene in eval_scenes:
        print(f"Plotting scene {scene.name} ({scene.timesteps} timesteps)")
        for timestep in range(2, scene.timesteps - 6, 3):
            print(f"    timestep {timestep}")
            plot_scene_timestep(scene, timestep)
            plot_latents_scene_timestep(scene, timestep)

