import numpy as np
from matplotlib.patches import Ellipse

def render_trajectories(coordinates, colors, plot_axis, ax):
    assert coordinates.ndim == 3, '''It is expected that coordinates are of size 
                                     [trajectories_qty x trajectory_length x coordinates_qty]'''
    assert coordinates.shape[2] == 2, 'It is expected that only two coordinates are tracked'

    min_ax, max_ax = np.min(coordinates[..., :2]), np.max(coordinates[..., :2])
    offset = (max_ax - min_ax) * 0.1
    square_axis_limits = np.array([min_ax - offset, max_ax + offset])
    for trajectory, color in zip(coordinates, colors):
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)
    if plot_axis == 'free':
        return
    if plot_axis is None:
        plot_axis = [square_axis_limits[0], square_axis_limits[1], square_axis_limits[0], square_axis_limits[1]]
    if plot_axis is not None:
        ax.axis(plot_axis)
        ax.set_aspect('equal')
    else:
        ax.axis('scaled')

def draw_2d_gaussians(means, stddevs, colors, ax, fill=False, alpha=0.8, use_adaptive_axis=False, draw_unit_gaussian=True, plot_axis=None):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)
    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(xy=(0, 0), width=2, height=2,
                          edgecolor='r', lw=1, facecolor='none', alpha=0.5)
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.]])
            stddev = np.concatenate([stddev, [0.1]])
        ellipse = Ellipse(xy=mean, width=stddev[0] * 2, height=stddev[1] * 2,
                          edgecolor=color, lw=1, facecolor='none' if not fill else color, alpha=alpha)
        ax.add_patch(ellipse)
        square_axis_limit = max(
                square_axis_limit,
                np.abs(mean[0] + stddev[0]),
                np.abs(mean[0] - stddev[0]),
                np.abs(mean[1] + stddev[1]),
                np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis('scaled')
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)

def calc_eval_metrics(coordinates, discretize_continuous_fn):
    eval_metrics = {}
    uniq_coords = np.unique(discretize_continuous_fn(coordinates).reshape(-1, coordinates.shape[-1]).astype(np.int32), axis=0)
    eval_metrics.update({
        'MjNumUniqueCoords': len(uniq_coords),
    })
    return eval_metrics