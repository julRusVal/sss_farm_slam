#!/usr/bin/env python3

import math
import numpy as np
import gtsam
import networkx as nx
import matplotlib.pyplot as plt


# ===== Genral stuff =====
def angle_between_rads(target_angle, source_angle):
    # Bound the angle [-pi, pi]
    target_angle = math.remainder(target_angle, 2 * np.pi)
    source_angle = math.remainder(source_angle, 2 * np.pi)

    diff_angle = target_angle - source_angle

    if diff_angle > np.pi:
        diff_angle = diff_angle - 2 * np.pi
    elif diff_angle < -1 * np.pi:
        diff_angle = diff_angle + 2 * np.pi

    return diff_angle


# ===== GTSAM Stuff =====
def create_Pose2(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
    rot3_yaw = rot3.yaw()
    # GTSAM Pose2: x, y, theta
    return gtsam.Pose2(input_pose[0], input_pose[1], rot3_yaw)


def pose2_list_to_nparray(pose_list):
    out_array = np.zeros((len(pose_list), 3))

    for i, pose2 in enumerate(pose_list):
        out_array[i, :] = pose2.x(), pose2.y(), pose2.theta()

    return out_array


def show_simple_graph_2d(graph, x_keys, b_keys, values, label, show_final=True):
    """
    Show Graph of Pose2 and Point2 elements
    This function does not display data association colors

    """
    plot_limits = [-12.5, 12.5, -5, 20]

    # Initialize network
    G = nx.Graph()
    for i in range(graph.size()):
        factor = graph.at(i)
        for key_id, key in enumerate(factor.keys()):
            # Test if key corresponds to a pose
            if key in x_keys.values():
                pos = (values.atPose2(key).x(), values.atPose2(key).y())
                G.add_node(key, pos=pos, color='black')

            # Test if key corresponds to points
            elif key in b_keys.values():
                pos = (values.atPoint2(key)[0], values.atPoint2(key)[1])
                G.add_node(key, pos=pos, color='yellow')
            else:
                print('There was a problem with a factor not corresponding to an available key')

            # Add edges that represent binary factor: Odometry or detection
            for key_2_id, key_2 in enumerate(factor.keys()):
                if key != key_2 and key_id < key_2_id:
                    # detections will have key corresponding to a landmark
                    if key in b_keys.values() or key_2 in b_keys.values():
                        G.add_edge(key, key_2, color='red')
                    else:
                        G.add_edge(key, key_2, color='blue')

    # ===== Plot the graph using matplotlib =====
    # Matplotlib options
    fig, ax = plt.subplots()
    plt.title(f'Factor Graph\n{label}')
    ax.set_aspect('equal', 'box')
    plt.axis(self.plot_limits)
    plt.grid(True)
    plt.xticks(np.arange(plot_limits[0], plot_limits[1] + 1, 2.5))

    # Networkx Options
    pos = nx.get_node_attributes(G, 'pos')
    e_colors = nx.get_edge_attributes(G, 'color').values()
    n_colors = nx.get_node_attributes(G, 'color').values()
    options = {'node_size': 25, 'width': 3, 'with_labels': False}

    # Plot
    nx.draw_networkx(G, pos, edge_color=e_colors, node_color=n_colors, **options)
    np.arange(plot_limits[0], plot_limits[1] + 1, 2.5)
    plt.show()
