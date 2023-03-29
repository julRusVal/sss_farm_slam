#!/usr/bin/env python3
"""
The Goal of this script is to load saved gt poses (base_link) and convert them to the camera poses in the map frame.
This is part of the work towards projecting images onto a planes to make algae farm maps.
"""
import gtsam
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from sam_slam_utils.sam_slam_helper_funcs import read_csv_to_array, read_csv_to_list
import gtsam.utils.plot as gtsam_plot

# 3D Plotting
from mayavi import mlab


# %% Functions
def create_Pose3(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(w=input_pose[3], x=input_pose[4], y=input_pose[5], z=input_pose[6])
    # rot3_xyz = rot3.xyz()
    return gtsam.Pose3(rot3, input_pose[0:3])


def convert_poses_to_Pose3(poses):
    """
    Poses is is of the form: [[x,y,z,q_w,q_x,q_,y,q_z]]
    """
    pose3s = []
    for pose in poses:
        pose3s.append(create_Pose3(pose))

    return pose3s


def apply_transformPoseFrom(pose3s, transform):
    """
    pose3s: [gtsam.Pose3]
    transform: gtsam.Pose3

    Apply the transform given in local coordinates, result is expressed in the world coords
    """
    transformed_pose3s = []
    for pose3 in pose3s:
        transformed_pose3 = pose3.transformPoseFrom(transform)
        transformed_pose3s.append(transformed_pose3)

    return transformed_pose3s


def projectPixelTo3dRay(u, v, cx, cy, fx, fy):
    """
    From ROS-perception
    https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/image_geometry/cameramodels.py
    Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
    using the camera :math:`P` matrix.
    This is the inverse of :math:`project3dToPixel`.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    norm = math.sqrt(x * x + y * y + 1)
    x /= norm
    y /= norm
    z = 1.0 / norm
    return x, y, z


class camera_model:
    """
    This mirrors the pinhole camera model of ROS perception.
    https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/image_geometry/cameramodels.py
    """

    def __init__(self, camera_info):
        # form k and P  matrices, plumb bob distortion model
        # K: 3x3
        self.K = np.array(camera_info[0], dtype=np.float64)
        self.K = np.reshape(self.K, (3, 3))
        # P: 3x4
        self.P = np.array(camera_info[1], dtype=np.float64)
        self.P = np.reshape(self.P, (3, 4))

        # Extract key parameters
        self.width = int(camera_info[2][0])
        self.height = int(camera_info[2][1])
        self.fx = self.P[0, 0]
        self.fy = self.P[1, 1]
        self.cx = self.P[0, 2]
        self.cy = self.P[1, 2]
        self.orig_img_corners = np.array([[0, 0],
                                          [self.width, 0],
                                          [0, self.height],
                                          [self.width, self.height]], dtype=np.int32)

    def projectPixelTo3dRay(self, u, v):
        """
        From ROS-perception
        https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/image_geometry/cameramodels.py
        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :math:`project3dToPixel`.
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        norm = math.sqrt(x * x + y * y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return x, y, z


class rope_section:
    def __init__(self, start_buoy, end_buoy, start_coord, end_coord, depth, spatial_2_pixel):
        self.start_buoy = start_buoy
        self.end_buoy = end_buoy

        self.start_coord = start_coord
        self.end_coord = end_coord

        self.start_bottom_coord = start_coord - np.array([0, 0, depth], dtype=np.float64)
        self.end_bottom_coord = end_coord - np.array([0, 0, depth], dtype=np.float64)

        # ===== Define plane =====
        # Basis vectors
        # x is the horizontal component and y is the vertical component
        self.v_x = self.end_coord - self.start_coord
        self.mag_x = np.sqrt(np.sum(np.multiply(self.v_x, self.v_x)))
        self.v_x = self.v_x / self.mag_x

        self.v_y = self.start_bottom_coord - self.start_coord
        self.mag_y = np.sqrt(np.sum(np.multiply(self.v_y, self.v_y)))
        self.v_y = self.v_y / self.mag_y

        # normal vector
        # Note:due to downward orientation of y the norm points into the plane
        self.normal = np.cross(self.v_x, self.v_y)

        # normalize normal
        norm_mag = np.sqrt(np.sum(np.multiply(self.normal, self.normal)))
        self.normal = self.normal / norm_mag

        # Point on plane
        self.Q = self.start_coord

        # Image info
        self.spatial_2_pixel = spatial_2_pixel  # 1 meter = N pixels
        self.pixel_width = int(self.mag_x * self.spatial_2_pixel // 1)
        self.pixel_height = int((self.mag_y * self.spatial_2_pixel) // 1)

        # ===== Storage for images and masks
        self.images = []
        self.masks = []
        self.distances = []

        self.final_image = None

    def find_intersection(self, point, direction):
        """
        Given the plane does the line described by the point and direction intersect.
        :param point: np.array([x, y, z])
        :param direction: np.array([delta_x, delta_y, delta_z])
        :return: status: uses the dot product of the plane norm and camera z-axis to exclude applying images to 'back'
        :return: w_coords:
        :return: p_coords:
        :return: in_bounds: true is central line of camera intersects with the bounds of the plane
        """

        # Check for intersection
        # Note the direction is also checked because we only want to consider intersections from the 'font'
        # of the plane
        if np.dot(self.normal, direction) <= 0:
            return False, False, False, False

        # Calculate the intersection point
        t = -np.dot(self.normal, (point - self.Q)) / np.dot(self.normal, direction)
        intersection_point = point + t * direction

        if t < 0:
            return False, False, False, False

        # Express the intersection point as a linear combination of v1 and v2 using the pseudoinverse
        A = np.vstack([self.v_x, self.v_y]).T
        b = intersection_point - self.Q
        x = np.linalg.pinv(A).dot(b)
        x_intercept, y_intercept = x
        # intersection_point_linear_combination = f"{s}*v1 + {t}*v2 + {Q}"

        if 0 <= x_intercept <= self.mag_x and 0 <= y_intercept <= self.mag_y:
            within_bounds = True
        else:
            within_bounds = False

        return True, intersection_point, (x_intercept, y_intercept), within_bounds


class ground_plane:
    def __init__(self, start_coord, x_width, y_width, depth, spatial_2_pixel):
        self.mag_x = abs(x_width)
        self.mag_y = abs(y_width)
        self.depth = abs(depth)

        # force start coordinate to be on the desired plane
        self.start_coord = np.array([start_coord[0], start_coord[1], -self.depth], dtype=np.float64)
        self.x_end_coord = self.start_coord + np.array([self.mag_x, 0, 0], dtype=np.float64)
        self.y_end_coord = self.start_coord + np.array([0, self.mag_y, 0], dtype=np.float64)
        self.x_y_end_coord = self.start_coord + np.array([self.mag_x, self.mag_y, 0], dtype=np.float64)

        # ===== Define plane =====
        # Basis vectors
        # x is the horizontal component and y is the vertical component
        self.v_x = self.x_end_coord - self.start_coord
        self.v_x = self.v_x / self.mag_x

        self.v_y = self.y_end_coord - self.start_coord
        self.v_y = self.v_y / self.mag_y

        # normal vector
        # Note:the norm of ground plane will point up
        self.normal = np.cross(self.v_x, self.v_y)

        # normalize normal
        norm_mag = np.sqrt(np.sum(np.multiply(self.normal, self.normal)))
        self.normal = self.normal / norm_mag

        # Point on plane
        self.Q = self.start_coord

        # Image info
        self.spatial_2_pixel = spatial_2_pixel  # 1 meter = N pixels
        self.pixel_x_width = int(self.mag_x * self.spatial_2_pixel // 1)
        self.pixel_y_width = int((self.mag_y * self.spatial_2_pixel) // 1)

        # ===== Storage for images and masks
        self.images = []
        self.masks = []
        self.distances = []

        self.final_image = None

    def find_intersection(self, point, direction):
        """
        Given the plane does the line described by the point and direction intersect.
        :param point: np.array([x, y, z])
        :param direction: np.array([delta_x, delta_y, delta_z])
        :return: status: uses the dot product of the plane norm and camera z-axis to exclude applying images to 'back'
        :return: w_coords:
        :return: p_coords:
        :return: in_bounds: true is central line of camera intersects with the bounds of the plane
        """

        # Check for intersection
        # Note the direction is also checked because we only want to consider intersections from the 'font'
        # of the plane
        if np.dot(self.normal, direction) == 0:
            return False, False, False, False

        # Calculate the intersection point
        t = -np.dot(self.normal, (point - self.Q)) / np.dot(self.normal, direction)
        intersection_point = point + t * direction

        if t < 0:
            return False, False, False, False

        # Express the intersection point as a linear combination of v1 and v2 using the pseudoinverse
        A = np.vstack([self.v_x, self.v_y]).T
        b = intersection_point - self.Q
        x = np.linalg.pinv(A).dot(b)
        x_intercept, y_intercept = x
        # intersection_point_linear_combination = f"{s}*v1 + {t}*v2 + {Q}"

        if 0 <= x_intercept <= self.mag_x and 0 <= y_intercept <= self.mag_y:
            within_bounds = True
        else:
            within_bounds = False

        return True, intersection_point, (x_intercept, y_intercept), within_bounds


class image_mapping:
    def __init__(self, gt_base_link_poses, base_link_poses, l_r_camera_info, buoy_info, ropes, rows):
        # ===== true base pose =====
        self.gt_base_pose = gt_base_link_poses
        self.gt_base_pose3s = convert_poses_to_Pose3(self.gt_base_pose)

        # ===== Base pose =====
        self.base_pose = base_link_poses
        self.base_pose3s = convert_poses_to_Pose3(self.base_pose)

        self.pose_ids = self.base_pose[:, -1]

        # ===== Cameras =====
        self.valid_camera_names = ["left", "right", "down"]  # This is a list of which cameras will be processed

        # many camera related data structures will be stored in dictionary
        self.cameras_info = {"left": l_r_camera_info[0],
                             "right": l_r_camera_info[1],
                             "down": l_r_camera_info[2]}

        self.cameras = {"left": camera_model(self.cameras_info["left"]),
                        "right": camera_model(self.cameras_info["right"]),
                        "down": camera_model(self.cameras_info["down"])}

        # Relative poses of camera wrt base_link, in gtsam:Pose3
        self.left_relative_camera_pose = self.return_left_relative_pose()
        self.right_relative_camera_pose = self.return_right_relative_pose()
        self.down_relative_camera_pose = self.return_down_relative_pose()

        # Ground truth camera poses
        # was self.gt_camera_pose3s
        self.gt_camera_pose3s = {"left": apply_transformPoseFrom(self.gt_base_pose3s, self.left_relative_camera_pose),
                                 "right": apply_transformPoseFrom(self.gt_base_pose3s, self.right_relative_camera_pose),
                                 "down": apply_transformPoseFrom(self.gt_base_pose3s, self.down_relative_camera_pose)}

        # Estimated camera poses
        # was self.camera_pose3s
        self.cameras_pose3s = {"left": apply_transformPoseFrom(self.base_pose3s, self.left_relative_camera_pose),
                               "right": apply_transformPoseFrom(self.base_pose3s, self.right_relative_camera_pose),
                               "down": apply_transformPoseFrom(self.base_pose3s, self.down_relative_camera_pose)}

        # Various rays used for associating images to planes
        # [upper left, upper right, lower left, lower right, center center,
        # center left-of-center, center right-of-center]
        self.fov_rays = {"left": self.find_fov_rays("left", offset_fraction=6),
                         "right": self.find_fov_rays("right", offset_fraction=6),
                         "down": self.find_fov_rays("right", offset_fraction=10)}

        # ===== Map info =====
        self.buoys = buoy_info
        self.ropes = ropes
        self.rows = rows
        self.depth = 7.5  # Used to define the vertical extent of the planes

        self.spatial_2_pixel = 100  # 1 meter = 100 pixels
        self.planes = []
        self.build_planes_from_buoys_ropes()

        self.ground_plane = ground_plane(start_coord=[-15, 0],
                                         x_width=30,
                                         y_width=20,
                                         depth=15,
                                         spatial_2_pixel=self.spatial_2_pixel)

    @staticmethod
    def return_left_relative_pose():
        """
        Relative pose, w.r.t. base_link, of Sam's left camera.

        :return: gtsam:pose3 of left camera pose relative to base_link
        """
        l_t_x = 1.313
        l_t_y = 0.048
        l_t_z = -0.007

        # ===== Quaternion values from ROS tf messages =====
        l_r_x = -0.733244
        l_r_y = 0.310005
        l_r_z = -0.235671
        l_r_w = 0.557413

        # Return gtsam.Pose3, Constructor expects [x,y,z,q_w,q_x,q_y,q_z]
        return create_Pose3([l_t_x, l_t_y, l_t_z, l_r_w, l_r_x, l_r_y, l_r_z])

    @staticmethod
    def return_right_relative_pose():
        """
        Relative pose, w.r.t. base_link, of Sam's right camera.

        :return: gtsam:pose3 of right camera pose relative to base_link
        """
        r_t_x = 1.313
        r_t_y = -0.048
        r_t_z = -0.007

        # ===== Quaternion values from ROS tf messages =====
        r_r_x = - 0.310012
        r_r_y = 0.733241
        r_r_z = - 0.557415
        r_r_w = 0.235668

        # Return gtsam.Pose3, Constructor expects [x,y,z,q_w,q_x,q_y,q_z]
        return create_Pose3([r_t_x, r_t_y, r_t_z, r_r_w, r_r_x, r_r_y, r_r_z])

    @staticmethod
    def return_down_relative_pose():
        """
        Relative pose, w.r.t. base_link, of Sam's down camera.

        :return: gtsam:pose3 of down camera pose relative to base_link
        """
        d_t_x = 1.1385
        d_t_y = 0.000
        d_t_z = -0.052

        # ===== Quaternion values from ROS tf messages =====
        d_r_x = 0.707141
        d_r_y = -0.707073
        d_r_z = 0.000003
        d_r_w = -0.000003

        # Return gtsam.Pose3, Constructor expects [x,y,z,q_w,q_x,q_y,q_z]
        return create_Pose3([d_t_x, d_t_y, d_t_z, d_r_w, d_r_x, d_r_y, d_r_z])

    def find_fov_rays(self, camera_name, offset_fraction=2):
        """
        Calculates and sets FOV points
        """

        if camera_name not in self.valid_camera_names:
            print("Invalid camera_name")
            return -1

        camera = self.cameras[camera_name]
        # Corners
        # (0,0,1), (width,0,1), (0, height, 1), (width, height, 1)
        p_0 = camera.projectPixelTo3dRay(0, 0)
        p_1 = camera.projectPixelTo3dRay(camera.width, 0)
        p_2 = camera.projectPixelTo3dRay(0, camera.height)
        p_3 = camera.projectPixelTo3dRay(camera.width, camera.height)

        # left and right centers
        if offset_fraction < 2:
            offset_fraction = 2
        horizontal_offset = int(camera.width // offset_fraction)
        p_lc = camera.projectPixelTo3dRay(camera.cx - horizontal_offset, camera.cy)
        p_rc = camera.projectPixelTo3dRay(camera.cx + horizontal_offset, camera.cy)
        # Center
        p_c = camera.projectPixelTo3dRay(camera.cx, camera.cy)

        return np.array([p_0, p_1, p_2, p_3, p_c, p_lc, p_rc], dtype=np.float64)

    def build_planes_from_buoys_ropes(self):

        for rope in self.ropes:
            rope_for = [rope[0], rope[1]]
            rope_rev = [rope[1], rope[0]]
            directions = [rope_for, rope_rev]
            for direction in directions:
                start_buoy = int(direction[0])
                end_buoy = int(direction[1])

                start_coord = self.buoys[start_buoy, :]
                end_coord = self.buoys[end_buoy, :]

                self.planes.append(rope_section(start_buoy=start_buoy,
                                                end_buoy=end_buoy,
                                                start_coord=start_coord,
                                                end_coord=end_coord,
                                                depth=self.depth,
                                                spatial_2_pixel=self.spatial_2_pixel))

    def plot_fancy(self, other_name=None):
        """
        This might need some work
        :param other_name:
        :return:
        """
        # Parameters
        fig_num = 0
        base_scale = .5
        other_scale = 1
        plot_base = [13, 14, 15]  # [8, 10, 11, 12, 13, 14]
        plot_other = [13, 14, 15]  # [8, 10, 11, 12, 13, 14]

        fig = plt.figure(fig_num)
        axes = fig.add_subplot(projection='3d')

        axes.set_xlabel("X axis")
        axes.set_ylabel("Y axis")
        axes.set_zlabel("Z axis")

        # Plot buoys and vertical 'ropes'
        for buoy in self.buoys:
            axes.scatter(buoy[0], buoy[1], buoy[2], c='b', linewidths=5)
            axes.plot([buoy[0], buoy[0]],
                      [buoy[1], buoy[1]],
                      [buoy[2], buoy[2] - 10], c='g')

        # plot base_link gt pose3s
        for i_base, pose3 in enumerate(self.base_pose3s):
            # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)
            if i_base in plot_base or len(plot_base) == 0:
                gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)

        # plot camera gt pose3s
        if other_name is not None:
            other_pose3s = self.cameras_pose3s[other_name]
            for i_other, pose3 in enumerate(other_pose3s):
                # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=other_scale)
                if i_other in plot_other or len(plot_other) == 0:
                    gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=other_scale)
                    # plot fov
                    for i_ray, ray in enumerate(self.fov_rays[other_name]):
                        point_end = pose3.transformFrom(5 * ray)
                        x_comp = [pose3.x(), point_end[0]]
                        y_comp = [pose3.y(), point_end[1]]
                        z_comp = [pose3.z(), point_end[2]]
                        # Plot (0,0) as magenta
                        if i_ray == 0:
                            axes.plot(x_comp, y_comp, z_comp, c='m')
                        # Plot center line as yellow
                        elif i_ray == 4:
                            axes.plot(x_comp, y_comp, z_comp, c='b')
                        # Other fov lines plotted as black
                        else:
                            axes.plot(x_comp, y_comp, z_comp, c='k')

                    # plot intersection
                    for plane in self.planes:
                        # Find the center ray in the camera frame and then find the world coord given pose
                        start, direction = self.camera_center_point_direction(camera_name=other_name, pose=pose3)

                        intrcpt_status, intrcpt_w_coords, _, in_bounds = plane.find_intersection(start,
                                                                                                 direction)

                        if in_bounds:
                            axes.scatter(intrcpt_w_coords[0], intrcpt_w_coords[1], intrcpt_w_coords[2], c='r')

        # plt.axis('equal')
        plt.title("Testing the transform")
        plt.show()

    def find_fov_corner_coords(self, camera_name, plane_id, pose_id):
        """
        Find the intersections of the fov with the defined plane, plane_id = -1 indicates the ground plane

        :param camera_name: "left" or " right" or "down"
        :param plane_id: index of plane in self.planes, -1 will use self.ground_plane
        :param pose_id:
        :return:
        """
        if plane_id >= len(self.planes):
            print("Invalid camera_name")
            return -1

        if camera_name not in self.valid_camera_names:
            print("Invalid camera_name")
            return -1

        # Camera pose
        # pose3 = self.camera_pose3s[pose_id]
        pose3 = self.cameras_pose3s[camera_name][pose_id]

        # intersection points of camera fov with plane
        # given in coords of the plane
        corner_cords = np.zeros((4, 2), dtype=np.float64)

        # Compute an intersection for each ray
        for i_ray in range(4):
            ray = self.fov_rays[camera_name][i_ray]
            ray_end_point_point3 = pose3.transformFrom(ray)

            end = np.array([ray_end_point_point3[0],
                            ray_end_point_point3[1],
                            ray_end_point_point3[2]], dtype=np.float64)

            start = np.array([pose3.x(),
                              pose3.y(),
                              pose3.z()], dtype=np.float64)

            direction = end - start

            if plane_id == -1:
                corner_status, corner_w_coords, corner_p_coords, _ = self.ground_plane.find_intersection(start,
                                                                                                         direction)
            else:
                corner_status, corner_w_coords, corner_p_coords, _ = self.planes[plane_id].find_intersection(start,
                                                                                                             direction)

            corner_cords[i_ray, :] = corner_p_coords

        return corner_cords

    def find_plane_corner_pixels(self, camera_name, plane_id, pose_id):
        """
        Each plane is defined by a start buoy and an end buoy, a lower buoy is defined as being below the start buoy.
        The fourth corner is placed below the end buoy. This function will return the pixel positions of these corners.
        """

        if camera_name not in self.valid_camera_names:
            print("Invalid camera_name")
            return -1

        corner_pixels = np.zeros((4, 2), dtype=np.float64)

        if plane_id == -1:  # work with ground plane
            # the world coordinates of the plane corners
            w_corner_coords = np.zeros((4, 3), dtype=np.float64)
            w_corner_coords[0, :] = self.ground_plane.start_coord
            w_corner_coords[1, :] = self.ground_plane.x_end_coord
            w_corner_coords[2, :] = self.ground_plane.y_end_coord
            w_corner_coords[3, :] = self.ground_plane.x_y_end_coord

            # Compute an intersection for each ray
            for i_corner in range(4):
                x_pixel, y_pixel = self.find_pixels_of_3d_point(camera_name=camera_name,
                                                                pose_id=pose_id,
                                                                map_point=w_corner_coords[i_corner, :])

                corner_pixels[i_corner, :] = x_pixel, y_pixel

        else:  # work with normal plane section
            # the world coordinates of the plane corners
            w_corner_coords = np.zeros((4, 3), dtype=np.float64)
            w_corner_coords[0, :] = self.planes[plane_id].start_coord
            w_corner_coords[1, :] = self.planes[plane_id].end_coord
            w_corner_coords[2, :] = self.planes[plane_id].start_bottom_coord
            w_corner_coords[3, :] = self.planes[plane_id].end_bottom_coord

            # Compute an intersection for each ray
            for i_corner in range(4):
                x_pixel, y_pixel = self.find_pixels_of_3d_point(camera_name=camera_name,
                                                                pose_id=pose_id,
                                                                map_point=w_corner_coords[i_corner, :])

                corner_pixels[i_corner, :] = x_pixel, y_pixel

        return corner_pixels

    def convert_spatial_corners_to_pixel(self, plane_id, corner_coords):
        """
        Currently only using for testing
        """
        corner_coords_pixels = corner_coords * self.spatial_2_pixel
        corner_coords_pixels = np.floor_divide(corner_coords_pixels, 1).astype(np.int32)

        # Find min and max and apply offset
        min_x = min(min(corner_coords_pixels[:, 0]), 0)
        min_y = min(min(corner_coords_pixels[:, 1]), 0)

        offset = np.array([min_x, min_y], dtype=np.int32)

        corner_coords_pixels = corner_coords_pixels - offset

        max_x = max(max(corner_coords_pixels[:, 0]), (self.planes[plane_id].mag_x * self.spatial_2_pixel) // 1 - 1)
        max_y = max(max(corner_coords_pixels[:, 1]), (self.planes[plane_id].mag_y * self.spatial_2_pixel) // 1 - 1)

        max_inds = np.array([max_x, max_y], dtype=np.int32)

        return corner_coords_pixels, offset, max_inds

    def find_pixels_of_3d_point(self, camera_name, pose_id, map_point):
        """

        :param pose_id:
        :param map_point: a point2, np.array (3x1) that hold the world coords of a point
        :return:
        """
        if camera_name not in self.valid_camera_names:
            print("Invalid camera_name")
            return -1

        # camera_pose3 = self.camera_pose3s[pose_id]
        camera_pose3 = self.cameras_pose3s[camera_name][pose_id]

        # express the provided point (given in map coords) in camera coords.
        camera_point3 = camera_pose3.transformTo(map_point)
        # convert to homogeneous coords
        # Convert m -> mm
        camera_point_hmg = np.ones((4,))
        camera_point_hmg[:3] = camera_point3[:] * 1000

        # Apply projection matrix
        uvw_array = np.matmul(self.cameras[camera_name].P, camera_point_hmg)

        u = uvw_array[0]
        v = uvw_array[1]
        w = uvw_array[2]

        if w != 0.0:
            return (u / w), (v / w)
        else:
            return float('nan'), float('nan')

    def camera_center_point_direction(self, camera_name, pose):
        """
        returns the center point and direction of central ray of camera given pose
        can accept pose_id or id
        """
        if camera_name not in self.valid_camera_names:
            print("Invalid camera_name")
            return -1

        if isinstance(pose, gtsam.Pose3):
            camera_pose3 = pose
        elif 0 <= int(pose) < len(self.cameras_pose3s[camera_name]):
            camera_pose3 = self.cameras_pose3s[camera_name][pose]
        else:
            print("Malformed request")
            return -1

        center_ray = self.fov_rays[camera_name][4]
        center_end_point_pose3 = camera_pose3.transformFrom(center_ray)

        end = np.array([center_end_point_pose3[0],
                        center_end_point_pose3[1],
                        center_end_point_pose3[2]], dtype=np.float64)

        start = np.array([camera_pose3.x(),
                          camera_pose3.y(),
                          camera_pose3.z()])

        direction = end - start

        return start, direction

    def camera_offset_point_directions(self, camera_name, pose):
        """
        returns the center point and direction of offset rays of camera given pose
        can accept pose_id or id

        Assumes that FOV rays are in the following format:
        [upper left, upper right, lower left, lower right, center center, center left-of-center, center right-of-center]

        """
        if camera_name not in self.valid_camera_names:
            print("Invalid camera_name")
            return -1

        if isinstance(pose, gtsam.Pose3):
            camera_pose3 = pose
        elif 0 <= int(pose) < len(self.cameras_pose3s[camera_name]):
            camera_pose3 = self.cameras_pose3s[camera_name][pose]
        else:
            print("Malformed request")
            return -1

        if len(self.fov_rays[camera_name]) != 7:
            print(f"camera_offset_point_directions() assumes self.fov_rays has length of 7, check format!")
            return

        start = np.array([camera_pose3.x(),
                          camera_pose3.y(),
                          camera_pose3.z()])

        offset_directions = []
        offset_indices = [-2, -1]  # dDepends on format of self.fov_arrays
        for offset_ind in offset_indices:
            offset_ray = self.fov_rays[camera_name][offset_ind]
            offset_end_point_pose3 = camera_pose3.transformFrom(offset_ray)

            end = np.array([offset_end_point_pose3[0],
                            offset_end_point_pose3[1],
                            offset_end_point_pose3[2]], dtype=np.float64)

            offset_direction = end - start
            offset_directions.append(offset_direction)

        return start, offset_directions

    def process_images(self, path_name, ignore_first=0, verbose=False):

        """
        process_images is used to map the gathered images and warp them on to the planes formed by the buoys.
        The 3d poses of the base link are provided as well as a list that describes the structure of the farm.

        :param path_name:
        :param image_path:
        :param ignore_first:
        :param verbose:
        :return:
        """
        if ignore_first < 0:
            start_index = int(0)
        elif ignore_first >= len(self.base_pose3s):
            print("All images ignored for processing")
            return -1
        else:
            start_index = int(ignore_first)

        """
        Loop over poses, at each pose process the data from each camera
        
        """

        for current_pose_id in range(start_index, len(self.base_pose3s)):
            # There is a mind melting index error between the poses and the img id :(
            # current_img_id = int(self.base_pose[current_pose_id][-1])

            check_x = self.gt_base_pose[current_pose_id][0]
            check_y = self.gt_base_pose[current_pose_id][1]
            check_img_id = int(self.base_pose[current_pose_id][-1])

            print(f"{current_pose_id}: {check_x} - {check_y} - {check_img_id}")

            for camera_name in self.valid_camera_names:
                camera_pose3 = self.cameras_pose3s[camera_name][current_pose_id]

                # Define rays are used to detect if a given plane is in view. Here are the rays are checked for
                # intersection with each plane.
                # Only one ray per pose should be associated with a plane but there can be multiple
                # associations between poses and plans due to the using multiple rays
                c_center, c_direction = self.camera_center_point_direction(camera_name=camera_name,
                                                                           pose=camera_pose3)
                o_center, o_directions = self.camera_offset_point_directions(camera_name=camera_name,
                                                                             pose=camera_pose3)
                directions = o_directions.copy()
                directions.append(c_direction)

                for plane_id, plane in enumerate(self.planes):
                    # Find which if any plane to apply the image to

                    for direction in directions:

                        status, w_coords, p_coords, in_bounds = plane.find_intersection(c_center, direction)

                        # properly oriented plane that is centrally located w.r.t. camera frame
                        if status and in_bounds:
                            corners = self.find_plane_corner_pixels(camera_name=camera_name, plane_id=plane_id,
                                                                    pose_id=current_pose_id)

                            # TODO figure why this increment is need, should not be
                            # looks like the down camera needs a different offset
                            if camera_name == "down":
                                frame_offset = 1
                            else:
                                frame_offset = 1
                            if current_pose_id + frame_offset < len(self.base_pose3s):
                                next_img_id = int(self.base_pose[current_pose_id + frame_offset][-1])
                            else:
                                continue

                            # mod_id = int(current_img_id + 1)
                            mod_id = next_img_id
                            img = cv2.imread(path_name + camera_name + f"/{mod_id}.jpg")

                            if not isinstance(img, np.ndarray):
                                continue

                            # Save
                            if verbose:
                                img_verbose = img.copy()

                                # Draw corners
                                for corner in corners:
                                    if math.isnan(corner[0]) or math.isnan(corner[1]):
                                        continue
                                    corner_x = int(corner[0] // 1)
                                    corner_y = int(corner[1] // 1)
                                    img_verbose = cv2.circle(img_verbose, (corner_x, corner_y), 5, (0, 0, 255), -1)

                                # Draw center
                                center = self.find_pixels_of_3d_point(camera_name=camera_name,
                                                                      pose_id=current_pose_id,
                                                                      map_point=w_coords)

                                if not math.isnan(center[0]) and not math.isnan(center[1]):
                                    center_x = int(center[0] // 1)
                                    center_y = int(center[1] // 1)
                                    img_verbose = cv2.circle(img_verbose, (center_x, center_y), 5, (255, 0, 255), -1)

                                # Draw other features
                                # p_0: offset from start buoy
                                # p_1: offset from end buoy
                                vertical_offset = 2.0  # Trying to mark the horizontal
                                w_p_0 = plane.start_coord - [0, 0, vertical_offset]
                                w_p_1 = plane.end_coord - [0, 0, vertical_offset]

                                p_0 = self.find_pixels_of_3d_point(camera_name=camera_name,
                                                                   pose_id=current_pose_id,
                                                                   map_point=w_p_0)

                                p_1 = self.find_pixels_of_3d_point(camera_name=camera_name,
                                                                   pose_id=current_pose_id,
                                                                   map_point=w_p_1)

                                if not math.isnan(p_0[0]) and not math.isnan(p_0[1]):
                                    p_0_x = int(p_0[0] // 1)
                                    p_0_y = int(p_0[1] // 1)
                                    img_verbose = cv2.circle(img_verbose, (p_0_x, p_0_y), 5, (0, 255, 0), -1)

                                if not math.isnan(p_1[0]) and not math.isnan(p_1[1]):
                                    p_1_x = int(p_1[0] // 1)
                                    p_1_y = int(p_1[1] // 1)
                                    img_verbose = cv2.circle(img_verbose, (p_1_x, p_1_y), 5, (0, 255, 0), -1)

                                cv2.imwrite(path_name + "images_registered/Processing_"
                                            + f"{camera_name}_{current_pose_id}_{mod_id}_{plane_id}.jpg",
                                            img_verbose)

                            # perform extraction
                            destination_corners = np.array([[0, 0],
                                                            [plane.pixel_width - 1, 0],
                                                            [0, plane.pixel_height - 1],
                                                            [plane.pixel_width - 1, plane.pixel_height - 1]],
                                                           dtype=np.float64)

                            homography = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                                                     destination_corners.astype(np.float32))
                            # Apply homography to image
                            img_warped = cv2.warpPerspective(img, homography,
                                                             (plane.pixel_width, plane.pixel_height))

                            # apply homography to form a mask
                            mask_warped = cv2.warpPerspective(np.ones_like(img) * 255, homography,
                                                              (plane.pixel_width, plane.pixel_height))

                            # Add warped image and mask to the plane
                            self.planes[plane_id].images.append(img_warped)
                            self.planes[plane_id].masks.append(mask_warped)

                            # Add the distance between the camera and the interesction w/ the plane
                            delta = c_center - w_coords
                            distance = np.sqrt(np.sum(np.square(delta)))

                            self.planes[plane_id].distances.append(distance)

                            # Save images and masks
                            cv2.imwrite(path_name + "images_warped/Warping_"
                                        + f"{camera_name}_{current_pose_id}_{mod_id}_{plane_id}.jpg",
                                        img_warped)

                            cv2.imwrite(path_name + "images_masked/Warping_"
                                        + f"{camera_name}_{current_pose_id}_{mod_id}_{plane_id}.jpg",
                                        mask_warped)

                            # Only associated each image with a plane once
                            break

    def process_ground_plane_images(self, path_name, ignore_first=0, verbose=False):
        """
        process_images is used to map the gathered images and warp them on to the ground plane.
        The 3d poses of the base link and definition of the ground plane are provided.

        :param path_name:
        :param image_path:
        :param ignore_first:
        :param verbose:
        :return:
        """
        # ===== Ignore some of the early data =====
        if ignore_first < 0:
            start_index = int(0)
        elif ignore_first >= len(self.base_pose3s):
            print("All images ignored for processing")
            return -1
        else:
            start_index = int(ignore_first)

        # ===== Parameters =====
        camera_name = "down"  # only use the downward camera
        plane = self.ground_plane
        plane_id = -1

        # ===== Loop over poses =====
        # at each pose process the data from each camera

        for current_pose_id in range(start_index, len(self.base_pose3s)):
            camera_pose3 = self.cameras_pose3s[camera_name][current_pose_id]

            # TODO figure why this increment is need, should not be
            # There is a mind melting index error between the poses and the img id :(
            # current_img_id = int(self.base_pose[current_pose_id][-1])
            # looks like the down camera needs a different offset
            if camera_name == "down":
                frame_offset = 1
            else:
                frame_offset = 1
            if current_pose_id + frame_offset < len(self.base_pose3s):
                next_img_id = int(self.base_pose[current_pose_id + frame_offset][-1])
            else:
                continue

            # ===== Select which image to use =====
            # current_img_id = int(self.base_pose[current_pose_id][-1])
            # used_img_id = current_img_id
            used_img_id = next_img_id
            img = cv2.imread(path_name + camera_name + f"/{used_img_id}.jpg")

            if not isinstance(img, np.ndarray):
                continue

            # Define rays are used to detect if a given plane is in view.
            c_center, c_direction = self.camera_center_point_direction(camera_name=camera_name,
                                                                       pose=camera_pose3)

            status, w_coords, p_coords, in_bounds = plane.find_intersection(c_center, c_direction)

            # properly oriented plane that is centrally located w.r.t. camera frame
            if status and in_bounds:

                # This will find the corners of the specified plane in the field of view
                corners = self.find_plane_corner_pixels(camera_name=camera_name, plane_id=plane_id,
                                                        pose_id=current_pose_id)

                # This will find the coor
                fov_corners = self.find_fov_corner_coords(camera_name=camera_name,
                                                          plane_id=plane_id,
                                                          pose_id=current_pose_id)

                fov_corners = fov_corners * self.ground_plane.spatial_2_pixel

                # Save
                if verbose:
                    img_verbose = img.copy()

                    # Draw corners
                    for corner in corners:
                        if math.isnan(corner[0]) or math.isnan(corner[1]):
                            continue
                        corner_x = int(corner[0] // 1)
                        corner_y = int(corner[1] // 1)
                        img_verbose = cv2.circle(img_verbose, (corner_x, corner_y), 5, (0, 0, 255), -1)

                    # Draw center
                    center = self.find_pixels_of_3d_point(camera_name=camera_name,
                                                          pose_id=current_pose_id,
                                                          map_point=w_coords)

                    if not math.isnan(center[0]) and not math.isnan(center[1]):
                        center_x = int(center[0] // 1)
                        center_y = int(center[1] // 1)
                        img_verbose = cv2.circle(img_verbose, (center_x, center_y), 5, (255, 0, 255),
                                                 -1)

                    cv2.imwrite(path_name + "images_ground/allignment_"
                                + f"{camera_name}_{current_pose_id}_{used_img_id}_{plane_id}.jpg",
                                img_verbose)

                # perform extraction
                # Old method for horizontal and small planes
                # destination_corners = np.array([[0, 0],
                #                                 [plane.pixel_x_width - 1, 0],
                #                                 [0, plane.pixel_x_width - 1],
                #                                 [plane.pixel_x_width - 1, plane.pixel_x_width - 1]],
                #                                dtype=np.float64)

                img_corners = self.cameras["down"].orig_img_corners

                homography = cv2.getPerspectiveTransform(img_corners.astype(np.float32),
                                                         fov_corners.astype(np.float32))
                # Apply homography to image
                img_warped = cv2.warpPerspective(img, homography,
                                                 (plane.pixel_x_width, plane.pixel_y_width))

                # apply homography to form a mask
                mask_warped = cv2.warpPerspective(np.ones_like(img) * 255, homography,
                                                  (plane.pixel_x_width, plane.pixel_y_width))

                # Add warped image and mask to the plane
                self.planes[plane_id].images.append(img_warped)
                self.planes[plane_id].masks.append(mask_warped)

                # Add the distance between the camera and the interesction w/ the plane
                delta = c_center - w_coords
                distance = np.sqrt(np.sum(np.square(delta)))

                self.planes[plane_id].distances.append(distance)

                # Save images and masks
                cv2.imwrite(path_name + "images_ground/img_Warp_"
                            + f"{camera_name}_{current_pose_id}_{used_img_id}_{plane_id}.jpg",
                            img_warped)

                cv2.imwrite(path_name + "images_ground/mask_warp_"
                            + f"{camera_name}_{current_pose_id}_{used_img_id}_{plane_id}.jpg",
                            mask_warped)

    def simple_stitch_planes_images(self, max_dist=np.inf, verbose=False):

        for plane_id, plane in enumerate(self.planes):
            if len(plane.images) == 0:
                continue

            sorted_data = sorted(zip(plane.distances, plane.images, plane.masks), reverse=True)

            final_img = np.zeros_like(plane.images[0])

            for distance, image, mask in sorted_data:
                # Exclude if the distance exceeds the provided threshold
                if distance > max_dist:
                    continue

                # Build the final plane image starting with the most distant images.
                # Nearer images will overwrite more distant images.
                # This method is very suboptimal
                final_img[mask > 0] = image[mask > 0]

            #
            plane.final_image = final_img
            cv2.imwrite(path_name + f"planes/final_{plane_id}.jpg",
                        final_img)

    def combine_row_images(self, fill_missing=False):
        for row_id, row in enumerate(self.rows):
            plane_width = 0
            plane_widths = []
            plane_height = 0
            # Find the size of the total row
            for plane_id in row:
                plane_width += self.planes[plane_id].pixel_width
                plane_widths.append(self.planes[plane_id].pixel_width)
                plane_height = self.planes[plane_id].pixel_height

            # Allocate
            row_img = np.zeros((plane_height, plane_width, 3))

            current_width_index = 0
            for i, plane_id in enumerate(row):
                if self.planes[plane_id].final_image is None:
                    current_width_index += plane_widths[i]
                else:
                    img_height, img_width, _ = self.planes[plane_id].final_image.shape
                    # Check if the img matches the expected size
                    if img_height != plane_height or img_width != plane_widths[i]:
                        print(" Mixmatch between expected and actual plane image size")
                        continue
                    next_width_index = current_width_index + plane_widths[i]
                    row_img[:, current_width_index:next_width_index, :] = self.planes[plane_id].final_image[:, :, :]
                    current_width_index = next_width_index

            # Save
            cv2.imwrite(path_name + f"rows/row_{row_id}.jpg",
                        row_img)

    def plot_3d_map(self, show_base=False):
        # Parameters
        plane_offset = 0.01
        stride = 5
        axis_length = 2.5

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Add axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for plane in self.planes:
            if plane.final_image is None:
                img = np.zeros((plane.pixel_height, plane.pixel_width, 3))
            else:
                # internally images are BGR (openCV) matplotlib expects RGB within range of 0-1.0
                img = np.flip(plane.final_image / 255, axis=2)
            # Will use the normal to apply small offset
            normal = plane.normal
            # Normalize, just in case
            normal_mag = np.sqrt(np.sum(np.multiply(normal, normal)))
            normal = normal / normal_mag

            # Offset slightly for plotting
            start_x, start_y, start_depth = plane.start_coord - normal * plane_offset
            end_x, end_y, _ = plane.end_coord - normal * plane_offset
            _, _, end_depth = plane.start_bottom_coord - normal * plane_offset

            # Define the resolution of the meshgrid based on image
            res_x = plane.pixel_width
            res_h = plane.pixel_height

            # Create the meshgrid
            x_linspace = np.linspace(start_x, end_x, res_x)
            y_linspace = np.linspace(start_y, end_y, res_x)
            h_linspace = np.linspace(start_depth, end_depth, res_h)

            X, Z = np.meshgrid(x_linspace, h_linspace)
            Y, _ = np.meshgrid(y_linspace, h_linspace)

            ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride,
                            facecolors=img)

        # plot base_link gt pose3s
        if show_base:
            for i_base, pose3 in enumerate(self.base_pose3s):
                # gtsam_plot.plot_pose3_on_axes(axes, pose3, axis_length=base_scale)
                gtsam_plot.plot_pose3_on_axes(ax, pose3, axis_length=axis_length)

        # Plot buoys
        for buoy in self.buoys:
            ax.scatter(buoy[0], buoy[1], buoy[2], c='b', linewidths=5)

        plt.title("Image Map")
        plt.show()

    def plot_3d_map_mayavi(self):
        # Parameters
        plane_offset = 0.01

        # ==== Plot planes =====
        for plane in self.planes:
            if plane.final_image is None:
                img = np.zeros((plane.pixel_height, plane.pixel_width))
            else:
                # internally images are BGR (openCV) matplotlib expects RGB within range of 0-1.0
                # img = np.flip(plane.final_image / 255, axis=2)
                img = cv2.cvtColor(plane.final_image, cv2.COLOR_BGR2GRAY)
                img = img.astype(np.float64) / 255
            # Will use the normal to apply small offset
            normal = plane.normal
            # Normalize, just in case
            normal_mag = np.sqrt(np.sum(np.multiply(normal, normal)))
            normal = normal / normal_mag

            # Offset slightly for plotting
            start_x, start_y, start_depth = plane.start_coord - normal * plane_offset
            end_x, end_y, _ = plane.end_coord - normal * plane_offset
            _, _, end_depth = plane.start_bottom_coord - normal * plane_offset

            # Define the resolution of the meshgrid based on image
            res_x = plane.pixel_width
            res_h = plane.pixel_height

            # Create the meshgrid
            x_linspace = np.linspace(start_x, end_x, res_x)
            y_linspace = np.linspace(start_y, end_y, res_x)
            h_linspace = np.linspace(start_depth, end_depth, res_h)

            X, Z = np.meshgrid(x_linspace, h_linspace)
            Y, _ = np.meshgrid(y_linspace, h_linspace)

            # mlab.mesh(X, Y, Z, color=(0, 1, 0))
            mlab.mesh(X, Y, Z, scalars=img, colormap="afmhot")

        # ===== Plot buoys =====
        for buoy in self.buoys:
            mlab.points3d(buoy[0], buoy[1], buoy[2], color=(0, 0, 1))

        # ===== Plot ground plane =====
        # Define the resolution of the meshgrid based on image
        res_x = self.ground_plane.pixel_x_width
        res_y = self.ground_plane.pixel_y_width

        start_x, start_y, start_z = self.ground_plane.start_coord

        # Create the meshgrid
        x_linspace = np.linspace(start_x, start_x + self.ground_plane.mag_x, res_x)
        y_linspace = np.linspace(start_y, start_y + self.ground_plane.mag_y, res_y)

        X, Y = np.meshgrid(x_linspace, y_linspace)
        Z = np.ones_like(X) * start_z

        mlab.mesh(X, Y, Z, color=(0, 1, 0))

        # ===== Other plotting option =====
        mlab.xlabel("X")
        mlab.ylabel("Y")
        mlab.zlabel("Z")

        mlab.axes(xlabel="X", x_axis_visibility=True,
                  ylabel="Y", y_axis_visibility=True,
                  zlabel="Z", z_axis_visibility=True,
                  color=(0, 0, 0))

        mlab.title("Visual Map")
        mlab.show()


# %% Load and process data
# This is the gt of the base_link indexed for the left images
# base_gt = read_csv_to_array('data/left_gt.csv')
# left_info = read_csv_to_list('data/left_info.csv')
# buoy_info = read_csv_to_array('data/buoy_info.csv')

# Mac
path_name = "/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/online_testing/"
# linux
# path_name = '/home/julian/catkin_ws/src/sam_slam/processing scripts/data/online_testing/'

gt_base = read_csv_to_array(path_name + 'camera_gt.csv')
base = read_csv_to_array(path_name + 'camera_gt.csv')  # change to camera_gt.csv, camera_dr.csv, or camera_est.csv

left_info = read_csv_to_list(path_name + 'left_info.csv')
right_info = read_csv_to_list(path_name + 'right_info.csv')
down_info = read_csv_to_list(path_name + 'down_info.csv')

# Select buoy position to use
# buoys.csv contains the ground truth locations of the buoys
# camera_buoys_est.csv contains the estimated locations of the buoys
# buoy_info = read_csv_to_array(path_name + 'buoys.csv')
buoy_info = read_csv_to_array(path_name + 'camera_buoys_est.csv')

# Define structure of the farm
# Define the connections between buoys
# list of start and stop indices of buoys
# TODO remove hard coded rope structure
ropes = [[0, 4], [4, 2],
         [1, 5], [5, 3]]

# Define which connections above form a row
# Note: each rope section forms two planes for example [0, 4] and [4, 0], so that both sides can be imaged separately
# TODO remove hard coded rows
rows = [[0, 2], [3, 1], [4, 6], [7, 5]]

img_map = image_mapping(gt_base_link_poses=gt_base,
                        base_link_poses=base,
                        l_r_camera_info=[left_info, right_info, down_info],
                        buoy_info=buoy_info,
                        ropes=ropes,
                        rows=rows)

# %% Plot
# img_map.plot_fancy(other_name=None)
# img_map.plot_fancy(other_name="left")
# img_map.plot_fancy(other_name="down")
# img_map.plot_fancy(img_map.gt_camera_pose3s)  # plot the ground ruth as other
# plot_fancy(base_gt_pose3s, left_gt_pose3s, buoy_info, points)
img_map.process_images(path_name, ignore_first=8, verbose=True)  #
img_map.process_ground_plane_images(path_name, ignore_first=8, verbose=True)
# img_map.simple_stitch_planes_images(max_dist=12)
# img_map.combine_row_images()
# img_map.plot_3d_map(show_base=True)
# img_map.plot_3d_map_mayavi()

# %% Testing parameters
do_testing_1 = False
do_testing_2 = False
do_testing_3 = False
do_testing_4 = False
# %% Testing 1
if do_testing_1:
    print("Testing 1")
    camera_name = "left"
    p_org = np.array([-5.0, 4.0, -0])
    p_x = np.array([-5.0, 9.0, -0])
    p_y = np.array([-5.0, 4.0, -15])

    point = np.array([-4.0, 3.0, -6])
    direction = np.array([-1.0, 0, 0])

    a, b, c, d = img_map.planes[0].find_intersection(point, direction)

    corner_coords = img_map.find_fov_corner_coords(camera_name=camera_name, plane_id=0, pose_id=25)

    corner_pixels, offset, max_inds = img_map.convert_spatial_corners_to_pixel(0, corner_coords)

    M = cv2.getPerspectiveTransform(img_map.cameras[camera_name].orig_img_corners.astype(np.float32),
                                    corner_pixels.astype(np.float32))

    img_id = int(img_map.base_pose[25, -1])

    img = cv2.imread(f"/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/left/l_{img_id}.jpg")

    new_img = cv2.warpPerspective(img, M, (max_inds[0] + 1, max_inds[1] + 1))

    # cv2.imshow('thing', new_img)

    # cv2.imwrite('new_img.jpg', new_img)

    # truncate warped image
    x_original = corner_pixels[0] + offset

    x_start = abs(offset[0]) + 76
    x_size = int((img_map.planes[0].mag_x * img_map.spatial_2_pixel) // 1)
    x_end = x_start + x_size

    y_start = abs(offset[1])
    y_size = int((img_map.planes[0].mag_y * img_map.spatial_2_pixel) // 1)
    y_end = y_start + y_size

    new_img = new_img[y_start:y_end, x_start:x_end]

    cv2.imwrite('new_img.jpg', new_img)

# %% Testing 2
if do_testing_2:
    print("Testing 2")
    camera_name = "left"
    pose_id = 25
    proper_img_id = int(img_map.base_pose[pose_id, -1])
    test_next_n = 3

    # buoys
    test_point_b0 = np.array([-5.0, 4.0, 0])
    test_point_b1 = np.array([-5.0, 9.0, 0])

    x_pix_0, y_pix_0 = img_map.find_pixels_of_3d_point(camera_name=camera_name,
                                                       pose_id=pose_id,
                                                       map_point=test_point_b0)

    x_pix_1, y_pix_1 = img_map.find_pixels_of_3d_point(camera_name=camera_name,
                                                       pose_id=pose_id,
                                                       map_point=test_point_b1)

    # Lower points
    depth = 7.5
    test_point_b0_deep = test_point_b0
    test_point_b0_deep[2] = -depth

    test_point_b1_deep = test_point_b1
    test_point_b1_deep[2] = -depth

    x_pix_0_d, y_pix_0_d = img_map.find_pixels_of_3d_point(camera_name=camera_name,
                                                           pose_id=pose_id,
                                                           map_point=test_point_b0_deep)
    x_pix_1_d, y_pix_1_d = img_map.find_pixels_of_3d_point(camera_name=camera_name,
                                                           pose_id=pose_id,
                                                           map_point=test_point_b1_deep)

    # Center
    # test_point_c = np.array([-5.0, 6.592, -0.706])
    # x_pix_c, y_pix_c = img_map.find_pixels_of_3d_point(pose_id, test_point_c)

    # Mark images
    for i in range(int(test_next_n + 1)):
        img_id = proper_img_id + i
        img = cv2.imread(f"/Users/julian/KTH/Degree project/sam_slam/processing scripts/data/left/l_{img_id}.jpg")

        # Buoys
        img_marked = cv2.circle(img, (int(x_pix_0 // 1), int(y_pix_0 // 1)), 5, (0, 0, 255), -1)
        img_marked = cv2.circle(img_marked, (int(x_pix_1 // 1), int(y_pix_1 // 1)), 5, (0, 0, 255), -1)

        # Lower points
        img_marked = cv2.circle(img, (int(x_pix_0_d // 1), int(y_pix_0_d // 1)), 5, (255, 0, 255), -1)
        img_marked = cv2.circle(img_marked, (int(x_pix_1_d // 1), int(y_pix_1_d // 1)), 5, (255, 0, 255), -1)

        # Center
        # img_marked = cv2.circle(img_marked, (int(x_pix_c // 1), int(y_pix_c // 1)), 5, (0, 255, 255), -1)

        img_marked[:, int(img_map.cameras[camera_name].cx), :] = (0, 255, 255)
        img_marked[int(img_map.cameras[camera_name].cy), :, :] = (0, 255, 255)

        #
        cv2.imshow(f"Marked Image: {img_id}", img_marked)
        cv2.imwrite(f"Test2:{proper_img_id}_{img_id}.jpg", img_marked)

    # Waits for a keystroke
    cv2.waitKey(0)

    # Destroys all the windows created
    cv2.destroyAllWindows()

# %% Testing 3 - checking rotation transforms and equivalences
if do_testing_3:
    """
    Testing for the transform between base_link and left camera
    """
    pose_id = 25
    base_pose3 = img_map.base_pose3s[pose_id]
    # Ros transform using quaternions reported by ros
    # stnfish using rpy in stonefish config -> quaternion w/ rpy_2_quat.py
    ros_b_2_c = img_map.return_left_relative_pose()
    stnfsh_b_2_c = img_map.return_left_relative_pose()

    ros_left_pose3 = base_pose3.transformPoseFrom(ros_b_2_c)
    stnfsh_left_pose3 = base_pose3.transformPoseFrom(stnfsh_b_2_c)

    # Convert matrix to rpy w/ gtsam
    ros_check = ros_b_2_c.rotation().rpy()
    stnfsh_check = stnfsh_b_2_c.rotation().rpy()

# %% Testing 4 - mark images with centers
if do_testing_4:
    camera_names = ["left", "right"]
    for pose in img_map.base_pose:
        img_id = int(pose[-1])

        for camera_name in camera_names:
            img = cv2.imread(path_name + camera_name + f"/{img_id}.jpg")

            # Check that image was able to be loaded
            if not isinstance(img, np.ndarray):
                continue

            center_x = img_map.cameras[camera_name].cx
            center_y = img_map.cameras[camera_name].cy

            img[:, int(center_x), :] = (0, 255, 255)
            img[int(center_y), :, :] = (0, 255, 255)

            cv2.imwrite(path_name + f"centers/centers_{img_id}.jpg", img)
