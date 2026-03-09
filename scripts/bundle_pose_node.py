#!/usr/bin/env python3
"""
AprilTag bundle pose estimation from multi-tag corner correspondences.

This node subscribes to AprilTag detections and camera intrinsics, then
solves a single PnP problem per bundle using all visible member tag corners.
"""

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from apriltag_msgs.msg import AprilTagDetectionArray
from tf2_ros import TransformBroadcaster


def pose_to_matrix(position, orientation):
    """Convert position (x,y,z) and quaternion (qx,qy,qz,qw) to 4x4 matrix."""
    x, y, z = position
    qx, qy, qz, qw = orientation

    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-10:
        return np.eye(4)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose(T):
    """Convert 4x4 matrix to ((x,y,z), (qx,qy,qz,qw))."""
    position = T[:3, 3]
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return tuple(position), (qx / norm, qy / norm, qz / norm, qw / norm)


def rvec_tvec_to_matrix(rvec, tvec):
    """Convert OpenCV Rodrigues rotation and translation to 4x4 matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def build_tag_corners_in_bundle(size, T_bundle_from_tag):
    """Return the 4 tag corners expressed in the bundle frame."""
    s = size / 2.0
    local_corners = np.array([
        [-s, -s, 0.0, 1.0],
        [s, -s, 0.0, 1.0],
        [s, s, 0.0, 1.0],
        [-s, s, 0.0, 1.0],
    ], dtype=np.float64)
    return (T_bundle_from_tag @ local_corners.T).T[:, :3]


class BundlePoseNode(Node):

    def __init__(self):
        super().__init__('bundle_pose_node')

        import os
        default_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cfg', 'bundle.yaml'
        )
        self.declare_parameter('bundle_config_file', default_config)
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_offset_z', -0.04)

        config_file = self.get_parameter('bundle_config_file').get_parameter_value().string_value
        if not config_file:
            self.get_logger().error('bundle_config_file parameter is required')
            raise RuntimeError('bundle_config_file parameter is required')

        self.camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        oz = self.get_parameter('object_offset_z').get_parameter_value().double_value
        self.T_bundle_object = pose_to_matrix((0.0, 0.0, oz), (0.0, 0.0, 0.0, 1.0))
        self.get_logger().info(f'Object offset from bundle origin: z={oz}m')

        self.bundles = self._load_bundles(config_file)
        self.get_logger().info(
            f'Loaded {len(self.bundles)} bundle(s): '
            f'{[b["name"] for b in self.bundles]}'
        )

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )
        self.sub = self.create_subscription(
            AprilTagDetectionArray, 'detections', self.detection_callback, 10
        )

        self.pose_pub = self.create_publisher(PoseStamped, 'bundle_pose', 10)
        self.distance_pub = self.create_publisher(Float64, 'bundle_distance', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'bundle_markers', 10)

    def camera_info_callback(self, msg):
        """Cache intrinsics for rectified images."""
        use_p = len(msg.p) >= 12 and msg.p[0] != 0.0 and msg.p[5] != 0.0
        if use_p:
            fx = float(msg.p[0])
            fy = float(msg.p[5])
            cx = float(msg.p[2])
            cy = float(msg.p[6])
        else:
            fx = float(msg.k[0])
            fy = float(msg.k[4])
            cx = float(msg.k[2])
            cy = float(msg.k[5])

        self.camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _load_bundles(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        bundles = []
        for bundle_def in config.get('tag_bundles', []):
            bundle = {'name': bundle_def['name'], 'members': {}}
            for tag in bundle_def['layout']:
                tag_id = int(tag['id'])
                size = float(tag['size'])
                x = float(tag.get('x', 0.0))
                y = float(tag.get('y', 0.0))
                z = float(tag.get('z', 0.0))
                qw = float(tag.get('qw', 1.0))
                qx = float(tag.get('qx', 0.0))
                qy = float(tag.get('qy', 0.0))
                qz = float(tag.get('qz', 0.0))

                # This transform maps points from the tag frame into the bundle frame.
                T_bundle_from_tag = pose_to_matrix((x, y, z), (qx, qy, qz, qw))

                bundle['members'][tag_id] = {
                    'size': size,
                    'corners_in_bundle': build_tag_corners_in_bundle(size, T_bundle_from_tag),
                }
                self.get_logger().info(
                    f'  tag {tag_id}: pos=({x:.4f}, {y:.4f}, {z:.4f})'
                )
            bundles.append(bundle)
        return bundles

    def _compute_reprojection_error(self, object_points, image_points, rvec, tvec):
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))

    def detection_callback(self, msg):
        if self.camera_matrix is None:
            self.get_logger().warn(
                'Waiting for camera_info before bundle pose estimation.',
                throttle_duration_sec=5.0
            )
            return

        detections_by_id = {det.id: det for det in msg.detections}

        for bundle in self.bundles:
            object_points = []
            image_points = []
            used_tag_ids = []

            for tag_id, member in bundle['members'].items():
                detection = detections_by_id.get(tag_id)
                if detection is None:
                    continue

                for object_corner, image_corner in zip(
                    member['corners_in_bundle'], detection.corners
                ):
                    object_points.append(object_corner)
                    image_points.append((float(image_corner.x), float(image_corner.y)))
                used_tag_ids.append(tag_id)

            if not object_points:
                continue

            object_points = np.asarray(object_points, dtype=np.float64)
            image_points = np.asarray(image_points, dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                self.get_logger().warn(
                    f'PnP failed for bundle {bundle["name"]}.',
                    throttle_duration_sec=1.0
                )
                continue

            T_camera_bundle = rvec_tvec_to_matrix(rvec, tvec)
            bundle_pos, bundle_quat = matrix_to_pose(T_camera_bundle)

            T_camera_object = T_camera_bundle @ self.T_bundle_object
            obj_pos, obj_quat = matrix_to_pose(T_camera_object)
            obj_pos = np.array(obj_pos)
            obj_quat = np.array(obj_quat)

            reproj_error = self._compute_reprojection_error(
                object_points, image_points, rvec, tvec
            )

            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = self.camera_frame
            pose_msg.pose.position.x = float(obj_pos[0])
            pose_msg.pose.position.y = float(obj_pos[1])
            pose_msg.pose.position.z = float(obj_pos[2])
            pose_msg.pose.orientation.x = float(obj_quat[0])
            pose_msg.pose.orientation.y = float(obj_quat[1])
            pose_msg.pose.orientation.z = float(obj_quat[2])
            pose_msg.pose.orientation.w = float(obj_quat[3])
            self.pose_pub.publish(pose_msg)

            tf_bundle = TransformStamped()
            tf_bundle.header = msg.header
            tf_bundle.header.frame_id = self.camera_frame
            tf_bundle.child_frame_id = bundle['name']
            tf_bundle.transform.translation.x = float(bundle_pos[0])
            tf_bundle.transform.translation.y = float(bundle_pos[1])
            tf_bundle.transform.translation.z = float(bundle_pos[2])
            tf_bundle.transform.rotation.x = float(bundle_quat[0])
            tf_bundle.transform.rotation.y = float(bundle_quat[1])
            tf_bundle.transform.rotation.z = float(bundle_quat[2])
            tf_bundle.transform.rotation.w = float(bundle_quat[3])

            tf_object = TransformStamped()
            tf_object.header = msg.header
            tf_object.header.frame_id = self.camera_frame
            tf_object.child_frame_id = 'object'
            tf_object.transform.translation.x = float(obj_pos[0])
            tf_object.transform.translation.y = float(obj_pos[1])
            tf_object.transform.translation.z = float(obj_pos[2])
            tf_object.transform.rotation.x = float(obj_quat[0])
            tf_object.transform.rotation.y = float(obj_quat[1])
            tf_object.transform.rotation.z = float(obj_quat[2])
            tf_object.transform.rotation.w = float(obj_quat[3])

            self.tf_broadcaster.sendTransform([tf_bundle, tf_object])

            dist = float(np.linalg.norm(obj_pos))
            dist_msg = Float64()
            dist_msg.data = dist
            self.distance_pub.publish(dist_msg)

            self._publish_markers(msg.header, obj_pos, obj_quat, dist, len(used_tag_ids))

            self.get_logger().info(
                f'Bundle {bundle["name"]}: {len(used_tag_ids)} tag(s), '
                f'ids={sorted(used_tag_ids)}, {len(object_points)} points, '
                f'reproj={reproj_error:.2f}px, '
                f'obj dist={dist:.3f}m',
                throttle_duration_sec=1.0
            )

    def _publish_markers(self, header, pos, quat, dist, num_tags):
        markers = MarkerArray()

        m = Marker()
        m.header.frame_id = self.camera_frame
        m.header.stamp = header.stamp
        m.ns = 'bundle'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.x = float(quat[0])
        m.pose.orientation.y = float(quat[1])
        m.pose.orientation.z = float(quat[2])
        m.pose.orientation.w = float(quat[3])
        m.scale.x = 0.08
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 0.6
        m.lifetime.nanosec = 500000000
        markers.markers.append(m)

        line = Marker()
        line.header.frame_id = self.camera_frame
        line.header.stamp = header.stamp
        line.ns = 'bundle'
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.003
        line.color.g = 1.0
        line.color.b = 1.0
        line.color.a = 0.8
        line.points.append(Point(x=0.0, y=0.0, z=0.0))
        line.points.append(Point(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
        line.lifetime.nanosec = 500000000
        markers.markers.append(line)

        text = Marker()
        text.header.frame_id = self.camera_frame
        text.header.stamp = header.stamp
        text.ns = 'bundle'
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(pos[0] / 2.0)
        text.pose.position.y = float(pos[1] / 2.0)
        text.pose.position.z = float(pos[2] / 2.0 + 0.03)
        text.scale.z = 0.03
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f'{dist:.3f}m ({num_tags} tags)'
        text.lifetime.nanosec = 500000000
        markers.markers.append(text)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = BundlePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
