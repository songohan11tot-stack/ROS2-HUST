
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()
        self.pub_target = self.create_publisher(PointStamped, '/target_point', 10)

        # Load YOLOv8 model
        self.model = YOLO('/home/kien/ROS2-HUST/best.pt')
        self.names = self.model.names

        # Subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)

        # Default intrinsics
        self.fx = 615.0; self.fy = 615.0
        self.cx = 319.5; self.cy = 239.5
        self.depth_image = None

        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    def info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

    def color_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Tính điểm trung tâm của bounding box
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2
            if self.depth_image is None:
                continue

            # Đọc chiều sâu (mm → m)
            z = float(self.depth_image[v, u]) / 1000.0
            if z == 0.0:
                continue

            # Tính toạ độ trong khung hình camera
            X_cam = (u - self.cx) * z / self.fx
            Y_cam = (v - self.cy) * z / self.fy
            Z_cam = z

            # Chuyển đổi sang base_link bằng công thức thủ công
            Xb, Yb, Zb = self.transform_point_camera_to_base(Z_cam, -X_cam, -Y_cam)

            # Publish trong base_link
            pt = PointStamped()
            pt.header.stamp = msg.header.stamp
            pt.header.frame_id = 'base_link'
            pt.point.x = float(Xb)
            pt.point.y = float(Yb)
            pt.point.z = float(Zb)
            self.pub_target.publish(pt)
            self.get_logger().info(f"📤 Publish BASE_LINK: X={Xb:.3f}, Y={Yb:.3f}, Z={Zb:.3f}")

            # Hiển thị thông tin
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (u, v), 4, (0,0,255), -1)
            label = f'{self.names[cls]}:{conf:.2f}'
            text3d = f'Xb={Xb:.2f} Yb={Yb:.2f} Zb={Zb:.2f}'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, text3d, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        cv2.imshow('Detection', frame)
        cv2.resizeWindow('Detection',800,600)
        cv2.waitKey(1)

    def transform_point_camera_to_base(self, x_c, y_c, z_c):
        # Các góc Euler: roll, pitch, yaw (rad)
        phi1 = 0.0    # roll
        phi2 = 0.14   # pitch
        phi3 = 0.0    # yaw

        # Vị trí camera so với base_link
        tx, ty, tz = -0.06, 0.08, 0.45

        # Ma trận quay Euler (Z * Y * X)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi1), -np.sin(phi1)],
            [0, np.sin(phi1),  np.cos(phi1)]
        ])
        R_y = np.array([
            [np.cos(phi2), 0, np.sin(phi2)],
            [0, 1, 0],
            [-np.sin(phi2), 0, np.cos(phi2)]
        ])
        R_z = np.array([
            [np.cos(phi3), -np.sin(phi3), 0],
            [np.sin(phi3),  np.cos(phi3), 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y @ R_x

        p_cam = np.array([[x_c], [y_c], [z_c]])
        p_base = R @ p_cam + np.array([[tx], [ty], [tz]])
        return p_base.flatten()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
