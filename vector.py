

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from synapse_msgs.msg import EdgeVectors
import cv2
import numpy as np


class EdgeVectorPublisher(Node):
    """
    ROS 2 Node that extracts black track lane boundary markings from camera frames
    using HSV color thresholding and publishes left/right EdgeVectors.
    Protected against T-junction horizontal bars and cross-line jumping.
    """

    def __init__(self):
        super().__init__('edge_vectors_publisher')

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            10)

        # Publishers for visual debugging in Foxglove / rviz
        self.publisher_thresh = self.create_publisher(
            CompressedImage,
            '/debug_images/thresh_image',
            10)
        self.publisher_vector_img = self.create_publisher(
            CompressedImage,
            '/debug_images/vector_image',
            10)

        self.get_logger().info("Edge Vectors Publisher Node started.")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        height, width, _ = image.shape

        # --- 1. HSV Segmentation for Black Road Markings ---
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 80], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Mask out top 50% of the frame (prevents distant T-junction horizontal lines from distorting vectors)
        mask[0:int(height * 0.50), :] = 0

        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- 2. Extract Contours & Form Lane Vectors ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        left_vector = None
        right_vector = None
        mid_x = width / 2.0

        valid_lines = []
        for c in contours:
            if cv2.contourArea(c) < 100:
                continue
            vx, vy, x0, y0 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            # SLOPE FILTER: Reject horizontal-ish lines (e.g., T-junction top bars or crosswalks)
            if abs(vy) < 0.35:
                continue

            # Calculate x at top (y=height*0.55) and bottom (y=height) of active ROI
            y_bottom = float(height)
            y_top = float(height * 0.55)
            x_bottom = x0 + (y_bottom - y0) * (vx / vy)
            x_top = x0 + (y_top - y0) * (vx / vy)

            valid_lines.append((x_bottom, x_top, y_bottom, y_top))

        # Assign closest valid line on the left and right side of frame center
        left_candidates = [l for l in valid_lines if l[0] < mid_x]
        right_candidates = [l for l in valid_lines if l[0] >= mid_x]

        if left_candidates:
            best_l = max(left_candidates, key=lambda l: l[0])
            left_vector = [best_l[1], y_top, best_l[0], y_bottom]

        if right_candidates:
            best_r = min(right_candidates, key=lambda l: l[0])
            right_vector = [best_r[1], y_top, best_r[0], y_bottom]

        # --- 3. Publish EdgeVectors Message ---
        msg = EdgeVectors()
        msg.image_height = height
        msg.image_width = width
        msg.vector_count = 0

        if left_vector is not None:
            msg.vector_count += 1
            p0 = Point(x=float(left_vector[0]), y=float(left_vector[1]), z=0.0)
            p1 = Point(x=float(left_vector[2]), y=float(left_vector[3]), z=0.0)
            msg.vector_1 = [p0, p1]

        if right_vector is not None:
            msg.vector_count += 1
            p0 = Point(x=float(right_vector[0]), y=float(right_vector[1]), z=0.0)
            p1 = Point(x=float(right_vector[2]), y=float(right_vector[3]), z=0.0)
            if msg.vector_count == 1 and left_vector is None:
                msg.vector_1 = [p0, p1]
            else:
                msg.vector_2 = [p0, p1]

        self.publisher_vectors.publish(msg)

        # --- 4. Visual Debug Publishing ---
        _, thresh_encoded = cv2.imencode('.jpg', mask)
        thresh_msg = CompressedImage()
        thresh_msg.format = 'jpeg'
        thresh_msg.data = np.array(thresh_encoded).tobytes()
        self.publisher_thresh.publish(thresh_msg)

        debug_img = image.copy()
        if left_vector:
            cv2.line(debug_img,
                     (int(left_vector[0]), int(left_vector[1])),
                     (int(left_vector[2]), int(left_vector[3])),
                     (0, 255, 0), 3)
        if right_vector:
            cv2.line(debug_img,
                     (int(right_vector[0]), int(right_vector[1])),
                     (int(right_vector[2]), int(right_vector[3])),
                     (0, 255, 0), 3)

        _, debug_encoded = cv2.imencode('.jpg', debug_img)
        vector_img_msg = CompressedImage()
        vector_img_msg.format = 'jpeg'
        vector_img_msg.data = np.array(debug_encoded).tobytes()
        self.publisher_vector_img.publish(vector_img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
