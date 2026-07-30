# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None


class QRDetector(Node):
    """
    ROS 2 Node that scans incoming camera frames for QR codes and publishes
    the decoded payload string (e.g. 'PATIENT_1', 'HOSPITAL_2') on
    /qr_detection.
    """

    def __init__(self):
        super().__init__('qr_detector')

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.cv_detector = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.get_logger().info("QR Detector Node started. Waiting for images...")
        if pyzbar is None:
            self.get_logger().warn(
                "pyzbar not installed - falling back to OpenCV QRCodeDetector. "
                "pip install pyzbar for better detection robustness.")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        qr_data = self.detect_qr_code(image)

        if qr_data:
            msg = String()
            msg.data = qr_data
            self.publisher_qr.publish(msg)
            self.get_logger().info(f"Published QR Data: {qr_data}")

    def _preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def detect_qr_code(self, image):
        # Method 1: OpenCV on raw image
        try:
            data, bbox, _ = self.cv_detector.detectAndDecode(image)
            if bbox is not None and data:
                return data
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR (raw) failed: {e}")

        # Method 2: OpenCV on enhanced grayscale
        enhanced = self._preprocess(image)
        try:
            data, bbox, _ = self.cv_detector.detectAndDecode(enhanced)
            if bbox is not None and data:
                return data
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR (enhanced) failed: {e}")

        # Method 3: pyzbar fallback
        if pyzbar is not None:
            try:
                decoded_objects = pyzbar.decode(enhanced)
                for obj in decoded_objects:
                    data = obj.data.decode('utf-8').strip()
                    if data:
                        return data
            except Exception as e:
                self.get_logger().debug(f"pyzbar QR failed: {e}")

        return None


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
