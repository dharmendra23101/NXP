

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

DEFAULT_CLASS_NAMES = {
    0: 'A', 1: 'B', 2: 'C',
    3: 'Left', 4: 'Right', 5: 'Straight',
    6: 'X', 7: 'Y', 8: 'Z',
}

LETTER_CLASSES = {'A', 'B', 'C', 'X', 'Y', 'Z'}
DIRECTION_CLASSES = {'Left', 'Right', 'Straight'}

CONFIDENCE_THRESHOLD = 0.50
MAX_COLUMN_X_DIFF_PX = 80       # Letters & arrows in one banner column must align horizontally
SIGN_CONFIRM_COUNT = 3


class ObjectRecognizer(Node):
    """
    ROS 2 Node that runs YOLOv8 inference on camera frames, pairs letters
    with their column-aligned direction arrow below them, debounces detections,
    and publishes combined labels (e.g. 'A_LEFT', 'Y_RIGHT').
    """

    def __init__(self):
        super().__init__('object_recognizer')

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        self.model = None
        self.class_names = DEFAULT_CLASS_NAMES

        if YOLO is None:
            self.get_logger().error(
                "ultralytics is not installed. Run: "
                "pip install ultralytics --break-system-packages")
        else:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_path, 'best.pt')
            if os.path.exists(model_path):
                try:
                    self.model = YOLO(model_path)
                    if getattr(self.model, 'names', None):
                        self.class_names = self.model.names
                    self.get_logger().info(f"Loaded YOLOv8 model from {model_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load YOLOv8 model: {e}")
            else:
                self.get_logger().error(
                    f"best.pt not found at {model_path} - place weights file in {dir_path}.")

        self._streak_label = None
        self._streak_count = 0

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None or self.model is None:
            return

        label = self.classify_sign(image)
        self._debounce_and_publish(label)

    def _debounce_and_publish(self, label):
        if label is None:
            self._streak_label = None
            self._streak_count = 0
            return

        if label == self._streak_label:
            self._streak_count += 1
        else:
            self._streak_label = label
            self._streak_count = 1

        if self._streak_count == SIGN_CONFIRM_COUNT:
            msg = String()
            msg.data = label
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"Detected Sign Board: {label}")

    def classify_sign(self, image):
        try:
            results = self.model.predict(image, verbose=False, conf=CONFIDENCE_THRESHOLD)
        except Exception as e:
            self.get_logger().debug(f"YOLO inference failed: {e}")
            return None

        if not results:
            return None

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        letters = []
        directions = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.class_names.get(cls_id, None)
            if name is None:
                continue

            xyxy = box.xyxy[0]
            cx = float((xyxy[0] + xyxy[2]) / 2.0)
            cy = float((xyxy[1] + xyxy[3]) / 2.0)

            if name in LETTER_CLASSES:
                letters.append({'name': name, 'conf': conf, 'cx': cx, 'cy': cy})
            elif name in DIRECTION_CLASSES:
                directions.append({'name': name, 'conf': conf, 'cx': cx, 'cy': cy})

        if not letters or not directions:
            return None

        best_pair = None
        best_score = -1.0

        # COLUMN-ALIGNED PAIRING: Pair letters ONLY with arrows below them in the same X-column
        for letter in letters:
            nearest_dir = None
            smallest_x_diff = float('inf')
            for direction in directions:
                x_diff = abs(letter['cx'] - direction['cx'])
                # The arrow must be horizontally aligned in the column and generally below/near the letter
                if x_diff < smallest_x_diff and x_diff < MAX_COLUMN_X_DIFF_PX and (direction['cy'] >= letter['cy'] - 25):
                    smallest_x_diff = x_diff
                    nearest_dir = direction

            if nearest_dir is None:
                continue

            score = letter['conf'] + nearest_dir['conf']
            if score > best_score:
                best_score = score
                best_pair = (letter['name'], nearest_dir['name'])

        if best_pair is None:
            return None

        letter_name, direction_name = best_pair
        return f"{letter_name}_{direction_name.upper()}"


def math_hypot(dx, dy):
    return (dx * dx + dy * dy) ** 0.5


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
