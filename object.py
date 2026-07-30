# b3rb_ros_object_recog.py

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

DEFAULT_CLASS_NAMES = {
    0: "A",
    1: "B",
    2: "C",
    3: "Left",
    4: "Right",
    5: "Straight",
    6: "X",
    7: "Y",
    8: "Z",
}

LETTER_CLASSES = {"A", "B", "C", "X", "Y", "Z"}
DIRECTION_CLASSES = {"Left", "Right", "Straight"}

CONFIDENCE_THRESHOLD = 0.50
MAX_COLUMN_X_DIFF_PX = 80
SIGN_CONFIRM_COUNT = 3

MODEL_PATH = (
    "/home/edith/cognipilot/cranium/src/"
    "b3rb_ros_line_follower/"
    "b3rb_ros_line_follower/"
    "b3rb_ros_line_follower/"
    "best.pt"
)


class ObjectRecognizer(Node):

    def __init__(self):
        super().__init__("object_recognizer")

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.camera_image_callback,
            10,
        )

        self.publisher_sign = self.create_publisher(
            String,
            "/sign_board_detection",
            10,
        )

        self.model = None
        self.class_names = DEFAULT_CLASS_NAMES

        if YOLO is None:
            self.get_logger().error(
                "Ultralytics not installed.\n"
                "Install using:\n"
                "pip install ultralytics --break-system-packages"
            )
        elif os.path.exists(MODEL_PATH):
            try:
                self.model = YOLO(MODEL_PATH)
                if getattr(self.model, "names", None):
                    self.class_names = self.model.names
                self.get_logger().info(f"Loaded YOLO model:\n{MODEL_PATH}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        else:
            self.get_logger().error(f"best.pt NOT FOUND:\n{MODEL_PATH}")

        self._streak_map = None
        self._streak_count = 0

        cv2.namedWindow("YOLO Sign Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Sign Detection", 900, 600)

        self.get_logger().info("Object Recognizer started.")

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None or self.model is None:
            return

        sign_map = self.classify_signs(image)
        self._debounce_and_publish(sign_map)

    def _debounce_and_publish(self, sign_map):
        if not sign_map:
            self._streak_map = None
            self._streak_count = 0
            return

        if sign_map == self._streak_map:
            self._streak_count += 1
        else:
            self._streak_map = sign_map
            self._streak_count = 1

        if self._streak_count == SIGN_CONFIRM_COUNT:
            payload = ",".join(
                f"{letter}_{direction.upper()}"
                for letter, direction in sorted(sign_map.items())
            )
            msg = String()
            msg.data = payload
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"Detected Sign Board: {payload}")

    def classify_signs(self, image):
        try:
            results = self.model.predict(
                image,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(str(e))
            return None

        if not results:
            return None

        result = results[0]

        annotated = result.plot()
        cv2.imshow("YOLO Sign Detection", annotated)
        cv2.waitKey(1)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        letters = []
        directions = []

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.class_names.get(cls)

            if name is None:
                continue

            xyxy = box.xyxy[0]
            cx = float((xyxy[0] + xyxy[2]) / 2.0)
            cy = float((xyxy[1] + xyxy[3]) / 2.0)

            detection = {"name": name, "conf": conf, "cx": cx, "cy": cy}

            if name in LETTER_CLASSES:
                letters.append(detection)
            elif name in DIRECTION_CLASSES:
                directions.append(detection)

        if not letters or not directions:
            return None

        sign_map = {}
        used_direction_idx = set()

        for letter in sorted(letters, key=lambda d: d["conf"], reverse=True):
            best_idx = None
            best_x_diff = float("inf")

            for idx, direction in enumerate(directions):
                if idx in used_direction_idx:
                    continue

                x_diff = abs(letter["cx"] - direction["cx"])

                if (
                    x_diff < MAX_COLUMN_X_DIFF_PX
                    and x_diff < best_x_diff
                    and direction["cy"] >= letter["cy"] - 25
                ):
                    best_x_diff = x_diff
                    best_idx = idx

            if best_idx is not None:
                used_direction_idx.add(best_idx)
                sign_map[letter["name"]] = directions[best_idx]["name"]

        return sign_map if sign_map else None


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
