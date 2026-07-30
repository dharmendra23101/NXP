





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


class ObjectRecognizer(Node):
    """
    Runs YOLOv8 on camera images and publishes
    labels like:
        A_LEFT
        B_RIGHT
        X_STRAIGHT
    """

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

        ###########################################################
        # Load YOLO Model
        ###########################################################

        if YOLO is None:
            self.get_logger().error(
                "Ultralytics not installed.\n"
                "Install using:\n"
                "pip install ultralytics --break-system-packages"
            )
        else:

            model_path = (
                "/home/edith/cognipilot/cranium/src/"
                "b3rb_ros_line_follower/"
                "b3rb_ros_line_follower/"
                "b3rb_ros_line_follower/"
                "best.pt"
            )

            if os.path.exists(model_path):

                try:
                    self.model = YOLO(model_path)

                    if getattr(self.model, "names", None):
                        self.class_names = self.model.names

                    self.get_logger().info(
                        f"Loaded YOLO model:\n{model_path}"
                    )

                except Exception as e:
                    self.get_logger().error(
                        f"Failed to load model: {e}"
                    )

            else:
                self.get_logger().error(
                    f"best.pt NOT FOUND:\n{model_path}"
                )

        ###########################################################

        self._streak_label = None
        self._streak_count = 0

        ###########################################################
        # OpenCV Window
        ###########################################################

        cv2.namedWindow("YOLO Sign Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO Sign Detection", 900, 600)

        self.get_logger().info(
            "Object Recognizer started."
        )

    ###############################################################

    def camera_image_callback(self, message):

        np_arr = np.frombuffer(message.data, np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        if self.model is None:
            return

        label = self.classify_sign(image)

        self._debounce_and_publish(label)

    ###############################################################

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

            self.get_logger().info(
                f"Detected Sign Board: {label}"
            )

    ###############################################################

    def classify_sign(self, image):

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

        ###########################################################
        # Display YOLO detections
        ###########################################################

        annotated = result.plot()

        cv2.imshow("YOLO Sign Detection", annotated)
        cv2.waitKey(1)

        ###########################################################

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

            detection = {
                "name": name,
                "conf": conf,
                "cx": cx,
                "cy": cy,
            }

            if name in LETTER_CLASSES:
                letters.append(detection)

            elif name in DIRECTION_CLASSES:
                directions.append(detection)

        if not letters or not directions:
            return None

        ###########################################################
        # Pair letters with nearest arrow
        ###########################################################

        best_pair = None
        best_score = -1

        for letter in letters:

            nearest = None
            smallest_x = float("inf")

            for direction in directions:

                x_diff = abs(letter["cx"] - direction["cx"])

                if (
                    x_diff < MAX_COLUMN_X_DIFF_PX
                    and x_diff < smallest_x
                    and direction["cy"] >= letter["cy"] - 25
                ):

                    smallest_x = x_diff
                    nearest = direction

            if nearest is None:
                continue

            score = letter["conf"] + nearest["conf"]

            if score > best_score:

                best_score = score

                best_pair = (
                    letter["name"],
                    nearest["name"],
                )

        if best_pair is None:
            return None

        letter, direction = best_pair

        return f"{letter}_{direction.upper()}"


###############################################################


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


###############################################################

if __name__ == "__main__":
    main()
