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
#
# ============================================================================
#  NXP CUP 2026 - Autonomous Medical Response
#  "runner" node  ==  the brain of the buggy (AMBULANCE HIGH-PERFORMANCE MODE)
# ============================================================================

import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# ---------------------------------------------------------------------------
# Control bounds (physical limits of the buggy)
# ---------------------------------------------------------------------------
SPEED_MIN = -1.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# ---------------------------------------------------------------------------
# TUNE: High-Performance "Ambulance" Velocity & Lane Gains
# ---------------------------------------------------------------------------
MAX_BOOST_SPEED = 0.80         # Top speed on wide-open straightaways
CRUISE_SPEED = 0.55            # Standard forward speed while lane-following
SLOW_SPEED = 0.15              # Speed while approaching a building / sharp corner
AVOID_MIN_SPEED = 0.12         # Speed floor while squeezing past tight obstacles

STEER_KP_BASE = 1.6            # Proportional steering gain at low/mid speeds
CURVE_KP = 2.0                 # Anticipation gain for upcoming road bends
SHARP_KP = 2.5                 # Quadratic correction for large drift recovery

CAMERA_CENTER_OFFSET_PX = 0.0  # Adjust if camera is slightly off-chassis center

# ---------------------------------------------------------------------------
# TUNE: Dynamic Physics Obstacle Avoidance (LIDAR)
# ---------------------------------------------------------------------------
LIDAR_FRONT_HALF_ANGLE_DEG = 85
LIDAR_SCAN_STEP_DEG = 2
LIDAR_RAY_WINDOW_DEG = 4
GAP_SAFE_DIST = 1.2
MIN_GAP_WIDTH_DEG = 18
FRONT_CENTER_HALF_ANGLE_DEG = 20

# Dynamic lookahead thresholds
AVOID_FAR_DIST_BASE = 1.8      # Base horizon at zero speed
AVOID_NEAR_DIST = 0.45         # Immediate obstacle threshold (100% avoidance)
LOOKAHEAD_VELOCITY_GAIN = 1.2  # Adds extra meters of lookahead per m/s speed

AVOID_STEER_SMOOTH_ALPHA = 0.40 # Filter responsiveness for gap tracking

# ---------------------------------------------------------------------------
# TUNE: Mission Zone & Sign Rules
# ---------------------------------------------------------------------------
QR_CONFIRM_COUNT = 3

SIGN_TO_PATIENT = {"A": "PATIENT_1", "B": "PATIENT_2", "C": "PATIENT_3"}
SIGN_TO_HOSPITAL = {"X": "HOSPITAL_1", "Y": "HOSPITAL_2", "Z": "HOSPITAL_3"}
FAKE_HOSPITALS = {"FAKE_HOSPITAL_1", "FAKE_HOSPITAL_2"}

ALL_PATIENTS = ["PATIENT_1", "PATIENT_2", "PATIENT_3"]

# ---------------------------------------------------------------------------
# Mission States
# ---------------------------------------------------------------------------
S_FIND_PATIENT = "FIND_PATIENT"
S_CONFIRM_PATIENT = "CONFIRM_PATIENT"
S_AWAIT_HOSPITAL_ASSIGN = "AWAIT_HOSPITAL"
S_FIND_HOSPITAL = "FIND_HOSPITAL"
S_CONFIRM_HOSPITAL = "CONFIRM_HOSPITAL"
S_AWAIT_NEXT_PATIENT = "AWAIT_NEXT_PATIENT"
S_MISSION_COMPLETE = "MISSION_COMPLETE"
S_EXIT_TO_PARKING = "EXIT_TO_PARKING"
S_PARKED = "PARKED"


class LineFollower(Node):
    """
    High-Performance Autonomous Medical Response Controller.
    Features dynamic velocity scaling, physics-informed lookahead obstacle
    avoidance, curvature-dependent braking, and full mission state handling.
    """

    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        self.subscription_vectors = self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication',
            self.server_communication_callback, QOS_PROFILE_DEFAULT)
        self.subscription_qr = self.create_subscription(
            String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)
        self.subscription_signs = self.create_subscription(
            String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # ------------------ Drive state ------------------
        self.target_speed = 0.0
        self.target_turn = 0.0
        self.current_speed_estimate = 0.0

        # ------------------ Perception state: lane ------------------
        self.last_sign = None
        self.qr_streak_label = None
        self.qr_streak_count = 0
        self.known_lane_half_width = None
        self.smoothed_offset = 0.0
        self._debug_log_counter = 0

        # ------------------ Perception state: obstacles / LIDAR ------------------
        self.front_min_dist = float('inf')
        self.left_side_dist = float('inf')
        self.right_side_dist = float('inf')
        self.has_valid_gap = True
        self.avoid_target_turn = 0.0
        self.avoidance_influence = 0.0
        self.dynamic_far_dist = AVOID_FAR_DIST_BASE

        # ------------------ Mission state ------------------
        self.state = S_FIND_PATIENT
        self.patients_remaining = list(ALL_PATIENTS)
        self.current_patient = self.patients_remaining[0]
        self.current_hospital = None
        self.delivered_count = 0

        # ------------------ Server protocol bookkeeping ------------------
        self.server_uid_counter = 1
        self.awaiting_ack_for_uid = None

        # 10 Hz control heartbeat
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"Ambulance Brain online. Targeting {self.current_patient} first.")

    # =========================================================================
    # LOW-LEVEL DRIVE
    # =========================================================================

    def publish_drive_commands(self):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        self.target_speed = float(max(min(speed, SPEED_MAX), SPEED_MIN))
        self.target_turn = float(max(min(turn, TURN_MAX), TURN_MIN))
        self.current_speed_estimate = abs(self.target_speed)

    # =========================================================================
    # PERCEPTION CALLBACKS
    # =========================================================================

    def edge_vectors_callback(self, message):
        self.last_vectors_msg = message

    def lidar_callback(self, message):
        """
        Finely sweeps front hemisphere, identifies passable gaps, and computes
        velocity-scaled avoidance influence so high-speed dodging starts earlier.
        """
        n = len(message.ranges)
        if n == 0 or message.angle_increment == 0:
            return

        angle_min = message.angle_min
        angle_inc = message.angle_increment

        def index_for_angle(angle_rad):
            idx = int(round((angle_rad - angle_min) / angle_inc))
            return max(0, min(n - 1, idx))

        def ray_dist(angle_deg, window_deg=LIDAR_RAY_WINDOW_DEG):
            center = math.radians(angle_deg)
            half = math.radians(window_deg)
            i_lo = index_for_angle(center - half)
            i_hi = index_for_angle(center + half)
            lo, hi = min(i_lo, i_hi), max(i_lo, i_hi)
            vals = [r for r in message.ranges[lo:hi + 1]
                    if r > 0.01 and not math.isinf(r) and not math.isnan(r)]
            return min(vals) if vals else float('inf')

        self.front_min_dist = ray_dist(0.0, FRONT_CENTER_HALF_ANGLE_DEG)
        self.left_side_dist = ray_dist(70.0, 15.0)
        self.right_side_dist = ray_dist(-70.0, 15.0)

        # Dynamic lookahead: extend detection horizon when moving fast
        self.dynamic_far_dist = AVOID_FAR_DIST_BASE + (LOOKAHEAD_VELOCITY_GAIN * self.current_speed_estimate)

        angles_deg = list(range(-LIDAR_FRONT_HALF_ANGLE_DEG,
                                LIDAR_FRONT_HALF_ANGLE_DEG + 1,
                                LIDAR_SCAN_STEP_DEG))
        distances = [ray_dist(a) for a in angles_deg]

        gaps = []
        run_start = None
        last_idx = len(distances) - 1
        for i, d in enumerate(distances):
            open_here = d > GAP_SAFE_DIST
            if open_here and run_start is None:
                run_start = i
            if run_start is not None and (not open_here or i == last_idx):
                run_end = i if open_here else i - 1
                width_deg = angles_deg[run_end] - angles_deg[run_start]
                if width_deg >= MIN_GAP_WIDTH_DEG:
                    center_angle = (angles_deg[run_start] + angles_deg[run_end]) / 2.0
                    gaps.append((center_angle, width_deg))
                run_start = None

        if gaps:
            best_gap_angle, _ = min(gaps, key=lambda g: abs(g[0]))
            self.has_valid_gap = True
        else:
            best_gap_angle = angles_deg[distances.index(max(distances))]
            self.has_valid_gap = False

        raw_avoid_turn = max(-1.0, min(1.0, best_gap_angle / 45.0))

        self.avoid_target_turn = (
            AVOID_STEER_SMOOTH_ALPHA * raw_avoid_turn
            + (1 - AVOID_STEER_SMOOTH_ALPHA) * self.avoid_target_turn
        )

        # Smooth repulsive blend based on dynamic speed-scaled horizon
        if self.front_min_dist >= self.dynamic_far_dist:
            self.avoidance_influence = 0.0
        elif self.front_min_dist <= AVOID_NEAR_DIST:
            self.avoidance_influence = 1.0
        else:
            span = self.dynamic_far_dist - AVOID_NEAR_DIST
            frac = (self.dynamic_far_dist - self.front_min_dist) / span
            # Quadratic curve gives smoother, physics-like repulsive pressure
            self.avoidance_influence = min(1.0, max(0.0, frac ** 1.5))

    def sign_board_callback(self, message):
        self.last_sign = message.data
        self.get_logger().info(f"Sign seen: {message.data}")

    def parse_sign(self, sign_str):
        if not sign_str or "_" not in sign_str:
            return None, None
        letter, direction = sign_str.split("_", 1)
        return letter.strip().upper(), direction.strip().upper()

    def qr_detection_callback(self, message):
        label = message.data
        if label == self.qr_streak_label:
            self.qr_streak_count += 1
        else:
            self.qr_streak_label = label
            self.qr_streak_count = 1

        if self.qr_streak_count >= QR_CONFIRM_COUNT:
            self.handle_confirmed_qr(label)

    def handle_confirmed_qr(self, label):
        if label in FAKE_HOSPITALS:
            self.get_logger().warn(f"FAKE HOSPITAL detected ({label}) - ignoring.")
            return

        if self.state == S_FIND_PATIENT and label == self.current_patient:
            self.get_logger().info(f"Confirmed patient QR: {label}")
            self.state = S_CONFIRM_PATIENT
        elif self.state == S_FIND_HOSPITAL and label == self.current_hospital:
            self.get_logger().info(f"Confirmed hospital QR: {label}")
            self.state = S_CONFIRM_HOSPITAL
        elif self.state == S_FIND_HOSPITAL and label.startswith("HOSPITAL") and label != self.current_hospital:
            self.get_logger().warn(f"Wrong hospital nearby ({label}), target is {self.current_hospital}.")

    # =========================================================================
    # SERVER COMMUNICATION
    # =========================================================================

    def send_server_update(self, text_msg):
        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        server_msg.uid = self.server_uid_counter
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.awaiting_ack_for_uid = self.server_uid_counter
        self.server_uid_counter += 1
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"-> Server: {text_msg} (uid={server_msg.uid})")

    def server_communication_callback(self, message):
        if message.dest != 1:
            return

        self.get_logger().info(f"<- Server: msg='{message.msg}' ack={message.ack} uid={message.uid}")

        if message.ack == 1 and message.uid == self.awaiting_ack_for_uid:
            self.awaiting_ack_for_uid = None
            self.process_server_payload(message.msg)
            return

        if message.msg:
            self.process_server_payload(message.msg)

    def process_server_payload(self, payload):
        payload = payload.strip().upper()
        if payload == "INVALID":
            self.get_logger().warn("Server said INVALID - likely outside zone.")
            return

        if payload.startswith("HOSPITAL_") and self.state == S_AWAIT_HOSPITAL_ASSIGN:
            self.current_hospital = payload
            self.get_logger().info(f"Assigned hospital: {self.current_hospital}")
            self.state = S_FIND_HOSPITAL
            return

        if payload.startswith("PATIENT_") and self.state == S_AWAIT_NEXT_PATIENT:
            if payload in self.patients_remaining:
                self.current_patient = payload
                self.get_logger().info(f"Next patient assigned: {self.current_patient}")
                self.state = S_FIND_PATIENT
            return

    # =========================================================================
    # MISSION STATE MACHINE
    # =========================================================================

    def control_loop(self):
        if self.state in (S_FIND_PATIENT, S_FIND_HOSPITAL):
            self.drive_lane_following()
        elif self.state == S_CONFIRM_PATIENT:
            self.rover_move_manual_mode(0.0, 0.0)
            if self.awaiting_ack_for_uid is None:
                self.send_server_update(self.current_patient)
                self.state = S_AWAIT_HOSPITAL_ASSIGN
        elif self.state == S_AWAIT_HOSPITAL_ASSIGN:
            self.rover_move_manual_mode(0.0, 0.0)
        elif self.state == S_CONFIRM_HOSPITAL:
            self.rover_move_manual_mode(0.0, 0.0)
            if self.awaiting_ack_for_uid is None:
                self.send_server_update(self.current_hospital)
                self.on_patient_delivered()
        elif self.state == S_AWAIT_NEXT_PATIENT:
            self.rover_move_manual_mode(0.0, 0.0)
        elif self.state == S_MISSION_COMPLETE:
            self.drive_lane_following()
            self.state = S_EXIT_TO_PARKING
        elif self.state == S_EXIT_TO_PARKING:
            self.drive_lane_following()
            if self.is_in_parking_area():
                self.rover_move_manual_mode(0.0, 0.0)
                self.send_server_update("PARKED")
                self.state = S_PARKED
        elif self.state == S_PARKED:
            self.rover_move_manual_mode(0.0, 0.0)

        self.publish_drive_commands()

    def is_in_parking_area(self):
        return False

    def on_patient_delivered(self):
        self.delivered_count += 1
        if self.current_patient in self.patients_remaining:
            self.patients_remaining.remove(self.current_patient)
        self.get_logger().info(
            f"Delivered {self.current_patient} -> {self.current_hospital} ({self.delivered_count}/3)")

        self.current_hospital = None
        if self.delivered_count >= 3:
            self.state = S_MISSION_COMPLETE
        else:
            self.state = S_AWAIT_NEXT_PATIENT

    # =========================================================================
    # DYNAMIC VELOCITY & PHYSICS-INFORMED LANE / AVOIDANCE CONTROLLER
    # =========================================================================

    def drive_lane_following(self):
        """
        Computes dynamic 'Ambulance' velocity and blends lane steering with
        LIDAR avoidance based on a velocity-dependent lookahead horizon.
        """
        vectors = getattr(self, 'last_vectors_msg', None)

        lane_turn = 0.0
        have_lane_signal = False
        curvature_signal = 0.0

        if vectors is not None and vectors.vector_count > 0:
            half_width = vectors.image_width / 2.0
            midpoint = None

            if vectors.vector_count == 2:
                left_near_x = vectors.vector_1[1].x
                right_near_x = vectors.vector_2[1].x
                midpoint = (left_near_x + right_near_x) / 2.0
                self.known_lane_half_width = abs(right_near_x - left_near_x) / 2.0

                left_tilt = vectors.vector_1[0].x - vectors.vector_1[1].x
                right_tilt = vectors.vector_2[0].x - vectors.vector_2[1].x
                curvature_signal = (left_tilt + right_tilt) / 2.0

            elif vectors.vector_count == 1:
                edge_x = vectors.vector_1[1].x
                half_lane = self.known_lane_half_width if self.known_lane_half_width else half_width * 0.5
                if edge_x < half_width:
                    midpoint = edge_x + half_lane
                else:
                    midpoint = edge_x - half_lane

            if midpoint is not None:
                target_center = half_width + CAMERA_CENTER_OFFSET_PX
                raw_offset = midpoint - target_center
                alpha = 0.4
                self.smoothed_offset = alpha * raw_offset + (1 - alpha) * self.smoothed_offset

                # Scale steering KP down slightly at high velocity to prevent oscillation
                speed_damping = max(0.65, 1.0 - (self.current_speed_estimate * 0.35))
                effective_steer_kp = STEER_KP_BASE * speed_damping

                position_term = -effective_steer_kp * (self.smoothed_offset / half_width)
                curvature_term = -CURVE_KP * (curvature_signal / half_width)
                normalized = self.smoothed_offset / half_width
                sharp_term = -math.copysign(SHARP_KP * (normalized ** 2), self.smoothed_offset)

                lane_turn = position_term + curvature_term + sharp_term
                have_lane_signal = True

                self._debug_log_counter += 1
                if self._debug_log_counter % 10 == 0:
                    self.get_logger().info(
                        f"[steer] spd={self.target_speed:.2f} turn={lane_turn:.2f} | "
                        f"front={self.front_min_dist:.2f}m infl={self.avoidance_influence:.2f}")

        if not have_lane_signal:
            lane_turn = 0.0

        lane_turn = self.apply_sign_guidance(lane_turn)

        # ---------------- Blend with gap-seeking avoidance ----------------
        influence = self.avoidance_influence
        turn = (1.0 - influence) * lane_turn + influence * self.avoid_target_turn
        turn = max(TURN_MIN, min(TURN_MAX, turn))

        # ---------------- Physics-Based Dynamic Speed Calculation ----------------
        front_dist = self.front_min_dist

        # 1. Obstacle Proximity Factor (Exponential/Quadratic braking)
        if front_dist >= self.dynamic_far_dist:
            proximity_factor = 1.0
        elif front_dist <= AVOID_NEAR_DIST:
            proximity_factor = AVOID_MIN_SPEED / CRUISE_SPEED
        else:
            span = self.dynamic_far_dist - AVOID_NEAR_DIST
            frac = (front_dist - AVOID_NEAR_DIST) / span
            # Smooth power curve lets it hold speed longer before decisive braking
            proximity_factor = (AVOID_MIN_SPEED / CRUISE_SPEED) + (1.0 - (AVOID_MIN_SPEED / CRUISE_SPEED)) * (frac ** 1.4)

        # 2. Centripetal Turn Factor (Quadratic curvature slowdown instead of linear cut)
        turn_factor = 1.0 / math.sqrt(1.0 + 3.0 * (turn ** 2))
        turn_factor = max(0.40, turn_factor)

        # 3. Straightaway Boost (When road is clear and straight, accelerate above cruise speed)
        is_clear_straightaway = (front_dist > 2.5) and (abs(turn) < 0.12) and (abs(curvature_signal) < 15.0)
        base_target_speed = MAX_BOOST_SPEED if is_clear_straightaway else CRUISE_SPEED

        speed = base_target_speed * proximity_factor * turn_factor
        speed = max(AVOID_MIN_SPEED, min(MAX_BOOST_SPEED, speed))

        self.rover_move_manual_mode(speed, turn)

    def apply_sign_guidance(self, base_turn):
        letter, direction = self.parse_sign(self.last_sign)
        if letter is None:
            return base_turn

        target_letter = None
        if self.state == S_FIND_PATIENT:
            target_letter = next(
                (k for k, v in SIGN_TO_PATIENT.items() if v == self.current_patient), None)
        elif self.state == S_FIND_HOSPITAL:
            target_letter = next(
                (k for k, v in SIGN_TO_HOSPITAL.items() if v == self.current_hospital), None)

        if letter != target_letter:
            return base_turn

        if direction == "LEFT":
            return max(base_turn, 0.5)
        elif direction == "RIGHT":
            return min(base_turn, -0.5)
        return base_turn


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
