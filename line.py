# b3rb_ros_line_follower.py

import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

SPEED_MIN = -1.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

MAX_BOOST_SPEED = 0.75
CRUISE_SPEED = 0.55
SLOW_SPEED = 0.22
QR_APPROACH_SPEED = 0.18
SIGN_TURN_SPEED = 0.30
AVOID_MIN_SPEED = 0.22

STEER_KP_BASE = 1.7
CURVE_KP = 2.1
SHARP_KP = 2.6

CAMERA_CENTER_OFFSET_PX = 0.0

SAFE_LANE_OFFSET_RATIO = 0.38
SIGN_LANE_OFFSET_RATIO = 0.18
SIGN_CURVE_BOOST = 1.6

LIDAR_FRONT_HALF_ANGLE_DEG = 85
LIDAR_SCAN_STEP_DEG = 2
LIDAR_RAY_WINDOW_DEG = 4
GAP_SAFE_DIST = 1.15
MIN_GAP_WIDTH_DEG = 18
FRONT_CENTER_HALF_ANGLE_DEG = 20
SIDE_RAY_ANGLE_DEG = 90
SIDE_RAY_WINDOW_DEG = 8

AVOID_FAR_DIST_BASE = 1.8
AVOID_NEAR_DIST = 0.45
LOOKAHEAD_VELOCITY_GAIN = 1.2

AVOID_STEER_SMOOTH_ALPHA = 0.40

NO_SIGNAL_RECOVERY_FRAMES = 6
NEAR_LINE_FRONT_DIST = 0.40
RECOVERY_MAX_TICKS = 25
RECOVERY_TURN = 0.85
RECOVERY_SPEED = 0.16

CURVATURE_STRAIGHT_THRESH = 6.0
STRAIGHT_CONFIRM_FRAMES = 5
TURN_LOCK_MAX_TICKS = 80

QR_CONFIRM_COUNT = 3
QR_STOP_TICKS = 20

ALIGN_SIDE_DIST = 0.90
ALIGN_CONFIRM_FRAMES = 3
APPROACH_MAX_TICKS = 60

SIGN_TO_PATIENT = {"A": "PATIENT_1", "B": "PATIENT_2", "C": "PATIENT_3"}
SIGN_TO_HOSPITAL = {"X": "HOSPITAL_1", "Y": "HOSPITAL_2", "Z": "HOSPITAL_3"}
FAKE_HOSPITALS = {"FAKE_HOSPITAL_1", "FAKE_HOSPITAL_2"}

ALL_PATIENTS = ["PATIENT_1", "PATIENT_2", "PATIENT_3"]

S_FIND_PATIENT = "FIND_PATIENT"
S_APPROACH_PATIENT = "APPROACH_PATIENT"
S_CONFIRM_PATIENT = "CONFIRM_PATIENT"
S_AWAIT_HOSPITAL_ASSIGN = "AWAIT_HOSPITAL"
S_FIND_HOSPITAL = "FIND_HOSPITAL"
S_APPROACH_HOSPITAL = "APPROACH_HOSPITAL"
S_CONFIRM_HOSPITAL = "CONFIRM_HOSPITAL"
S_MISSION_COMPLETE = "MISSION_COMPLETE"
S_EXIT_TO_PARKING = "EXIT_TO_PARKING"
S_AWAIT_PARK_CONFIRM = "AWAIT_PARK_CONFIRM"
S_PARKED = "PARKED"

LANE_DRIVE_STATES = (
    S_FIND_PATIENT, S_APPROACH_PATIENT,
    S_FIND_HOSPITAL, S_APPROACH_HOSPITAL,
)


class LineFollower(Node):

    def __init__(self):
        super().__init__('line_follower')

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

        self.publisher_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)
        self.publisher_mission_status = self.create_publisher(
            String, '/mission_status', QOS_PROFILE_DEFAULT)

        self.target_speed = 0.0
        self.target_turn = 0.0
        self.current_speed_estimate = 0.0

        self.sign_map = {}
        self.sign_timeout_counter = 0
        self.qr_streak_label = None
        self.qr_streak_count = 0
        self.qr_seen_recently_counter = 0
        self.known_lane_half_width = None
        self.smoothed_offset = 0.0
        self._debug_log_counter = 0
        self.last_vectors_msg = None

        self.front_min_dist = float('inf')
        self.left_side_dist = float('inf')
        self.right_side_dist = float('inf')
        self.left_90_dist = float('inf')
        self.right_90_dist = float('inf')
        self.has_valid_gap = True
        self.avoid_target_turn = 0.0
        self.avoidance_influence = 0.0
        self.dynamic_far_dist = AVOID_FAR_DIST_BASE

        self.no_lane_signal_counter = 0
        self.recovering = False
        self.recovery_ticks = 0

        self.turn_lock_direction = None
        self.turn_lock_straight_counter = 0
        self.turn_lock_ticks = 0

        self.approach_ticks = 0
        self.align_counter = 0

        self.dwell_ticks_remaining = 0

        self.state = S_FIND_PATIENT
        self.patients_remaining = list(ALL_PATIENTS)
        self.current_patient = self.patients_remaining[0]
        self.current_hospital = None
        self.delivered_count = 0

        self.server_uid_counter = 1
        self.awaiting_ack_for_uid = None
        self.patient_request_sent = False
        self.hospital_confirm_sent = False

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f"Ambulance Brain online. Targeting {self.current_patient} first.")

    def publish_drive_commands(self):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        self.target_speed = float(max(min(speed, SPEED_MAX), SPEED_MIN))
        self.target_turn = float(max(min(turn, TURN_MAX), TURN_MIN))
        self.current_speed_estimate = abs(self.target_speed)

    def edge_vectors_callback(self, message):
        self.last_vectors_msg = message

    def lidar_callback(self, message):
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
        self.left_90_dist = ray_dist(SIDE_RAY_ANGLE_DEG, SIDE_RAY_WINDOW_DEG)
        self.right_90_dist = ray_dist(-SIDE_RAY_ANGLE_DEG, SIDE_RAY_WINDOW_DEG)

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

        if self.front_min_dist >= self.dynamic_far_dist:
            self.avoidance_influence = 0.0
        elif self.front_min_dist <= AVOID_NEAR_DIST:
            self.avoidance_influence = 1.0
        else:
            span = self.dynamic_far_dist - AVOID_NEAR_DIST
            frac = (self.dynamic_far_dist - self.front_min_dist) / span
            self.avoidance_influence = min(1.0, max(0.0, frac ** 1.5))

    def sign_board_callback(self, message):
        parsed = {}
        for pair in message.data.split(","):
            pair = pair.strip()
            if "_" not in pair:
                continue
            letter, direction = pair.split("_", 1)
            parsed[letter.strip().upper()] = direction.strip().upper()

        if parsed:
            self.sign_map = parsed
            self.sign_timeout_counter = 25
            self.get_logger().info(f"Sign board seen: {parsed}")

    def get_target_letter(self):
        if self.state in (S_FIND_PATIENT, S_APPROACH_PATIENT):
            return next((k for k, v in SIGN_TO_PATIENT.items() if v == self.current_patient), None)
        if self.state in (S_FIND_HOSPITAL, S_APPROACH_HOSPITAL):
            return next((k for k, v in SIGN_TO_HOSPITAL.items() if v == self.current_hospital), None)
        return None

    def get_raw_sign_direction(self):
        if self.sign_timeout_counter <= 0:
            return None
        target_letter = self.get_target_letter()
        if target_letter is None:
            return None
        return self.sign_map.get(target_letter)

    def qr_detection_callback(self, message):
        label = message.data
        self.qr_seen_recently_counter = 20

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
            self.get_logger().info(f"Confirmed target patient QR: {label}, approaching for side alignment.")
            self.state = S_APPROACH_PATIENT
            self.approach_ticks = 0
            self.align_counter = 0
        elif self.state == S_FIND_HOSPITAL and label == self.current_hospital:
            self.get_logger().info(f"Confirmed target hospital QR: {label}, approaching for side alignment.")
            self.state = S_APPROACH_HOSPITAL
            self.approach_ticks = 0
            self.align_counter = 0
        elif self.state == S_FIND_HOSPITAL and label.startswith("HOSPITAL") and label != self.current_hospital:
            self.get_logger().warn(f"Wrong hospital nearby ({label}), target is {self.current_hospital}.")

    def _next_uid(self):
        uid = self.server_uid_counter
        self.server_uid_counter = (self.server_uid_counter + 1) % 256
        return uid

    def send_server_update(self, text_msg):
        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        server_msg.uid = self._next_uid()
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.awaiting_ack_for_uid = server_msg.uid
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"-> Server: {text_msg} (uid={server_msg.uid})")

    def send_ack(self, uid):
        ack_msg = ServerCommunication()
        ack_msg.src = 1
        ack_msg.dest = 2
        ack_msg.uid = uid
        ack_msg.ack = 1
        ack_msg.msg = ""
        self.publisher_server.publish(ack_msg)
        self.get_logger().info(f"-> Server: ACK (uid={uid})")

    def server_communication_callback(self, message):
        if message.dest != 1:
            return

        self.get_logger().info(
            f"<- Server: msg='{message.msg}' ack={message.ack} uid={message.uid}")

        if message.ack == 1:
            if message.uid == self.awaiting_ack_for_uid:
                self.awaiting_ack_for_uid = None
            if message.msg:
                self.process_server_payload(message.msg)
            return

        self.send_ack(message.uid)
        if message.msg:
            self.process_server_payload(message.msg)

    def process_server_payload(self, payload):
        payload = payload.strip().upper()

        if payload == "INVALID":
            if self.state == S_AWAIT_PARK_CONFIRM:
                self.get_logger().warn("Server says parking INVALID - realigning.")
                self.state = S_EXIT_TO_PARKING
            else:
                self.get_logger().warn("Server said INVALID - likely outside zone.")
            return

        if payload == "OK" and self.state == S_AWAIT_PARK_CONFIRM:
            self.get_logger().info("Server confirmed parking OK. Run complete.")
            self.state = S_PARKED
            return

        hospital_name = None
        if payload.startswith("HOSPITAL_"):
            hospital_name = payload
        elif payload in SIGN_TO_HOSPITAL:
            hospital_name = SIGN_TO_HOSPITAL[payload]

        if hospital_name and self.state == S_AWAIT_HOSPITAL_ASSIGN:
            self.current_hospital = hospital_name
            self.get_logger().info(f"Assigned hospital: {self.current_hospital}")
            self.state = S_FIND_HOSPITAL
            return

    def publish_mission_status(self):
        target_letter = self.get_target_letter()
        msg = String()
        msg.data = (
            f"STATE:{self.state} | PATIENT:{self.current_patient} | "
            f"HOSPITAL:{self.current_hospital} | TARGET_LETTER:{target_letter} | "
            f"DELIVERED:{self.delivered_count}/3"
        )
        self.publisher_mission_status.publish(msg)

    def control_loop(self):
        if self.sign_timeout_counter > 0:
            self.sign_timeout_counter -= 1
        else:
            self.sign_map = {}

        if self.qr_seen_recently_counter > 0:
            self.qr_seen_recently_counter -= 1

        if self.state in LANE_DRIVE_STATES:
            self.drive_lane_following()
        elif self.state == S_CONFIRM_PATIENT:
            self.rover_move_manual_mode(0.0, 0.0)
            if self.dwell_ticks_remaining > 0:
                self.dwell_ticks_remaining -= 1
            elif not self.patient_request_sent:
                self.send_server_update(self.current_patient)
                self.patient_request_sent = True
                self.state = S_AWAIT_HOSPITAL_ASSIGN
        elif self.state == S_AWAIT_HOSPITAL_ASSIGN:
            self.rover_move_manual_mode(0.0, 0.0)
        elif self.state == S_CONFIRM_HOSPITAL:
            self.rover_move_manual_mode(0.0, 0.0)
            if self.dwell_ticks_remaining > 0:
                self.dwell_ticks_remaining -= 1
            elif not self.hospital_confirm_sent:
                self.send_server_update(self.current_hospital)
                self.hospital_confirm_sent = True
            elif self.awaiting_ack_for_uid is None:
                self.on_patient_delivered()
        elif self.state == S_MISSION_COMPLETE:
            self.drive_lane_following()
            self.state = S_EXIT_TO_PARKING
        elif self.state == S_EXIT_TO_PARKING:
            self.drive_lane_following()
            if self.is_in_parking_area():
                self.rover_move_manual_mode(0.0, 0.0)
                if self.awaiting_ack_for_uid is None:
                    self.send_server_update("PARKED")
                    self.state = S_AWAIT_PARK_CONFIRM
        elif self.state == S_AWAIT_PARK_CONFIRM:
            self.rover_move_manual_mode(0.0, 0.0)
        elif self.state == S_PARKED:
            self.rover_move_manual_mode(0.0, 0.0)

        self.publish_drive_commands()
        self.publish_mission_status()

    def is_in_parking_area(self):
        return (self.left_side_dist < 0.65 and self.right_side_dist < 0.65)

    def on_patient_delivered(self):
        self.delivered_count += 1
        if self.current_patient in self.patients_remaining:
            self.patients_remaining.remove(self.current_patient)
        self.get_logger().info(
            f"Delivered {self.current_patient} -> {self.current_hospital} ({self.delivered_count}/3)")

        self.current_hospital = None

        if self.delivered_count >= 3 or not self.patients_remaining:
            self.state = S_MISSION_COMPLETE
        else:
            self.current_patient = self.patients_remaining[0]
            self.get_logger().info(f"Next target: {self.current_patient}")
            self.state = S_FIND_PATIENT

    def update_turn_lock(self, raw_direction, curvature_signal, half_width, vector_valid):
        if self.turn_lock_direction is None and raw_direction in ("LEFT", "RIGHT"):
            self.turn_lock_direction = raw_direction
            self.turn_lock_straight_counter = 0
            self.turn_lock_ticks = 0

        if self.turn_lock_direction is None:
            return None

        self.turn_lock_ticks += 1

        if vector_valid:
            is_straight = (
                abs(curvature_signal) < CURVATURE_STRAIGHT_THRESH
                and abs(self.smoothed_offset) < (half_width * 0.15)
            )
            if is_straight:
                self.turn_lock_straight_counter += 1
            else:
                self.turn_lock_straight_counter = 0

        if (self.turn_lock_straight_counter >= STRAIGHT_CONFIRM_FRAMES
                or self.turn_lock_ticks >= TURN_LOCK_MAX_TICKS):
            self.turn_lock_direction = None
            self.turn_lock_straight_counter = 0
            self.turn_lock_ticks = 0
            return None

        return self.turn_lock_direction

    def check_side_alignment(self):
        self.approach_ticks += 1
        min_side = min(self.left_90_dist, self.right_90_dist)

        if min_side < ALIGN_SIDE_DIST:
            self.align_counter += 1
        else:
            self.align_counter = 0

        if (self.align_counter >= ALIGN_CONFIRM_FRAMES
                or self.approach_ticks >= APPROACH_MAX_TICKS):
            self.rover_move_manual_mode(0.0, 0.0)
            next_state = S_CONFIRM_PATIENT if self.state == S_APPROACH_PATIENT else S_CONFIRM_HOSPITAL
            self.get_logger().info(
                f"Aligned beside target (side_dist={min_side:.2f}m) - stopping to contact server.")
            self.state = next_state
            self.dwell_ticks_remaining = QR_STOP_TICKS
            self.patient_request_sent = False
            self.hospital_confirm_sent = False
            self.approach_ticks = 0
            self.align_counter = 0

    def drive_lane_following(self):
        vectors = self.last_vectors_msg

        lane_turn = 0.0
        have_lane_signal = False
        curvature_signal = 0.0
        half_width = 1.0

        raw_sign_direction = self.get_raw_sign_direction()

        if vectors is not None and vectors.vector_count > 0:
            half_width = vectors.image_width / 2.0
            midpoint = None

            follow_direction = self.turn_lock_direction or raw_sign_direction
            offset_ratio = SIGN_LANE_OFFSET_RATIO if follow_direction in ("LEFT", "RIGHT") else SAFE_LANE_OFFSET_RATIO

            if vectors.vector_count == 2:
                left_near_x = vectors.vector_1[1].x
                right_near_x = vectors.vector_2[1].x
                self.known_lane_half_width = abs(right_near_x - left_near_x) / 2.0

                if follow_direction == "LEFT":
                    midpoint = left_near_x + (self.known_lane_half_width * (1.0 - offset_ratio))
                elif follow_direction == "RIGHT":
                    midpoint = right_near_x - (self.known_lane_half_width * (1.0 - offset_ratio))
                else:
                    midpoint = (left_near_x + right_near_x) / 2.0

                left_tilt = vectors.vector_1[0].x - vectors.vector_1[1].x
                right_tilt = vectors.vector_2[0].x - vectors.vector_2[1].x
                curvature_signal = (left_tilt + right_tilt) / 2.0

            elif vectors.vector_count == 1:
                edge_x = vectors.vector_1[1].x
                half_lane = self.known_lane_half_width if self.known_lane_half_width else half_width * 0.5
                if edge_x < half_width:
                    midpoint = edge_x + half_lane * (1.0 - offset_ratio) * 2.0
                else:
                    midpoint = edge_x - half_lane * (1.0 - offset_ratio) * 2.0

            if midpoint is not None:
                target_center = half_width + CAMERA_CENTER_OFFSET_PX
                raw_offset = midpoint - target_center
                alpha = 0.45
                self.smoothed_offset = alpha * raw_offset + (1 - alpha) * self.smoothed_offset

                speed_damping = max(0.65, 1.0 - (self.current_speed_estimate * 0.35))
                effective_steer_kp = STEER_KP_BASE * speed_damping
                curve_kp = CURVE_KP * (SIGN_CURVE_BOOST if follow_direction in ("LEFT", "RIGHT") else 1.0)

                position_term = -effective_steer_kp * (self.smoothed_offset / half_width)
                curvature_term = -curve_kp * (curvature_signal / half_width)
                normalized = self.smoothed_offset / half_width
                sharp_term = -math.copysign(SHARP_KP * (normalized ** 2), self.smoothed_offset)

                lane_turn = position_term + curvature_term + sharp_term
                have_lane_signal = True

                self._debug_log_counter += 1
                if self._debug_log_counter % 10 == 0:
                    self.get_logger().info(
                        f"[steer] spd={self.target_speed:.2f} turn={lane_turn:.2f} | "
                        f"front={self.front_min_dist:.2f}m infl={self.avoidance_influence:.2f} "
                        f"sign_guide={follow_direction} lock={self.turn_lock_direction}")

        locked_direction = self.update_turn_lock(
            raw_sign_direction, curvature_signal, half_width, have_lane_signal)
        follow_direction = locked_direction or raw_sign_direction

        if not have_lane_signal:
            lane_turn = 0.0
            self.no_lane_signal_counter += 1
        else:
            self.no_lane_signal_counter = 0

        near_dead_end = self.front_min_dist < NEAR_LINE_FRONT_DIST
        should_start_recovery = (
            not self.recovering
            and self.turn_lock_direction is None
            and follow_direction is None
            and (self.no_lane_signal_counter >= NO_SIGNAL_RECOVERY_FRAMES or near_dead_end)
        )

        if should_start_recovery:
            self.recovering = True
            self.recovery_ticks = 0
            self.get_logger().warn("Junction ambiguous / line too close - blind rotating left.")

        if self.recovering:
            self.recovery_ticks += 1
            self.rover_move_manual_mode(RECOVERY_SPEED, RECOVERY_TURN)

            recovered = have_lane_signal and self.front_min_dist > NEAR_LINE_FRONT_DIST
            if recovered or self.recovery_ticks >= RECOVERY_MAX_TICKS:
                self.recovering = False
                self.no_lane_signal_counter = 0
            return

        influence = self.avoidance_influence
        turn = (1.0 - influence) * lane_turn + influence * self.avoid_target_turn
        turn = max(TURN_MIN, min(TURN_MAX, turn))

        front_dist = self.front_min_dist

        if front_dist >= self.dynamic_far_dist:
            proximity_factor = 1.0
        elif front_dist <= AVOID_NEAR_DIST:
            proximity_factor = AVOID_MIN_SPEED / CRUISE_SPEED
        else:
            span = self.dynamic_far_dist - AVOID_NEAR_DIST
            frac = (front_dist - AVOID_NEAR_DIST) / span
            proximity_factor = (AVOID_MIN_SPEED / CRUISE_SPEED) + (1.0 - (AVOID_MIN_SPEED / CRUISE_SPEED)) * (frac ** 1.4)

        turn_factor = 1.0 / math.sqrt(1.0 + 3.0 * (turn ** 2))
        turn_factor = max(0.40, turn_factor)

        is_clear_straightaway = (front_dist > 2.5) and (abs(turn) < 0.12) and (abs(curvature_signal) < 15.0)
        base_target_speed = MAX_BOOST_SPEED if is_clear_straightaway else CRUISE_SPEED

        if self.state in (S_APPROACH_PATIENT, S_APPROACH_HOSPITAL):
            speed = QR_APPROACH_SPEED
        elif self.qr_seen_recently_counter > 0:
            speed = QR_APPROACH_SPEED
        elif follow_direction in ("LEFT", "RIGHT"):
            speed = SIGN_TURN_SPEED
        else:
            speed = base_target_speed * proximity_factor * turn_factor
            speed = max(AVOID_MIN_SPEED, min(MAX_BOOST_SPEED, speed))

        self.rover_move_manual_mode(speed, turn)

        if self.state in (S_APPROACH_PATIENT, S_APPROACH_HOSPITAL):
            self.check_side_alignment()


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
