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
#  "runner" node  ==  the brain of the buggy
# ============================================================================
#
#  Fixed obstacle avoidance with a proper 4-stage state machine:
#
#    STAGE 0 - NORMAL LANE FOLLOW
#    STAGE 1 - APPROACH  : obstacle detected at > OBSTACLE_WARN_DIST,
#                           slow down and find the open gap side
#    STAGE 2 - TURN      : steer hard toward the gap side until front clears
#    STAGE 3 - PASS      : go straight beside obstacle until it disappears
#                          on the side we dodged toward
#    STAGE 4 - MERGE     : steer back toward lane center until
#                          lane offset is small again
#
#  Key improvements vs the original skeleton:
#  - Gap direction is chosen by scanning the widest open sector across the
#    whole left/right arc (not just front-left vs front-right which fails
#    when an obstacle occupies 75%+ of the road)
#  - Speed tapers as we approach and is capped low during each stage
#  - "Hysteresis" on stage transitions prevents flip-flopping
#  - All state lives inside self.avoid_* so it never races with callbacks
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
# Control bounds (do not change - these are the buggy's physical limits)
# ---------------------------------------------------------------------------
SPEED_MIN = -1.0
SPEED_MAX = 1.0
TURN_MIN  = -1.0
TURN_MAX  = 1.0

# ---------------------------------------------------------------------------
# TUNE: lane following gains / speeds
# ---------------------------------------------------------------------------
CRUISE_SPEED = 0.22   # normal forward speed while lane-following
SLOW_SPEED   = 0.10   # creep speed when no lane visible / stopping
STEER_KP     = 1.6    # proportional gain
CURVE_KP     = 2.0    # curvature anticipation gain
SHARP_KP     = 2.5    # extra correction for large offsets (quadratic)

# Camera center calibration (pixels).  0.0 = camera is perfectly centered.
# Positive = shift target right  (if buggy drifts left)
# Negative = shift target left   (if buggy drifts right)
CAMERA_CENTER_OFFSET_PX = 0.0

# ---------------------------------------------------------------------------
# TUNE: obstacle avoidance distances (metres)
# ---------------------------------------------------------------------------
OBSTACLE_WARN_DIST  = 1.5   # start slowing & picking gap direction
OBSTACLE_STOP_DIST  = 0.80  # front is "blocked" -> begin TURN stage
OBSTACLE_CLEAR_DIST = 1.30  # front must be this clear before PASS stage
SIDE_CLEAR_DIST     = 1.00  # side must be this open to declare obstacle gone

# Steering intensity while dodging
AVOID_STEER = 0.70   # how hard to steer away from obstacle (0-1)
MERGE_STEER = 0.45   # gentler steer when merging back to lane

# Speed limits per avoidance stage
AVOID_APPROACH_SPD = 0.14   # stage 1 - slowing down
AVOID_TURN_SPD     = 0.10   # stage 2 - turning
AVOID_PASS_SPD     = 0.15   # stage 3 - passing beside obstacle
AVOID_MERGE_SPD    = 0.16   # stage 4 - merging back

# Minimum frames gap must stay clear before advancing stage (hysteresis)
HYSTERESIS_FRAMES = 3

# ---------------------------------------------------------------------------
# TUNE: QR confirm count
# ---------------------------------------------------------------------------
QR_CONFIRM_COUNT = 3

# ---------------------------------------------------------------------------
# Sign -> building lookup
# ---------------------------------------------------------------------------
SIGN_TO_PATIENT  = {"A": "PATIENT_1",  "B": "PATIENT_2",  "C": "PATIENT_3"}
SIGN_TO_HOSPITAL = {"X": "HOSPITAL_1", "Y": "HOSPITAL_2", "Z": "HOSPITAL_3"}
FAKE_HOSPITALS   = {"FAKE_HOSPITAL_1", "FAKE_HOSPITAL_2"}
ALL_PATIENTS     = ["PATIENT_1", "PATIENT_2", "PATIENT_3"]

# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------
S_FIND_PATIENT          = "FIND_PATIENT"
S_CONFIRM_PATIENT       = "CONFIRM_PATIENT"
S_AWAIT_HOSPITAL_ASSIGN = "AWAIT_HOSPITAL"
S_FIND_HOSPITAL         = "FIND_HOSPITAL"
S_CONFIRM_HOSPITAL      = "CONFIRM_HOSPITAL"
S_AWAIT_NEXT_PATIENT    = "AWAIT_NEXT_PATIENT"
S_MISSION_COMPLETE      = "MISSION_COMPLETE"
S_EXIT_TO_PARKING       = "EXIT_TO_PARKING"
S_PARKED                = "PARKED"


class LineFollower(Node):
    """
    Core controller node.  Owns the mission state machine and drives the buggy
    by combining: lane-vector steering, LIDAR obstacle avoidance (4-stage),
    sign-guided turning, QR-triggered zone actions, and server comms.
    """

    def __init__(self):
        super().__init__('line_follower')

        # â”€â”€ Subscriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.subscription_vectors = self.create_subscription(
            EdgeVectors, '/edge_vectors',
            self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan',
            self.lidar_callback, QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication',
            self.server_communication_callback, QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String, '/qr_detection',
            self.qr_detection_callback, QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String, '/sign_board_detection',
            self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # â”€â”€ Publishers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # â”€â”€ Drive state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.target_speed = 0.0
        self.target_turn  = 0.0

        # â”€â”€ Raw LIDAR readings (updated every scan callback) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.lidar_front       = float('inf')
        self.lidar_front_left  = float('inf')
        self.lidar_front_right = float('inf')
        self.lidar_left        = float('inf')
        self.lidar_right       = float('inf')
        self.lidar_gap_left    = float('inf')
        self.lidar_gap_right   = float('inf')

        # â”€â”€ Obstacle avoidance state machine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # avoid_stage:
        #   0 = normal lane follow
        #   1 = APPROACH  - slowing, gap selected, waiting for STOP_DIST
        #   2 = TURN      - steering hard toward gap
        #   3 = PASS      - going straight beside obstacle
        #   4 = MERGE     - returning to lane center
        self.avoid_stage     = 0
        self.avoid_direction = 0.0   # +1 = gap is LEFT, -1 = gap is RIGHT
        self._hysteresis     = 0     # counts clear frames for stage advance

        # â”€â”€ Lane-following perception state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.last_vectors_msg      = None
        self.known_lane_half_width = None
        self.smoothed_offset       = 0.0
        self._debug_log_counter    = 0

        # â”€â”€ Sign / QR state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.last_sign       = None
        self.qr_streak_label = None
        self.qr_streak_count = 0

        # â”€â”€ Mission state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self.state                = S_FIND_PATIENT
        self.patients_remaining   = list(ALL_PATIENTS)
        self.current_patient      = self.patients_remaining[0]
        self.current_hospital     = None
        self.delivered_count      = 0
        self.server_uid_counter   = 1
        self.awaiting_ack_for_uid = None

        # 10 Hz control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            f"LineFollower brain online. Targeting {self.current_patient} first.")

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
        self.target_turn  = float(max(min(turn,  TURN_MAX), TURN_MIN))

    # =========================================================================
    # PERCEPTION CALLBACKS
    # =========================================================================

    def edge_vectors_callback(self, message):
        """Store the latest lane geometry; steering is computed in control_loop."""
        self.last_vectors_msg = message

    def lidar_callback(self, message):
        """
        Parse the LIDAR scan into named distance sectors and update the
        obstacle-avoidance state machine.

        LIDAR index 0 = directly ahead in Gazebo convention.
        Positive fraction = counter-clockwise (left in robot frame).
        Negative fraction = clockwise (right in robot frame).

        We use fraction-of-360 addressing so the code works regardless of
        how many beams the sensor has (360, 720, 1800 ...).
        """
        n = len(message.ranges)
        if n == 0:
            return

        def sector_min(center_frac, width_frac):
            """Minimum (closest) valid range in a circular sector."""
            c  = int(n * center_frac) % n
            hw = max(1, int(n * width_frac / 2))
            indices = [(c + i) % n for i in range(-hw, hw + 1)]
            vals = [message.ranges[i] for i in indices
                    if 0 < message.ranges[i] < float('inf')
                    and not math.isnan(message.ranges[i])]
            return min(vals) if vals else float('inf')

        def sector_max(center_frac, width_frac):
            """Maximum (farthest) valid range in a sector - used for gap finding."""
            c  = int(n * center_frac) % n
            hw = max(1, int(n * width_frac / 2))
            indices = [(c + i) % n for i in range(-hw, hw + 1)]
            vals = [message.ranges[i] for i in indices
                    if 0 < message.ranges[i] < float('inf')
                    and not math.isnan(message.ranges[i])]
            return max(vals) if vals else float('inf')

        # â”€â”€ Named sectors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # fractions:  0.0=front, 0.25=left, 0.5=rear, 0.75=right
        self.lidar_front       = sector_min(0.00, 0.10)   # Â±18 deg dead ahead
        self.lidar_front_left  = sector_min(0.08, 0.08)   # ~30 deg left of ahead
        self.lidar_front_right = sector_min(0.92, 0.08)   # ~30 deg right of ahead
        self.lidar_left        = sector_min(0.25, 0.10)   # 90 deg left side
        self.lidar_right       = sector_min(0.75, 0.10)   # 90 deg right side

        # â”€â”€ Gap scanning: find the OPEN side across the full forward arc â”€â”€â”€
        # Scan 8 sectors spanning the left half (0-135 deg) and 8 spanning
        # the right half (225-360 deg).  Sum their maximum ranges.
        # The side with more total open space is the correct escape direction
        # even when an obstacle covers 75%+ of the road.
        left_gap  = sum(sector_max(i / 80.0, 0.04) for i in range(1, 9))
        right_gap = sum(sector_max(1.0 - i / 80.0, 0.04) for i in range(1, 9))
        self.lidar_gap_left  = left_gap
        self.lidar_gap_right = right_gap

        # â”€â”€ Update avoidance FSM â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._update_avoid_fsm()

    def _update_avoid_fsm(self):
        """
        Called every LIDAR scan.  Drives the 4-stage avoidance FSM.

        Transitions:
          0 -> 1  front < WARN_DIST        (start slowing)
          1 -> 2  front < STOP_DIST        (begin hard turn)
          2 -> 3  front > CLEAR_DIST       (obstacle cleared, go straight)
          3 -> 4  side gap open            (obstacle passed, merge back)
          4 -> 0  lane offset small        (done - handled in control_loop)
        """
        front = self.lidar_front

        if self.avoid_stage == 0:
            if front < OBSTACLE_WARN_DIST:
                # Pick gap direction NOW while we still have reaction time
                if self.lidar_gap_left >= self.lidar_gap_right:
                    self.avoid_direction = 1.0    # LEFT gap bigger -> dodge left
                else:
                    self.avoid_direction = -1.0   # RIGHT gap bigger -> dodge right
                self.avoid_stage = 1
                self._hysteresis = 0
                self.get_logger().info(
                    f"[AVOID] Stage 1 APPROACH: front={front:.2f}m "
                    f"gap_L={self.lidar_gap_left:.1f} gap_R={self.lidar_gap_right:.1f} "
                    f"-> dodge {'LEFT' if self.avoid_direction > 0 else 'RIGHT'}")

        elif self.avoid_stage == 1:
            # Re-evaluate gap direction while still approaching (not yet committed)
            if self.lidar_gap_left >= self.lidar_gap_right:
                self.avoid_direction = 1.0
            else:
                self.avoid_direction = -1.0

            if front < OBSTACLE_STOP_DIST:
                self.avoid_stage = 2
                self._hysteresis = 0
                self.get_logger().info(
                    f"[AVOID] Stage 2 TURN: front={front:.2f}m "
                    f"-> hard steer {'LEFT' if self.avoid_direction > 0 else 'RIGHT'}")
            elif front >= OBSTACLE_WARN_DIST:
                # Obstacle disappeared or was a false positive
                self.avoid_stage = 0
                self.get_logger().info("[AVOID] Stage 0 RESUME: obstacle gone")

        elif self.avoid_stage == 2:
            # Hard turn - wait until front opens up
            if front > OBSTACLE_CLEAR_DIST:
                self._hysteresis += 1
                if self._hysteresis >= HYSTERESIS_FRAMES:
                    self.avoid_stage = 3
                    self._hysteresis = 0
                    self.get_logger().info(
                        f"[AVOID] Stage 3 PASS: front={front:.2f}m")
            else:
                self._hysteresis = 0

        elif self.avoid_stage == 3:
            # Going straight beside obstacle.
            # Check the side we dodged TOWARD for clearing.
            side_dist = self.lidar_left if self.avoid_direction > 0 else self.lidar_right
            if side_dist > SIDE_CLEAR_DIST:
                self._hysteresis += 1
                if self._hysteresis >= HYSTERESIS_FRAMES:
                    self.avoid_stage = 4
                    self._hysteresis = 0
                    self.get_logger().info(
                        f"[AVOID] Stage 4 MERGE: side={side_dist:.2f}m")
            else:
                self._hysteresis = 0

        # Stage 4 -> 0 transition handled in drive_lane_following()

    def sign_board_callback(self, message):
        self.last_sign = message.data
        self.get_logger().info(f"Sign seen: {message.data}")

    def parse_sign(self, sign_str):
        """Returns (letter, direction) or (None, None)."""
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
            self.get_logger().warn(
                f"FAKE HOSPITAL detected ({label}) - ignoring.")
            return
        if self.state == S_FIND_PATIENT and label == self.current_patient:
            self.get_logger().info(f"Confirmed patient QR: {label}")
            self.state = S_CONFIRM_PATIENT
        elif self.state == S_FIND_HOSPITAL and label == self.current_hospital:
            self.get_logger().info(f"Confirmed hospital QR: {label}")
            self.state = S_CONFIRM_HOSPITAL
        elif (self.state == S_FIND_HOSPITAL
              and label.startswith("HOSPITAL")
              and label != self.current_hospital):
            self.get_logger().warn(
                f"Wrong hospital ({label}), target={self.current_hospital}. Continuing.")

    # =========================================================================
    # SERVER COMMUNICATION
    # =========================================================================

    def send_server_update(self, text_msg):
        server_msg = ServerCommunication()
        server_msg.src  = 1
        server_msg.dest = 2
        server_msg.uid  = self.server_uid_counter
        server_msg.ack  = 0
        server_msg.msg  = text_msg
        self.awaiting_ack_for_uid = self.server_uid_counter
        self.server_uid_counter  += 1
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"-> Server: {text_msg} (uid={server_msg.uid})")

    def server_communication_callback(self, message):
        if message.dest != 1:
            return
        self.get_logger().info(
            f"<- Server: msg='{message.msg}' ack={message.ack} uid={message.uid}")
        if message.ack == 1 and message.uid == self.awaiting_ack_for_uid:
            self.awaiting_ack_for_uid = None
            self.process_server_payload(message.msg)
            return
        if message.msg:
            self.process_server_payload(message.msg)

    def process_server_payload(self, payload):
        payload = payload.strip().upper()
        if payload == "INVALID":
            self.get_logger().warn("Server said INVALID - re-check position.")
            return
        if payload.startswith("HOSPITAL_") and self.state == S_AWAIT_HOSPITAL_ASSIGN:
            self.current_hospital = payload
            self.get_logger().info(f"Assigned hospital: {self.current_hospital}")
            self.state = S_FIND_HOSPITAL
            return
        if payload.startswith("PATIENT_") and self.state == S_AWAIT_NEXT_PATIENT:
            if payload in self.patients_remaining:
                self.current_patient = payload
                self.get_logger().info(f"Next patient: {self.current_patient}")
                self.state = S_FIND_PATIENT

    # =========================================================================
    # MISSION STATE MACHINE
    # =========================================================================

    def control_loop(self):
        """Runs at 10 Hz. Reads current state + sensors, decides drive command."""

        if self.state in (S_FIND_PATIENT, S_FIND_HOSPITAL):
            self.drive_lane_following()

        elif self.state == S_CONFIRM_PATIENT:
            self.rover_move_manual_mode(0.0, 0.0)
            if self.awaiting_ack_for_uid is None:
                self.send_server_update(self.current_patient)
                self.state = S_AWAIT_HOSPITAL_ASSIGN

        elif self.state == S_AWAIT_HOSPITAL_ASSIGN:
            # Stay inside zone until hospital is assigned (penalty to leave early)
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
        """
        Placeholder - replace with real detection, e.g.:
        - both left_side_dist and right_side_dist below a small threshold, or
        - a specific QR/sign marker at the parking entrance.
        """
        return False

    def on_patient_delivered(self):
        self.delivered_count += 1
        if self.current_patient in self.patients_remaining:
            self.patients_remaining.remove(self.current_patient)
        self.get_logger().info(
            f"Delivered {self.current_patient} -> {self.current_hospital} "
            f"({self.delivered_count}/3)")
        self.current_hospital = None
        if self.delivered_count >= 3:
            self.state = S_MISSION_COMPLETE
        else:
            self.state = S_AWAIT_NEXT_PATIENT

    # =========================================================================
    # LANE FOLLOWING + 4-STAGE OBSTACLE AVOIDANCE + SIGN-GUIDED TURNING
    # =========================================================================

    def drive_lane_following(self):
        """
        Priority order (highest first):
          1. Active obstacle avoidance stages 1-4
          2. Normal lane centering + sign guidance
        """

        # â”€â”€ Stage 1: APPROACH - slow down & pre-steer toward gap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.avoid_stage == 1:
            # Taper speed linearly as we close in
            dist_range = OBSTACLE_WARN_DIST - OBSTACLE_STOP_DIST
            proximity  = max(0.0, 1.0 - (self.lidar_front - OBSTACLE_STOP_DIST) / dist_range)
            speed      = CRUISE_SPEED - proximity * (CRUISE_SPEED - AVOID_APPROACH_SPD)
            pre_steer  = 0.25 * self.avoid_direction   # gentle early lean
            self.rover_move_manual_mode(speed, pre_steer)
            self._log_avoid(1, speed, pre_steer)
            return

        # â”€â”€ Stage 2: TURN - hard steer toward gap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.avoid_stage == 2:
            steer = AVOID_STEER * self.avoid_direction
            self.rover_move_manual_mode(AVOID_TURN_SPD, steer)
            self._log_avoid(2, AVOID_TURN_SPD, steer)
            return

        # â”€â”€ Stage 3: PASS - creep straight beside obstacle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.avoid_stage == 3:
            # Tiny counter-steer keeps the buggy from drifting into the obstacle
            gentle_steer = 0.15 * self.avoid_direction
            self.rover_move_manual_mode(AVOID_PASS_SPD, gentle_steer)
            self._log_avoid(3, AVOID_PASS_SPD, gentle_steer)
            return

        # â”€â”€ Stage 4: MERGE - return to lane center â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.avoid_stage == 4:
            merge_steer = -MERGE_STEER * self.avoid_direction   # opposite to dodge
            self.rover_move_manual_mode(AVOID_MERGE_SPD, merge_steer)
            self._log_avoid(4, AVOID_MERGE_SPD, merge_steer)
            # Exit when the lane offset is back to near-zero
            if abs(self.smoothed_offset) < 15.0:
                self.avoid_stage     = 0
                self.avoid_direction = 0.0
                self.get_logger().info("[AVOID] Stage 0 RESUME: back in lane")
            return

        # â”€â”€ Stage 0: NORMAL LANE FOLLOWING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        vectors = self.last_vectors_msg
        if vectors is None or vectors.vector_count == 0:
            self.rover_move_manual_mode(SLOW_SPEED, 0.0)
            return

        half_width       = vectors.image_width / 2.0
        turn             = 0.0
        midpoint         = None
        curvature_signal = 0.0

        if vectors.vector_count == 2:
            # Use the NEAR endpoint of each edge (index 1 = closest to buggy)
            left_near_x  = vectors.vector_1[1].x
            right_near_x = vectors.vector_2[1].x
            midpoint     = (left_near_x + right_near_x) / 2.0
            self.known_lane_half_width = abs(right_near_x - left_near_x) / 2.0

            # Curvature anticipation: compare far vs near x on each edge
            left_tilt        = vectors.vector_1[0].x - vectors.vector_1[1].x
            right_tilt       = vectors.vector_2[0].x - vectors.vector_2[1].x
            curvature_signal = (left_tilt + right_tilt) / 2.0

        elif vectors.vector_count == 1:
            edge_x    = vectors.vector_1[1].x
            half_lane = (self.known_lane_half_width
                         if self.known_lane_half_width else half_width * 0.5)
            if edge_x < half_width:
                midpoint = edge_x + half_lane   # left edge -> center is rightward
            else:
                midpoint = edge_x - half_lane   # right edge -> center is leftward

        if midpoint is not None:
            target_center = half_width + CAMERA_CENTER_OFFSET_PX
            raw_offset    = midpoint - target_center
            # Exponential moving average (alpha=0.4: responsive but smoothed)
            alpha = 0.4
            self.smoothed_offset = (alpha * raw_offset
                                    + (1 - alpha) * self.smoothed_offset)

            position_term  = -STEER_KP * (self.smoothed_offset / half_width)
            curvature_term = -CURVE_KP * (curvature_signal / half_width)
            normalized     = self.smoothed_offset / half_width
            sharp_term     = -math.copysign(
                SHARP_KP * (normalized ** 2), self.smoothed_offset)

            turn = position_term + curvature_term + sharp_term

            self._debug_log_counter += 1
            if self._debug_log_counter % 10 == 0:
                self.get_logger().info(
                    f"[steer] mid={midpoint:.1f} ctr={target_center:.1f} "
                    f"off={self.smoothed_offset:.1f} curv={curvature_signal:.1f} "
                    f"sharp={sharp_term:.2f} turn={turn:.2f} "
                    f"front={self.lidar_front:.2f}m")

        # Sign-guided bias at intersections
        turn = self.apply_sign_guidance(turn)
        turn = max(TURN_MIN, min(TURN_MAX, turn))

        # Speed taper: slow when steering hard
        speed = CRUISE_SPEED * max(0.5, 1.0 - abs(turn) * 0.6)

        # Belt-and-suspenders: extra slow-down if something is in front
        # even though avoidance hasn't triggered stage 1 yet
        if self.lidar_front < OBSTACLE_WARN_DIST:
            proximity_factor = self.lidar_front / OBSTACLE_WARN_DIST
            speed *= max(0.5, proximity_factor)

        self.rover_move_manual_mode(speed, turn)

    def _log_avoid(self, stage, speed, steer):
        """Throttled avoidance debug log."""
        self._debug_log_counter += 1
        if self._debug_log_counter % 5 == 0:
            side = 'LEFT' if self.avoid_direction > 0 else 'RIGHT'
            self.get_logger().info(
                f"[AVOID stg={stage}] spd={speed:.2f} steer={steer:.2f} "
                f"dir={side} front={self.lidar_front:.2f} "
                f"L={self.lidar_left:.2f} R={self.lidar_right:.2f}")

    def apply_sign_guidance(self, base_turn):
        """
        Nudge steering toward the sign-indicated direction when the sign
        matches our current navigation target letter.
        """
        letter, direction = self.parse_sign(self.last_sign)
        if letter is None:
            return base_turn

        target_letter = None
        if self.state == S_FIND_PATIENT:
            target_letter = next(
                (k for k, v in SIGN_TO_PATIENT.items()
                 if v == self.current_patient), None)
        elif self.state == S_FIND_HOSPITAL:
            target_letter = next(
                (k for k, v in SIGN_TO_HOSPITAL.items()
                 if v == self.current_hospital), None)

        if letter != target_letter:
            return base_turn

        if direction == "LEFT":
            return max(base_turn, 0.5)
        elif direction == "RIGHT":
            return min(base_turn, -0.5)
        return base_turn   # STRAIGHT


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
