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
#  CHANGE LOG (obstacle avoidance rewrite):
#  The old lidar_callback only checked two crude 10%-wide cones and picked a
#  side with a hard boolean (front_blocked True/False). That's why a 75%
#  blocked road with only a 25% gap on one side could still get misjudged -
#  there was no concept of "how wide is the opening", just "which single
#  ray is bigger right now".
#
#  This version:
#    1. Sweeps the LIDAR finely (every 2 degrees) across the front hemisphere.
#    2. Finds every contiguous "open" angular region (a real gap), and
#       throws away any gap too narrow for the buggy to actually fit through.
#    3. Picks the gap closest to straight-ahead (minimizes unnecessary swerve).
#    4. Blends the resulting avoidance steering with normal lane-following
#       steering SMOOTHLY based on how close the nearest obstacle is -
#       no more instant on/off snapping between the two behaviors.
#    5. Speed ramps down continuously as obstacles get closer (and ramps
#       back up as they clear), instead of a binary "fast" / "slow" flag.
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
TURN_MIN = -1.0
TURN_MAX = 1.0

# ---------------------------------------------------------------------------
# TUNE: lane following gains / speeds
# ---------------------------------------------------------------------------
CRUISE_SPEED = 0.20            # normal forward speed while lane-following, clear road
SLOW_SPEED = 0.10              # speed while approaching a building / turning
STEER_KP = 1.6                 # proportional gain: bigger = more aggressive steering
CURVE_KP = 2.0                 # anticipation gain: reacts to how much the lane is
                                # bending ahead, so the buggy turns in before it
                                # drifts off-center (prevents corner-cutting)

SHARP_KP = 2.5                 # extra correction that only kicks in when the
                                # offset is already large (e.g. mid-corner,
                                # recovering from a drift). Grows quadratically
                                # with offset, so small centering wobble is
                                # untouched but a 40-50px deviation gets a much
                                # stronger pull back than plain STEER_KP alone.

# TUNE / CALIBRATE: if the buggy consistently hugs one side even when the
# code reports offset ~= 0, the camera is likely not mounted exactly on the
# chassis's true centerline. This shifts what pixel column counts as "center"
# to compensate.
#
# Positive value = shift the target center to the RIGHT in the image
# (use this if the buggy drifts LEFT of the road).
# Negative value = shift target center to the LEFT
# (use this if the buggy drifts RIGHT of the road).
CAMERA_CENTER_OFFSET_PX = 0.0

# ---------------------------------------------------------------------------
# TUNE: obstacle avoidance - gap seeking
# ---------------------------------------------------------------------------
# How wide (in degrees, centered on "straight ahead") to sweep the LIDAR
# looking for open gaps. +/- 85 covers almost the whole front hemisphere,
# which matters when an obstacle takes up 75% of the road width - the real
# gap can appear at a fairly wide angle off-center.
LIDAR_FRONT_HALF_ANGLE_DEG = 85

# Resolution of the sweep. Smaller = finer detection of narrow gaps, but
# more compute. 2 degrees is plenty for this track scale.
LIDAR_SCAN_STEP_DEG = 2

# Each sampled ray actually averages a small +/- window around it, to reduce
# single-ray noise/dropouts from being misread as a false gap or false wall.
LIDAR_RAY_WINDOW_DEG = 4

# A ray farther than this counts as "open" in that direction.
GAP_SAFE_DIST = 1.1

# TUNE: a candidate gap must span at least this many degrees to be trusted
# as "wide enough for the buggy to actually drive through". This is the key
# fix for the 75%-blocked case: a stray single open ray from noise will
# never satisfy this width, only a real opening will.
MIN_GAP_WIDTH_DEG = 18

# How far ahead we check to decide "how close is the nearest thing directly
# in front of us" - drives both the speed ramp-down and how strongly
# avoidance overrides lane-following.
FRONT_CENTER_HALF_ANGLE_DEG = 20

# Distance thresholds for the smooth blend between pure lane-following and
# pure gap-seeking avoidance:
#   front_dist >= AVOID_FAR_DIST  -> 0% avoidance influence (pure lane-follow)
#   front_dist <= AVOID_NEAR_DIST -> 100% avoidance influence (pure gap-seek)
#   in between                    -> linear blend
AVOID_FAR_DIST = 2.0
AVOID_NEAR_DIST = 0.45

# Speed floor while squeezing past something close - never fully stop while
# still trying to drive around an obstacle (would break lane-following context).
AVOID_MIN_SPEED = 0.07

# Low-pass filter strength on the avoidance steering target itself, so the
# chosen gap angle doesn't jump frame-to-frame as the sweep noise changes.
# Higher = more responsive but twitchier; lower = smoother but slower to react.
AVOID_STEER_SMOOTH_ALPHA = 0.35

# Kept for logging / backwards compatibility only - no longer gates behavior
# with a hard on/off switch (see avoidance_influence instead).
OBSTACLE_FRONT_DIST = 0.8

# ---------------------------------------------------------------------------
# TUNE: how close counts as "at the building" (for zone-proxy logic)
# ---------------------------------------------------------------------------
QR_CONFIRM_COUNT = 3            # need N consecutive matching QR reads before acting
                                 # (guards against a misread QR triggering a false zone-enter)

# ---------------------------------------------------------------------------
# Sign -> building lookup, exactly as specified in the challenge doc
# ---------------------------------------------------------------------------
SIGN_TO_PATIENT = {"A": "PATIENT_1", "B": "PATIENT_2", "C": "PATIENT_3"}
SIGN_TO_HOSPITAL = {"X": "HOSPITAL_1", "Y": "HOSPITAL_2", "Z": "HOSPITAL_3"}
FAKE_HOSPITALS = {"FAKE_HOSPITAL_1", "FAKE_HOSPITAL_2"}

ALL_PATIENTS = ["PATIENT_1", "PATIENT_2", "PATIENT_3"]

# ---------------------------------------------------------------------------
# Mission states - this is the whole "brain" in one enum
# ---------------------------------------------------------------------------
S_FIND_PATIENT = "FIND_PATIENT"                # driving, looking for current target patient
S_CONFIRM_PATIENT = "CONFIRM_PATIENT"          # QR matched, confirming before acting
S_AWAIT_HOSPITAL_ASSIGN = "AWAIT_HOSPITAL"     # sent patient id, waiting on server reply
S_FIND_HOSPITAL = "FIND_HOSPITAL"              # driving toward assigned hospital
S_CONFIRM_HOSPITAL = "CONFIRM_HOSPITAL"        # QR matched hospital, confirming
S_AWAIT_NEXT_PATIENT = "AWAIT_NEXT_PATIENT"    # delivered, waiting for server to say "go"
S_MISSION_COMPLETE = "MISSION_COMPLETE"        # all 3 delivered
S_EXIT_TO_PARKING = "EXIT_TO_PARKING"          # bonus: driving to parking
S_PARKED = "PARKED"                            # bonus: sent PARKED, done


class LineFollower(Node):
    """
    Core controller node. Owns the mission state machine and drives the buggy
    by combining: lane-vector steering, LIDAR gap-seeking obstacle avoidance
    (smoothly blended, not a hard switch), sign-guided turning at
    intersections, QR-triggered zone actions, and server comms.
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

        # ------------------ Perception state: lane ------------------
        self.last_sign = None          # e.g. "A_LEFT"
        self.qr_streak_label = None
        self.qr_streak_count = 0

        # Lane-width memory: remembers how wide the road was (in pixels) the
        # last time BOTH edges were visible. Used to estimate the center when
        # only one edge is currently visible, instead of a blind fixed bias.
        self.known_lane_half_width = None

        # Smoothed steering offset - reduces frame-to-frame jitter from noisy
        # contour detection so small per-frame noise doesn't accumulate into
        # a visible directional bias.
        self.smoothed_offset = 0.0
        self._debug_log_counter = 0

        # ------------------ Perception state: obstacles / LIDAR ------------------
        self.front_min_dist = float('inf')     # closest thing in a narrow forward cone
        self.left_side_dist = float('inf')     # kept for compatibility / debugging
        self.right_side_dist = float('inf')
        self.front_blocked = False              # legacy flag, logging only now
        self.has_valid_gap = True
        self.avoid_target_turn = 0.0            # smoothed steer target from gap-seeking
        self.avoidance_influence = 0.0          # 0 = pure lane-follow, 1 = pure avoidance

        # ------------------ Mission state ------------------
        self.state = S_FIND_PATIENT
        self.patients_remaining = list(ALL_PATIENTS)   # order server may re-sequence this
        self.current_patient = self.patients_remaining[0]
        self.current_hospital = None
        self.delivered_count = 0

        # ------------------ Server protocol bookkeeping ------------------
        self.server_uid_counter = 1
        self.awaiting_ack_for_uid = None

        # 10 Hz control loop - the "heartbeat" of the brain
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
        self.target_turn = float(max(min(turn, TURN_MAX), TURN_MIN))

    # =========================================================================
    # PERCEPTION CALLBACKS - these just update state; they don't drive directly.
    # Keeping "sense" separate from "act" (which happens in control_loop) avoids
    # race conditions between camera-rate and lidar-rate callbacks.
    # =========================================================================

    def edge_vectors_callback(self, message):
        """Store the latest lane geometry; steering is computed in control_loop."""
        self.last_vectors_msg = message

    def lidar_callback(self, message):
        """
        Gap-seeking obstacle detection.

        Instead of checking 2-3 crude cones and picking a side with a hard
        boolean, this sweeps the front hemisphere finely, finds every
        contiguous "open" angular run, discards any that are too narrow to
        actually drive through, and picks the widest-appropriate gap closest
        to straight-ahead. That's what correctly handles the case where an
        obstacle covers 75% of the road and only a narrow strip is open -
        the fixed-width old cones could easily misjudge which side was truly
        clearer, or fail to notice a gap at all if it wasn't in either fixed
        cone's direction.

        NOTE ON LIDAR ANGLE CONVENTION (verify this once in Foxglove/rviz):
        Standard ROS convention (REP-103) has angle 0 = straight ahead, and
        angle increases counter-clockwise, i.e. POSITIVE angle = LEFT of the
        buggy. This code assumes that convention when converting a chosen gap
        angle into a steering value (steer positive = left, matching
        msg.axes[3]). If your buggy consistently dodges the WRONG way in
        testing, the fix is a one-line sign flip - see raw_avoid_turn below.
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

        # --- 1. How close is the nearest thing directly ahead? ---
        # Drives both the speed ramp-down and how strongly avoidance
        # overrides lane-following (see avoidance_influence below).
        self.front_min_dist = ray_dist(0.0, FRONT_CENTER_HALF_ANGLE_DEG)
        self.front_blocked = self.front_min_dist < OBSTACLE_FRONT_DIST  # legacy/logging only

        # Kept for compatibility with anything else that might reference these.
        self.left_side_dist = ray_dist(70.0, 15.0)
        self.right_side_dist = ray_dist(-70.0, 15.0)

        # --- 2. Fine sweep across the front hemisphere to find real gaps ---
        angles_deg = list(range(-LIDAR_FRONT_HALF_ANGLE_DEG,
                                 LIDAR_FRONT_HALF_ANGLE_DEG + 1,
                                 LIDAR_SCAN_STEP_DEG))
        distances = [ray_dist(a) for a in angles_deg]

        gaps = []  # list of (center_angle_deg, width_deg)
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
            # Prefer whichever valid (wide-enough) gap requires the LEAST
            # deviation from straight-ahead - avoids unnecessary swerving
            # when the road ahead is actually fine.
            best_gap_angle, best_gap_width = min(gaps, key=lambda g: abs(g[0]))
            self.has_valid_gap = True
        else:
            # No opening wide enough for the buggy anywhere in the scanned
            # cone. As a last resort, lean toward whichever single ray is
            # farthest - better than driving dead straight into the obstacle.
            best_gap_angle = angles_deg[distances.index(max(distances))]
            self.has_valid_gap = False

        # Positive angle = left (see convention note above). Normalize by a
        # reasonable max useful deflection (45 deg) into a [-1, 1] steer value.
        raw_avoid_turn = max(-1.0, min(1.0, best_gap_angle / 45.0))

        # Low-pass filter so the avoidance target doesn't jump frame-to-frame
        # as sweep noise fluctuates - this is what makes the dodge a smooth
        # lean instead of a jerky snap.
        self.avoid_target_turn = (
            AVOID_STEER_SMOOTH_ALPHA * raw_avoid_turn
            + (1 - AVOID_STEER_SMOOTH_ALPHA) * self.avoid_target_turn
        )

        # --- 3. How strongly should avoidance override lane-following right now? ---
        # Smooth linear blend based on proximity - no hard on/off switch.
        if self.front_min_dist >= AVOID_FAR_DIST:
            self.avoidance_influence = 0.0
        elif self.front_min_dist <= AVOID_NEAR_DIST:
            self.avoidance_influence = 1.0
        else:
            span = AVOID_FAR_DIST - AVOID_NEAR_DIST
            self.avoidance_influence = (AVOID_FAR_DIST - self.front_min_dist) / span

    def sign_board_callback(self, message):
        """
        Stores the most recent sign reading.
        # TODO: confirm the exact string format your `detect` node publishes
        # (e.g. "A_LEFT" vs separate letter/arrow topics) and adjust
        # parse_sign() below to match.
        """
        self.last_sign = message.data
        self.get_logger().info(f"Sign seen: {message.data}")

    def parse_sign(self, sign_str):
        """Returns (letter, direction) or (None, None) if unparseable."""
        if not sign_str or "_" not in sign_str:
            return None, None
        letter, direction = sign_str.split("_", 1)
        return letter.strip().upper(), direction.strip().upper()

    def qr_detection_callback(self, message):
        """
        Debounce QR reads: require QR_CONFIRM_COUNT consecutive identical
        reads before treating it as "confirmed" - a single misread frame
        should not trigger a server message (that's a penalty risk).
        """
        label = message.data
        if label == self.qr_streak_label:
            self.qr_streak_count += 1
        else:
            self.qr_streak_label = label
            self.qr_streak_count = 1

        if self.qr_streak_count >= QR_CONFIRM_COUNT:
            self.handle_confirmed_qr(label)

    def handle_confirmed_qr(self, label):
        """Reacts to a QR code we're confident about, based on current state."""
        if label in FAKE_HOSPITALS:
            self.get_logger().warn(f"FAKE HOSPITAL detected ({label}) - ignoring, do not approach.")
            return

        if self.state == S_FIND_PATIENT and label == self.current_patient:
            self.get_logger().info(f"Confirmed patient QR: {label}")
            self.state = S_CONFIRM_PATIENT

        elif self.state == S_FIND_HOSPITAL and label == self.current_hospital:
            self.get_logger().info(f"Confirmed hospital QR: {label}")
            self.state = S_CONFIRM_HOSPITAL

        elif self.state == S_FIND_HOSPITAL and label.startswith("HOSPITAL") \
                and label != self.current_hospital:
            # Scanned a real hospital, but the wrong one - don't stop/deliver here.
            self.get_logger().warn(
                f"Wrong hospital nearby ({label}), target is {self.current_hospital}. Continuing.")

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
            return  # not for us

        self.get_logger().info(
            f"<- Server: msg='{message.msg}' ack={message.ack} uid={message.uid}")

        # --- Case 1: server is ACKing something we sent ---
        if message.ack == 1 and message.uid == self.awaiting_ack_for_uid:
            self.awaiting_ack_for_uid = None
            # The payload alongside the ack is where the actual instruction lives
            self.process_server_payload(message.msg)
            return

        # --- Case 2: server is pushing a message we didn't explicitly ask for ---
        if message.msg:
            self.process_server_payload(message.msg)

    def process_server_payload(self, payload):
        """
        Interprets the server's instruction text.
        # TODO: confirm exact payload strings from your server build
        # (e.g. "HOSPITAL_2", "INVALID", "PATIENT_2", "NEXT:PATIENT_3", ...)
        # and extend this parser accordingly.
        """
        payload = payload.strip().upper()

        if payload == "INVALID":
            self.get_logger().warn("Server said INVALID - likely outside zone. Re-check position.")
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
    # MISSION STATE MACHINE - the actual "brain" logic tying everything together
    # =========================================================================

    def control_loop(self):
        """Runs at 10 Hz. Reads current state + sensors, decides drive command."""

        if self.state in (S_FIND_PATIENT, S_FIND_HOSPITAL):
            self.drive_lane_following()

        elif self.state == S_CONFIRM_PATIENT:
            # We're at the patient building - stop and register with the server.
            self.rover_move_manual_mode(0.0, 0.0)
            if self.awaiting_ack_for_uid is None:
                self.send_server_update(self.current_patient)
                self.state = S_AWAIT_HOSPITAL_ASSIGN

        elif self.state == S_AWAIT_HOSPITAL_ASSIGN:
            # IMPORTANT: stay put (inside the zone) until hospital is assigned.
            # Doc explicitly penalizes leaving the patient zone before assignment.
            self.rover_move_manual_mode(0.0, 0.0)

        elif self.state == S_CONFIRM_HOSPITAL:
            self.rover_move_manual_mode(0.0, 0.0)
            if self.awaiting_ack_for_uid is None:
                self.send_server_update(self.current_hospital)
                self.on_patient_delivered()

        elif self.state == S_AWAIT_NEXT_PATIENT:
            self.rover_move_manual_mode(0.0, 0.0)

        elif self.state == S_MISSION_COMPLETE:
            # Bonus task: head for the exit / parking area.
            self.drive_lane_following()
            # TODO: add logic to detect "inside parking area" (e.g. via a
            # dedicated LIDAR/geometry check or a specific sign/marker) and
            # transition to S_EXIT_TO_PARKING -> send "PARKED".
            self.state = S_EXIT_TO_PARKING

        elif self.state == S_EXIT_TO_PARKING:
            self.drive_lane_following()
            # TODO: replace this placeholder trigger with a real "am I in the
            # parking box" check (LIDAR walls on both sides + low speed, or a
            # parking marker QR/sign).
            if self.is_in_parking_area():
                self.rover_move_manual_mode(0.0, 0.0)
                self.send_server_update("PARKED")
                self.state = S_PARKED

        elif self.state == S_PARKED:
            self.rover_move_manual_mode(0.0, 0.0)

        self.publish_drive_commands()

    def is_in_parking_area(self):
        """
        Placeholder. Replace with real detection, e.g.:
        - both left_side_dist and right_side_dist below a small threshold
          (buggy boxed in by parking lines), or
        - a specific QR/sign marker at the parking entrance seen recently.
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
    # LANE FOLLOWING + OBSTACLE AVOIDANCE (smoothly blended) + SIGN-GUIDED TURN
    # =========================================================================

    def drive_lane_following(self):
        """
        Computes lane-centering steering (lane_turn) and gap-seeking
        avoidance steering (self.avoid_target_turn, from lidar_callback),
        then BLENDS them proportionally to how close the nearest obstacle is
        (self.avoidance_influence). This replaces the old hard switch
        ("if front_blocked: avoid, else: lane-follow") which caused abrupt,
        easily-wrong-direction dodges. Speed is similarly a smooth function
        of obstacle proximity and current turn sharpness, not a binary flag.
        """
        vectors = getattr(self, 'last_vectors_msg', None)

        # ---------------- Lane-following steering ----------------
        lane_turn = 0.0
        have_lane_signal = False

        if vectors is not None and vectors.vector_count > 0:
            half_width = vectors.image_width / 2.0
            midpoint = None
            curvature_signal = 0.0

            if vectors.vector_count == 2:
                # Use the NEAR endpoint of each side (index 1 = max-y = closest
                # to the buggy), not an average of both endpoints - avoids a
                # length-asymmetry bias when left/right contours span
                # different y-ranges (perspective, partial occlusion, etc).
                left_near_x = vectors.vector_1[1].x
                right_near_x = vectors.vector_2[1].x
                midpoint = (left_near_x + right_near_x) / 2.0
                self.known_lane_half_width = abs(right_near_x - left_near_x) / 2.0

                # Curvature anticipation: compare each line's FAR point to its
                # own NEAR point, kept separate from position so it can't
                # reintroduce the asymmetry bias.
                left_tilt = vectors.vector_1[0].x - vectors.vector_1[1].x
                right_tilt = vectors.vector_2[0].x - vectors.vector_2[1].x
                curvature_signal = (left_tilt + right_tilt) / 2.0

            elif vectors.vector_count == 1:
                # Only one edge visible - estimate the missing edge using the
                # last known lane width rather than steering toward the
                # visible edge directly (that caused "hugs the line" behavior).
                edge_x = vectors.vector_1[1].x
                half_lane = self.known_lane_half_width if self.known_lane_half_width else half_width * 0.5
                if edge_x < half_width:
                    midpoint = edge_x + half_lane   # visible edge is LEFT -> center is to its right
                else:
                    midpoint = edge_x - half_lane   # visible edge is RIGHT -> center is to its left

            if midpoint is not None:
                target_center = half_width + CAMERA_CENTER_OFFSET_PX
                raw_offset = midpoint - target_center
                alpha = 0.4  # EMA smoothing on the raw pixel offset
                self.smoothed_offset = alpha * raw_offset + (1 - alpha) * self.smoothed_offset

                position_term = -STEER_KP * (self.smoothed_offset / half_width)
                curvature_term = -CURVE_KP * (curvature_signal / half_width)
                normalized = self.smoothed_offset / half_width
                sharp_term = -math.copysign(SHARP_KP * (normalized ** 2), self.smoothed_offset)

                lane_turn = position_term + curvature_term + sharp_term
                have_lane_signal = True

                self._debug_log_counter += 1
                if self._debug_log_counter % 10 == 0:
                    self.get_logger().info(
                        f"[steer] offset={self.smoothed_offset:.1f} lane_turn={lane_turn:.2f} | "
                        f"front={self.front_min_dist:.2f}m gap_ok={self.has_valid_gap} "
                        f"avoid_turn={self.avoid_target_turn:.2f} "
                        f"influence={self.avoidance_influence:.2f}")

        if not have_lane_signal:
            # No lane visible at all - default to straight; avoidance blend
            # below still applies on top of this.
            lane_turn = 0.0

        lane_turn = self.apply_sign_guidance(lane_turn)

        # ---------------- Blend with gap-seeking avoidance ----------------
        influence = self.avoidance_influence
        avoid_turn = self.avoid_target_turn

        turn = (1.0 - influence) * lane_turn + influence * avoid_turn
        turn = max(TURN_MIN, min(TURN_MAX, turn))

        # ---------------- Speed: smooth ramp-down near obstacles ----------------
        front_dist = self.front_min_dist
        min_speed_ratio = AVOID_MIN_SPEED / CRUISE_SPEED

        if front_dist >= AVOID_FAR_DIST:
            proximity_factor = 1.0
        elif front_dist <= AVOID_NEAR_DIST:
            proximity_factor = min_speed_ratio
        else:
            span = AVOID_FAR_DIST - AVOID_NEAR_DIST
            frac = (front_dist - AVOID_NEAR_DIST) / span
            proximity_factor = min_speed_ratio + frac * (1.0 - min_speed_ratio)

        # Still slow down for sharp turns on top of the obstacle-proximity ramp.
        turn_factor = max(0.5, 1.0 - abs(turn) * 0.6)

        speed = CRUISE_SPEED * proximity_factor * turn_factor
        speed = max(AVOID_MIN_SPEED, speed)

        self.rover_move_manual_mode(speed, turn)

    def apply_sign_guidance(self, base_turn):
        """
        If the last sign we saw corresponds to our current target
        (patient letter while FIND_PATIENT, hospital letter while
        FIND_HOSPITAL), nudge the steering toward the indicated direction.
        Lane-vector following still keeps us inside the lane; this just
        biases which fork we take at an intersection.
        """
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
            return base_turn  # sign is for a different destination - ignore it

        if direction == "LEFT":
            return max(base_turn, 0.5)
        elif direction == "RIGHT":
            return min(base_turn, -0.5)
        # STRAIGHT -> just keep lane-centering value
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