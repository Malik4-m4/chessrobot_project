#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
board_calibration.py
====================
معايرة لوحة الشطرنج وبناء مواقع المربعات لذراع Franka Panda.

المحتويات:
- ثوابت هندسة اللوحة + اعدادات الـprobing
- TF helpers (init_tf, get_current_pose_in_ref)
- دوال حركة مشتركة (move_to_pose, make_down_pose, gripper_full_close)
- ForceMonitor: مراقبة القوة الخارجية + عزوم المفاصل عبر FrankaState
- BoardCalibration: يمسك حالة المعايرة + يبنى خرائط المواقع
  + ينفذ الـprobing والمعايرة (two-pass) + save/load/show/test

تعديلات هذه النسخة:
- Multi-criteria contact detection (F_xy, F_z, τ_max, τ_wrist).
- USE_WRIST_DEFLECTION_CORRECTION صار ثابت بأول الملف (default=False).
- المعايرة تبدأ من الجنوب (S1/S2) لحساب الزاوية الأولية + الترتيب كامل.
- الروبوت يبدأ من موقع READY ويرجعله بالنهاية.
- ⭐ Fine-tuning offsets (SQUARE_OFFSET_U/V) في إطار اللوحة:
    * بتنطبق على كل المربعات + promotion + graveyard
    * بتتحفظ في YAML فأي ملف يستدعي load() بياخدها تلقائياً
    * مفيش حاجة إعادة معايرة بعد تعديلها
- ⭐ READY pose صارت معرّفة بقيم joint مباشرة (READY_JOINT_VALUES) بدل
    الاعتماد على الـSRDF، عشان تجنب اصطدام الروبوتين.
"""

import os
import copy
import threading

import rospy
import numpy as np
import yaml
import tf2_ros
import tf2_geometry_msgs  # لازم للـPoseStamped.transform()

from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from franka_gripper.msg import MoveGoal
from franka_msgs.msg import FrankaState


# =====================================================================
# --- الاطار المرجعي ---
# =====================================================================
REFERENCE_FRAME = "world"

# =====================================================================
# --- ثوابت الحركة والقابض ---
# =====================================================================
V_SLOW = 0.9
A_SLOW = 0.9
SAFE_H = 0.12

OPEN_WIDTH    = 0.04
CLOSE_WIDTH   = 0.025
GRIPPER_SPEED = 0.1

# =====================================================================
# --- وضعية الـREADY (joint-space) ---
# --- ⭐ بدل ما نعتمد على الـSRDF، الوضعية صارت معرّفة بقيم joint مباشرة ---
# --- القيم مأخوذة يدوياً من المختبر بحيث ما يصطدمش مع الروبوت التاني ---
# --- لتغيير الوضعية: عدّل القيم هنا فقط (radians). ---
# =====================================================================
READY_JOINT_VALUES = [
    -0.06985691071725911,   # panda1_joint1
    -1.1998518015580744,    # panda1_joint2
     0.02219807691630491,   # panda1_joint3
    -2.692064207143927,     # panda1_joint4
     0.019521287538939048,  # panda1_joint5
     1.5087595141071353,    # panda1_joint6
     0.7624571674186945,    # panda1_joint7
]


# =====================================================================
# --- ثوابت هندسة اللوحة ---
# =====================================================================
BOARD_OUTER_SIZE = 0.42
PLAY_AREA_SIZE   = 0.360
MARGIN           = (BOARD_OUTER_SIZE - PLAY_AREA_SIZE) / 2.0  # 0.030m

ORIGIN_X    = 0.3981098174137588
ORIGIN_Y    = -6.41297277950631e-05
ORIGIN_Z    = 0
SQUARE_SIZE = 0.045

# =====================================================================
# --- اعدادات الـprobing ---
# =====================================================================
PROBE_Z              = 0.07
PROBE_SPEED_SCALE    = 0.02
PROBE_ACCEL_SCALE    = 0.02
PROBE_MAX_TRAVEL     = 0.6
PROBE_RETREAT        = 0.020
PROBE_POST_LIFT      = 0.010
PROBE_TRANSIT_Z      = SAFE_H + 0.080
PROBE_FORCE_THRESH   = 8   # عتبة |F_xy| لـW probes  (N)
PROBE_FORCE_THRESH_S = 8   # عتبة |F_xy| لـS probes  (N)
PROBE_BIAS_SAMPLES   = 50
PROBE_BIAS_RATE_HZ   = 100
PROBE_CONTACT_CONSEC = 1     # عدد العينات المتتالية لتأكيد اللمس
PROBE_MIN_TRAVEL     = 0.005 # أقل مسافة قبل قبول contact (تجاهل spike البداية)


# --- معايير contact إضافية (احتياطية لما F_xy يفشل في configuration معين) ---
PROBE_FORCE_THRESH_Z   = 8     # عتبة |F_z|         (N)
PROBE_TORQUE_THRESH    = 2.5   # عتبة max(|τ_ext|) على أي مفصل  (Nm)
PROBE_TORQUE_THRESH_W  = 1.7   # عتبة max(|τ_ext|) على J5/J6/J7 (Nm)
PROBE_INITIAL_SETTLE_S = 2   # ثواني نتجاهل فيها أي trigger في بداية الحركة

# =====================================================================
# --- خيارات قابلة للتعديل (للـoptimization tuning) ---
# =====================================================================
USE_WRIST_DEFLECTION_CORRECTION = False

# --- ازاحة طرف القابض عن مركز TCP (لكل محور على حدة) ---
GRIPPER_TIP_OFFSET_S = (0.018 / 2) - 0.00   # أوفسيت الجنوب (S probes)
GRIPPER_TIP_OFFSET_W = (0.018 / 2) - 0.00   # أوفسيت الغرب (W probes)

# =====================================================================
# --- Fine-tuning offsets (board frame) ---
# --- ⭐ source of truth: عدّلهم هنا مباشرة. مش بيتحفظوا/يتقروا من YAML ---
# --- عشان لما تعدل القيم تتطبق فوراً في كل الملفات اللي بتستورد الموديول ---
# ---   U direction: e_h  (من a -> h)
# ---   V direction: e_N  (من row1 -> row8)
# =====================================================================
#
# Layer 1: Global U/V offset (إزاحة uniform على كل اللوحة).
#          استخدمها لما الإزاحة ثابتة في كل المربعات.
#
SQUARE_OFFSET_U = 0.00   # m, موجب → نحو h
SQUARE_OFFSET_V = 0.000   # m, موجب → نحو row 8

#
# Layer 2: Per-corner offsets (bilinear interpolation للداخل).
#          استخدمها لما الخطأ بيختلف من ركن لآخر (gradient/skew/scale).
#          القيم بالـmillimeters (mm) عشان أسهل للقراءة.
#          الإشارة: (dU_mm, dV_mm) - بإطار اللوحة.
#
# المنطق:
#   - الـ4 corners هي قياس مباشر لمراكز a1, h1, a8, h8.
#   - الـ60 مربع التانيين بيتحسبوا بـbilinear interpolation:
#       * على الـedges: خط مستقيم بين الزاويتين
#       * في الداخل: bilinear surface ناعمة
#   - لو كل القيم = (0,0) → لا يوجد bilinear correction (الـcalibration الأصلية).
#   - لو كل القيم متساوية = نفس الـSQUARE_OFFSET_U/V (redundant).
#   - الـtheta و axes يفضلوا من probing - مش بيتغيروا.
#
CORNER_OFFSETS = {
    'a1': (0.0, 0.0),   # (dU_mm, dV_mm) white 
    'h1': (0, 0.0),
    'a8': (0.0, 0.0),
    'h8': (0.0, 0.0),
}

# --- نقاط بدء الـprobing ---
W1_START_XY = (0.50,  0.15)
W2_START_XY = (0.55,  0.15)
W_PROBE_DIR = (0.0, -1.0)
W_PROBE_YAW = np.pi / 2.0

S1_START_XY = (0.30, 0.01)
S2_START_XY = (0.30, -0.15)
S_PROBE_DIR = (+1.0, 0.0)
S_PROBE_YAW = 0.0

CALIB_FILE = os.path.expanduser("~/board_calibration_P1_v2.yaml")


# =====================================================================
# --- TF buffer عالمي ---
# =====================================================================
_tf_buffer   = None
_tf_listener = None

def init_tf(timeout=5.0):
    """ينشئ TF buffer/listener ويتحقق من التحويل panda1_link0 -> world."""
    global _tf_buffer, _tf_listener
    if _tf_buffer is not None:
        return _tf_buffer

    _tf_buffer   = tf2_ros.Buffer()
    _tf_listener = tf2_ros.TransformListener(_tf_buffer)
    rospy.loginfo(f"Waiting for TF: panda1_link0 <-> {REFERENCE_FRAME} ...")
    try:
        _tf_buffer.can_transform(REFERENCE_FRAME, "panda1_link0",
                                 rospy.Time(0), rospy.Duration(timeout))
        rospy.loginfo(f"  TF available: panda1_link0 -> {REFERENCE_FRAME}")
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"  TF NOT available: {e}")
        rospy.logerr("  Check that static_transform_publisher or URDF "
                     "defines the link between panda1_link0 and "
                     f"{REFERENCE_FRAME}.")
    return _tf_buffer


def get_current_pose_in_ref(move_group, timeout=1.0):
    """
    ترجع current pose في REFERENCE_FRAME (world) بدل الـplanning frame.
    ترجع: geometry_msgs/Pose (بدون header).
    """
    ps_planning = move_group.get_current_pose()
    if _tf_buffer is None:
        return ps_planning.pose
    try:
        ps_ref = _tf_buffer.transform(ps_planning, REFERENCE_FRAME,
                                      rospy.Duration(timeout))
        return ps_ref.pose
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"get_current_pose_in_ref: TF transform failed: {e}")
        return ps_planning.pose

# =====================================================================
# --- دوال حركة مساعدة ---
# =====================================================================
def make_down_pose(x, y, z, yaw):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    pose.orientation = Quaternion(*q)
    return pose


def move_to_pose(move_group, x, y, z, vf, af, yaw=0.0):
    """حركة Cartesian بسيطة (pointing down) مع retime."""
    pose = make_down_pose(x, y, z, yaw)
    waypoints = [copy.deepcopy(pose)]
    plan, fraction = move_group.compute_cartesian_path(waypoints, 0.05, False)
    if fraction > 0.65:
        plan = move_group.retime_trajectory(
            move_group.get_current_state(), plan, vf, af,
            "iterative_time_parameterization")
        move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

def move_to_ready(move_group):
    """
    يحرك الروبوت لوضعية الـREADY (joint-space) المعرّفة بقيم joint مباشرة في
    READY_JOINT_VALUES (radians) في أعلى الملف.

    هذه النسخة لا تعتمد على الـSRDF (set_named_target) - بدل كده
    بتحدد الـ7 joints مباشرة، مما يضمن أن الوضعية ثابتة ومعروفة
    وما تتعارضش مع الروبوت التاني.

    Parameters
    ----------
    move_group : moveit_commander.MoveGroupCommander
        الـMoveGroupCommander الخاص بالذراع (مثلاً panda1_arm).

    Returns
    -------
    bool
        True إذا وصل بنجاح، False غير ذلك.

    Usage (من ملف تاني)
    -------------------
    >>> from board_calibration import move_to_ready
    >>> import moveit_commander
    >>> arm = moveit_commander.MoveGroupCommander("panda1_arm")
    >>> move_to_ready(arm)   # بدل move_group.set_named_target("ready")

    لتغيير الوضعية: عدّل القيم في READY_JOINT_VALUES في أعلى الملف.
    """
    rospy.loginfo("[READY] Moving to joint-space ready pose:")
    rospy.loginfo(f"        joints (rad) = "
                  f"[{', '.join(f'{v:+.4f}' for v in READY_JOINT_VALUES)}]")
    move_group.set_joint_value_target(list(READY_JOINT_VALUES))
    success = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    if not success:
        rospy.logerr("[READY] Failed to reach ready joint pose!")
    return success

def gripper_full_close(move_client, speed=None):
    """
    يسكر القابض لأقصى حد ممكن (width=0) عشان لما يلامس اللوحة أثناء
    الـprobing ما ينفتح. يُنادى قبل كل probe.
    """
    goal = MoveGoal()
    goal.width = 0.0
    goal.speed = float(GRIPPER_SPEED if speed is None else speed)
    move_client.send_goal(goal)
    move_client.wait_for_result()


def _rot_2d(v, angle):
    """دوران متجه 2D بزاوية angle (CCW) حول نقطة الأصل."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])

def _yaw_from_pose(pose):
    """يستخرج yaw من geometry_msgs/Pose.orientation (rad). [DIAG helper]"""
    q = pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return float(yaw)

def _wrap_angle(a):
    """يلف الزاوية إلى [-pi, +pi] لتفادي قفزات ±360°. [DIAG helper]"""
    return float(np.arctan2(np.sin(a), np.cos(a)))


# =====================================================================
# --- مراقب القوة + العزوم ---
# =====================================================================
class ForceMonitor:
    """
    يقرأ من FrankaState:
      - O_F_ext_hat_K        : قوة خارجية تقديرية في الـbase frame (Fx,Fy,Fz,...).
      - tau_ext_hat_filtered : عزم خارجي تقديري على كل مفصل (7 قيم).
    وين بيوفر:
      - get_xy_magnitude()        : |F_xy| بعد bias  [legacy].
      - get_force_components()    : (Fx, Fy, Fz) بعد bias.
      - get_max_joint_torque()    : max(|τ_ext_i|) بعد bias على كل المفاصل.
      - get_max_wrist_torque()    : max(|τ_ext_i|) بعد bias على J5/J6/J7.
    """
    def __init__(self, topic="/panda1/franka_state_controller/franka_states"):
        self._lock = threading.Lock()
        self._force_xyz = np.zeros(3)
        self._bias_xyz  = np.zeros(3)
        self._tau_ext   = np.zeros(7)
        self._tau_bias  = np.zeros(7)
        self._got_msg   = False
        self._sub = rospy.Subscriber(topic, FrankaState, self._cb, queue_size=1)

    def _cb(self, msg):
        f = msg.O_F_ext_hat_K
        with self._lock:
            self._force_xyz = np.array([f[0], f[1], f[2]], dtype=float)
            self._tau_ext = np.array(msg.tau_ext_hat_filtered, dtype=float)
            self._got_msg = True


    def wait_for_data(self, timeout=5.0):
        start = rospy.Time.now()
        rate = rospy.Rate(50)
        while not self._got_msg and not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > timeout:
                raise RuntimeError("No FrankaState message received (topic?)")
            rate.sleep()

    def zero_bias(self, n_samples=None, rate_hz=None):
        if n_samples is None: n_samples = PROBE_BIAS_SAMPLES
        if rate_hz   is None: rate_hz   = PROBE_BIAS_RATE_HZ
        rate = rospy.Rate(rate_hz)
        f_samples, tau_samples = [], []
        for _ in range(n_samples):
            with self._lock:
                f_samples.append(self._force_xyz.copy())
                tau_samples.append(self._tau_ext.copy())
            rate.sleep()
        self._bias_xyz = np.mean(f_samples, axis=0)
        self._tau_bias = np.mean(tau_samples, axis=0)
        rospy.loginfo(f"  [ForceMonitor] F_bias = "
                      f"({self._bias_xyz[0]:+.2f}, {self._bias_xyz[1]:+.2f}, "
                      f"{self._bias_xyz[2]:+.2f}) N | "
                      f"|τ_bias|_max = "
                      f"{float(np.max(np.abs(self._tau_bias))):.3f} Nm")

    # --- legacy ---
    def get_xy_magnitude(self):
        with self._lock:
            f = self._force_xyz - self._bias_xyz
        return float(np.hypot(f[0], f[1]))

    # --- جديد ---
    def get_force_components(self):
        """ترجع (F_x, F_y, F_z) بعد طرح الـbias."""
        with self._lock:
            f = self._force_xyz - self._bias_xyz
        return float(f[0]), float(f[1]), float(f[2])

    def get_max_joint_torque(self):
        """أكبر |τ_ext| على أي من المفاصل السبعة (بعد bias)."""
        with self._lock:
            tau = self._tau_ext - self._tau_bias
        return float(np.max(np.abs(tau)))

    def get_max_wrist_torque(self):
        """أكبر |τ_ext| على مفاصل الـwrist (J5,J6,J7) (بعد bias)."""
        with self._lock:
            tau = self._tau_ext - self._tau_bias
        return float(np.max(np.abs(tau[4:7])))


# =====================================================================
# --- فئة المعايرة وبناء المواقع ---
# =====================================================================
class BoardCalibration:
    """
    يمسك حالة المعايرة (corner, axes, theta, fine-tune offsets) ويبنى عليها:
      - square_positions    : a1..h8
      - mirrored_squares    : reversed mapping (الـindex بيستخدم ده)
      - promotion_positions : q, r, b, n
      - graveyard_positions : 24 مربع
      - hx, hy, hz          : نقطة الـhome
    ويوفر probing + calibrate (two-pass) + save/load/show/test.

    Fine-tuning offsets:
      self.square_offset_u : إزاحة على محور e_h (a -> h).
      self.square_offset_v : إزاحة على محور e_N (row1 -> row8).
    تنطبق على كل المربعات بعد build_positions(). تتحفظ وتُقرأ من YAML.
    """

    def __init__(self):
        # حالة افتراضية قبل المعايرة
        self.board_corner = np.array([
            ORIGIN_X + 7.5 * SQUARE_SIZE + MARGIN,
            ORIGIN_Y - 7.5 * SQUARE_SIZE - MARGIN,
        ])
        self.e_h_axis    = np.array([0.0, 1.0])
        self.e_N_axis    = np.array([-1.0, 0.0])
        self.board_theta = 0.0

        # --- Fine-tuning offsets (board frame). تُهيأ من الثوابت default. ---
        self.square_offset_u = float(SQUARE_OFFSET_U)
        self.square_offset_v = float(SQUARE_OFFSET_V)

        # --- Per-corner offsets (mm, board frame). نسخة instance من الـconstants. ---
        # المفاتيح physical: a1, h1, a8, h8 (مش حسب لون اللاعب).
        self.corner_offsets = {k: (float(v[0]), float(v[1]))
                               for k, v in CORNER_OFFSETS.items()}

        self.square_positions    = {}
        self.mirrored_squares    = {}
        self.promotion_positions = {}
        self.graveyard_positions = []
        self.hx = self.hy = self.hz = 0.0

        self.build_positions()


    # ------------------------------------------------------------------
    # --- helper: bilinear offset لكل مربع ---
    # ------------------------------------------------------------------
    def _bilinear_offset_mm(self, col_idx, row_idx):
        """
        ترجع (du_mm, dv_mm) للمربع المعطى عبر bilinear interpolation
        بين الـ4 corners (a1, h1, a8, h8).

        col_idx: 0..7 (a..h)
        row_idx: 0..7 (row1..row8)
        """
        u = col_idx / 7.0   # 0=a, 1=h
        v = row_idx / 7.0   # 0=row1, 1=row8
        c = self.corner_offsets

        du = ((1.0 - u) * (1.0 - v) * c['a1'][0] +
              u         * (1.0 - v) * c['h1'][0] +
              (1.0 - u) * v         * c['a8'][0] +
              u         * v         * c['h8'][0])
        dv = ((1.0 - u) * (1.0 - v) * c['a1'][1] +
              u         * (1.0 - v) * c['h1'][1] +
              (1.0 - u) * v         * c['a8'][1] +
              u         * v         * c['h8'][1])
        return du, dv


    # ------------------------------------------------------------------
    # --- بناء خرائط المواقع ---
    # ------------------------------------------------------------------
    def build_positions(self, corner=None, eh=None, eN=None):
        if corner is None: corner = self.board_corner
        if eh     is None: eh     = self.e_h_axis
        if eN     is None: eN     = self.e_N_axis

        corner = np.asarray(corner, dtype=float)
        eh     = np.asarray(eh,     dtype=float)
        eN     = np.asarray(eN,     dtype=float)

        # --- Layer 1: global U/V offset (uniform). ---
        global_fine = self.square_offset_u * eh + self.square_offset_v * eN

        # --- Layer 2: bilinear per-corner offset (per-square). ---
        # ينطبق على المربعات الـ64 فقط (داخل اللوحة).
        # promotion + graveyard خارج اللوحة، فيستخدموا global_fine فقط.

        sq_pos = {}
        for col_idx, col in enumerate('abcdefgh'):
            for row in range(1, 9):
                u_pos = MARGIN + (col_idx + 0.5) * SQUARE_SIZE
                v_pos = MARGIN + (row - 1 + 0.5) * SQUARE_SIZE

                # bilinear corner offset لهذا المربع (mm → m)
                du_mm, dv_mm = self._bilinear_offset_mm(col_idx, row - 1)
                bilinear_fine = (du_mm / 1000.0) * eh + (dv_mm / 1000.0) * eN

                fine = global_fine + bilinear_fine
                p = corner + u_pos * eh + v_pos * eN + fine
                sq_pos[f"{col}{row}"] = (float(p[0]), float(p[1]), ORIGIN_Z)

        board_u_far = MARGIN + 8 * SQUARE_SIZE + 0.01

        u_promo = board_u_far + 3.5 * SQUARE_SIZE
        v_by_letter = {
            'q': MARGIN + 0.5 * SQUARE_SIZE,
            'r': MARGIN + 1.5 * SQUARE_SIZE,
            'b': MARGIN + 2.5 * SQUARE_SIZE,
            'n': MARGIN + 3.5 * SQUARE_SIZE,
        }
        promo = {}
        for letter, v in v_by_letter.items():
            # promotion خارج اللوحة → global offset فقط (مش bilinear)
            p = corner + u_promo * eh + v * eN + global_fine
            promo[letter] = (float(p[0]), float(p[1]), ORIGIN_Z)

        graves = []
        for col_g in range(3):
            u = board_u_far + (col_g + 0.5) * SQUARE_SIZE
            for row_g in range(8):
                v = MARGIN + (0.5 + row_g) * SQUARE_SIZE
                # graveyard خارج اللوحة → global offset فقط
                p = corner + u * eh + v * eN + global_fine
                graves.append((float(p[0]), float(p[1]), ORIGIN_Z))

        self.square_positions    = sq_pos
        self.promotion_positions = promo
        self.graveyard_positions = graves

        original_keys = list(self.square_positions.keys())
        reversed_keys = list(reversed(original_keys))
        self.mirrored_squares = {
            original_keys[i]: self.square_positions[reversed_keys[i]]
            for i in range(len(original_keys))
        }

        base_hx, base_hy, _ = self.square_positions["h8"]
        self.hx = base_hx - 0.1
        self.hy = base_hy + 0.3
        self.hz = SAFE_H


    # ------------------------------------------------------------------
    # --- API صغير لضبط الـoffset برمجياً ---
    # ------------------------------------------------------------------
    def set_offsets(self, offset_u=None, offset_v=None, rebuild=True):
        """
        يضبط الـfine-tuning offsets (بالأمتار، بإطار اللوحة) ويعيد بناء
        المواقع إذا rebuild=True.

        ⚠️ التعديل ده مؤقت (in-memory فقط). عشان يبقى دائم لازم تعدّل
        الـconstants SQUARE_OFFSET_U/V في أعلى الملف يدوياً.
        """
        if offset_u is not None:
            self.square_offset_u = float(offset_u)
        if offset_v is not None:
            self.square_offset_v = float(offset_v)
        if rebuild:
            self.build_positions()
        rospy.loginfo(f"[OFFSET] in-memory: u={self.square_offset_u*1000:+.2f}mm "
                      f"v={self.square_offset_v*1000:+.2f}mm "
                      f"({'rebuilt' if rebuild else 'pending'})")
        rospy.loginfo(f"[OFFSET] لتثبيت دائم: عدّل SQUARE_OFFSET_U/V "
                      f"في أعلى board_calibration_p1_v1.py")


    # ------------------------------------------------------------------
    # --- API لضبط corner offsets برمجياً ---
    # ------------------------------------------------------------------
    def set_corner_offset(self, corner_name, du_mm=None, dv_mm=None,
                          rebuild=True):
        """
        يضبط offset لركن واحد من الـ4. corner_name physical:
        'a1', 'h1', 'a8', 'h8'.

        - du_mm: إزاحة على محور U (a→h) بالمليمتر، أو None لعدم التغيير.
        - dv_mm: إزاحة على محور V (row1→row8) بالمليمتر، أو None.
        - rebuild: لو True يعيد بناء square_positions تلقائياً.

        ⚠️ التعديل مؤقت (in-memory). لتثبيت دائم: عدّل CORNER_OFFSETS
           في أعلى board_calibration_p1_v1.py.
        """
        if corner_name not in self.corner_offsets:
            rospy.logerr(f"[CORNER_OFFSET] Invalid corner: {corner_name!r}. "
                         f"Must be one of: "
                         f"{sorted(self.corner_offsets.keys())}")
            return False

        cur_du, cur_dv = self.corner_offsets[corner_name]
        new_du = float(du_mm) if du_mm is not None else cur_du
        new_dv = float(dv_mm) if dv_mm is not None else cur_dv
        self.corner_offsets[corner_name] = (new_du, new_dv)

        if rebuild:
            self.build_positions()

        rospy.loginfo(f"[CORNER_OFFSET] {corner_name} = "
                      f"({new_du:+.2f}, {new_dv:+.2f}) mm "
                      f"({'rebuilt' if rebuild else 'pending'})")
        return True

    def reset_corner_offsets(self, rebuild=True):
        """يصفّر كل الـ4 corner offsets."""
        for cname in ('a1', 'h1', 'a8', 'h8'):
            self.corner_offsets[cname] = (0.0, 0.0)
        if rebuild:
            self.build_positions()
        rospy.loginfo(f"[CORNER_OFFSET] all reset to (0, 0)")

    def print_corner_offsets_constants(self):
        """يطبع الـcorner offsets الحالية بصيغة constants جاهزة للنسخ."""
        print("# انسخ ده فوق board_calibration_p1_v1.py:")
        print("CORNER_OFFSETS = {")
        for cname in ('a1', 'h1', 'a8', 'h8'):
            du, dv = self.corner_offsets[cname]
            print(f"    '{cname}': ({du:+.4f}, {dv:+.4f}),")
        print("}")


    # ------------------------------------------------------------------
    # --- probing ---
    # ------------------------------------------------------------------
    def _probe_linear(self, move_group, force_monitor, direction_xy,
                      max_travel=PROBE_MAX_TRAVEL,
                      force_threshold=PROBE_FORCE_THRESH):
        """
        يتحرك خطياً حتى يلامس عقبة. الـcontact يتم رصده بـOR بين
        أربع معايير:
            |F_xy| > force_threshold
            |F_z|  > PROBE_FORCE_THRESH_Z
            max(|τ_ext|) > PROBE_TORQUE_THRESH        (كل المفاصل)
            max(|τ_ext_wrist|) > PROBE_TORQUE_THRESH_W (J5/J6/J7)
        ويرجع (نقطة التلامس [x, y], delta_yaw_settled) أو None.
        """
        cur = get_current_pose_in_ref(move_group)
        start = np.array([cur.position.x, cur.position.y])
        z = cur.position.z

        yaw_start = _yaw_from_pose(cur)

        d = np.asarray(direction_xy, dtype=float)
        d = d / np.linalg.norm(d)
        target_xy = start + d * max_travel

        target_pose = copy.deepcopy(cur)
        target_pose.position.x = float(target_xy[0])
        target_pose.position.y = float(target_xy[1])
        target_pose.position.z = float(z)

        plan, fraction = move_group.compute_cartesian_path(
            [target_pose], 0.005, False)
        if fraction < 0.65:
            rospy.logerr(f"  [probe] cartesian plan failed: fraction={fraction:.2f}")
            return None

        plan = move_group.retime_trajectory(
            move_group.get_current_state(), plan,
            PROBE_SPEED_SCALE, PROBE_ACCEL_SCALE,
            "iterative_time_parameterization")

        rospy.loginfo("  [probe] zeroing force/torque bias ...")
        force_monitor.zero_bias()

        rospy.loginfo(f"  [probe] moving: dir=({d[0]:+.2f},{d[1]:+.2f}), "
                      f"max={max_travel*1000:.0f}mm")
        rospy.loginfo(f"  [probe] thresholds: |F_xy|>{force_threshold:.1f}N | "
                      f"|F_z|>{PROBE_FORCE_THRESH_Z:.1f}N | "
                      f"|τ|_max>{PROBE_TORQUE_THRESH:.2f}Nm | "
                      f"|τ_wrist|>{PROBE_TORQUE_THRESH_W:.2f}Nm")
        rospy.loginfo(f"  [probe][DIAG] yaw_start              = "
                      f"{np.degrees(yaw_start):+.4f} deg")
        move_group.execute(plan, wait=False)


        rate = rospy.Rate(200)
        contact = False
        consec = 0
        contact_reason = None
        t0 = rospy.Time.now()
        timeout = 30.0
        last_log_t = rospy.Time.now()

        fxy_max     = 0.0
        fz_max      = 0.0
        tau_max_max = 0.0
        tau_w_max   = 0.0

        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() < timeout:
            elapsed = (rospy.Time.now() - t0).to_sec()

            fx, fy, fz = force_monitor.get_force_components()
            fmag_xy = float(np.hypot(fx, fy))
            fmag_z  = float(abs(fz))
            tau_max = force_monitor.get_max_joint_torque()
            tau_w   = force_monitor.get_max_wrist_torque()

            fxy_max     = max(fxy_max, fmag_xy)
            fz_max      = max(fz_max,  fmag_z)
            tau_max_max = max(tau_max_max, tau_max)
            tau_w_max   = max(tau_w_max,   tau_w)

            cur_now = get_current_pose_in_ref(move_group)
            traveled = np.hypot(cur_now.position.x - start[0],
                                cur_now.position.y - start[1])

            if (rospy.Time.now() - last_log_t).to_sec() >= 0.25:
                rospy.loginfo(f"  [probe][LIVE] F=({fx:+.2f},{fy:+.2f},{fz:+.2f}) "
                              f"|F_xy|={fmag_xy:.2f} |F_z|={fmag_z:.2f} "
                              f"τ_max={tau_max:.2f} τ_wrist={tau_w:.2f} "
                              f"trav={traveled*1000:.1f}mm")
                last_log_t = rospy.Time.now()

            in_settle = (elapsed < PROBE_INITIAL_SETTLE_S
                         or traveled < PROBE_MIN_TRAVEL)

            if not in_settle:
                triggers = []
                if fmag_xy > force_threshold:
                    triggers.append(f"|F_xy|={fmag_xy:.2f}N")
                if fmag_z > PROBE_FORCE_THRESH_Z:
                    triggers.append(f"|F_z|={fmag_z:.2f}N")
                if tau_max > PROBE_TORQUE_THRESH:
                    triggers.append(f"|τ_max|={tau_max:.2f}Nm")
                if tau_w > PROBE_TORQUE_THRESH_W:
                    triggers.append(f"|τ_wrist|={tau_w:.2f}Nm")

                if triggers:
                    consec += 1
                    if consec >= PROBE_CONTACT_CONSEC:
                        move_group.stop()
                        contact_reason = " + ".join(triggers)
                        rospy.loginfo(f"  [probe] CONTACT! [{contact_reason}] "
                                      f"traveled={traveled*1000:.1f}mm")
                        contact = True
                        break
                else:
                    consec = 0

            if np.hypot(target_xy[0]-cur_now.position.x,
                        target_xy[1]-cur_now.position.y) < 0.002:
                rospy.logwarn(f"  [probe] reached max_travel without contact")
                rospy.logwarn(f"  [probe] MAX seen: |F_xy|={fxy_max:.2f}N | "
                              f"|F_z|={fz_max:.2f}N | "
                              f"|τ|_max={tau_max_max:.2f}Nm | "
                              f"|τ_wrist|={tau_w_max:.2f}Nm")
                break
            rate.sleep()


        move_group.stop()
        move_group.clear_pose_targets()
        if not contact:
            return None

        cur_loaded = get_current_pose_in_ref(move_group)
        yaw_loaded = _yaw_from_pose(cur_loaded)
        d_yaw_loaded = _wrap_angle(yaw_loaded - yaw_start)
        tip_err_loaded_mm = abs(GRIPPER_TIP_OFFSET_S *
                                np.sin(d_yaw_loaded)) * 1000.0
        rospy.loginfo(f"  [probe][DIAG] yaw_loaded   (contact) = "
                      f"{np.degrees(yaw_loaded):+.4f} deg")
        rospy.loginfo(f"  [probe][DIAG] delta_yaw    (loaded)  = "
                      f"{np.degrees(d_yaw_loaded):+.4f} deg "
                      f"-> tip err ~ {tip_err_loaded_mm:.3f} mm")

        rospy.sleep(0.25)

        final = get_current_pose_in_ref(move_group)
        yaw_settled = _yaw_from_pose(final)
        d_yaw_settled = _wrap_angle(yaw_settled - yaw_start)
        tip_err_settled_mm = abs(GRIPPER_TIP_OFFSET_S *
                                 np.sin(d_yaw_settled)) * 1000.0
        rospy.loginfo(f"  [probe][DIAG] yaw_settled            = "
                      f"{np.degrees(yaw_settled):+.4f} deg")
        rospy.loginfo(f"  [probe][DIAG] delta_yaw    (settled) = "
                      f"{np.degrees(d_yaw_settled):+.4f} deg "
                      f"-> tip err ~ {tip_err_settled_mm:.3f} mm")

        if abs(np.degrees(d_yaw_settled)) > 3.0:
            rospy.logwarn(f"  [probe][DIAG] LARGE wrist deflection: "
                          f"{np.degrees(d_yaw_settled):+.3f} deg")

        return np.array([final.position.x, final.position.y]), float(d_yaw_settled)


    def _do_probe(self, arm, force_monitor, gripper_client,
                  name, start_xy, direction, probe_yaw,
                  force_threshold=PROBE_FORCE_THRESH):
        """approach + touch + retreat.
        ترجع (نقطة التلامس, delta_yaw_settled) أو None.
        """
        rospy.loginfo(f"[Probe] {name}")

        if gripper_client is not None:
            rospy.loginfo("  [probe] full-closing gripper before probe")
            gripper_full_close(gripper_client)

        cur = get_current_pose_in_ref(arm)
        move_to_pose(arm, cur.position.x, cur.position.y, PROBE_TRANSIT_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        rospy.loginfo(f"  move -> start ({start_xy[0]:+.4f}, {start_xy[1]:+.4f}) "
                      f"@ Z={PROBE_TRANSIT_Z:.3f} yaw={np.degrees(probe_yaw):+.2f}deg")
        move_to_pose(arm, start_xy[0], start_xy[1], PROBE_TRANSIT_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        rospy.loginfo(f"  descend -> Z={PROBE_Z:.3f}")
        move_to_pose(arm, start_xy[0], start_xy[1], PROBE_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        result = self._probe_linear(arm, force_monitor, direction,
                                    max_travel=PROBE_MAX_TRAVEL,
                                    force_threshold=force_threshold)
        if result is None:
            rospy.logerr(f"{name}: no contact detected within "
                         f"{PROBE_MAX_TRAVEL*1000:.0f}mm.")
            cur = get_current_pose_in_ref(arm)
            move_to_pose(arm, cur.position.x, cur.position.y, PROBE_TRANSIT_Z,
                         V_SLOW, A_SLOW, yaw=probe_yaw)
            return None

        contact, d_yaw = result
        rospy.loginfo(f"  contact @ ({contact[0]:+.4f}, {contact[1]:+.4f})")

        d_norm = np.asarray(direction, dtype=float)
        d_norm = d_norm / np.linalg.norm(d_norm)
        back_xy = contact - d_norm * PROBE_RETREAT
        rospy.loginfo(f"  retreat -> ({back_xy[0]:+.4f}, {back_xy[1]:+.4f})")
        move_to_pose(arm, back_xy[0], back_xy[1], PROBE_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        lift_z = PROBE_Z + PROBE_POST_LIFT
        move_to_pose(arm, back_xy[0], back_xy[1], lift_z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        move_to_pose(arm, back_xy[0], back_xy[1], PROBE_TRANSIT_Z,
                     V_SLOW, A_SLOW, yaw=probe_yaw)

        return contact, d_yaw


    # ------------------------------------------------------------------
    # --- المعايرة الرئيسية (Two-pass) - تبدأ من الجنوب ---
    # ------------------------------------------------------------------
    def calibrate(self, arm, force_monitor, gripper_client=None, confirm=True):
        rospy.loginfo("=" * 62)
        rospy.loginfo("Starting board calibration (two-pass, South-first)")
        rospy.loginfo(f"  probe height Z   : {PROBE_Z:.3f} m")
        rospy.loginfo(f"  transit height Z : {PROBE_TRANSIT_Z:.3f} m")
        rospy.loginfo(f"  post-contact lift: {PROBE_POST_LIFT*1000:.0f} mm")
        rospy.loginfo(f"  probe speed      : {PROBE_SPEED_SCALE*100:.0f}% of max")
        rospy.loginfo(f"  |F_xy| thresh    : {PROBE_FORCE_THRESH:.1f} N")
        rospy.loginfo(f"  |F_z|  thresh    : {PROBE_FORCE_THRESH_Z:.1f} N")
        rospy.loginfo(f"  |τ|_max thresh   : {PROBE_TORQUE_THRESH:.2f} Nm")
        rospy.loginfo(f"  |τ_wrist| thresh : {PROBE_TORQUE_THRESH_W:.2f} Nm")
        rospy.loginfo(f"  max travel       : {PROBE_MAX_TRAVEL*1000:.0f} mm")
        rospy.loginfo(f"  wrist-deflection correction: "
                      f"{'ENABLED' if USE_WRIST_DEFLECTION_CORRECTION else 'DISABLED'} "
                      f"(constant @ top of file)")
        rospy.loginfo(f"  fine-tune offset : "
                      f"({self.square_offset_u*1000:+.2f}, "
                      f"{self.square_offset_v*1000:+.2f}) mm "
                      f"(preserved across calibration)")
        rospy.loginfo("=" * 62)

        if confirm:
            ans = input("Make sure the workspace is CLEAR. Proceed? [y/N]: "
                        ).strip().lower()
            if ans != 'y':
                rospy.loginfo("Calibration aborted by user.")
                return False

        # --- الذهاب لموقع الـREADY أولاً ---
        move_to_ready(arm)

        use_deflection_corr = USE_WRIST_DEFLECTION_CORRECTION
        if use_deflection_corr:
            rospy.loginfo(">>> Wrist-deflection correction: ENABLED")
        else:
            rospy.loginfo(">>> Wrist-deflection correction: DISABLED (default)")


        # =============================================================
        # PASS 1: S1 + S2 (جنوب) لتقدير θ الأولي
        # =============================================================
        rospy.loginfo(">>> PASS 1: estimating board angle from S1 & S2 (South)")
        res = self._do_probe(arm, force_monitor, gripper_client,
                             "S1 (pass1)", S1_START_XY,
                             S_PROBE_DIR, S_PROBE_YAW,
                             force_threshold=PROBE_FORCE_THRESH_S)
        if res is None:
            move_to_ready(arm)
            return False
        S1_p1, _ = res

        res = self._do_probe(arm, force_monitor, gripper_client,
                             "S2 (pass1)", S2_START_XY,
                             S_PROBE_DIR, S_PROBE_YAW,
                             force_threshold=PROBE_FORCE_THRESH_S)
        if res is None:
            move_to_ready(arm)
            return False
        S2_p1, _ = res

        # المحور الجنوبي يمشي بين S1 و S2 (اتجاه e_N تقريباً)
        dS = S2_p1 - S1_p1
        # theta_est: زاوية اللوحة. e_h عمودي على dS.
        # dS هو تقريباً اتجاه e_N (row1->row8)، فـtheta = atan2(dS_x, -dS_y) + corr
        # لكن أسهل: نحسب اتجاه الضلع الجنوبي (dS) ونستنتج theta منه.
        # الضلع الجنوبي موازي لـe_N، فـe_h عمودي عليه: e_h = rot90(dS/|dS|)
        theta_est = float(np.arctan2(dS[0], -dS[1]))
        rospy.loginfo(f"  S2 - S1 = ({dS[0]:+.4f}, {dS[1]:+.4f})")
        rospy.loginfo(f"  Estimated board theta (from South) = "
                      f"{np.degrees(theta_est):+.3f} deg")


        # =============================================================
        # PASS 2: الرجوع بـyaw/dir مصححين بـθ (الجنوب أولاً ثم الغرب)
        # =============================================================
        rospy.loginfo(">>> PASS 2: re-probing with yaw/dir rotated by theta "
                      "(South first, then West)")
        S_dir_rot  = _rot_2d(np.array(S_PROBE_DIR, dtype=float), theta_est)
        W_dir_rot  = _rot_2d(np.array(W_PROBE_DIR, dtype=float), theta_est)
        S_yaw_corr = S_PROBE_YAW + theta_est
        W_yaw_corr = W_PROBE_YAW + theta_est

        rospy.loginfo(f"  S: yaw={np.degrees(S_yaw_corr):+.2f}deg, "
                      f"dir=({S_dir_rot[0]:+.3f},{S_dir_rot[1]:+.3f})")
        rospy.loginfo(f"  W: yaw={np.degrees(W_yaw_corr):+.2f}deg, "
                      f"dir=({W_dir_rot[0]:+.3f},{W_dir_rot[1]:+.3f})")

        # ترتيب: S1, S2, W1, W2 (الجنوب أولاً)
        probes_p2 = [
            ("S1 (pass2)", S1_START_XY, S_dir_rot, S_yaw_corr, PROBE_FORCE_THRESH_S),
            ("S2 (pass2)", S2_START_XY, S_dir_rot, S_yaw_corr, PROBE_FORCE_THRESH_S),
            ("W1 (pass2)", W1_START_XY, W_dir_rot, W_yaw_corr, PROBE_FORCE_THRESH),
            ("W2 (pass2)", W2_START_XY, W_dir_rot, W_yaw_corr, PROBE_FORCE_THRESH),
        ]

        contacts = []
        probe_dirs = []
        delta_yaws = []
        for name, start_xy, direction, probe_yaw, thresh in probes_p2:
            res = self._do_probe(arm, force_monitor, gripper_client,
                                 name, start_xy, direction, probe_yaw,
                                 force_threshold=thresh)
            if res is None:
                rospy.logerr(f"{name}: aborting calibration.")
                move_to_ready(arm)
                return False
            contact, d_yaw = res
            contacts.append(contact)
            probe_dirs.append(direction / np.linalg.norm(direction))
            delta_yaws.append(d_yaw)

        # الترتيب الآن: S1, S2, W1, W2
        S1, S2, W1, W2 = contacts


        # --- حساب المحاور ---
        dN = W2 - W1
        eN_meas = dN / np.linalg.norm(dN)
        dH = S2 - S1
        eh_meas = dH / np.linalg.norm(dH)

        dot = float(np.dot(eh_meas, eN_meas))
        perp_err_deg = abs(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) - 90.0)
        rospy.loginfo(f"Measured axes (pass2):")
        rospy.loginfo(f"  e_h_meas = ({eh_meas[0]:+.5f}, {eh_meas[1]:+.5f})")
        rospy.loginfo(f"  e_N_meas = ({eN_meas[0]:+.5f}, {eN_meas[1]:+.5f})")
        rospy.loginfo(f"  perpendicularity error = {perp_err_deg:.2f} deg")
        if perp_err_deg > 3.0:
            rospy.logwarn("  WARNING: perp. error large. Verify probe contacts.")

        eN_as_eh = np.array([eN_meas[1], -eN_meas[0]])
        eh_avg = (eh_meas + eN_as_eh) / 2.0
        eh_avg = eh_avg / np.linalg.norm(eh_avg)
        eN_fix = np.array([-eh_avg[1], eh_avg[0]])

        # --- تصحيح نقاط التلامس بازاحة نصف عرض الجريبر (لكل محور أوفسيت خاص) ---
        # probes_p2 ترتيب: S1, S2, W1, W2
        # S probes تستخدم GRIPPER_TIP_OFFSET_S، W probes تستخدم GRIPPER_TIP_OFFSET_W
        tip_offsets = [GRIPPER_TIP_OFFSET_S, GRIPPER_TIP_OFFSET_S,
                       GRIPPER_TIP_OFFSET_W, GRIPPER_TIP_OFFSET_W]

        if use_deflection_corr:
            rospy.loginfo(f"Applying gripper tip offsets: "
                          f"S={GRIPPER_TIP_OFFSET_S*1000:.1f}mm, "
                          f"W={GRIPPER_TIP_OFFSET_W*1000:.1f}mm "
                          f"(WITH wrist-deflection correction)")
            offset_dirs = [_rot_2d(d, dy)
                           for d, dy in zip(probe_dirs, delta_yaws)]
        else:
            rospy.loginfo(f"Applying gripper tip offsets: "
                          f"S={GRIPPER_TIP_OFFSET_S*1000:.1f}mm, "
                          f"W={GRIPPER_TIP_OFFSET_W*1000:.1f}mm "
                          f"(legacy: nominal direction)")
            offset_dirs = list(probe_dirs)

        contacts_corr = [c + d * off
                         for c, d, off in zip(contacts, offset_dirs, tip_offsets)]
        S1, S2, W1, W2 = contacts_corr
        for (name, *_rest), c_raw, c_cor, d_yaw in zip(
                probes_p2, contacts, contacts_corr, delta_yaws):
            rospy.loginfo(f"  {name}: raw=({c_raw[0]:+.4f},{c_raw[1]:+.4f}) "
                          f"-> corr=({c_cor[0]:+.4f},{c_cor[1]:+.4f}) "
                          f"[Δyaw={np.degrees(d_yaw):+.3f}deg]")


        # اعادة حساب المحاور من النقاط المصححة
        eN_meas = (W2 - W1) / np.linalg.norm(W2 - W1)
        eh_meas = (S2 - S1) / np.linalg.norm(S2 - S1)
        eN_as_eh = np.array([eN_meas[1], -eN_meas[0]])
        eh_avg = (eh_meas + eN_as_eh) / 2.0
        eh_avg = eh_avg / np.linalg.norm(eh_avg)
        eN_fix = np.array([-eh_avg[1], eh_avg[0]])

        A = np.column_stack([eN_fix, -eh_avg])
        b = S1 - W1
        try:
            uv = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            rospy.logerr("Line intersection failed (axes nearly parallel).")
            move_to_ready(arm)
            return False
        corner_meas = W1 + uv[0] * eN_fix

        theta_meas = float(np.arctan2(eh_avg[0], -eh_avg[1]))

        self.board_corner = corner_meas
        self.e_h_axis     = eh_avg
        self.e_N_axis     = eN_fix
        self.board_theta  = theta_meas

        rospy.loginfo("=" * 62)
        rospy.loginfo("Calibration complete.")
        rospy.loginfo(f"  theta_est  (pass1, South) = "
                      f"{np.degrees(theta_est):+.3f} deg")
        rospy.loginfo(f"  theta_final(pass2)        = "
                      f"{np.degrees(theta_meas):+.3f} deg")
        rospy.loginfo(f"  corner (outer, near a1)   = "
                      f"({corner_meas[0]:+.4f}, {corner_meas[1]:+.4f}) m")
        rospy.loginfo(f"  e_h                       = "
                      f"({eh_avg[0]:+.5f}, {eh_avg[1]:+.5f})")
        rospy.loginfo(f"  e_N                       = "
                      f"({eN_fix[0]:+.5f}, {eN_fix[1]:+.5f})")
        rospy.loginfo(f"  fine-tune offset (preserved)= "
                      f"({self.square_offset_u*1000:+.2f}, "
                      f"{self.square_offset_v*1000:+.2f}) mm")
        rospy.loginfo("=" * 62)

        self.build_positions()

        # --- الرجوع لموقع الـREADY ---
        move_to_ready(arm)

        return True


    # ------------------------------------------------------------------
    # --- حفظ / تحميل ---
    # ------------------------------------------------------------------
    def save(self, path=CALIB_FILE):
        data = {
            'corner_x':         float(self.board_corner[0]),
            'corner_y':         float(self.board_corner[1]),
            'theta_rad':        float(self.board_theta),
            'e_h_x':            float(self.e_h_axis[0]),
            'e_h_y':            float(self.e_h_axis[1]),
            'e_N_x':            float(self.e_N_axis[0]),
            'e_N_y':            float(self.e_N_axis[1]),
            'square_size':      float(SQUARE_SIZE),
            'margin':           float(MARGIN),
            'board_outer_size': float(BOARD_OUTER_SIZE),
            # NOTE: square_offset_u/v مش بيتحفظوا في YAML.
            # الـsource of truth هو الـconstants في أعلى الملف.
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
        rospy.loginfo(f"Calibration saved -> {path}")
        rospy.loginfo(f"  current square_offset (in-memory) = "
                      f"({self.square_offset_u*1000:+.2f}, "
                      f"{self.square_offset_v*1000:+.2f}) mm "
                      f"[NOT persisted; edit constants to make permanent]")

    def load(self, path=CALIB_FILE):
        if not os.path.exists(path):
            rospy.logwarn(f"No calibration file at {path}")
            return False
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self.board_corner = np.array([data['corner_x'], data['corner_y']])
        self.board_theta  = float(data['theta_rad'])
        self.e_h_axis     = np.array([data['e_h_x'], data['e_h_y']])
        self.e_N_axis     = np.array([data['e_N_x'], data['e_N_y']])
        # NOTE: square_offset_u/v ما بيتقروش من YAML.
        # القيم بتفضل من __init__ (الـconstants) - دي الـsource of truth.
        self.build_positions()
        rospy.loginfo(f"Calibration loaded <- {path}")
        rospy.loginfo(f"  corner=({self.board_corner[0]:+.4f}, "
                      f"{self.board_corner[1]:+.4f}), "
                      f"theta={np.degrees(self.board_theta):+.3f} deg")
        rospy.loginfo(f"  square_offset (from constants) = "
                      f"({self.square_offset_u*1000:+.2f}, "
                      f"{self.square_offset_v*1000:+.2f}) mm")
        return True


    # ------------------------------------------------------------------
    # --- عرض / اختبار ---
    # ------------------------------------------------------------------
    def show(self):
        print("--- Current calibration ---")
        print(f"  corner (outer near a1): "
              f"({self.board_corner[0]:+.4f}, {self.board_corner[1]:+.4f}) m")
        print(f"  theta (board yaw)     : "
              f"{np.degrees(self.board_theta):+.3f} deg  "
              f"({self.board_theta:+.5f} rad)")
        print(f"  e_h (a -> h)          : "
              f"({self.e_h_axis[0]:+.5f}, {self.e_h_axis[1]:+.5f})")
        print(f"  e_N (row1 -> row8)    : "
              f"({self.e_N_axis[0]:+.5f}, {self.e_N_axis[1]:+.5f})")
        print(f"  square_offset (U,V)   : "
              f"({self.square_offset_u*1000:+.2f}, "
              f"{self.square_offset_v*1000:+.2f}) mm")
        print(f"  corner offsets (mm)   :")
        for cname in ('a1', 'h1', 'a8', 'h8'):
            du, dv = self.corner_offsets[cname]
            print(f"      {cname}: U={du:+.2f}  V={dv:+.2f}")
        print(f"  a1 center             : "
              f"({self.square_positions['a1'][0]:+.4f}, "
              f"{self.square_positions['a1'][1]:+.4f})")
        print(f"  h8 center             : "
              f"({self.square_positions['h8'][0]:+.4f}, "
              f"{self.square_positions['h8'][1]:+.4f})")
        print(f"  home                  : "
              f"({self.hx:+.4f}, {self.hy:+.4f}, {self.hz:+.4f})")

    def test(self, arm, safe_h=None):
        if safe_h is None:
            safe_h = SAFE_H
        for sq in ('a1', 'h1', 'h8', 'a8'):
            p = self.mirrored_squares[sq]
            rospy.loginfo(f"  test -> {sq} @ ({p[0]:+.4f}, {p[1]:+.4f})")
            move_to_pose(arm, p[0], p[1], safe_h,
                         V_SLOW, A_SLOW, yaw=self.board_theta)
            rospy.sleep(0.3)
        rospy.loginfo("  test -> home")
        move_to_pose(arm, self.hx, self.hy, self.hz,
                     V_SLOW, A_SLOW, yaw=self.board_theta)
