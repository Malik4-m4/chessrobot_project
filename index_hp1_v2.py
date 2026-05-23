#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_hp1_v2.py
===============
تحكم يدوي بذراع Franka Panda لمشروع روبوت الشطرنج.

نسخة v2: نفس v4 + أمر `tune` color-aware لضبط CORNER_OFFSETS
بدون مجهود ذهني للترجمة بين اللون والـphysical position.

هذا الملف بيركز فقط على:
- تهيئة MoveIt + TF + gripper client
- حلقة التحكم اليدوي (input loop)
- لون اللاعب (player color) واختيار خريطة المربعات المناسبة
- تنفيذ حركة الذراع لمربعات اللوحة والـgraveyard/promotion
- ⭐ ضبط الـCORNER_OFFSETS بالعين عبر أمر `tune` (يفهم اللون تلقائياً)

أي كود خاص بالمعايرة أو بناء المواقع أو الـprobing موجود في
`board_calibration_p1_v1.py`.

لون اللاعب (color):
- black (الافتراضي): "a1" → physical h8 (mirrored).
- white: "a1" → physical a1 (direct).

الـtune workflow:
- المستخدم يكتب أسماء chess (a1, h1, a8, h8) من منظور لونه الحالي.
- الـtool بيترجم تلقائياً للـphysical corner ويحفظ هناك.
- يعني تقدر تضبط في أي لون والنتيجة شغّالة في الاتنين.

أوامر tune:
  tune                           -- يعرض corner offsets الحالية
  tune <corner> <du_mm> <dv_mm>  -- يضبط (corner من منظور لونك)
  tune reset                     -- يصفّر كل الـ4 corners
  tune save                      -- يطبع القيم بصيغة constants للنسخ
"""

import sys
import threading

import rospy
import moveit_commander
import actionlib
import numpy as np

from std_msgs.msg import String
from franka_gripper.msg import MoveAction, MoveGoal

from board_calibration_p1_v1 import (
    BoardCalibration,
    ForceMonitor,
    init_tf,
    move_to_pose,
    REFERENCE_FRAME,
    V_SLOW, A_SLOW, SAFE_H,
    OPEN_WIDTH, CLOSE_WIDTH, GRIPPER_SPEED,
)


# =====================================================================
# --- لون اللاعب (player color) ---
# =====================================================================
PLAYER_COLOR_TOPIC = "/chess/player_color"
DEFAULT_PLAYER_COLOR = "black"   # يحافظ على سلوك v3 (mirrored map)
VALID_COLORS = ("white", "black")

_color_lock = threading.Lock()
_player_color = DEFAULT_PLAYER_COLOR


def set_player_color(color):
    """يضبط لون اللاعب. بيرجع True لو اللون صالح."""
    global _player_color
    color = (color or "").strip().lower()
    if color not in VALID_COLORS:
        rospy.logwarn(f"[color] Ignoring invalid color: {color!r} "
                      f"(valid: {VALID_COLORS})")
        return False
    with _color_lock:
        changed = (color != _player_color)
        _player_color = color
    if changed:
        rospy.loginfo(f"[color] player_color = {color}")
    return True


def get_player_color():
    with _color_lock:
        return _player_color


def get_square_map(board):
    """
    يرجع الخريطة المناسبة للون الحالي:
      - white ⇒ square_positions (direct)
      - black ⇒ mirrored_squares (180° mirror)
    """
    if get_player_color() == "white":
        return board.square_positions
    return board.mirrored_squares


def _color_topic_cb(msg):
    set_player_color(msg.data)


# =====================================================================
# --- ترجمة اللون → physical (للـtune command) ---
# =====================================================================
_BLACK_MIRROR_MAP = {
    'a1': 'h8',
    'h1': 'a8',
    'a8': 'h1',
    'h8': 'a1',
}


def chess_corner_to_physical(corner_user, color):
    """
    يحوّل اسم corner من منظور المستخدم إلى الـphysical corner المقابل.
    
    - في الأبيض: نفس الاسم (direct mapping).
    - في الأسود: 180° mirror.
    
    corner_user in {'a1', 'h1', 'a8', 'h8'}
    """
    if color == "white":
        return corner_user
    return _BLACK_MIRROR_MAP[corner_user]


# =====================================================================
# --- تحكم القابض ---
# =====================================================================
def gripper_control(move_client, action_type):
    goal = MoveGoal()
    if action_type == "open":
        goal.width = float(OPEN_WIDTH)
        rospy.loginfo(f"Opening to: {OPEN_WIDTH}m")
    else:
        goal.width = float(CLOSE_WIDTH)
        rospy.loginfo(f"Closing to: {CLOSE_WIDTH}m")
    goal.speed = float(GRIPPER_SPEED)
    move_client.send_goal(goal)
    move_client.wait_for_result()


# =====================================================================
# --- handler لأمر tune ---
# =====================================================================
def handle_tune_command(board, parts):
    """
    يعالج كل صيغ أمر tune:
      tune                          -- show
      tune <corner> <du_mm> <dv_mm> -- set (color-aware)
      tune reset                    -- zero all
      tune save                     -- print constants block
    """
    valid_corners = ('a1', 'h1', 'a8', 'h8')

    # tune (بدون args) → show
    if len(parts) == 1:
        color = get_player_color()
        print(f"--- Corner offsets (physical, mm) ---")
        for cname in valid_corners:
            du, dv = board.corner_offsets[cname]
            print(f"  {cname}: U={du:+6.2f}  V={dv:+6.2f}")
        print(f"--- From your view ({color}) ---")
        for user_name in valid_corners:
            phys = chess_corner_to_physical(user_name, color)
            du, dv = board.corner_offsets[phys]
            print(f"  {user_name} (= physical {phys}): "
                  f"U={du:+6.2f}  V={dv:+6.2f}")
        return

    # tune reset
    if len(parts) == 2 and parts[1] == 'reset':
        board.reset_corner_offsets()
        return

    # tune save
    if len(parts) == 2 and parts[1] == 'save':
        board.print_corner_offsets_constants()
        return

    # tune <corner> <du_mm> <dv_mm>
    if len(parts) == 4:
        corner_user = parts[1]
        if corner_user not in valid_corners:
            print(f"Corner must be one of: {valid_corners}  (got {corner_user!r})")
            return
        try:
            du = float(parts[2])
            dv = float(parts[3])
        except ValueError:
            print(f"Invalid numbers: {parts[2]!r} {parts[3]!r}")
            return

        color = get_player_color()
        physical = chess_corner_to_physical(corner_user, color)
        board.set_corner_offset(physical, du, dv)
        print(f"  [tune] {corner_user} (your {color} view) "
              f"→ physical {physical}: U={du:+.2f}mm V={dv:+.2f}mm")
        return

    # غير معروف
    print("Usage:")
    print("  tune                              -- show current offsets")
    print("  tune <corner> <du_mm> <dv_mm>     -- set (e.g. tune a1 2 -1)")
    print("  tune reset                        -- zero all 4 corners")
    print("  tune save                         -- print constants block")


# =====================================================================
# --- حلقة التحكم اليدوي ---
# =====================================================================
def manual_control():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('manual_robot_control_globals', anonymous=True)

    # --- TF listener ---
    init_tf()

    # --- MoveIt commander ---
    arm = moveit_commander.MoveGroupCommander(
        "panda1_manipulator",
        robot_description="/panda1/robot_description",
        ns="/panda1")

    rospy.loginfo(f"Setting pose reference frame to: {REFERENCE_FRAME}")
    arm.set_pose_reference_frame(REFERENCE_FRAME)
    rospy.loginfo(f"  planning frame (base) : {arm.get_planning_frame()}")
    rospy.loginfo(f"  pose reference frame  : {arm.get_pose_reference_frame()}")

    # --- gripper action client ---
    gripper_client = actionlib.SimpleActionClient(
        '/panda1/franka_gripper/move', MoveAction)
    gripper_client.wait_for_server()

    # --- force monitor ---
    force_monitor = ForceMonitor()
    try:
        force_monitor.wait_for_data(timeout=5.0)
    except RuntimeError as e:
        rospy.logwarn(f"ForceMonitor: {e}")

    # --- board state + محاولة تحميل آخر معايرة ---
    board = BoardCalibration()
    board.load()

    # --- subscriber للون اللاعب ---
    rospy.Subscriber(PLAYER_COLOR_TOPIC, String,
                     _color_topic_cb, queue_size=1)
    rospy.loginfo(f"[color] subscribed to {PLAYER_COLOR_TOPIC} "
                  f"(default = {get_player_color()})")

    print(f"--- Manual Control (Open: {OPEN_WIDTH}, Close: {CLOSE_WIDTH}) ---")
    print(f"Home Coords : X={board.hx:.3f}, Y={board.hy:.3f}, Z={board.hz:.3f}")
    print(f"Board theta : {np.degrees(board.board_theta):+.3f} deg")
    print(f"Player color: {get_player_color()}  "
          f"(topic: {PLAYER_COLOR_TOPIC})")
    print("Commands: a1..h8 | pq pr pb pn | x0..x23 | open | close | ready")
    print("          home | calibrate | save | load | show | test")
    print("          color | color white | color black")
    print("          tune | tune <corner> <du_mm> <dv_mm> | tune reset | tune save")
    print("          exit")

    while not rospy.is_shutdown():
        cmd = input("Enter Target/Action: ").strip().lower()
        if cmd == 'exit':
            break
        if cmd == '':
            continue

        # --- أوامر القابض ---
        if cmd == 'open':
            gripper_control(gripper_client, "open"); continue
        if cmd == 'close':
            gripper_control(gripper_client, "close"); continue

        # --- ready / home ---
        if cmd == 'ready':
            arm.set_named_target('ready'); arm.go(wait=True); continue
        if cmd == 'home':
            move_to_pose(arm, board.hx, board.hy, board.hz,
                         V_SLOW, A_SLOW, yaw=board.board_theta)
            continue

        # --- لون اللاعب ---
        if cmd.startswith('color'):
            parts = cmd.split()
            if len(parts) == 1:
                color = get_player_color()
                map_kind = 'direct' if color == 'white' else 'mirrored'
                print(f"Player color: {color}  (square map: {map_kind})")
            elif len(parts) == 2 and parts[1] in VALID_COLORS:
                set_player_color(parts[1])
            else:
                print("Usage: color | color white | color black")
            continue

        # --- ⭐ tune (color-aware corner offsets) ---
        if cmd.startswith('tune'):
            parts = cmd.split()
            handle_tune_command(board, parts)
            continue

        # --- المعايرة + حفظ/تحميل/عرض/اختبار ---
        if cmd in ('calibrate', 'calib'):
            ok = board.calibrate(arm, force_monitor, gripper_client)
            if ok:
                ans = input("Save calibration to YAML? [y/N]: ").strip().lower()
                if ans == 'y':
                    board.save()
            continue

        if cmd == 'save':
            board.save(); continue
        if cmd == 'load':
            board.load(); continue
        if cmd == 'show':
            board.show()
            print(f"  player color         : {get_player_color()}")
            continue
        if cmd == 'test':
            board.test(arm); continue

        # --- مربع على اللوحة / promotion / graveyard ---
        # ملاحظة: square lookup يعتمد على اللون الحالي.
        # promotion + graveyard مواقع فيزيائية ثابتة.
        target_pos = None
        sq_map = get_square_map(board)
        if cmd in sq_map:
            target_pos = sq_map[cmd]
        elif (cmd.startswith('p') and len(cmd) == 2
              and cmd[1] in board.promotion_positions):
            target_pos = board.promotion_positions[cmd[1]]
        elif cmd.startswith('x'):
            try:
                idx = int(cmd[1:])
                if 0 <= idx < len(board.graveyard_positions):
                    target_pos = board.graveyard_positions[idx]
            except ValueError:
                pass

        if target_pos is not None:
            move_to_pose(arm, target_pos[0], target_pos[1], SAFE_H,
                         V_SLOW, A_SLOW, yaw=board.board_theta)
        else:
            print("Invalid Input!")


if __name__ == '__main__':
    try:
        manual_control()
    except rospy.ROSInterruptException:
        pass
