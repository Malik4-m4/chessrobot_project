#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_hp1.py
===============
تحكم يدوي بذراع Franka Panda لمشروع روبوت الشطرنج.

هذا الملف بيركز فقط على:
- تهيئة MoveIt + TF + gripper client
- حلقة التحكم اليدوي (input loop)
- لون اللاعب (player color) واختيار خريطة المربعات المناسبة
- تنفيذ حركة الذراع لمربعات اللوحة والـgraveyard/promotion

أي كود خاص بالمعايرة أو بناء المواقع أو الـprobing موجود في
`board_calibration.py`.

لون اللاعب (color):
- black (الافتراضي): المستخدم بيكتب "a1" من منظور الأسود ⇒
  الروبوت يروح للمربع الفيزيائي h8 (mirrored map).
- white: المستخدم بيكتب "a1" من منظور الأبيض ⇒
  الروبوت يروح للمربع الفيزيائي a1 مباشرة (direct map).

اللون قابل للتغيير:
- من CLI: "color" لعرضه، "color white" / "color black" للتغيير.
- من ROS topic: نشر std_msgs/String على /chess/player_color.

ملاحظة: مواقع الـpromotion/graveyard/home مواقع فيزيائية ثابتة بالنسبة
للروبوت ومش بتتأثر بلون اللاعب.

ملاحظة: ⭐ وضعية الـ"ready" بقت معرّفة بقيم joint مباشرة في
board_calibration_p1_v3 (READY_JOINT_VALUES) عبر دالة move_to_ready،
بدل الاعتماد على الـSRDF، عشان تجنب اصطدام الروبوتين.
"""

import sys
import threading

import rospy
import moveit_commander
import actionlib
import numpy as np

from std_msgs.msg import String
from franka_gripper.msg import MoveAction, MoveGoal

from board_calibration_p1_v3 import (
    BoardCalibration,
    ForceMonitor,
    init_tf,
    move_to_pose,
    move_to_ready,
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
# --- حلقة التحكم اليدوي ---
# =====================================================================
def manual_control():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('manual_robot_control_globals', anonymous=True)



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
    print("          color | color white | color black | exit")

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
        # ⭐ ready دلوقتي بيستخدم READY_JOINT_VALUES (joint-space)
        # من board_calibration_p1_v3 بدل الـSRDF "ready" القديمة.
        if cmd == 'ready':
            move_to_ready(arm)
            continue
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
