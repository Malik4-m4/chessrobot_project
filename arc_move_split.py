#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arc_move (Two-phase split version)
==================================
نسخة معدّلة من arc_move بتقسيم الحركة لمرحلتين:
  - Phase 1 (Arc): lift + bezier arc حتى فوق المربع المستهدف.
                   سرعة عالية (vf=1, af=1).
  - Phase 2 (Descent): نزول رأسي من h_travel إلى pick_h.
                       سرعة منخفضة وثابتة (vf=0.15, af=0.15).

هذا يضمن:
  ✅ القوس سريع (الجزء الأطول من الحركة).
  ✅ النزول بطيء وآمن في الصفوف البعيدة (لا يعتمد على Jacobian).
  ✅ الانتقال بين المرحلتين smooth (كل trajectory ينتهي ويبدأ بـv=0).
"""

import copy
import rospy
import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler

from board_calibration_p1_v3 import get_current_pose_in_ref


def arc_move(move_group,
             sx, sy, ex, ey,
             pick_h,
             h_travel,
             arc_extra=0.2,
             vf=1.0, af=1.0,                      # ⭐ للـarc (سريع)
             vf_descent=0.15, af_descent=0.15,    # ⭐ للنزول (بطيء وثابت)
             N_up=10, N_curve=40, N_down=10,
             eef_step=0.01,
             yaw=0.0):
    """
    Pick-and-place arc motion، النزول بسرعة منخفضة بغض النظر عن مكان المربع.

    Parameters
    ----------
    sx, sy        : موقع الالتقاط (XY).
    ex, ey        : موقع الإيداع (XY).
    pick_h        : ارتفاع الإمساك (نهاية النزول).
    h_travel      : ارتفاع الانتقال (قمة القوس).
    arc_extra     : زيادة الـapex فوق h_travel (control points للـBezier).
    vf, af        : سرعة وتسارع الـarc (عادة 1.0, 1.0).
    vf_descent    : سرعة النزول. 0.15 = 15% من سرعة المفصل القصوى.
    af_descent    : تسارع النزول.
    N_up/curve/down : عدد نقاط كل مرحلة.
    eef_step      : خطوة الـCartesian interpolation (متر).
    yaw           : زاوية الـyaw للقابض (rad).

    Returns
    -------
    bool : True إذا نُفذتا الخطتان بنجاح.
    """
    q = quaternion_from_euler(np.pi, 0.0, float(yaw))
    ori = Quaternion(*q)

    sx, sy, ex, ey = float(sx), float(sy), float(ex), float(ey)
    pick_h = float(pick_h)
    h_travel = float(h_travel)
    arc_extra = float(arc_extra)

    def lerp(a, b, t):
        return a + (b - a) * t

    def bezier(p0, p1, p2, p3, t):
        u = 1.0 - t
        return (u*u*u)*p0 + 3*(u*u)*t*p1 + 3*u*(t*t)*p2 + (t*t*t)*p3

    # =================================================================
    # PHASE 1: lift + arc (سريع)
    # =================================================================
    waypoints_arc = []

    # نقطة البداية الحالية
    cur = get_current_pose_in_ref(move_group)
    cur.orientation = ori
    waypoints_arc.append(copy.deepcopy(cur))

    # رفع رأسي من الـpick height إلى h_travel
    z0 = float(cur.position.z)
    for i in range(max(2, int(N_up))):
        t = i / float(N_up - 1)
        p = Pose()
        p.position.x = sx
        p.position.y = sy
        p.position.z = lerp(z0, h_travel, t)
        p.orientation = ori
        waypoints_arc.append(copy.deepcopy(p))

    # قوس Bezier: من فوق sx,sy إلى فوق ex,ey
    P0 = (sx, sy, h_travel)
    P1 = (sx, sy, h_travel + arc_extra)
    P2 = (ex, ey, h_travel + arc_extra)
    P3 = (ex, ey, h_travel)

    for i in range(max(2, int(N_curve))):
        t = i / float(N_curve - 1)
        p = Pose()
        p.position.x = float(bezier(P0[0], P1[0], P2[0], P3[0], t))
        p.position.y = float(bezier(P0[1], P1[1], P2[1], P3[1], t))
        p.position.z = float(bezier(P0[2], P1[2], P2[2], P3[2], t))
        p.orientation = ori
        if i == 0:
            continue
        waypoints_arc.append(copy.deepcopy(p))

    # نخطّط ونحرك الـarc
    move_group.set_start_state_to_current_state()
    plan_arc, frac_arc = move_group.compute_cartesian_path(
        waypoints_arc, float(eef_step), False)

    if frac_arc < 0.9:
        rospy.logwarn(f"[arc_move] arc fraction={frac_arc:.2f} (<0.9)")
        return False

    plan_arc = move_group.retime_trajectory(
        move_group.get_current_state(), plan_arc,
        velocity_scaling_factor=float(vf),
        acceleration_scaling_factor=float(af),
        algorithm="iterative_spline_parameterization")

    rospy.loginfo(f"[arc_move] executing arc phase "
                  f"(vf={vf:.2f}, af={af:.2f}, "
                  f"{len(plan_arc.joint_trajectory.points)} pts)")
    move_group.execute(plan_arc, wait=True)
    move_group.stop()

    # =================================================================
    # PHASE 2: descent فقط (بطيء وثابت)
    # =================================================================
    waypoints_desc = []

    # البداية: فوق المربع المستهدف على h_travel (نقطة الـcurrent بعد الـarc)
    start_desc = Pose()
    start_desc.position.x = ex
    start_desc.position.y = ey
    start_desc.position.z = h_travel
    start_desc.orientation = ori
    waypoints_desc.append(copy.deepcopy(start_desc))

    # نزول رأسي
    for i in range(1, max(2, int(N_down))):
        t = i / float(N_down - 1)
        p = Pose()
        p.position.x = ex
        p.position.y = ey
        p.position.z = lerp(h_travel, pick_h, t)
        p.orientation = ori
        waypoints_desc.append(copy.deepcopy(p))

    move_group.set_start_state_to_current_state()
    plan_desc, frac_desc = move_group.compute_cartesian_path(
        waypoints_desc, float(eef_step), False)

    if frac_desc < 0.9:
        rospy.logwarn(f"[arc_move] descent fraction={frac_desc:.2f} (<0.9)")
        return False

    plan_desc = move_group.retime_trajectory(
        move_group.get_current_state(), plan_desc,
        velocity_scaling_factor=float(vf_descent),
        acceleration_scaling_factor=float(af_descent),
        algorithm="iterative_time_parameterization")   # ⭐ IPTP لضمان final accel=0

    rospy.loginfo(f"[arc_move] executing descent phase "
                  f"(vf={vf_descent:.2f}, af={af_descent:.2f}, IPTP, "
                  f"{len(plan_desc.joint_trajectory.points)} pts)")
    move_group.execute(plan_desc, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    return True
