#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arc_move (Three-phase split + concatenation)
=============================================
نسخة معدّلة من arc_move بتقسيم الحركة لـ3 مراحل:
  - Phase 1 (Lift):    رفع رأسي من pick_h إلى h_travel.
                       بطيء (vf_lift=0.15, af_lift=0.15) + IPTP.
  - Phase 2 (Arc):     bezier arc من فوق sx,sy إلى فوق ex,ey.
                       سريع (vf=1.0, af=1.0) + ISP (smooth).
  - Phase 3 (Descent): نزول رأسي من h_travel إلى pick_h.
                       بطيء (vf_descent=0.15, af_descent=0.15) + IPTP.

الـ3 plans بتتدمج في trajectory واحدة وبتنفّذ كـexecute واحد فقط.
ده بيحل:
  ✅ مشكلة الـJacobian في الطلوع (مش بس النزول).
  ✅ Discontinuity بين الـphases (واحدة execute = مفيش action client overhead).
  ✅ خطأ "Final acceleration out of bounds" (IPTP في endpoints).
  ✅ Smoothness في الـarc (ISP).
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
             vf_lift=0.15, af_lift=0.15,          # ⭐ للطلوع (بطيء)
             vf_descent=0.15, af_descent=0.15,    # ⭐ للنزول (بطيء)
             N_up=10, N_curve=40, N_down=10,
             eef_step=0.01,
             yaw=0.0):
    """
    Pick-and-place arc motion على 3 مراحل مدموجة في trajectory واحدة.

    Parameters
    ----------
    sx, sy        : موقع الالتقاط (XY).
    ex, ey        : موقع الإيداع (XY).
    pick_h        : ارتفاع الإمساك.
    h_travel      : ارتفاع الانتقال.
    arc_extra     : زيادة الـapex فوق h_travel.
    vf, af        : سرعة وتسارع الـarc (سريع).
    vf_lift, af_lift       : سرعة وتسارع الطلوع (بطيء).
    vf_descent, af_descent : سرعة وتسارع النزول (بطيء).
    N_up/curve/down : عدد نقاط كل مرحلة.
    eef_step      : خطوة الـCartesian interpolation (متر).
    yaw           : زاوية الـyaw للقابض.

    Returns
    -------
    bool : True إذا نُفّذت كل الـ3 مراحل بنجاح.
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

    # ------------------------------------------------------------------
    # Helper: يبني RobotState عند آخر نقطة في الـplan
    # ------------------------------------------------------------------
    def _state_at_end_of_plan(plan):
        state = copy.deepcopy(move_group.get_current_state())
        last_pt = plan.joint_trajectory.points[-1]
        joint_names = plan.joint_trajectory.joint_names

        name_to_idx = {n: i for i, n in enumerate(state.joint_state.name)}
        positions = list(state.joint_state.position)
        for jname, jpos in zip(joint_names, last_pt.positions):
            if jname in name_to_idx:
                positions[name_to_idx[jname]] = jpos
        state.joint_state.position = tuple(positions)
        return state

    # ------------------------------------------------------------------
    # Helper: يدمج عدة RobotTrajectory في trajectory واحدة
    # ------------------------------------------------------------------
    def _concatenate(plans):
        combined = copy.deepcopy(plans[0])
        for plan in plans[1:]:
            t_offset = combined.joint_trajectory.points[-1].time_from_start
            for pt in plan.joint_trajectory.points[1:]:  # نتخطى أول نقطة (مكررة)
                new_pt = copy.deepcopy(pt)
                new_pt.time_from_start = pt.time_from_start + t_offset
                combined.joint_trajectory.points.append(new_pt)
        return combined

    # ==================================================================
    # PHASE 1: LIFT (slow, IPTP)
    # ==================================================================
    cur = get_current_pose_in_ref(move_group)
    cur.orientation = ori
    z0 = float(cur.position.z)

    waypoints_lift = [copy.deepcopy(cur)]
    for i in range(1, max(2, int(N_up))):
        t = i / float(N_up - 1)
        p = Pose()
        p.position.x = sx
        p.position.y = sy
        p.position.z = lerp(z0, h_travel, t)
        p.orientation = ori
        waypoints_lift.append(copy.deepcopy(p))

    move_group.set_start_state_to_current_state()
    plan_lift, frac_lift = move_group.compute_cartesian_path(
        waypoints_lift, float(eef_step), False)
    if frac_lift < 0.9:
        rospy.logwarn(f"[arc_move] lift fraction={frac_lift:.2f} (<0.9)")
        return False
    plan_lift = move_group.retime_trajectory(
        move_group.get_current_state(), plan_lift,
        velocity_scaling_factor=float(vf_lift),
        acceleration_scaling_factor=float(af_lift),
        algorithm="iterative_time_parameterization")

    # ==================================================================
    # PHASE 2: ARC (fast, ISP)
    # ==================================================================
    state_after_lift = _state_at_end_of_plan(plan_lift)

    P0 = (sx, sy, h_travel)
    P1 = (sx, sy, h_travel + arc_extra)
    P2 = (ex, ey, h_travel + arc_extra)
    P3 = (ex, ey, h_travel)

    p_arc_start = Pose()
    p_arc_start.position.x = sx
    p_arc_start.position.y = sy
    p_arc_start.position.z = h_travel
    p_arc_start.orientation = ori
    waypoints_arc = [copy.deepcopy(p_arc_start)]

    for i in range(1, max(2, int(N_curve))):
        t = i / float(N_curve - 1)
        p = Pose()
        p.position.x = float(bezier(P0[0], P1[0], P2[0], P3[0], t))
        p.position.y = float(bezier(P0[1], P1[1], P2[1], P3[1], t))
        p.position.z = float(bezier(P0[2], P1[2], P2[2], P3[2], t))
        p.orientation = ori
        waypoints_arc.append(copy.deepcopy(p))

    move_group.set_start_state(state_after_lift)
    plan_arc, frac_arc = move_group.compute_cartesian_path(
        waypoints_arc, float(eef_step), False)
    if frac_arc < 0.9:
        rospy.logwarn(f"[arc_move] arc fraction={frac_arc:.2f} (<0.9)")
        move_group.set_start_state_to_current_state()
        return False
    plan_arc = move_group.retime_trajectory(
        state_after_lift, plan_arc,
        velocity_scaling_factor=float(vf),
        acceleration_scaling_factor=float(af),
        algorithm="iterative_spline_parameterization")

    # ==================================================================
    # PHASE 3: DESCENT (slow, IPTP)
    # ==================================================================
    state_after_arc = _state_at_end_of_plan(plan_arc)

    p_desc_start = Pose()
    p_desc_start.position.x = ex
    p_desc_start.position.y = ey
    p_desc_start.position.z = h_travel
    p_desc_start.orientation = ori
    waypoints_desc = [copy.deepcopy(p_desc_start)]

    for i in range(1, max(2, int(N_down))):
        t = i / float(N_down - 1)
        p = Pose()
        p.position.x = ex
        p.position.y = ey
        p.position.z = lerp(h_travel, pick_h, t)
        p.orientation = ori
        waypoints_desc.append(copy.deepcopy(p))

    move_group.set_start_state(state_after_arc)
    plan_desc, frac_desc = move_group.compute_cartesian_path(
        waypoints_desc, float(eef_step), False)
    if frac_desc < 0.9:
        rospy.logwarn(f"[arc_move] descent fraction={frac_desc:.2f} (<0.9)")
        move_group.set_start_state_to_current_state()
        return False
    plan_desc = move_group.retime_trajectory(
        state_after_arc, plan_desc,
        velocity_scaling_factor=float(vf_descent),
        acceleration_scaling_factor=float(af_descent),
        algorithm="iterative_time_parameterization")

    # نرجّع الـstart state للحالة الحالية
    move_group.set_start_state_to_current_state()

    # ==================================================================
    # CONCATENATE & EXECUTE (واحدة بس)
    # ==================================================================
    combined = _concatenate([plan_lift, plan_arc, plan_desc])

    rospy.loginfo(f"[arc_move] executing combined trajectory: "
                  f"lift({len(plan_lift.joint_trajectory.points)}) + "
                  f"arc({len(plan_arc.joint_trajectory.points)}) + "
                  f"descent({len(plan_desc.joint_trajectory.points)}) "
                  f"= {len(combined.joint_trajectory.points)} pts")
    move_group.execute(combined, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    return True
