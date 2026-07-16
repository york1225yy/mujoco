"""
【中文说明】
NutAssembly 螺母装配演示（脚本化控制）

场景说明：
  桌面上有两颗螺母和两根固定插销：
    SquareNut（方形螺母，黄铜色）→ peg1（方形插销，x≈0.23, y≈+0.10）
    RoundNut （圆形螺母，钢色）  → peg2（圆形插销，x≈0.23, y≈−0.10）

  任务目标：依次抓起每颗螺母，将其孔洞对准对应插销顶端，下压套入，
  直到螺母落至桌面层（判定为装配成功）。

装配流程（每颗螺母分 8 个阶段）：
  1. PRE_GRASP     → 末端移到螺母正上方
  2. DESCEND       → 下降到抓取高度
  3. GRASP         → 关闭夹爪夹住螺母
  4. LIFT          → 垂直提起到搬运高度
  5. ALIGN_PEG     → 平移到插销正上方（XY 对准）
  6. DESCEND_PLACE → 孔洞对准插销，从上向下套入
  7. RELEASE       → 张开夹爪，螺母落在插销上
  8. RETRACT       → 抬起末端，准备抓取下一颗

【功能参数说明】
  --robot      机械臂型号（Kinova3/Panda/UR5e/IIWA/Sawyer/xArm7 等）
  --gripper    独立选择夹爪型号（default=机器人默认）
  --order      装配顺序：square_first（默认）或 round_first
  --plan       启用 Cartesian RRT 路径规划（自动避障）
  --nreset     重复次数
  --speed      速度缩放
  --save-video 录制视频保存路径（mp4）
  --camera     录制相机名称
  --show-camera 实时显示相机画面（需要 opencv-python）

【运行示例】
  # 默认模式（Kinova3，方形螺母先）
  python -m robosuite.demos.demo_nut_assembly

  # 指定 Panda + 规划模式 + 录制
  python -m robosuite.demos.demo_nut_assembly \\
      --robot Panda --plan --save-video nut_assembly.mp4

  # 圆形螺母先 + 3 次重复
  python -m robosuite.demos.demo_nut_assembly \\
      --order round_first --nreset 3
"""

import argparse
import os
import time

import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ─────────────────────────────────────────────────────────────
# 超参数
# ─────────────────────────────────────────────────────────────
MAX_STEP          = 0.05    # 每步末端最大位移（米）
GRASP_OFFSET_Z    = 0.15    # 预抓取 / 搬运时高出目标的距离（m）
GRASP_Z_OFFSET    = -0.005  # 抓取时末端轻微嵌入螺母（m）
REACH_THRESH      = 0.018   # 到达判定阈值（m）
GRASP_STEPS       = 100     # 关闭夹爪持续步数
RELEASE_STEPS     = 100     # 张开夹爪持续步数
RETRACT_STEPS     = 80      # 收尾抬臂步数
MAX_FR            = 30      # 最大帧率（fps）
VIDEO_W           = 640     # 视频宽度
VIDEO_H           = 480     # 视频高度

# 套入插销时螺母中心的目标高度 = table_z + 此偏移
# 螺母半高 ~0.010 m，下降到桌面以上 0.020 m 即可触发 on_peg 判定
PEG_PLACE_Z_ABOVE_TABLE = 0.020


# ─────────────────────────────────────────────────────────────
# Cartesian RRT 路径规划器（与 demo_grasp_scripted.py 相同）
# ─────────────────────────────────────────────────────────────
class CartesianRRT:
    """
    三维笛卡尔空间 RRT（Rapidly-exploring Random Tree）路径规划器。
    规划成功后对路径进行贪心平滑，删除不必要的中间路径点。
    如果超过最大迭代次数，退化为直线路径并打印警告。
    """

    def __init__(self, table_z, ws_xy_half=0.55, z_max_above=0.65,
                 step_size=0.07, max_iter=3000, goal_bias=0.20,
                 safety_margin=0.04):
        self.step_size = step_size
        self.max_iter  = max_iter
        self.goal_bias = goal_bias
        self.lo = np.array([-ws_xy_half, -ws_xy_half, table_z + safety_margin])
        self.hi = np.array([ ws_xy_half,  ws_xy_half, table_z + z_max_above])

    def _valid(self, pos):
        return bool(np.all(pos >= self.lo) and np.all(pos <= self.hi))

    def _segment_valid(self, p1, p2, n=8):
        for i in range(n + 1):
            t = i / n
            if not self._valid(p1 + t * (p2 - p1)):
                return False
        return True

    def _steer(self, from_pos, to_pos):
        d    = to_pos - from_pos
        dist = np.linalg.norm(d)
        return to_pos.copy() if dist <= self.step_size else from_pos + (d / dist) * self.step_size

    def plan(self, start, goal):
        start = np.array(start, dtype=float)
        goal  = np.array(goal,  dtype=float)
        start[2] = np.clip(start[2], self.lo[2] + 0.005, self.hi[2])
        goal[2]  = np.clip(goal[2],  self.lo[2] + 0.005, self.hi[2])

        if self._segment_valid(start, goal, n=20):
            print(f"    [RRT] 直线路径可行，跳过迭代")
            return [start.copy(), goal.copy()]

        nodes  = [start.copy()]
        parent = [-1]

        for it in range(self.max_iter):
            sample = goal.copy() if np.random.rand() < self.goal_bias \
                     else np.random.uniform(self.lo, self.hi)
            dists   = np.array([np.linalg.norm(n - sample) for n in nodes])
            nn_idx  = int(np.argmin(dists))
            new_pos = self._steer(nodes[nn_idx], sample)

            if not self._valid(new_pos):
                continue
            if not self._segment_valid(nodes[nn_idx], new_pos):
                continue

            nodes.append(new_pos)
            parent.append(nn_idx)

            if np.linalg.norm(new_pos - goal) <= self.step_size:
                if self._segment_valid(new_pos, goal):
                    nodes.append(goal.copy())
                    parent.append(len(nodes) - 2)
                    path, idx = [], len(nodes) - 1
                    while idx >= 0:
                        path.append(nodes[idx])
                        idx = parent[idx]
                    path.reverse()
                    path = self._smooth(path)
                    print(f"    [RRT] 规划成功：{it + 1} 次迭代，平滑后 {len(path)} 个路径点")
                    return path

        print(f"    [RRT] 警告：{self.max_iter} 次迭代未找到路径，退化为直线路径")
        return [start.copy(), goal.copy()]

    def _smooth(self, path):
        if len(path) <= 2:
            return path
        smoothed, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._segment_valid(path[i], path[j], n=15):
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed


# ─────────────────────────────────────────────────────────────
# 视频帧捕获
# ─────────────────────────────────────────────────────────────
def _capture_frame(obs, camera, video_writer):
    """从 obs 字典提取相机图像写入视频。"""
    img_key = f"{camera}_image"
    if img_key in obs:
        video_writer.append_data(obs[img_key])


# ─────────────────────────────────────────────────────────────
# 相机实时可视化
# ─────────────────────────────────────────────────────────────
def show_camera_frame(obs, camera_name="frontview"):
    """使用 OpenCV 实时显示相机画面。"""
    import cv2
    img_key = f"{camera_name}_image"
    if img_key not in obs:
        return
    img = obs[img_key].copy()
    if macros.IMAGE_CONVENTION == "opencv":
        img = img[::-1]
    else:
        img = img[::-1, :, ::-1]
    cv2.imshow("相机视图", img)
    cv2.waitKey(1)


# ─────────────────────────────────────────────────────────────
# 核心控制函数
# ─────────────────────────────────────────────────────────────
def move_to(env, obs, target_pos, max_steps=600, gripper_cmd=-1.0,
            verbose=True, video_writer=None, camera=None, show_camera_name=None):
    """
    比例控制：将末端执行器移动到 target_pos。

    每步：
        error  = target_pos − obs["robot0_eef_pos"]
        delta  = clip(error, −MAX_STEP, MAX_STEP)
        action = [delta_x, delta_y, delta_z, 0, 0, 0, gripper_cmd, ...]

    返回：(obs, reached)
    """
    for step in range(max_steps):
        start = time.time()
        eef   = obs["robot0_eef_pos"]
        error = np.array(target_pos) - eef
        dist  = np.linalg.norm(error)

        if verbose and step % 20 == 0:
            print(f"  d={dist:.4f}m | eef={eef.round(3)} → tgt={np.array(target_pos).round(3)}")

        if dist < REACH_THRESH:
            if verbose:
                print(f"  ✓ 到达（{step+1}步，残差={dist:.4f}m）")
            return obs, True

        action = np.zeros(env.action_dim)
        action[:3]  = np.clip(error, -MAX_STEP, MAX_STEP)
        action[3:6] = 0.0
        action[6:]  = gripper_cmd

        obs, _, _, _ = env.step(action)
        env.render()
        if video_writer is not None:
            _capture_frame(obs, camera, video_writer)
        if show_camera_name is not None:
            show_camera_frame(obs, show_camera_name)

        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    if verbose:
        print(f"  ✗ 超时（{max_steps}步），残差={dist:.4f}m")
    return obs, False


def hold_gripper(env, obs, gripper_cmd, steps, stage_name="",
                 video_writer=None, camera=None, show_camera_name=None):
    """末端静止，只执行夹爪开合（+1=关闭抓取，-1=张开释放）。"""
    if stage_name:
        print(f"\n{stage_name}")
    for _ in range(steps):
        start = time.time()
        action = np.zeros(env.action_dim)
        action[6:] = gripper_cmd
        obs, _, _, _ = env.step(action)
        env.render()
        if video_writer is not None:
            _capture_frame(obs, camera, video_writer)
        if show_camera_name is not None:
            show_camera_frame(obs, show_camera_name)
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
    return obs


def execute_rrt_path(env, obs, waypoints, gripper_cmd, verbose=False,
                     video_writer=None, camera=None, show_camera_name=None):
    """按 RRT 规划的路径点列表逐段移动末端执行器。"""
    kw = dict(video_writer=video_writer, camera=camera, show_camera_name=show_camera_name)
    n = len(waypoints)
    print(f"    执行规划路径：共 {n} 个路径点")
    for i, wp in enumerate(waypoints[1:], 1):
        if verbose:
            print(f"    路径点 {i}/{n-1}: {wp.round(3)}")
        obs, _ = move_to(env, obs, wp,
                         max_steps=600,
                         gripper_cmd=gripper_cmd,
                         verbose=False,
                         **kw)
    return obs


# ─────────────────────────────────────────────────────────────
# 单颗螺母装配
# ─────────────────────────────────────────────────────────────
def assemble_nut(env, obs, nut_obs_key, peg_pos, table_z,
                 nut_label, handle_site_id=None, planner=None,
                 video_writer=None, camera=None, show_camera_name=None):
    """
    抓取一颗螺母并将其套入对应插销（8 阶段流程）。

    参数：
        obs            : 当前环境观测字典
        nut_obs_key    : 螺母在 obs 中的位置键，如 "SquareNut_pos"
        peg_pos        : 插销在世界坐标系中的位置 (3,)
        table_z        : 桌面 Z 坐标（用于计算套入高度）
        nut_label      : 打印标签，用于区分两颗螺母
        handle_site_id : 螺母 handle_site 在仿真中的 site id（由
                         env.object_site_ids[i] 取得）。
                         螺母是空心环，夹爪若移到质心（孔中心）会空夹，
                         因此需偏移到 handle_site 所在的实体边缘处抓取。
                         搬运时对插销对准坐标施加反向补偿，确保孔对准插销。
        planner        : CartesianRRT 实例（None 表示不使用规划）

    返回：
        (obs, success)  success=True 表示螺母已成功套入插销
    """
    kw = dict(video_writer=video_writer, camera=camera, show_camera_name=show_camera_name)
    nut_pos     = obs[nut_obs_key].copy()
    place_z     = table_z + PEG_PLACE_Z_ABOVE_TABLE
    transport_z = nut_pos[2] + GRASP_OFFSET_Z

    # ── 计算夹取点 XY 及插销对准补偿 ──────────────────────
    # 螺母 body 中心 = 孔中心，夹爪若对准质心会落入空洞夹不住螺母。
    # handle_site 定义在螺母实体边缘（XML 中偏移约 0.054~0.06 m），
    # 夹爪对准 handle 才能夹到实体。
    # 抓取后夹爪在 handle 处，孔中心 = handle - xy_offset；
    # 为使孔对准插销 XY，夹爪目标 XY = peg_xy + xy_offset。
    if handle_site_id is not None:
        handle_pos   = env.sim.data.site_xpos[handle_site_id].copy()
        grasp_xy     = handle_pos[:2]
        xy_offset    = handle_pos[:2] - nut_pos[:2]   # handle 相对孔中心的偏移
    else:
        grasp_xy  = nut_pos[:2]
        xy_offset = np.zeros(2)

    # 搬运/下压时夹爪目标 XY（补偿后让孔对准插销）
    peg_align_xy = peg_pos[:2] + xy_offset

    print(f"\n  螺母质心位置 : {nut_pos.round(3)}")
    print(f"  夹取点 XY   : {grasp_xy.round(4)}  偏移: {xy_offset.round(4)}")
    print(f"  插销位置     : {peg_pos.round(3)}")
    print(f"  插销对准 XY  : {peg_align_xy.round(4)}（= 插销 XY + 偏移）")
    print(f"  套入目标 Z   : {place_z:.3f}m  桌面 Z: {table_z:.3f}m")
    print(f"  运动模式     : {'RRT 路径规划' if planner else '固定 8 阶段'}")

    # ── 阶段1：移到夹取点正上方（PRE_GRASP）──────────────
    # XY 对准 handle_site（螺母实体边缘），Z 悬停在螺母上方
    pre_grasp = np.array([grasp_xy[0], grasp_xy[1], nut_pos[2] + GRASP_OFFSET_Z])

    if planner:
        print(f"\n=== [{nut_label}] 阶段1：[RRT] 规划到预抓取位置（handle 边缘上方）===")
        path = planner.plan(obs["robot0_eef_pos"].copy(), pre_grasp)
        obs = execute_rrt_path(env, obs, path, gripper_cmd=-1.0, **kw)
    else:
        print(f"\n=== [{nut_label}] 阶段1：PRE_GRASP — 移到螺母边缘（handle）上方 ===")
        obs, _ = move_to(env, obs, pre_grasp, max_steps=600, gripper_cmd=-1.0, **kw)

    # ── 阶段2：下降到抓取高度（DESCEND）───────────────────
    print(f"\n=== [{nut_label}] 阶段2：DESCEND — 下降到螺母边缘抓取高度 ===")
    nut_pos = obs[nut_obs_key].copy()   # 重新读取（防止漂移）
    # 同步刷新 handle 位置（螺母可能因接触轻微位移）
    if handle_site_id is not None:
        handle_pos   = env.sim.data.site_xpos[handle_site_id].copy()
        grasp_xy     = handle_pos[:2]
        xy_offset    = handle_pos[:2] - nut_pos[:2]
        peg_align_xy = peg_pos[:2] + xy_offset
    grasp_pos = np.array([grasp_xy[0], grasp_xy[1], nut_pos[2] + GRASP_Z_OFFSET])
    obs, _ = move_to(env, obs, grasp_pos, max_steps=600, gripper_cmd=-1.0, **kw)

    # ── 阶段3：关闭夹爪（GRASP）──────────────────────────
    obs = hold_gripper(env, obs, +1.0, GRASP_STEPS,
                       f"=== [{nut_label}] 阶段3：GRASP — 关闭夹爪 ===", **kw)

    # ── 阶段4：垂直提起（LIFT）───────────────────────────
    print(f"\n=== [{nut_label}] 阶段4：LIFT — 垂直提起螺母 ===")
    lift_pos = obs["robot0_eef_pos"].copy()
    lift_pos[2] = transport_z
    obs, _ = move_to(env, obs, lift_pos, max_steps=600, gripper_cmd=+1.0,
                     verbose=False, **kw)

    # ── 阶段5：平移到插销正上方（ALIGN_PEG）──────────────
    # 夹爪在 handle 处（偏离孔中心 xy_offset），
    # 目标 XY = peg_xy + xy_offset，使螺母孔正对插销
    peg_above = np.array([peg_align_xy[0], peg_align_xy[1], transport_z])

    if planner:
        print(f"\n=== [{nut_label}] 阶段5：[RRT] 规划搬运路径 → 插销对准上方 ===")
        print(f"    {obs['robot0_eef_pos'].round(3)} → {peg_above.round(3)}")
        path = planner.plan(obs["robot0_eef_pos"].copy(), peg_above)
        obs = execute_rrt_path(env, obs, path, gripper_cmd=+1.0, **kw)
    else:
        print(f"\n=== [{nut_label}] 阶段5：ALIGN_PEG — 平移到插销对准位置上方 ===")
        print(f"    目标 XY={peg_above[:2].round(4)}（插销 XY={peg_pos[:2].round(4)} + 偏移={xy_offset.round(4)}）")
        obs, _ = move_to(env, obs, peg_above, max_steps=600, gripper_cmd=+1.0, **kw)

    # ── 阶段6：下降套入插销（DESCEND_PLACE）──────────────
    # XY 保持补偿后的对准坐标，Z 下降到桌面层让孔从插销顶滑入
    print(f"\n=== [{nut_label}] 阶段6：DESCEND_PLACE — 孔洞套入插销 ===")
    place_pos = np.array([peg_align_xy[0], peg_align_xy[1], place_z])
    obs, _ = move_to(env, obs, place_pos, max_steps=800, gripper_cmd=+1.0, **kw)

    # ── 阶段7：释放夹爪（RELEASE）────────────────────────
    obs = hold_gripper(env, obs, -1.0, RELEASE_STEPS,
                       f"=== [{nut_label}] 阶段7：RELEASE — 释放螺母 ===", **kw)

    # ── 阶段8：收尾抬起（RETRACT）────────────────────────
    print(f"\n=== [{nut_label}] 阶段8：RETRACT — 抬起末端 ===")
    retract = obs["robot0_eef_pos"].copy()
    retract[2] += GRASP_OFFSET_Z
    obs, _ = move_to(env, obs, retract, max_steps=RETRACT_STEPS,
                     gripper_cmd=-1.0, verbose=False, **kw)

    # ── 成功判定 ─────────────────────────────────────────
    # NutAssembly.on_peg 判定：XY 偏差 < 0.03m 且 Z < table_z + 0.05
    nut_final = obs[nut_obs_key].copy()
    xy_dist   = np.linalg.norm(nut_final[:2] - peg_pos[:2])
    z_ok      = nut_final[2] < (table_z + 0.05)
    success   = (xy_dist < 0.03) and z_ok

    print(f"\n  [{nut_label}] {'✓ 装配成功！' if success else '✗ 装配失败'}")
    print(f"  螺母最终位置 : {nut_final.round(3)}")
    print(f"  XY 偏差      : {xy_dist:.4f}m（阈值 0.030m）")
    print(f"  Z 位置       : {nut_final[2]:.4f}m（阈值 {table_z + 0.05:.3f}m）")

    return obs, success


# ─────────────────────────────────────────────────────────────
# 主流程：完整装配任务
# ─────────────────────────────────────────────────────────────
def run_nut_assembly(env, order="square_first",
                     video_writer=None, camera=None,
                     planner=None, show_camera_name=None):
    """
    执行一次完整的 NutAssembly 任务：
    依次抓起两颗螺母并将其套入对应插销。

    顺序由 order 参数控制：
      "square_first" : SquareNut → peg1, 然后 RoundNut → peg2
      "round_first"  : RoundNut  → peg2, 然后 SquareNut → peg1
    """
    obs = env.reset()

    # 从仿真获取插销世界坐标
    # peg1（方形插销）对应 SquareNut，peg2（圆形插销）对应 RoundNut
    peg1_pos = env.sim.data.body_xpos[env.peg1_body_id].copy()
    peg2_pos = env.sim.data.body_xpos[env.peg2_body_id].copy()
    table_z  = env.table_offset[2]   # 桌面 Z 坐标

    print(f"\n  观测键（螺母位置）: {[k for k in obs.keys() if 'Nut' in k and 'pos' in k]}")
    print(f"  peg1 位置（方形）: {peg1_pos.round(3)}")
    print(f"  peg2 位置（圆形）: {peg2_pos.round(3)}")
    print(f"  桌面 Z           : {table_z:.3f}m")
    print(f"  SquareNut 位置   : {obs['SquareNut_pos'].round(3)}")
    print(f"  RoundNut  位置   : {obs['RoundNut_pos'].round(3)}")
    print(f"  末端初始位置     : {obs['robot0_eef_pos'].round(3)}")
    print(f"  装配顺序         : {order}")

    kw = dict(video_writer=video_writer, camera=camera,
              planner=planner, show_camera_name=show_camera_name)

    # 两颗螺母的装配参数：(obs_key, peg_pos, label, handle_site_id)
    # env.object_site_ids[0] = SquareNut 的 handle_site
    # env.object_site_ids[1] = RoundNut  的 handle_site
    tasks = {
        "square": ("SquareNut_pos", peg1_pos, "SquareNut→peg1（方形）", env.object_site_ids[0]),
        "round":  ("RoundNut_pos",  peg2_pos, "RoundNut→peg2（圆形）",  env.object_site_ids[1]),
    }
    task_order = ["square", "round"] if order == "square_first" else ["round", "square"]

    results = {}
    for key in task_order:
        nut_obs_key, peg_pos, label, handle_site_id = tasks[key]
        print(f"\n\n{'━'*54}")
        print(f"  装配：{label}")
        print(f"{'━'*54}")
        obs, ok = assemble_nut(env, obs, nut_obs_key, peg_pos, table_z,
                               label, handle_site_id=handle_site_id, **kw)
        results[key] = ok

    # 保持一秒渲染最终状态
    for _ in range(int(MAX_FR)):
        action = np.zeros(env.action_dim)
        obs, _, _, _ = env.step(action)
        env.render()
        if video_writer is not None:
            _capture_frame(obs, camera, video_writer)
        if show_camera_name is not None:
            show_camera_frame(obs, show_camera_name)
        time.sleep(1 / MAX_FR)

    sq_ok = results["square"]
    rn_ok = results["round"]
    all_ok = sq_ok and rn_ok

    print(f"\n  方形螺母（→peg1）: {'✓ 成功' if sq_ok else '✗ 失败'}")
    print(f"  圆形螺母（→peg2）: {'✓ 成功' if rn_ok else '✗ 失败'}")

    return all_ok


# ─────────────────────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="NutAssembly 脚本化螺母装配 Demo（基于坐标控制，无需 RL）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--robot", type=str, default="Kinova3",
                        help="机器人型号（Kinova3/Panda/UR5e/IIWA/Sawyer/xArm7 等）")
    parser.add_argument("--gripper", type=str, default="PandaGripper",
                        help="夹爪型号（PandaGripper | Robotiq85Gripper | RethinkGripper 等）")
    parser.add_argument("--order", type=str, default="square_first",
                        choices=["square_first", "round_first"],
                        help="装配顺序：square_first（默认）或 round_first")
    parser.add_argument("--plan", action="store_true",
                        help="启用 Cartesian RRT 路径规划（自动避障）")
    parser.add_argument("--plan-step", type=float, default=0.07,
                        help="RRT 步长（m），默认 0.07")
    parser.add_argument("--plan-iter", type=int, default=3000,
                        help="RRT 最大迭代次数，默认 3000")
    parser.add_argument("--nreset", type=int, default=1,
                        help="重复装配次数（默认 1）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="速度缩放（默认 1.0，越大越快但精度下降）")
    parser.add_argument("--save-video", type=str, default=None, metavar="PATH",
                        help="视频保存路径（如 nut_assembly.mp4）")
    parser.add_argument("--camera", type=str, default="frontview",
                        help="录制相机名称（默认 frontview）")
    parser.add_argument("--show-camera", action="store_true",
                        help="实时显示相机画面（需要 opencv-python）")
    args = parser.parse_args()

    MAX_STEP    = 0.05 * args.speed
    save_video  = args.save_video is not None
    show_camera = args.show_camera
    need_offscreen = save_video or show_camera

    if save_video:
        macros.IMAGE_CONVENTION = "opencv"

    print(f"\n{'='*58}")
    print(f"  NutAssembly 螺母装配 Demo")
    print(f"  机器人 : {args.robot}   夹爪 : {args.gripper}")
    print(f"  装配顺序: {args.order}")
    print(f"  运动模式: {'RRT 路径规划' if args.plan else '固定 8 阶段'}")
    if args.plan:
        print(f"  RRT 参数: step={args.plan_step}m  iter={args.plan_iter}")
    print(f"  速度   : {args.speed}x  重复 : {args.nreset} 次")
    if save_video:
        print(f"  录制   : {args.save_video}  相机 : {args.camera}")
    if show_camera:
        print(f"  相机可视化: {args.camera}")
    print(f"{'='*58}")

    # ── 控制器配置（OSC_POSE：末端位移增量控制）─────────
    arm_ctrl = suite.load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(
        arm_ctrl, args.robot, ["right", "left"]
    )

    # ── 环境配置 ─────────────────────────────────────────
    make_kwargs = dict(
        env_name="NutAssembly",
        robots=args.robot,
        controller_configs=ctrl_cfg,
        has_renderer=True,
        has_offscreen_renderer=need_offscreen,
        use_camera_obs=need_offscreen,
        use_object_obs=True,
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
        hard_reset=False,
        single_object_mode=0,   # 两颗螺母都出现
    )
    if need_offscreen:
        cam_list = list(dict.fromkeys(
            ([args.camera] if save_video else []) +
            ([args.camera] if show_camera else [])
        ))
        make_kwargs.update(
            camera_names=cam_list,
            camera_heights=VIDEO_H,
            camera_widths=VIDEO_W,
        )
    if args.gripper != "default":
        make_kwargs["gripper_types"] = args.gripper
        print(f"\n  指定夹爪: {args.gripper}")

    # ── 创建环境 ─────────────────────────────────────────
    print("\n  正在初始化 NutAssembly 环境...")
    env = suite.make(**make_kwargs)
    obs = env.reset()

    # ── 初始化 RRT 规划器（如果启用）────────────────────
    planner = None
    if args.plan:
        # 从第一颗螺母推算桌面高度
        sq_z     = obs["SquareNut_pos"][2]
        table_z  = sq_z - 0.010   # 螺母半高约 0.010m
        planner  = CartesianRRT(
            table_z   = table_z,
            step_size = args.plan_step,
            max_iter  = args.plan_iter,
            goal_bias = 0.20,
            safety_margin = 0.04,
        )
        print(f"\n  RRT 规划器已初始化（桌面高度={table_z:.3f}m）")
        print(f"  工作空间 Z: [{table_z+0.04:.3f}, {table_z+0.69:.3f}]m")

    # ── 视频录制 ─────────────────────────────────────────
    video_writer = None
    if save_video:
        import imageio
        video_writer = imageio.get_writer(args.save_video, fps=MAX_FR)
        print(f"\n  ✓ 视频录制中 → {args.save_video}（{VIDEO_W}×{VIDEO_H}，相机: {args.camera}）")

    # ── 执行多次装配 ──────────────────────────────────────
    results = []
    for i in range(args.nreset):
        print(f"\n\n{'─'*52}")
        print(f"  第 {i+1} / {args.nreset} 次装配任务")
        print(f"{'─'*52}")
        success = run_nut_assembly(
            env,
            order=args.order,
            video_writer=video_writer,
            camera=args.camera if (save_video or show_camera) else None,
            planner=planner,
            show_camera_name=args.camera if show_camera else None,
        )
        results.append(success)

    # ── 保存视频 ─────────────────────────────────────────
    if video_writer is not None:
        video_writer.close()
        size_mb = os.path.getsize(args.save_video) / 1024 / 1024
        print(f"\n  ✓ 视频已保存: {args.save_video}（{size_mb:.1f} MB）")

    # ── 汇总结果 ─────────────────────────────────────────
    n_ok = sum(results)
    print(f"\n{'='*58}")
    print(f"  结果汇总：{n_ok} / {args.nreset} 次完整装配成功")
    for i, s in enumerate(results):
        print(f"  第 {i+1} 次: {'✓ 全部成功' if s else '✗ 部分/全部失败'}")
    print(f"{'='*58}\n")

    if show_camera:
        import cv2
        cv2.destroyAllWindows()

    env.close()
