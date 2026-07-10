"""
【中文说明】
脚本化机械臂抓取+放置演示（增强版 v3）

基于坐标控制，无需强化学习，自动完成完整的抓取-搬运-放置流程。

【功能参数说明】
  --robot      机械臂型号（Panda/UR5e/IIWA/xArm7 等）
  --gripper    独立选择夹爪型号（default=机器人默认）
  --place-pos  放置目标坐标 X Y Z（桌面约 z=0.82）
  --plan       启用 Cartesian RRT 路径规划（自动避障，替代固定运动序列）
  --nreset     重复次数
  --speed      速度缩放
  --save-video 录制视频保存路径（mp4）
  --camera     录制相机名称

【默认模式（不加 --plan）】
  固定 6 阶段流程：预抓取→下降→抓取→搬运→下降放置→释放

【规划模式（加 --plan）】
  使用 Cartesian RRT 自动规划接近路径和搬运路径：
    阶段1 规划并执行到预抓取点的路径
    阶段2 下降+抓取（固定，短距离不需要规划）
    阶段3 规划并执行到放置目标上方的路径（夹爪关闭）
    阶段4 下降+放置+收尾

【运行示例】
  # 默认模式
  python -m robosuite.demos.demo_grasp_scripted

  # 规划模式 + 自定义放置坐标 + 录制
  python -m robosuite.demos.demo_grasp_scripted \
      --robot UR5e --gripper Robotiq85Gripper \
      --place-pos 0.15 0.15 0.82 \
      --plan --save-video output.mp4
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
# 超参数（可直接修改）
# ─────────────────────────────────────────────────────────────
MAX_STEP       = 0.05    # 每步末端最大位移（米）
GRASP_OFFSET_Z = 0.15   # 预抓取/搬运时高出目标的距离（m）
GRASP_Z_OFFSET = -0.005  # 抓取时末端轻微嵌入方块（m）
REACH_THRESH   = 0.018   # 到达判定阈值（m）
GRASP_STEPS    = 100      # 关闭夹爪持续步数
RELEASE_STEPS  = 100      # 张开夹爪持续步数
RETRACT_STEPS  = 100      # 放置后抬臂步数
MAX_FR         = 30      # 最大帧率（fps）
VIDEO_W        = 640     # 视频宽度
VIDEO_H        = 480     # 视频高度


# ─────────────────────────────────────────────────────────────
# Cartesian RRT 路径规划器
# ─────────────────────────────────────────────────────────────
class CartesianRRT:
    """
    三维笛卡尔空间 RRT（Rapidly-exploring Random Tree）路径规划器。

    在末端执行器工作空间内规划从起点到终点的无碰撞路径。
    碰撞检测使用几何约束：
      - 必须高于桌面（table_z + safety_margin）
      - 必须在 XYZ 工作空间边界内

    规划成功后对路径进行贪心平滑，删除不必要的中间路径点。
    如果 RRT 在最大迭代次数内未找到路径，退化为直线路径（并打印警告）。

    参数：
        table_z        : 桌面 Z 坐标（米），从 obs["cube_pos"] 推算
        ws_xy_half     : XY 工作空间半径（m），默认 0.55m
        z_max_above    : 桌面以上最大高度（m），默认 0.65m
        step_size      : 每步扩展距离（m），越小路径越精细但迭代更多
        max_iter       : 最大 RRT 迭代次数
        goal_bias      : 直接朝目标采样的概率（0~1）
        safety_margin  : 桌面安全裕度（m）
    """

    def __init__(self, table_z, ws_xy_half=0.55, z_max_above=0.65,
                 step_size=0.07, max_iter=3000, goal_bias=0.20,
                 safety_margin=0.04):
        self.step_size = step_size
        self.max_iter  = max_iter
        self.goal_bias = goal_bias

        # 工作空间边界（下界/上界）
        self.lo = np.array([-ws_xy_half, -ws_xy_half, table_z + safety_margin])
        self.hi = np.array([ ws_xy_half,  ws_xy_half, table_z + z_max_above])

    # ── 内部工具方法 ────────────────────────────────────────

    def _valid(self, pos):
        """检查单个点是否在安全工作空间内。"""
        return bool(np.all(pos >= self.lo) and np.all(pos <= self.hi))

    def _segment_valid(self, p1, p2, n=8):
        """检查路径段 p1→p2 是否无碰撞（沿线采样 n+1 个点检查）。"""
        for i in range(n + 1):
            t = i / n
            if not self._valid(p1 + t * (p2 - p1)):
                return False
        return True

    def _steer(self, from_pos, to_pos):
        """从 from_pos 向 to_pos 延伸一步（不超过 step_size）。"""
        d    = to_pos - from_pos
        dist = np.linalg.norm(d)
        return to_pos.copy() if dist <= self.step_size else from_pos + (d / dist) * self.step_size

    # ── 主规划方法 ──────────────────────────────────────────

    def plan(self, start, goal):
        """
        从 start 到 goal 运行 RRT 规划。

        参数：
            start, goal : np.ndarray 形状 (3,)，世界坐标系（m）

        返回：
            list[np.ndarray]  路径点列表（含起止点），已平滑
            如果完全失败，返回 [start, goal]（直线退化路径）
        """
        start = np.array(start, dtype=float)
        goal  = np.array(goal,  dtype=float)

        # 将起/终点 Z 修正到安全高度以内（防止输入在边界外）
        start[2] = np.clip(start[2], self.lo[2] + 0.005, self.hi[2])
        goal[2]  = np.clip(goal[2],  self.lo[2] + 0.005, self.hi[2])

        # 快速检查：直线是否可行（大多数空旷场景下直接返回）
        if self._segment_valid(start, goal, n=20):
            print(f"    [RRT] 直线路径可行，跳过迭代")
            return [start.copy(), goal.copy()]

        # RRT 主循环
        nodes  = [start.copy()]
        parent = [-1]

        for it in range(self.max_iter):
            # 采样：goal_bias 概率直接采样目标，否则均匀随机采样
            sample = goal.copy() if np.random.rand() < self.goal_bias \
                     else np.random.uniform(self.lo, self.hi)

            # 找距 sample 最近的树节点
            dists   = np.array([np.linalg.norm(n - sample) for n in nodes])
            nn_idx  = int(np.argmin(dists))
            new_pos = self._steer(nodes[nn_idx], sample)

            # 有效性检查
            if not self._valid(new_pos):
                continue
            if not self._segment_valid(nodes[nn_idx], new_pos):
                continue

            nodes.append(new_pos)
            parent.append(nn_idx)

            # 检查是否足够接近目标，且直线可连接
            if np.linalg.norm(new_pos - goal) <= self.step_size:
                if self._segment_valid(new_pos, goal):
                    nodes.append(goal.copy())
                    parent.append(len(nodes) - 2)

                    # 回溯路径
                    path, idx = [], len(nodes) - 1
                    while idx >= 0:
                        path.append(nodes[idx])
                        idx = parent[idx]
                    path.reverse()

                    path = self._smooth(path)
                    print(f"    [RRT] 规划成功：{it + 1} 次迭代，平滑后 {len(path)} 个路径点")
                    return path

        # 超时退化为直线
        print(f"    [RRT] 警告：{self.max_iter} 次迭代未找到路径，退化为直线路径")
        return [start.copy(), goal.copy()]

    def _smooth(self, path):
        """
        贪心路径平滑：从路径头部出发，尽量跳到最远可直连节点，
        跳过不必要的中间路径点。
        """
        if len(path) <= 2:
            return path
        smoothed, i = [path[0]], 0
        while i < len(path) - 1:
            # 从当前点向后找最远可直连节点
            j = len(path) - 1
            while j > i + 1 and not self._segment_valid(path[i], path[j], n=15):
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed


# ─────────────────────────────────────────────────────────────
# 视频帧捕获（功能：--save-video）
# ─────────────────────────────────────────────────────────────
def _capture_frame(obs, camera, video_writer):
    """从 obs 字典提取相机图像写入视频（需启用 use_camera_obs=True）。"""
    img_key = f"{camera}_image"
    if img_key in obs:
        video_writer.append_data(obs[img_key])


# ─────────────────────────────────────────────────────────────
# 核心控制函数
# ─────────────────────────────────────────────────────────────
def move_to(env, obs, target_pos, max_steps=600, gripper_cmd=-1.0,
            verbose=True, video_writer=None, camera=None):
    """
    比例控制：将末端执行器移动到 target_pos。

    每步：
        error  = target_pos - obs["robot0_eef_pos"]
        delta  = clip(error, -MAX_STEP, MAX_STEP)
        action = [delta_x, delta_y, delta_z, 0, 0, 0, gripper_cmd, ...]

    action[6:] = gripper_cmd 适配任意 DOF 夹爪（多指手也兼容）。

    返回：
        (obs, reached)  reached=True 表示到达目标
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

        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    if verbose:
        print(f"  ✗ 超时（{max_steps}步），残差={dist:.4f}m")
    return obs, False


def hold_gripper(env, obs, gripper_cmd, steps, stage_name="",
                 video_writer=None, camera=None):
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
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
    return obs


def execute_rrt_path(env, obs, waypoints, gripper_cmd, verbose=False,
                     video_writer=None, camera=None):
    """
    按 RRT 规划的路径点列表逐段移动末端执行器。

    对每个路径点（跳过第一个，即当前位置）调用 move_to()。
    路径点间距约为 step_size（默认0.07m），每段 max_steps=80 足够。

    参数：
        waypoints  : CartesianRRT.plan() 返回的路径点列表
        gripper_cmd: 执行过程中的夹爪状态
    """
    kw = dict(video_writer=video_writer, camera=camera)
    n = len(waypoints)
    print(f"    执行规划路径：共 {n} 个路径点")
    for i, wp in enumerate(waypoints[1:], 1):   # 跳过起始点（当前位置）
        if verbose:
            print(f"    路径点 {i}/{n-1}: {wp.round(3)}")
        obs, _ = move_to(env, obs, wp,
                         max_steps=600,
                         gripper_cmd=gripper_cmd,
                         verbose=False,
                         **kw)
    return obs


# ─────────────────────────────────────────────────────────────
# 主流程：抓取+放置
# ─────────────────────────────────────────────────────────────
def run_pick_and_place(env, place_pos, video_writer=None, camera=None, planner=None):
    """
    执行一次完整的抓取+放置。

    默认模式（planner=None）：固定 6 阶段运动序列
      1. PRE_GRASP     → 末端移到方块正上方
      2. DESCEND       → 下降到抓取高度
      3. GRASP         → 关闭夹爪
      4. TRANSPORT     → 垂直提起后水平搬运
      5. DESCEND_PLACE → 下降到放置高度
      6. RELEASE       → 张开夹爪

    规划模式（planner=CartesianRRT 实例）：
      RRT 自动规划阶段1（接近方块）和阶段4（搬运到目标）的路径，
      其余阶段（短距离运动）仍使用直接控制。

    成功判定：放置后方块与目标 XY 距离 < 8cm 且 Z 偏差 < 6cm
    """
    obs = env.reset()

    print("\n  观测键（位置类）:", [k for k in obs.keys() if "pos" in k])
    cube_pos  = obs["cube_pos"].copy()
    place_arr = np.array(place_pos, dtype=float)
    use_plan  = planner is not None

    print(f"\n  方块初始位置 : {cube_pos.round(3)}")
    print(f"  放置目标位置 : {place_arr.round(3)}")
    print(f"  末端初始位置 : {obs['robot0_eef_pos'].round(3)}")
    print(f"  运动模式     : {'RRT 路径规划' if use_plan else '固定 6 阶段'}")

    kw = dict(video_writer=video_writer, camera=camera)

    # ── 阶段1：到达预抓取位置（方块正上方）─────────────
    pre_grasp = cube_pos.copy()
    pre_grasp[2] += GRASP_OFFSET_Z

    if use_plan:
        print("\n=== 阶段1：[RRT] 规划并执行到预抓取位置 ===")
        path = planner.plan(obs["robot0_eef_pos"].copy(), pre_grasp)
        obs = execute_rrt_path(env, obs, path, gripper_cmd=-1.0, **kw)
    else:
        print("\n=== 阶段1：PRE_GRASP — 移到方块正上方 ===")
        obs, _ = move_to(env, obs, pre_grasp, max_steps=600, gripper_cmd=-1.0, **kw)

    # ── 阶段2：下降到抓取高度（直接控制，短距离）────────
    print("\n=== 阶段2：DESCEND — 下降到抓取高度 ===")
    cube_pos = obs["cube_pos"].copy()   # 重新读取（防止漂移）
    grasp_pos = cube_pos.copy()
    grasp_pos[2] += GRASP_Z_OFFSET
    obs, _ = move_to(env, obs, grasp_pos, max_steps=600, gripper_cmd=-1.0, **kw)

    # ── 阶段3：关闭夹爪 ──────────────────────────────────
    obs = hold_gripper(env, obs, +1.0, GRASP_STEPS,
                       "=== 阶段3：GRASP — 关闭夹爪 ===", **kw)

    # ── 阶段4：搬运到放置目标上方 ────────────────────────
    # 先垂直提起到安全搬运高度（两种模式都需要）
    lift_z    = cube_pos[2] + GRASP_OFFSET_Z
    lift_pos  = obs["robot0_eef_pos"].copy()
    lift_pos[2] = lift_z

    if use_plan:
        print("\n=== 阶段4：[RRT] 垂直提起，然后规划搬运路径 ===")
        # 先垂直提起（短距离，直接控制）
        obs, _ = move_to(env, obs, lift_pos, max_steps=600, gripper_cmd=+1.0,
                         verbose=False, **kw)
        # 规划从提起位置到放置目标上方
        place_above = place_arr.copy()
        place_above[2] = lift_z
        print(f"  规划搬运路径：{obs['robot0_eef_pos'].round(3)} → {place_above.round(3)}")
        path = planner.plan(obs["robot0_eef_pos"].copy(), place_above)
        obs = execute_rrt_path(env, obs, path, gripper_cmd=+1.0, **kw)
    else:
        print("\n=== 阶段4：TRANSPORT — 提起并搬运到放置目标上方 ===")
        obs, _ = move_to(env, obs, lift_pos, max_steps=600, gripper_cmd=+1.0,
                         verbose=False, **kw)
        transport_tgt = place_arr.copy()
        transport_tgt[2] = lift_z
        obs, _ = move_to(env, obs, transport_tgt, max_steps=600, gripper_cmd=+1.0, **kw)

    # ── 阶段5：下降到放置高度（直接控制）────────────────
    print("\n=== 阶段5：DESCEND_PLACE — 下降到放置高度 ===")
    place_tgt = place_arr.copy()
    place_tgt[2] += GRASP_Z_OFFSET + 0.01
    obs, _ = move_to(env, obs, place_tgt, max_steps=600, gripper_cmd=+1.0, **kw)

    # ── 阶段6：释放夹爪 ──────────────────────────────────
    obs = hold_gripper(env, obs, -1.0, RELEASE_STEPS,
                       "=== 阶段6：RELEASE — 张开夹爪放下方块 ===", **kw)

    # ── 收尾：抬起末端 ────────────────────────────────────
    print("\n  收尾：抬起末端...")
    retract = obs["robot0_eef_pos"].copy()
    retract[2] += GRASP_OFFSET_Z
    obs, _ = move_to(env, obs, retract, max_steps=RETRACT_STEPS,
                     gripper_cmd=-1.0, verbose=False, **kw)

    # ── 成功判定 ─────────────────────────────────────────
    final_cube = obs["cube_pos"]
    xy_dist = np.linalg.norm(final_cube[:2] - place_arr[:2])
    z_diff  = abs(final_cube[2] - place_arr[2])
    success = (xy_dist < 0.08) and (z_diff < 0.06)

    print(f"\n{'  ✓ 放置成功！' if success else '  ✗ 放置失败'}")
    print(f"  方块最终位置 : {final_cube.round(3)}")
    print(f"  XY 偏差      : {xy_dist:.3f}m（阈值 0.08m）")
    print(f"  Z  偏差      : {z_diff:.3f}m（阈值 0.06m）")

    for _ in range(int(MAX_FR)):
        action = np.zeros(env.action_dim)
        obs, _, _, _ = env.step(action)
        env.render()
        if video_writer is not None:
            _capture_frame(obs, camera, video_writer)
        time.sleep(1 / MAX_FR)

    return success


# ─────────────────────────────────────────────────────────────
# 主程序入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="脚本化机械臂抓取+放置 Demo（基于坐标控制，无需 RL）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--robot", type=str, default="Kinova3",
                        help="机器人型号（Kinova3/Panda/UR5e/IIWA/Sawyer/xArm7 等）")
    parser.add_argument("--gripper", type=str, default="RobotiqThreeFingerGripper",
                        help="夹爪型号（RobotiqThreeFingerGripper | PandaGripper | "
                             "Robotiq85Gripper | Robotiq140Gripper | RethinkGripper）")
    parser.add_argument("--place-pos", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="放置目标坐标（米），如 --place-pos 0.15 0.15 0.82")
    parser.add_argument("--plan", action="store_true",
                        help="启用 Cartesian RRT 路径规划（自动避障）替代固定运动序列")
    parser.add_argument("--plan-step", type=float, default=0.07,
                        help="RRT 步长（m），越小路径越精细但规划更慢（默认 0.07）")
    parser.add_argument("--plan-iter", type=int, default=3000,
                        help="RRT 最大迭代次数（默认 3000）")
    parser.add_argument("--nreset", type=int, default=3,
                        help="重复次数（默认 3）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="速度缩放（默认 1.0，越大越快但精度下降）")
    parser.add_argument("--save-video", type=str, default=None, metavar="PATH",
                        help="视频保存路径（如 output.mp4），不填则不录制")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="录制相机名称（默认 agentview）")
    args = parser.parse_args()

    MAX_STEP   = 0.05 * args.speed
    save_video = args.save_video is not None

    if save_video:
        macros.IMAGE_CONVENTION = "opencv"

    print(f"\n{'='*58}")
    print(f"  脚本化抓取+放置 Demo")
    print(f"  机器人 : {args.robot}   夹爪 : {args.gripper}")
    print(f"  运动模式: {'RRT 路径规划' if args.plan else '固定 6 阶段'}")
    if args.plan:
        print(f"  RRT 参数: step={args.plan_step}m  iter={args.plan_iter}")
    print(f"  速度   : {args.speed}x  重复 : {args.nreset} 次")
    if args.place_pos:
        print(f"  放置点 : {args.place_pos}")
    if save_video:
        print(f"  录制   : {args.save_video}  相机 : {args.camera}")
    print(f"{'='*58}")

    # ── 控制器配置（OSC_POSE：末端位移增量控制）─────────
    arm_ctrl = suite.load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(
        arm_ctrl, args.robot, ["right", "left"]
    )

    # ── 环境配置 ─────────────────────────────────────────
    make_kwargs = dict(
        env_name="Lift",
        robots=args.robot,
        controller_configs=ctrl_cfg,
        has_renderer=True,
        has_offscreen_renderer=save_video,
        use_camera_obs=save_video,
        use_object_obs=True,
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
        hard_reset=False,
    )
    if save_video:
        make_kwargs.update(
            camera_names=args.camera,
            camera_heights=VIDEO_H,
            camera_widths=VIDEO_W,
        )
    if args.gripper != "default":
        make_kwargs["gripper_types"] = args.gripper
        print(f"\n  指定夹爪: {args.gripper}")

    # ── 创建环境 ─────────────────────────────────────────
    print("\n  正在初始化环境...")
    env = suite.make(**make_kwargs)
    obs = env.reset()

    # ── 确定放置目标坐标 ─────────────────────────────────
    if args.place_pos is not None:
        place_pos = np.array(args.place_pos, dtype=float)
    else:
        cube_pos_init = obs["cube_pos"].copy()
        place_pos = np.array([cube_pos_init[0] + 0.20, cube_pos_init[1], cube_pos_init[2]])
        print(f"\n  自动设定放置目标: {place_pos.round(3)}")
        print(f"  （方块初始右侧 20cm，可用 --place-pos X Y Z 自定义）")

    # ── 初始化 RRT 规划器（如果启用）────────────────────
    planner = None
    if args.plan:
        # 从 cube_pos 推算桌面高度：cube 底面 = cube_pos[2] - 半尺寸
        cube_z   = obs["cube_pos"][2]
        table_z  = cube_z - 0.025   # cube 半尺寸约 0.02~0.025m
        planner  = CartesianRRT(
            table_z    = table_z,
            step_size  = args.plan_step,
            max_iter   = args.plan_iter,
            goal_bias  = 0.20,
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

    # ── 执行多次抓取放置 ──────────────────────────────────
    results = []
    for i in range(args.nreset):
        print(f"\n\n{'─'*52}")
        print(f"  第 {i+1} / {args.nreset} 次抓取放置")
        print(f"{'─'*52}")
        success = run_pick_and_place(
            env, place_pos,
            video_writer=video_writer,
            camera=args.camera if save_video else None,
            planner=planner,
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
    print(f"  结果汇总：{n_ok} / {args.nreset} 次放置成功")
    for i, s in enumerate(results):
        print(f"  第 {i+1} 次: {'✓ 成功' if s else '✗ 失败'}")
    print(f"{'='*58}\n")

    env.close()
