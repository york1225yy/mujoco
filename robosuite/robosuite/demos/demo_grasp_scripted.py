"""
【中文说明】
基于坐标的脚本化机械臂抓取演示

这是 robosuite 官方 demo 中缺失的一个关键示例：
不依赖强化学习，不依赖遥操作，而是通过「读取目标坐标 → 计算运动方向 → 发送控制指令」
的方式，让机械臂自动完成完整的「接近 → 下降 → 抓取 → 提起」抓取流程。

【核心思路】
  使用 OSC_POSE（操作空间控制器），其动作格式为：
    action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
  其中 dx/dy/dz 是末端执行器的位移增量（单位：米），gripper 是夹爪开合（-1=张开, +1=关闭）

  每一步控制逻辑：
    误差 = 目标位置 - 当前末端位置
    动作 = clip(误差 * 增益, 最大步长)   ← 比例控制

【抓取流程（4个阶段）】
  阶段1 PRE_GRASP：末端移动到方块正上方（预抓取位置）
  阶段2 DESCEND  ：末端垂直下降到方块高度（接近阶段）
  阶段3 GRASP    ：关闭夹爪（抓紧方块）
  阶段4 LIFT     ：将方块提起到安全高度

【运行方式】
  python -m robosuite.demos.demo_grasp_scripted
  可选参数：
    --robot   Panda/UR5e/IIWA 等（默认 Panda）
    --nreset  重复抓取次数（默认 3）
    --speed   控制步长缩放比例（默认 1.0，越大越快越不精确）
"""

import argparse
import time

import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ─────────────────────────────────────────────
# 超参数：可根据机器人和任务调整
# ─────────────────────────────────────────────
MAX_STEP = 0.05          # 每步末端最大位移（米），越小越精确但越慢
GRASP_OFFSET_Z = 0.15    # 预抓取悬停高度（方块上方 15cm）
GRASP_Z = -0.005         # 抓取时末端相对方块的 Z 偏移（轻微嵌入）
LIFT_HEIGHT = 0.25       # 提升目标高度（相对桌面）
REACH_THRESH = 0.018     # 到达判定阈值（距目标小于此值视为到达）
GRASP_STEPS = 25         # 关闭夹爪持续步数
LIFT_STEPS = 60          # 提升持续步数
MAX_FR = 30              # 最大帧率限制


def move_to(env, obs, target_pos, max_steps=200, gripper_cmd=-1.0, verbose=True):
    """
    【子函数】将末端执行器移动到目标位置（比例控制）

    参数：
        env        : robosuite 环境实例
        obs        : 当前观测字典
        target_pos : 目标位置，形状 (3,)，世界坐标系
        max_steps  : 最大尝试步数（防止卡死）
        gripper_cmd: 夹爪指令，-1=张开（默认），+1=关闭
        verbose    : 是否打印距离信息

    返回：
        obs        : 执行后的最新观测
        reached    : 是否成功到达目标
    """
    for step in range(max_steps):
        start = time.time()

        # 读取当前末端位置（键名格式：robot0_eef_pos）
        eef_pos = obs["robot0_eef_pos"]

        # 计算位置误差（目标 - 当前）
        error = target_pos - eef_pos
        dist = np.linalg.norm(error)

        if verbose and step % 20 == 0:
            print(f"  距目标: {dist:.4f}m | 当前: {eef_pos.round(3)} | 目标: {target_pos.round(3)}")

        # 判断是否到达目标
        if dist < REACH_THRESH:
            if verbose:
                print(f"  ✓ 到达目标（{step+1}步，残差={dist:.4f}m）")
            return obs, True

        # 比例控制：计算位移增量，clip 到最大步长
        delta_pos = np.clip(error, -MAX_STEP, MAX_STEP)

        # 构建 OSC_POSE 动作向量：[dx, dy, dz, droll, dpitch, dyaw, gripper]
        # 姿态保持不变（旋转分量全为0），只控制位置
        action = np.zeros(7)
        action[:3] = delta_pos       # 位置增量
        action[3:6] = 0.0            # 姿态增量（保持当前姿态）
        action[6] = gripper_cmd      # 夹爪指令

        obs, reward, done, _ = env.step(action)
        env.render()

        # 帧率限制
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    if verbose:
        print(f"  ✗ 超过最大步数 ({max_steps}) 未到达")
    return obs, False


def hold_action(env, obs, gripper_cmd, steps, description=""):
    """
    【子函数】保持末端静止，仅执行夹爪动作（用于抓取/保持阶段）

    参数：
        gripper_cmd: -1=张开, +1=关闭
        steps      : 持续步数
    """
    if description:
        print(f"\n{description}")

    eef_pos = obs["robot0_eef_pos"]  # 保持当前末端位置不动

    for _ in range(steps):
        start = time.time()

        # 位置增量为0，只改变夹爪状态
        action = np.zeros(7)
        action[6] = gripper_cmd

        obs, reward, done, _ = env.step(action)
        env.render()

        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    return obs


def run_grasp_episode(env):
    """
    【核心函数】执行一次完整的脚本化抓取流程

    返回：
        success: 是否成功提起方块
    """
    # ── 初始化 ──────────────────────────────────────────
    obs = env.reset()

    # 打印当前观测键（帮助调试）
    print("\n【可用观测键】:", [k for k in obs.keys() if "pos" in k or "quat" in k])

    # 读取方块位置（Lift 任务的观测键为 "cube_pos"）
    cube_pos = obs["cube_pos"].copy()
    print(f"\n方块位置: {cube_pos.round(3)}")
    print(f"末端初始位置: {obs['robot0_eef_pos'].round(3)}")

    # ── 阶段1：PRE_GRASP 移动到方块正上方 ──────────────
    print("\n=== 阶段1：移动到预抓取位置（方块上方）===")
    pre_grasp_pos = cube_pos.copy()
    pre_grasp_pos[2] += GRASP_OFFSET_Z   # 提高 Z 到方块上方 15cm
    obs, reached = move_to(env, obs, pre_grasp_pos, max_steps=200, gripper_cmd=-1.0)

    if not reached:
        print("警告：未能到达预抓取位置，继续尝试")

    # ── 阶段2：DESCEND 下降到抓取高度 ──────────────────
    print("\n=== 阶段2：下降到抓取高度 ===")
    # 每次重新读取方块位置（方块可能因物理仿真轻微移动）
    cube_pos = obs["cube_pos"].copy()
    grasp_pos = cube_pos.copy()
    grasp_pos[2] += GRASP_Z              # 下降到方块高度（轻微嵌入）
    obs, reached = move_to(env, obs, grasp_pos, max_steps=150, gripper_cmd=-1.0)

    # ── 阶段3：GRASP 关闭夹爪 ──────────────────────────
    obs = hold_action(env, obs, gripper_cmd=+1.0, steps=GRASP_STEPS,
                      description="=== 阶段3：关闭夹爪（抓取方块）===")

    # ── 阶段4：LIFT 提起方块 ────────────────────────────
    print("\n=== 阶段4：提起方块 ===")
    lift_pos = obs["robot0_eef_pos"].copy()
    lift_pos[2] = cube_pos[2] + LIFT_HEIGHT   # 提升到方块原始高度 + 25cm
    obs, reached = move_to(env, obs, lift_pos, max_steps=LIFT_STEPS,
                           gripper_cmd=+1.0)   # 提升过程中保持夹爪关闭

    # ── 判断是否成功 ────────────────────────────────────
    final_cube_z = obs["cube_pos"][2]
    success_thresh = cube_pos[2] + 0.04   # 方块比初始位置高4cm以上视为成功
    success = final_cube_z > success_thresh

    print(f"\n{'✓ 抓取成功！' if success else '✗ 抓取失败'}")
    print(f"  方块初始Z={cube_pos[2]:.3f}  最终Z={final_cube_z:.3f}  "
          f"提升={final_cube_z - cube_pos[2]:.3f}m")

    # 成功后展示1秒
    for _ in range(30):
        action = np.zeros(7)
        action[6] = +1.0   # 保持夹爪关闭
        obs, _, _, _ = env.step(action)
        env.render()
        time.sleep(1 / MAX_FR)

    return success


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="脚本化机械臂抓取 Demo（基于坐标，无需 RL）")
    parser.add_argument("--robot", type=str, default="Panda",
                        help="机器人型号（Panda/UR5e/IIWA/Sawyer/xArm7 等）")
    parser.add_argument("--nreset", type=int, default=3,
                        help="重复抓取次数")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="速度缩放（越大越快，但精度下降）")
    args = parser.parse_args()

    # 根据 speed 参数调整步长（MAX_STEP 已在模块顶层定义，此处直接覆盖即可）
    MAX_STEP = 0.05 * args.speed

    print(f"\n{'='*50}")
    print(f"  脚本化抓取 Demo")
    print(f"  机器人: {args.robot}  重复: {args.nreset}次  速度: {args.speed}x")
    print(f"{'='*50}")

    # ── 创建环境 ────────────────────────────────────────
    # 使用 OSC_POSE 控制器：接受末端位移增量，内部自动转换为关节力矩
    arm_controller_config = suite.load_part_controller_config(
        default_controller="OSC_POSE"   # 操作空间控制器，抓取任务最佳选择
    )
    controller_config = refactor_composite_controller_config(
        arm_controller_config, args.robot, ["right", "left"]
    )

    env = suite.make(
        env_name="Lift",               # 举起方块任务（含方块+桌面场景）
        robots=args.robot,             # 机械臂型号
        controller_configs=controller_config,
        has_renderer=True,             # 开启可视化窗口
        has_offscreen_renderer=False,
        use_camera_obs=False,          # 不需要图像观测，只用数值状态
        use_object_obs=True,           # 必须开启：获取 cube_pos 等物体状态
        reward_shaping=False,          # 不需要奖励（非 RL）
        control_freq=20,               # 控制频率 20Hz
        ignore_done=True,              # 忽略任务完成信号，持续运行
    )

    # ── 多次运行 ─────────────────────────────────────────
    results = []
    for i in range(args.nreset):
        print(f"\n\n{'─'*40}")
        print(f"第 {i+1}/{args.nreset} 次抓取")
        print(f"{'─'*40}")
        success = run_grasp_episode(env)
        results.append(success)

    # ── 汇总结果 ─────────────────────────────────────────
    n_success = sum(results)
    print(f"\n{'='*50}")
    print(f"  抓取结果汇总：{n_success}/{args.nreset} 次成功")
    for i, s in enumerate(results):
        print(f"  第{i+1}次: {'✓ 成功' if s else '✗ 失败'}")
    print(f"{'='*50}\n")

    env.close()
