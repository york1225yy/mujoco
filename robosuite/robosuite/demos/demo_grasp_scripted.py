"""
【中文说明】
脚本化机械臂抓取+放置演示（增强版）

基于坐标控制，无需强化学习，自动完成完整的抓取-搬运-放置流程。

【新增功能（相比基础版）】
  1. --gripper   独立选择夹爪，不受机械臂型号限制
  2. --place-pos 指定放置目标坐标，场景中用绿色球体+圆盘标记可视化
  3. --save-video 保存完整执行过程为 mp4 视频
  4. 完整 6 阶段流程：预抓取→下降→抓取→搬运→下降放置→释放

【运行方式】
  # 基础运行（Panda，自动设定放置点，不录制）
  python -m robosuite.demos.demo_grasp_scripted

  # 指定机械臂 + 夹爪 + 放置坐标 + 录制
  python -m robosuite.demos.demo_grasp_scripted \
      --robot UR5e --gripper Robotiq85Gripper \
      --place-pos 0.15 0.15 0.82 \
      --save-video output.mp4 --nreset 2

【可用夹爪 --gripper 值】
  default / PandaGripper / Robotiq85Gripper / Robotiq140Gripper /
  RethinkGripper / WiperGripper / PushingGripper

【放置目标坐标参考（Lift 任务桌面约 z=0.82）】
  方块通常出现在 x≈0, y≈0, z≈0.82
  建议放置在桌面范围内：x∈[-0.3,0.3], y∈[-0.3,0.3], z≈0.82
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)

# ─────────────────────────────────────────────────────────────
# 超参数（可直接修改调整行为）
# ─────────────────────────────────────────────────────────────
MAX_STEP       = 0.05    # 每步末端最大位移（米）
GRASP_OFFSET_Z = 0.15   # 预抓取/搬运高出目标的距离（m）
GRASP_Z_OFFSET = -0.005  # 抓取时末端轻微嵌入方块的偏移（m）
REACH_THRESH   = 0.018   # 到达判定阈值（m）
GRASP_STEPS    = 100      # 关闭夹爪持续步数
RELEASE_STEPS  = 100      # 张开夹爪持续步数
RETRACT_STEPS  = 100      # 放置后抬起手臂步数
MAX_FR         = 30      # 最大帧率（fps）
VIDEO_W        = 640     # 录制视频宽度
VIDEO_H        = 480     # 录制视频高度


# ─────────────────────────────────────────────────────────────
# 功能1：放置目标可视化标记
# ─────────────────────────────────────────────────────────────
def add_placement_marker(env, place_pos):
    """
    在场景 XML 中添加放置目标可视化标记（绿色球体+圆盘），然后重新加载仿真。

    只需在首次 env.reset() 后调用一次。
    之后 hard_reset=False 的 env.reset() 会复用含 marker 的模型。

    参数：
        env       : robosuite 环境（已 reset）
        place_pos : 放置目标 [x, y, z]
    """
    xml_str = env.model.get_xml()
    tree = ET.fromstring(xml_str)
    worldbody = tree.find(".//worldbody")

    px, py, pz = float(place_pos[0]), float(place_pos[1]), float(place_pos[2])

    marker_body = ET.SubElement(
        worldbody, "body",
        name="place_target_vis",
        pos=f"{px:.4f} {py:.4f} {pz:.4f}",
    )
    # 绿色半透明球体（主标记）
    ET.SubElement(
        marker_body, "geom",
        type="sphere", size="0.025",
        rgba="0.05 0.90 0.15 0.65",
        contype="0", conaffinity="0",
    )
    # 绿色扁圆柱（落点范围指示圆盘）
    ET.SubElement(
        marker_body, "geom",
        type="cylinder", size="0.055 0.003",
        pos="0 0 -0.025",
        rgba="0.05 0.85 0.15 0.30",
        contype="0", conaffinity="0",
    )

    modified_xml = ET.tostring(tree, encoding="unicode")
    # reset_from_xml_string 内部调用 reset()（deterministic_reset=True 时为软重置）
    # 之后 hard_reset=False 的 env.reset() 复用此模型，marker 持久存在
    env.reset_from_xml_string(modified_xml)
    print(f"  ✓ 可视化标记已添加至 ({px:.3f}, {py:.3f}, {pz:.3f})")


# ─────────────────────────────────────────────────────────────
# 功能3：视频帧捕获
# ─────────────────────────────────────────────────────────────
def _capture_frame(obs, camera, video_writer):
    """
    从观测字典提取相机图像并写入视频。
    需 suite.make() 启用 use_camera_obs=True 和 has_offscreen_renderer=True。
    macros.IMAGE_CONVENTION="opencv" 已在主程序中设置，图像方向正确无需翻转。
    """
    img_key = f"{camera}_image"
    if img_key in obs:
        video_writer.append_data(obs[img_key])


# ─────────────────────────────────────────────────────────────
# 核心控制函数
# ─────────────────────────────────────────────────────────────
def move_to(env, obs, target_pos, max_steps=200, gripper_cmd=-1.0,
            verbose=True, video_writer=None, camera=None):
    """
    使用比例控制将末端移动到目标位置。

    每步逻辑：
        error  = target_pos - current_eef_pos
        delta  = clip(error, -MAX_STEP, MAX_STEP)
        action = [delta_x, delta_y, delta_z, 0, 0, 0, gripper_cmd, ...]

    参数：
        gripper_cmd : -1=张开, +1=关闭
    返回：
        (obs, reached)
    """
    for step in range(max_steps):
        start = time.time()

        eef_pos = obs["robot0_eef_pos"]
        error   = target_pos - eef_pos
        dist    = np.linalg.norm(error)

        if verbose and step % 20 == 0:
            print(f"  d={dist:.4f}m | eef={eef_pos.round(3)} → tgt={np.array(target_pos).round(3)}")

        if dist < REACH_THRESH:
            if verbose:
                print(f"  ✓ 到达目标（{step+1}步，残差={dist:.4f}m）")
            return obs, True

        delta_pos = np.clip(error, -MAX_STEP, MAX_STEP)
        action = np.zeros(env.action_dim)
        action[:3]  = delta_pos   # 末端位移增量（OSC_POSE 格式）
        action[3:6] = 0.0         # 姿态保持不变
        action[6:]  = gripper_cmd # 适配任意 DOF 夹爪（包括多指手）

        obs, _, _, _ = env.step(action)
        env.render()

        if video_writer is not None:
            _capture_frame(obs, camera, video_writer)

        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)

    if verbose:
        print(f"  ✗ 超过最大步数 ({max_steps})，残差={dist:.4f}m")
    return obs, False


def hold_gripper(env, obs, gripper_cmd, steps, stage_name="",
                 video_writer=None, camera=None):
    """
    末端静止，只执行夹爪开合。
      gripper_cmd=+1.0 → 关闭（抓取）
      gripper_cmd=-1.0 → 张开（释放）
    """
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


# ─────────────────────────────────────────────────────────────
# 6 阶段抓取+放置主流程
# ─────────────────────────────────────────────────────────────
def run_pick_and_place(env, place_pos, video_writer=None, camera=None):
    """
    执行一次完整的抓取+放置（6 阶段）。

    阶段1 PRE_GRASP     - 末端移到方块正上方（夹爪张开）
    阶段2 DESCEND       - 末端下降到抓取高度（夹爪张开）
    阶段3 GRASP         - 关闭夹爪抓住方块
    阶段4 TRANSPORT     - 先垂直提起，再水平搬运到放置目标上方（夹爪关闭）
    阶段5 DESCEND_PLACE - 下降到放置高度（夹爪关闭）
    阶段6 RELEASE       - 张开夹爪，方块落下

    成功判定：方块与目标 XY 距离 < 8cm 且 Z 偏差 < 6cm
    """
    obs = env.reset()   # 软重置，含 marker 的模型保持不变

    print("\n  观测键（位置类）:", [k for k in obs.keys() if "pos" in k])
    cube_pos  = obs["cube_pos"].copy()
    place_arr = np.array(place_pos, dtype=float)

    print(f"\n  方块初始位置 : {cube_pos.round(3)}")
    print(f"  放置目标位置 : {place_arr.round(3)}")
    print(f"  末端初始位置 : {obs['robot0_eef_pos'].round(3)}")

    kw = dict(video_writer=video_writer, camera=camera)

    # ── 阶段1：PRE_GRASP ─────────────────────────────────
    print("\n=== 阶段1：PRE_GRASP — 移到方块正上方 ===")
    tgt = cube_pos.copy()
    tgt[2] += GRASP_OFFSET_Z
    obs, _ = move_to(env, obs, tgt, max_steps=500, gripper_cmd=-1.0, **kw)

    # ── 阶段2：DESCEND ───────────────────────────────────
    print("\n=== 阶段2：DESCEND — 下降到抓取高度 ===")
    cube_pos = obs["cube_pos"].copy()   # 重新读取防止漂移
    tgt = cube_pos.copy()
    tgt[2] += GRASP_Z_OFFSET
    obs, _ = move_to(env, obs, tgt, max_steps=500, gripper_cmd=-1.0, **kw)

    # ── 阶段3：GRASP ─────────────────────────────────────
    obs = hold_gripper(env, obs, +1.0, GRASP_STEPS,
                       "=== 阶段3：GRASP — 关闭夹爪 ===", **kw)

    # ── 阶段4：TRANSPORT ─────────────────────────────────
    print("\n=== 阶段4：TRANSPORT — 提起并搬运到放置目标上方 ===")
    # 4a: 先垂直提起（防止碰桌面物体）
    lift_z = cube_pos[2] + GRASP_OFFSET_Z
    lift_pos = obs["robot0_eef_pos"].copy()
    lift_pos[2] = lift_z
    obs, _ = move_to(env, obs, lift_pos, max_steps=500, gripper_cmd=+1.0, **kw)
    # 4b: 水平移动到放置目标正上方（保持高度）
    transport_tgt = place_arr.copy()
    transport_tgt[2] = lift_z
    obs, _ = move_to(env, obs, transport_tgt, max_steps=500, gripper_cmd=+1.0, **kw)

    # ── 阶段5：DESCEND_PLACE ────────────────────────────
    print("\n=== 阶段5：DESCEND_PLACE — 下降到放置高度 ===")
    place_tgt = place_arr.copy()
    place_tgt[2] += GRASP_Z_OFFSET + 0.01
    obs, _ = move_to(env, obs, place_tgt, max_steps=500, gripper_cmd=+1.0, **kw)

    # ── 阶段6：RELEASE ───────────────────────────────────
    obs = hold_gripper(env, obs, -1.0, RELEASE_STEPS,
                       "=== 阶段6：RELEASE — 张开夹爪放下方块 ===", **kw)

    # 收尾：抬起末端
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

    # 展示 ~1 秒
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
    parser.add_argument("--robot", type=str, default="Panda",
                        help="机器人型号（Panda/UR5e/IIWA/Sawyer/xArm7 等）")
    parser.add_argument("--gripper", type=str, default="default",
                        help="夹爪型号（default=机器人默认 | PandaGripper | "
                             "Robotiq85Gripper | Robotiq140Gripper | RethinkGripper）")
    parser.add_argument("--place-pos", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="放置目标坐标（米），如 --place-pos 0.15 0.15 0.82")
    parser.add_argument("--nreset", type=int, default=3,
                        help="重复次数（默认 3）")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="速度缩放（默认 1.0，越大越快但精度下降）")
    parser.add_argument("--save-video", type=str, default=None, metavar="PATH",
                        help="视频保存路径（如 output.mp4），不填则不录制")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="录制相机名称（默认 agentview）")
    args = parser.parse_args()

    MAX_STEP = 0.05 * args.speed
    save_video = args.save_video is not None

    if save_video:
        macros.IMAGE_CONVENTION = "opencv"   # 图像坐标正向，imageio 直接使用

    print(f"\n{'='*58}")
    print(f"  脚本化抓取+放置 Demo（6 阶段）")
    print(f"  机器人 : {args.robot}   夹爪 : {args.gripper}")
    print(f"  速度   : {args.speed}x  重复 : {args.nreset} 次")
    if args.place_pos:
        print(f"  放置点 : {args.place_pos}")
    if save_video:
        print(f"  录制   : {args.save_video}  相机 : {args.camera}")
    print(f"{'='*58}")

    # ── 控制器：OSC_POSE（末端位移增量控制）─────────────
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
        has_offscreen_renderer=save_video,   # 录制时开启离屏渲染
        use_camera_obs=save_video,           # 录制时将相机图像纳入 obs
        use_object_obs=True,                 # 必须开启：获取 cube_pos
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
        hard_reset=False,                    # 软重置：保留含 marker 的模型
    )
    if save_video:
        make_kwargs.update(
            camera_names=args.camera,
            camera_heights=VIDEO_H,
            camera_widths=VIDEO_W,
        )
    # 功能1：独立夹爪选择（不传则使用机器人默认夹爪）
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
        place_pos = np.array([
            cube_pos_init[0] + 0.20,
            cube_pos_init[1],
            cube_pos_init[2],
        ])
        print(f"\n  自动设定放置目标: {place_pos.round(3)}")
        print(f"  （方块初始位置 +20cm X 方向，可用 --place-pos X Y Z 自定义）")

    # ── 功能2：添加放置目标可视化标记 ─────────────────────
    print("\n  添加绿色目标标记到场景...")
    add_placement_marker(env, place_pos)

    # ── 功能3：初始化视频录制 ─────────────────────────────
    video_writer = None
    if save_video:
        import imageio
        video_writer = imageio.get_writer(args.save_video, fps=MAX_FR)
        print(f"  ✓ 视频录制中 → {args.save_video}（{VIDEO_W}×{VIDEO_H}，相机: {args.camera}）")

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
