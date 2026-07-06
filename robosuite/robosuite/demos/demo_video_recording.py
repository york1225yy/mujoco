"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

# 【中文说明】
# 功能：视频录制演示 —— 使用离屏渲染将仿真过程保存为 mp4 视频
# 关键点：
#   - has_renderer=False + has_offscreen_renderer 由 use_camera_obs=True 隐式开启
#   - 相机图像通过 obs["相机名_image"] 访问，shape=(H, W, 3) uint8
#   - imageio 库负责将帧序列写入视频文件
# 运行方式：python -m robosuite.demos.demo_video_recording --environment Lift --robots Panda
import argparse

import imageio           # 外部库：帧序列 → 视频文件
import numpy as np

import robosuite.macros as macros  # 全局配置（图像格式/仿真步长等）
from robosuite import make

# 设置图像坐标约定为 OpenCV 格式（y轴向下），使 imageio 保存的视频方向正确
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    # 命令行参数：支持灵活指定环境/机器人/相机/输出路径/分辨率
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stack")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--camera", type=str, default="agentview")  # 录制的相机视角
    parser.add_argument("--video_path", type=str, default="video.mp4")  # 输出视频路径
    parser.add_argument("--timesteps", type=int, default=500)           # 录制总帧数
    parser.add_argument("--height", type=int, default=512)              # 视频分辨率高
    parser.add_argument("--width", type=int, default=512)               # 视频分辨率宽
    parser.add_argument("--skip_frame", type=int, default=1)            # 每N帧保存一帧（降采样）
    args = parser.parse_args()

    # 创建环境：关闭实时窗口，开启相机图像观测（自动激活离屏渲染）
    env = make(
        args.environment,
        args.robots,
        has_renderer=False,          # 关闭实时显示窗口（无需人看）
        ignore_done=True,
        use_camera_obs=True,         # 开启相机观测（obs 中包含图像数组）
        use_object_obs=False,        # 关闭物体状态观测（节省计算）
        camera_names=args.camera,    # 指定录制的相机
        camera_heights=args.height,
        camera_widths=args.width,
    )

    obs = env.reset()
    ndim = env.action_dim

    # 创建视频写入器（fps=20 对应仿真实时速度）
    writer = imageio.get_writer(args.video_path, fps=20)

    frames = []
    for i in range(args.timesteps):
        action = 0.5 * np.random.randn(ndim)  # 小幅随机动作（0.5倍缩放避免过激烈运动）
        obs, reward, done, info = env.step(action)

        # 每 skip_frame 步保存一帧（默认每步都保存）
        if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"]  # 从观测字典中取相机图像，shape=(H,W,3)
            writer.append_data(frame)
            print("Saving frame #{}".format(i))

        if done:
            break

    writer.close()  # 必须调用，否则视频文件不完整
    print(f"Video saved to {args.video_path}")
