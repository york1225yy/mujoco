"""
【中文说明】
功能：夹爪选择演示 —— 依次遍历系统内所有夹爪，展示每种夹爪在 Lift 任务中的外观和行为
核心参数：gripper_types —— suite.make() 的参数，用字符串名称指定夹爪型号
可用夹爪列表来自 robosuite/models/grippers/__init__.py 中注册的 ALL_GRIPPERS
运行方式：python -m robosuite.demos.demo_gripper_selection
"""
import time

import numpy as np

import robosuite as suite
from robosuite import ALL_GRIPPERS  # 所有已注册夹爪的名称列表（约10款）

MAX_FR = 25  # 最大帧率限制

if __name__ == "__main__":

    # 遍历系统中所有注册的夹爪型号
    for gripper in ALL_GRIPPERS:

        print("Using gripper {}...".format(gripper))

        # 关键：通过 gripper_types 参数切换夹爪，其余配置固定（Panda + Lift任务）
        # 这说明「夹爪」是独立于「机械臂」的可替换模块
        env = suite.make(
            "Lift",
            robots="Panda",
            gripper_types=gripper,   # 指定夹爪型号（覆盖 Panda 的默认夹爪）
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=50,         # 较高频率使动画更流畅
            camera_names="frontview",
        )

        env.reset()

        # 获取动作空间边界（由夹爪自由度决定上下限）
        low, high = env.action_spec

        # 每款夹爪运行100步随机动作后切换下一款
        for t in range(100):
            start = time.time()
            env.render()
            action = np.random.uniform(low, high)  # 均匀分布随机动作（在合法范围内）
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

            elapsed = time.time() - start
            diff = 1 / MAX_FR - elapsed
            if diff > 0:
                time.sleep(diff)

        env.close()  # 关闭窗口，切换到下一个夹爪
