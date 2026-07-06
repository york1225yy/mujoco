"""
Script to showcase domain randomization functionality.
"""

# 【中文说明】
# 功能：域随机化演示 —— 每次 reset 时随机改变场景外观（纹理/光照/相机/物理参数）
# 目的：提升视觉策略从仿真到真实世界的迁移能力（Sim-to-Real Transfer）
# 随机化选项：
#   randomize_color=True    → 随机化物体/机器人颜色和材质
#   randomize_camera=True   → 随机化相机位置和视角
#   randomize_lighting=True → 随机化光源方向和强度
#   randomize_dynamics=True → 随机化物理参数（摩擦/质量等）
# 运行方式：python -m robosuite.demos.demo_domain_randomization
import time

import robosuite.macros as macros
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper  # 域随机化包装器

# 启用实例随机化：整组几何体（geom group）作为整体随机化，视觉上更自然
macros.USING_INSTANCE_RANDOMIZATION = True

if __name__ == "__main__":
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))
    # If a humanoid environment has been chosen, choose humanoid robots
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        hard_reset=False,  # TODO: Not setting this flag to False brings up a segfault on macos or glfw error on linux
    )
    env = DomainRandomizationWrapper(
        env,
        randomize_color=False,  # randomize_color currently only works for mujoco==3.1.1
        randomize_camera=False,  # less jarring when visualizing
        randomize_dynamics=False,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    max_frame_rate = 20  # Set the desired maximum frame rate

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(100):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1 / max_frame_rate)
