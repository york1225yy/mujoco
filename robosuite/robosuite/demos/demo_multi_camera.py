# 【中文说明】
# 功能：多相机视角演示 —— 同时显示多个相机的实时画面
# 与 demo_random_action 几乎相同，唯一区别是调用了 set_camera(camera_name=[...]) 传入列表
# 可用相机名："agentview"（正视）/ "birdview"（俯视）/ "frontview" / "sideview"
#             / "robot0_eye_in_hand"（腕部相机）
# 运行方式：python -m robosuite.demos.demo_multi_camera
import time

from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *

MAX_FR = 25

if __name__ == "__main__":

    options = {}
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # 步骤1：选择环境和机器人（与 demo_random_action 完全相同）
    options["env_name"] = choose_environment()

    if "TwoArm" in options["env_name"]:
        options["env_configuration"] = choose_multi_arm_config()
        if options["env_configuration"] == "single-robot":
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        else:
            options["robots"] = []
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
    elif "Humanoid" in options["env_name"]:
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    # 步骤2：创建环境（renderer="mujoco" 支持多视角分屏）
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mujoco",  # 使用 MuJoCo 原生渲染器（支持多相机分屏）
    )
    env.reset()

    # 步骤3：关键区别 —— 传入相机名称列表实现多视角同时显示
    camera_name = ["agentview", "birdview"]  # 可添加更多："frontview", "robot0_eye_in_hand" 等
    env.viewer.set_camera(camera_name=camera_name)
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)

    # 步骤4：主循环（与 demo_random_action 完全相同）
    for i in range(10000):
        start = time.time()
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, _ = env.step(action)
        env.render()

        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
