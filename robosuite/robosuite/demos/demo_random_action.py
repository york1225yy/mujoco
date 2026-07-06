# 【中文说明】
# 功能：随机动作演示 —— robosuite 最简单的入口示例
# 流程：终端交互选择「任务环境」和「机器人」→ suite.make() 创建环境 → 循环执行随机动作
# 核心理念：只需「选配置」，framework 负责所有物理仿真细节
# 运行方式：python -m robosuite.demos.demo_random_action
import time

from robosuite.robots import MobileRobot
from robosuite.utils.input_utils import *  # 导入交互菜单函数（choose_environment / choose_robots 等）

MAX_FR = 25  # 最大帧率限制（帧/秒），避免仿真运行过快

if __name__ == "__main__":

    # 用字典收集所有环境创建参数，最终传给 suite.make()
    options = {}

    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # 步骤1：终端菜单选择任务环境（Lift / PickPlace / Stack / Door 等）
    options["env_name"] = choose_environment()

    # 步骤2：根据环境类型选择对应机器人
    if "TwoArm" in options["env_name"]:
        # 双臂环境：先选配置（并排/对立/单机器人双臂）
        options["env_configuration"] = choose_multi_arm_config()

        if options["env_configuration"] == "single-robot":
            # 单机器人双臂配置（如 Baxter）
            options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True, exclude_single_arm=True)
        else:
            options["robots"] = []
            # 分别选择两个机器人
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=False, use_humanoids=True))
    elif "Humanoid" in options["env_name"]:
        # 人形机器人环境：只能选人形机器人（GR1 等）
        options["robots"] = choose_robots(use_humanoids=True)
    else:
        # 普通单臂环境：选择任意单臂机器人（Panda/UR5e/IIWA 等）
        options["robots"] = choose_robots(exclude_bimanual=False, use_humanoids=True)

    # 步骤3：suite.make() 是核心 —— 根据配置自动构建完整仿真场景
    # 内部完成：加载机器人MJCF → 构建场景XML → 初始化MuJoCo引擎 → 创建控制器
    env = suite.make(
        **options,
        has_renderer=True,           # 开启实时渲染窗口
        has_offscreen_renderer=False, # 不需要离屏渲染（无图像观测时关闭以节省资源）
        ignore_done=True,            # 忽略任务完成标志，持续运行
        use_camera_obs=False,        # 不使用相机图像作为观测（只用数值状态）
        control_freq=20,             # 控制频率：每秒执行20次动作
    )
    env.reset()                      # 初始化仿真状态，必须在第一次 step 前调用
    env.viewer.set_camera(camera_id=0)  # 设置默认相机视角
    for robot in env.robots:
        if isinstance(robot, MobileRobot):
            robot.enable_parts(legs=False, base=False)  # 移动机器人：仅控制手臂，禁用腿部和底座

    # 步骤4：主循环 —— 每步采样随机动作并执行
    for i in range(10000):
        start = time.time()
        # 从动作空间随机采样（高斯分布），维度由控制器类型决定（OSC_POSE=7维：dx/dy/dz/droll/dpitch/dyaw/gripper）
        action = np.random.randn(*env.action_spec[0].shape)
        obs, reward, done, _ = env.step(action)  # 执行动作，获取观测/奖励/终止信号
        env.render()  # 刷新渲染窗口

        # 帧率控制：若仿真比实时快，则休眠补齐
        elapsed = time.time() - start
        diff = 1 / MAX_FR - elapsed
        if diff > 0:
            time.sleep(diff)
