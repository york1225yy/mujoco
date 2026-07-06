"""
This script shows how to adapt an environment to be compatible
with the Gymnasium API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Basic Usage" section of the Gymnasium documentation

The following snippet was used to demo basic functionality.

    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            env.close()

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

# 【中文说明】
# 功能：Gymnasium 接口演示 —— 将 robosuite 包装成标准 Gym 接口
# 核心：GymWrapper 让 robosuite 环境兼容所有支持 Gymnasium API 的 RL 训练库
#       （如 Stable-Baselines3、CleanRL、RLlib 等）
# 包装后的接口变化：
#   reset() → (obs, info)          [标准 Gym 格式]
#   step()  → (obs, reward, terminated, truncated, info)
#   env.observation_space / env.action_space  [Box 空间，可直接用于神经网络]
# 运行方式：python -m robosuite.demos.demo_gym_functionality
import robosuite as suite
from robosuite.wrappers import GymWrapper  # 关键：Gymnasium 兼容包装器

if __name__ == "__main__":

    # GymWrapper 套在 suite.make() 外层，将 robosuite 接口转换为标准 Gym 格式
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="Sawyer",
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=True,
            reward_shaping=True,   # 使用密集奖励（更适合 RL 训练）
            control_freq=20,
        )
    )

    env.reset(seed=0)  # 支持设置随机种子（标准 Gym 特性）

    for i_episode in range(20):
        observation = env.reset()  # 返回 numpy 数组（已将观测字典展平为向量）
        for t in range(500):
            env.render()
            action = env.action_space.sample()  # 从动作空间均匀采样（Box空间）
            # 标准 Gym step 接口：terminated=任务完成，truncated=超时
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                break
        env.close()
