"""Gripper interaction demo.

This script illustrates the process of importing grippers into a scene and making it interact
with the objects with actuators. It also shows how to procedurally generate a scene with the
APIs of the MJCF utility functions.

Example:
    $ python run_gripper_test.py
"""

import xml.etree.ElementTree as ET

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.grippers import PandaGripper, RethinkGripper
from robosuite.models.objects import BoxObject
from robosuite.renderers.viewer import OpenCVViewer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from robosuite.utils.mjcf_utils import new_actuator, new_joint

if __name__ == "__main__":
    # ----------------------------------------------------------------
    # 【中文说明】此 Demo 展示「底层直接构建」方式，不使用 suite.make()
    # 适合需要完全控制场景XML细节的高级用法
    # 流程：手动创建 world → 添加 arena/gripper/object → 编译为 MuJoCo 模型 → 控制仿真
    # ----------------------------------------------------------------

    # 步骤1：创建空白 MuJoCo 世界容器（对应 <mujoco> 根标签）
    world = MujocoWorldBase()

    # 步骤2：添加桌面场景（table_arena.xml 提供桌子几何体）
    arena = TableArena(table_full_size=(0.4, 0.4, 0.05), table_offset=(0, 0, 1.1), has_legs=False)
    world.merge(arena)  # merge() 将 arena 的 XML 节点合并到 world

    # 步骤3：添加夹爪
    gripper = RethinkGripper()  # 加载 Rethink 夹爪的 MJCF 定义
    # 创建一个带滑动关节的虚拟 body，用于控制夹爪的上下位置
    gripper_body = ET.Element("body", name="gripper_base")
    gripper_body.set("pos", "0 0 1.3")
    gripper_body.set("quat", "0 0 1 0")  # 翻转z轴使夹爪朝下
    gripper_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 1", damping="50"))  # 竖直滑动关节
    world.worldbody.append(gripper_body)  # 添加到世界body
    world.merge(gripper, merge_body="gripper_base")  # 夹爪作为虚拟body的子节点
    # 为滑动关节创建位置控制器（kp=500 是位置增益）
    world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

    # 步骤4：添加目标抓取物体（红色小方块）
    mujoco_object = BoxObject(
        name="box", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001]
    ).get_obj()  # get_obj() 返回 XML Element
    mujoco_object.set("pos", "0 0 1.11")  # 放在桌面上
    world.worldbody.append(mujoco_object)

    # 添加坐标参考物体（绿色=x轴方向，蓝色=y轴方向，仅视觉，无物理碰撞）
    x_ref = BoxObject(
        name="x_ref", size=[0.01, 0.01, 0.01], rgba=[0, 1, 0, 1], obj_type="visual", joints=None
    ).get_obj()
    x_ref.set("pos", "0.2 0 1.105")
    world.worldbody.append(x_ref)
    y_ref = BoxObject(
        name="y_ref", size=[0.01, 0.01, 0.01], rgba=[0, 0, 1, 1], obj_type="visual", joints=None
    ).get_obj()
    y_ref.set("pos", "0 0.2 1.105")
    world.worldbody.append(y_ref)

    # 步骤5：将 XML 模型编译为 MuJoCo 模型并初始化仿真
    model = world.get_model(mode="mujoco")  # 将 XML 转为 MjModel 对象

    sim = MjSim(model)                          # 创建仿真实例
    viewer = OpenCVViewer(sim)                  # OpenCV 渲染窗口
    render_context = MjRenderContextOffscreen(sim, device_id=-1)  # 离屏渲染上下文
    sim.add_render_context(render_context)

    sim_state = sim.get_state()

    # for gravity correction
    gravity_corrected = ["gripper_z_joint"]
    _ref_joint_vel_indexes = [sim.model.get_joint_qvel_addr(x) for x in gravity_corrected]

    # Set gripper parameters
    gripper_z_id = sim.model.actuator_name2id("gripper_z")
    gripper_z_low = 0.07
    gripper_z_high = -0.02
    gripper_z_is_low = False

    gripper_jaw_ids = [sim.model.actuator_name2id(x) for x in gripper.actuators]
    gripper_open = [-0.0115, 0.0115]
    gripper_closed = [0.020833, -0.020833]
    gripper_is_closed = True

    # hardcode sequence for gripper looping trajectory
    seq = [(False, False), (True, False), (True, True), (False, True)]

    sim.set_state(sim_state)
    step = 0
    T = 500
    while True:
        if step % 100 == 0:
            print("step: {}".format(step))

            # Get contact information
            for contact in sim.data.contact[0 : sim.data.ncon]:

                geom_name1 = sim.model.geom_id2name(contact.geom1)
                geom_name2 = sim.model.geom_id2name(contact.geom2)
                if geom_name1 == "floor" and geom_name2 == "floor":
                    continue

                print("geom1: {}, geom2: {}".format(geom_name1, geom_name2))
                print("contact id {}".format(id(contact)))
                print("friction: {}".format(contact.friction))
                print("normal: {}".format(contact.frame[0:3]))

        # Iterate through gripping trajectory
        if step % T == 0:
            plan = seq[int(step / T) % len(seq)]
            gripper_z_is_low, gripper_is_closed = plan
            print("changing plan: gripper low: {}, gripper closed {}".format(gripper_z_is_low, gripper_is_closed))

        # Control gripper
        if gripper_z_is_low:
            sim.data.ctrl[gripper_z_id] = gripper_z_low
        else:
            sim.data.ctrl[gripper_z_id] = gripper_z_high
        if gripper_is_closed:
            sim.data.ctrl[gripper_jaw_ids] = gripper_closed
        else:
            sim.data.ctrl[gripper_jaw_ids] = gripper_open

        # Step through sim
        sim.step()
        sim.data.qfrc_applied[_ref_joint_vel_indexes] = sim.data.qfrc_bias[_ref_joint_vel_indexes]
        viewer.render()
        step += 1
