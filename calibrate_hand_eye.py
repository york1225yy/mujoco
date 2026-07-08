"""眼在手上 (Eye-in-Hand) 手眼标定 — 标定与验证脚本。

1. 采集 15 组 (R_gripper2base, t_gripper2base) 和 (R_target2cam, t_target2cam)
2. 手眼标定 (Tsai 方法) 求解 X = cam→gripper
3. 从 MuJoCo 模型树读取真值 (Ground Truth) 并对比误差

注意:
  - 标定方程: inv(T_g2b_{i+1}) @ T_g2b_i @ X = X @ T_t2c_{i+1} @ inv(T_t2c_i)
    解得 X = T_cam→gripper (相机在法兰坐标系下的位姿)
  - cv2.calibrateHandEye 在 OpenCV 5.x 中不可用, 使用手动 Tsai 实现
  - MuJoCo xmat/cam_xmat 为 C-order (行主序), 使用默认 reshape(3,3)
"""

import argparse
from pathlib import Path

import cv2
import cv2.aruco as aruco
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 常量 (与 collect_calibration_data.py 保持一致)
# ---------------------------------------------------------------------------
ARUCO_DICT = aruco.DICT_5X5_100
ARUCO_MARKER_ID = 23
BOARD_SIZE = 0.18
BOARD_Z_OFFSET = 0.55
W, H = 640, 480

_ARUCO_MARKER = aruco.generateImageMarker(
    aruco.getPredefinedDictionary(ARUCO_DICT), ARUCO_MARKER_ID, 512, borderBits=1)

_POSES = np.array([
    [-0.1376, 0.1242, 3.1416, -2.4457, 0.0, 1.1064, 1.5708],
    [0.0447, 0.1176, 3.1416, -2.3521, 0.0, 0.9065, 1.5708],
    [0.0370, 0.0804, 3.1416, -2.2259, 0.0, 0.8281, 1.5708],
    [-0.1512, 0.2599, 3.1416, -2.4552, 0.0, 1.1237, 1.5708],
    [0.1758, 0.4197, 3.1416, -2.2298, 0.0, 1.1287, 1.5708],
    [0.1085, 0.0914, 3.1416, -2.3255, 0.0, 0.8063, 1.5708],
    [0.1043, 0.2863, 3.1416, -2.1605, 0.0, 0.9574, 1.5708],
    [0.1630, 0.1615, 3.1416, -2.3048, 0.0, 1.0622, 1.5708],
    [-0.1085, 0.0926, 3.1416, -2.3530, 0.0, 0.8244, 1.5708],
    [0.1719, 0.3850, 3.1416, -2.2156, 0.0, 1.1085, 1.5708],
    [0.0075, 0.3430, 3.1416, -2.3235, 0.0, 1.1486, 1.5708],
    [0.1850, 0.1625, 3.1416, -2.2700, 0.0, 0.8803, 1.5708],
    [-0.0529, 0.3147, 3.1416, -2.2155, 0.0, 0.9742, 1.5708],
    [0.0048, 0.1524, 3.1416, -2.2109, 0.0, 0.8297, 1.5708],
    [0.0764, 0.2165, 3.1416, -2.0942, 0.0, 0.8149, 1.5708],
])

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _rot(body_id, data):
    return data.xmat[body_id].reshape(3, 3)


def _cam_rot(cam_id, data):
    return data.cam_xmat[cam_id].reshape(3, 3)


def compute_intrinsics(fovy_deg: float, w: int, h: int) -> np.ndarray:
    f = (h / 2) / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def get_gripper2base(model, data):

    t_g = data.xpos[model.body("bracelet_link").id]
    t_b = data.xpos[model.body("base_link").id]
    R_b_inv = _rot(model.body("base_link").id, data).T
    return R_b_inv @ _rot(model.body("bracelet_link").id, data), R_b_inv @ (t_g - t_b)


def project_and_overlay(model, data, cam_name, K):
    cid = model.camera(cam_name).id
    bid = model.body("calibration_board_body").id
    R_cw = _cam_rot(cid, data)
    t_cw = data.cam_xpos[cid]
    R_bw = _rot(bid, data)
    t_bw = data.xpos[bid]

    half = BOARD_SIZE / 2
    corners_local = np.array(
        [[0, -half, half], [0, half, half], [0, half, -half], [0, -half, -half]],
        dtype=np.float64)
    corners_w = (R_bw @ corners_local.T).T + t_bw + [0, 0, BOARD_Z_OFFSET]

    F = np.diag([1.0, -1.0, -1.0])
    corners_cv = (F @ R_cw.T @ (corners_w - t_cw).T).T
    if np.any(corners_cv[:, 2] <= 0):
        return None, None

    corners_pix = (K @ corners_cv.T).T
    corners_pix = corners_pix[:, :2] / corners_pix[:, 2:3]

    w_ = max(np.linalg.norm(corners_pix[1] - corners_pix[0]),
             np.linalg.norm(corners_pix[2] - corners_pix[3]))
    h_ = max(np.linalg.norm(corners_pix[3] - corners_pix[0]),
             np.linalg.norm(corners_pix[2] - corners_pix[1]))
    x0, y0 = corners_pix.min(axis=0)
    x1, y1 = corners_pix.max(axis=0)
    if w_ < 30 or h_ < 30 or x0 < -30 or y0 < -30 or x1 > W + 30 or y1 > H + 30:
        return None, None

    v01 = corners_pix[1] - corners_pix[0]
    v12 = corners_pix[2] - corners_pix[1]
    if v01[0] * v12[1] - v01[1] * v12[0] < 0:
        corners_pix = corners_pix[::-1]

    M = cv2.getPerspectiveTransform(
        np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype=np.float32),
        corners_pix.astype(np.float32))
    warped = cv2.warpPerspective(_ARUCO_MARKER, M, (W, H), flags=cv2.INTER_LINEAR)
    inside = cv2.warpPerspective(
        np.ones((512, 512), dtype=np.uint8) * 255, M, (W, H),
        flags=cv2.INTER_LINEAR) > 0

    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    img[inside] = np.stack([warped[inside]] * 3, axis=-1)
    return img, corners_pix


def detect_aruco(img: np.ndarray, K: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.markerBorderBits = 1
    corners, ids, _ = aruco.ArucoDetector(
        aruco.getPredefinedDictionary(ARUCO_DICT), params).detectMarkers(gray)

    if ids is None or ARUCO_MARKER_ID not in ids:
        return None, None

    idx = int(np.where(ids.flatten() == ARUCO_MARKER_ID)[0][0])
    half = BOARD_SIZE / 2
    # 标定板角点在 board 中心坐标系 (原点已含 BOARD_Z_OFFSET 偏移, 无需再加)
    obj = np.array([[0, -half, half], [0, half, half],
                    [0, half, -half], [0, -half, -half]], dtype=np.float64)
    _, rvec, tvec = cv2.solvePnP(obj, corners[idx].reshape(4, 2).astype(np.float64), K, None)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


# ---------------------------------------------------------------------------
# 真值计算: 从 MuJoCo 模型树直接读取相机→法兰的相对位姿
# ---------------------------------------------------------------------------

def get_ground_truth_cam2gripper(model, data):
    gid = model.body("bracelet_link").id
    cam_name = "d435i_rgb_camera"
    cid = model.camera(cam_name).id

    R_g = _rot(gid, data)            # gripper → world
    t_g = data.xpos[gid]
    R_c = _cam_rot(cid, data)         # camera sensor → world (MuJoCo z-backward)
    t_c = data.cam_xpos[cid]

    # T_cam(OpenCV)→gripper = T_world→gripper⁻¹ @ T_world→cam(OpenCV)
    # R_c2g = R_g.T @ R_c @ F   (F 将 MuJoCo 相机坐标系转为 OpenCV 约定 z-forward)
    # t_c2g = R_g.T @ (t_c - t_g)
    F = np.diag([1.0, -1.0, -1.0])
    R_cam2gripper = R_g.T @ R_c @ F
    t_cam2gripper = R_g.T @ (t_c - t_g)

    return R_cam2gripper, t_cam2gripper


# ---------------------------------------------------------------------------
# 手眼标定: Tsai 方法手动实现 (OpenCV 5.x 移除了 Python 绑定)
# ---------------------------------------------------------------------------

def _quat_from_R(R):
    q = np.zeros(4)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def _R_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def calibrate_hand_eye_tsai(R_g2b, t_g2b, R_t2c, t_t2c):
    """手眼标定 (四元数法): AX = XB. 等价于 OpenCV calibHandEye (Tsai).
    让AI写的

    参数:
      R_g2b: list of (3,3) — gripper→base 旋转矩阵 (T_gripper→base)
      t_g2b: list of (3,1) — gripper→base 平移向量
      R_t2c: list of (3,3) — target→cam 旋转矩阵 (T_target→camera)
      t_t2c: list of (3,1) — target→cam 平移向量

    返回:
      R_cam2gripper: (3,3) — 相机坐标系→法兰坐标系的旋转矩阵
      t_cam2gripper: (3,1) — 相机在法兰坐标系下的位置 (m)

    推导说明:
      眼在手上, 标定板固定在世界系中 (base 系).
      设 T_g2b = T_gripper→base, T_t2c = T_target→camera, X = T_cam→gripper.

      标定板上一点 p_target: 它在 base 系下的坐标是固定的.
        p_base = T_g2b @ X @ T_t2c @ p_target  (对任意帧不变)

      对帧 i 和 j:
        T_g2b(i) @ X @ T_t2c(i) = T_g2b(j) @ X @ T_t2c(j)
        → T_g2b(j)⁻¹ @ T_g2b(i) @ X = X @ T_t2c(j) @ T_t2c(i)⁻¹
        → A @ X = X @ B

      其中:
        A = inv(T_g2b(j)) @ T_g2b(i)      (法兰在帧 i→j 之间的运动)
        B = T_t2c(j) @ inv(T_t2c(i))       (相机在帧 i→j 之间的运动)
        X = T_cam→gripper                  (待求解的标定矩阵)

      B 的约定说明:
        OpenCV 文档中 B = inv(T_t2c_i) @ T_t2c_{i+1}, 但经过严格验证,
        正确的 B 应为 T_t2c_j @ inv(T_t2c_i). 因为 T_t2c = target→camera,
        用 T_t2c_j @ inv(T_t2c_i) 消去 target, 正好得到 cam_i→cam_j.

      旋转求解: 四元数零空间法.
        R_A @ R_X = R_X @ R_B  →  用四元数表示为:
        q_A ⊗ q_X = q_X ⊗ q_B  →  (M_L(q_A) - M_R(q_B)) @ q_X = 0
        对每对 (i,j) 堆叠得到 H @ q_X = 0, 用 SVD 求零空间向量.

      退化处理:
        当所有法兰运动 R_A 的旋转轴在同一平面内时, 零空间维度 > 1,
        x 轴旋转分量无法从方程中确定. 此时在零空间内扫描所有可能的
        旋转, 选择使 T_g2b @ X @ T_t2c 跨帧最一致 (方差最小) 的解.
    """
    n = len(R_g2b)

    # 使用所有位姿对 (包含非相邻帧), 提供更多约束
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pairs)

    R_A_list, t_A_list = [], []
    R_B_list, t_B_list = [], []

    for i, j in pairs:
        # A = inv(T_g2b(j)) @ T_g2b(i)
        R_A = R_g2b[j].T @ R_g2b[i]
        t_A = R_g2b[j].T @ (t_g2b[i].ravel() - t_g2b[j].ravel())
        R_A_list.append(R_A)
        t_A_list.append(t_A)

        # B = T_t2c(j) @ inv(T_t2c(i))
        R_B = R_t2c[j] @ R_t2c[i].T
        t_B = t_t2c[j].ravel() - R_B @ t_t2c[i].ravel()
        R_B_list.append(R_B)
        t_B_list.append(t_B)

    H = np.zeros((4 * m, 4))
    for k, (R_A, R_B) in enumerate(zip(R_A_list, R_B_list)):
        q_A = _quat_from_R(R_A)
        q_B = _quat_from_R(R_B)

        wa, xa, ya, za = q_A
        M_L = np.array([
            [ wa, -xa, -ya, -za],
            [ xa,  wa, -za,  ya],
            [ ya,  za,  wa, -xa],
            [ za, -ya,  xa,  wa],
        ])

        wb, xb, yb, zb = q_B

        M_R = np.array([
            [ wb, -xb, -yb, -zb],
            [ xb,  wb,  zb, -yb],
            [ yb, -zb,  wb,  xb],
            [ zb,  yb, -xb,  wb],
        ])

        H[4*k:4*k+4, :] = M_L - M_R


    _, S, Vt = np.linalg.svd(H.astype(np.float64))


    degenerated = (S[-1] / S[0]) < 0.1


    C_all = np.vstack([R_A - np.eye(3) for R_A in R_A_list])
    _, S_C, _ = np.linalg.svd(C_all)
    cond = S_C[0] / S_C[-1] if S_C[-1] > 1e-12 else np.inf

    print(f"\n  H 奇异值: {S}")
    print(f"  min/max 奇异值比: {S[-1]/S[0]:.4f} "
          f"({'< 0.1, 可能退化' if degenerated else '>= 0.1, 良态'})")

    if not degenerated:
        q_X = Vt[-1, :]
        q_X = q_X / np.linalg.norm(q_X)
        R_X = _R_from_quat(q_X)

        d_all = np.concatenate([(R_X @ t_B - t_A).ravel() for t_A, t_B in zip(t_A_list, t_B_list)])
        t_X, _, _, _ = np.linalg.lstsq(C_all, d_all, rcond=None)
        t_X = t_X.reshape(3, 1)
    else:
        print(f"  退化处理: 扫描零空间寻找最优解...")
        v1 = Vt[-2]
        v2 = Vt[-1]

        best_cost = np.inf
        best_R_X = None
        best_t_X = None
        best_theta = None

        n_scan = 720  # 扫描分辨率
        for k in range(n_scan):
            theta = k * np.pi / n_scan  # [0, π)
            q_X = np.cos(theta) * v1 + np.sin(theta) * v2
            q_X = q_X / np.linalg.norm(q_X)
            R_X_cand = _R_from_quat(q_X)

            # 对当前 R_X 求解平移
            d_all = np.concatenate([(R_X_cand @ t_B - t_A).ravel()
                                    for t_A, t_B in zip(t_A_list, t_B_list)])
            t_X_cand, _, _, _ = np.linalg.lstsq(C_all, d_all, rcond=None)
            t_X_cand = t_X_cand.reshape(3)

            # 评估代价: T_g2b(i) @ X @ T_t2c(i) 的平移部分跨帧方差
            # 正确的 X 应使该值为常数 (标定板在 base 系下不动)
            p_base_list = []
            for i in range(n):
                p_base = (R_g2b[i] @ (R_X_cand @ t_t2c[i].ravel() + t_X_cand)
                          + t_g2b[i].ravel())
                p_base_list.append(p_base)
            p_base_arr = np.array(p_base_list)  # (n, 3)
            cost = np.mean(np.std(p_base_arr, axis=0))

            if cost < best_cost:
                best_cost = cost
                best_R_X = R_X_cand
                best_t_X = t_X_cand.reshape(3, 1)
                best_theta = theta

        R_X = best_R_X
        t_X = best_t_X
        q_X = np.cos(best_theta) * v1 + np.sin(best_theta) * v2
        q_X = q_X / np.linalg.norm(q_X)
        print(f"  最优 theta = {np.rad2deg(best_theta):.1f}°")
        print(f"  最优 q_X = [{q_X[0]:.6f}, {q_X[1]:.6f}, {q_X[2]:.6f}, {q_X[3]:.6f}]")
        print(f"  跨帧一致性 cost = {best_cost:.6f} m (越小越好)")

    print(f"  约束矩阵 C 的奇异值: {S_C}")
    print(f"  条件数 (cond): {cond:.2f} (越小越好, < 100 说明平移解可靠)")

    return R_X, t_X


# ---------------------------------------------------------------------------
# 误差计算
# ---------------------------------------------------------------------------

def rotation_error_deg(R_est, R_gt):
    R_err = R_est @ R_gt.T
    trace_val = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_val)))


def translation_error_mm(t_est, t_gt):
    return float(np.linalg.norm(t_est.ravel() - t_gt.ravel())) * 1000.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="眼在手上手眼标定与验证")
    parser.add_argument("--scene", type=Path,
                        default=Path(__file__).resolve().parent / "scene.xml")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # 相机内参粗略计算
    fovy = model.cam_fovy[model.camera("d435i_rgb_camera").id]
    K = compute_intrinsics(fovy, W, H)

    # ---- 第一步: 数据采集 ----
    print("=" * 60)
    print("Step 1: 采集 15 组位姿数据")
    print("=" * 60)

    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []
    R_t2c_gt_list, t_t2c_gt_list = [], []  # GT 对比用

    for i, qpos in enumerate(_POSES):
        data.qpos[:7] = qpos
        mujoco.mj_forward(model, data)

        R_g2b, t_g2b = get_gripper2base(model, data)
        composited, corners_pix = project_and_overlay(model, data, "d435i_rgb_camera", K)

        if corners_pix is not None:
            R_t2c, t_t2c = detect_aruco(composited, K)
            flag = "检测" if R_t2c is not None else "真值"
        else:
            R_t2c, t_t2c = None, None
            flag = "真值"

        if R_t2c is None:
            R_t2c, t_t2c = compute_target2cam(model, data, "d435i_rgb_camera")

        # GT target2cam for comparison
        R_t2c_gt, t_t2c_gt = compute_target2cam(model, data, "d435i_rgb_camera")
        R_t2c_gt_list.append(R_t2c_gt)
        t_t2c_gt_list.append(t_t2c_gt.reshape(3, 1))

        R_g2b_list.append(R_g2b)
        t_g2b_list.append(t_g2b.reshape(3, 1))
        R_t2c_list.append(R_t2c)
        t_t2c_list.append(t_t2c.reshape(3, 1))

        # ArUco vs GT 差异
        dz = (t_t2c[2] - t_t2c_gt[2]) * 1000
        flag_text = "(检测)" if flag == "检测" else "(真值-回退)"
        print(f"  [{i+1:2d}/15] {flag_text}  t_g2b=({t_g2b[0]:+.3f},{t_g2b[1]:+.3f},{t_g2b[2]:+.3f})  "
              f"t_t2c_z={t_t2c[2]:.3f} (GT={t_t2c_gt[2]:.3f}, dZ={dz:+.1f}mm)")

    # 对比 ArUco 与 GT 的 t_t2c 差异
    dz_all = np.array([(t_t2c_list[i][2] - t_t2c_gt_list[i][2]) * 1000 for i in range(len(_POSES))])
    print(f"\n  ArUco t_t2c Z 误差: mean={np.mean(dz_all):.1f}mm, std={np.std(dz_all):.1f}mm")

    # ---- 第二步: 手眼标定 ----
    print("\n" + "=" * 60)
    print("Step 2: 手眼标定 (Tsai 方法)")
    print("=" * 60)

    print("\n手眼标定方程 AX = XB:")
    print("  A_i: 法兰在帧 i→j 的运动 (= inv(T_g2b_j) @ T_g2b_i)")
    print("  B_i: 相机在帧 i→j 的运动 (= T_t2c_j @ inv(T_t2c_i))    — 经过严格验证的B约定")
    print("  X:   待求解的 cam→gripper 位姿 (相机在法兰坐标系下的位姿)")
    print("\n推导:")
    print("  标定板固定在世界系 → T_g2b @ X @ T_t2c = 常量")
    print("  T_g2b(i) @ X @ T_t2c(i) = T_g2b(j) @ X @ T_t2c(j)")
    print("  → inv(T_g2b_j) @ T_g2b(i) @ X = X @ T_t2c(j) @ inv(T_t2c_i)")
    print("\n注意:")
    print("  本实现使用所有位姿对 (C(15,2)=105 对), 提供比相邻帧更多的约束.")
    print("  旋转使用四元数零空间法 (SVD), 平移使用最小二乘法.")
    print("  当机器人运动轴多样性不足时, 旋转解可能退化.")

    R_cam2gripper, t_cam2gripper = calibrate_hand_eye_tsai(
        R_g2b_list, t_g2b_list,
        R_t2c_list, t_t2c_list,
    )

    print(f"\n标定结果 — 相机在法兰坐标系下的位姿 (X = cam→gripper):")
    print(f"  旋转矩阵 R_cam2gripper:")
    for row in R_cam2gripper:
        print(f"    [{row[0]:+12.8f}  {row[1]:+12.8f}  {row[2]:+12.8f}]")
    print(f"  平移向量 t_cam2gripper (相机在法兰坐标系下的位置, m):")
    print(f"    [{t_cam2gripper[0][0]:+12.8f}  {t_cam2gripper[1][0]:+12.8f}  {t_cam2gripper[2][0]:+12.8f}]")

    # ---- 对比: 用 MuJoCo GT 数据标定 (验证求解器正确性) ----
    print("\n" + "-" * 40)
    print("[对比] 使用 MuJoCo GT 数据进行标定 (无 ArUco 检测噪声):")
    R_gt_calib, t_gt_calib = calibrate_hand_eye_tsai(
        R_g2b_list, t_g2b_list,
        R_t2c_gt_list, t_t2c_gt_list,
    )
    print(f"  GT 标定结果 t_cam2gripper:")
    print(f"    [{t_gt_calib[0][0]:+12.8f}  {t_gt_calib[1][0]:+12.8f}  {t_gt_calib[2][0]:+12.8f}]")

    # ---- 第三步: 真值对比 ----
    print("\n" + "=" * 60)
    print("Step 3: 真值对比")
    print("=" * 60)

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    R_gt, t_gt = get_ground_truth_cam2gripper(model, data)

    print(f"\nGround Truth (从 MuJoCo 模型树读取):")
    print(f"  R_cam2gripper (真值):")
    for row in R_gt:
        print(f"    [{row[0]:+12.8f}  {row[1]:+12.8f}  {row[2]:+12.8f}]")
    print(f"  t_cam2gripper (真值, 相机在法兰坐标系下的位置, m):")
    print(f"    [{t_gt[0]:+12.8f}  {t_gt[1]:+12.8f}  {t_gt[2]:+12.8f}]")

    rot_err = rotation_error_deg(R_cam2gripper, R_gt)
    trans_err = translation_error_mm(t_cam2gripper, t_gt)

    print(f"\n误差分析:")
    print(f"  平移误差: {trans_err:.3f} mm")
    print(f"  旋转误差: {rot_err:.4f} °")

    t_est = t_cam2gripper.ravel()
    t_truth = t_gt.ravel()
    print(f"\n标定结果 vs 真值 (ArUco 检测数据):")
    print(f"  平移误差: {trans_err:.3f} mm (总)")
    print(f"  旋转误差: {rot_err:.4f} °")
    print(f"  各轴误差: X={abs(t_est[0]-t_truth[0])*1000:.3f} mm, "
          f"Y={abs(t_est[1]-t_truth[1])*1000:.3f} mm, "
          f"Z={abs(t_est[2]-t_truth[2])*1000:.3f} mm")

    # GT 数据标定误差 (用于验证求解器)
    rot_err_gt = rotation_error_deg(R_gt_calib, R_gt)
    trans_err_gt = translation_error_mm(t_gt_calib, t_gt)
    t_gt_est = t_gt_calib.ravel()
    print(f"\nGT 数据标定 vs 真值 (验证求解器):")
    print(f"  平移误差: {trans_err_gt:.3f} mm (总)")
    print(f"  旋转误差: {rot_err_gt:.4f} °")
    print(f"  各轴误差: X={abs(t_gt_est[0]-t_truth[0])*1000:.3f} mm, "
          f"Y={abs(t_gt_est[1]-t_truth[1])*1000:.3f} mm, "
          f"Z={abs(t_gt_est[2]-t_truth[2])*1000:.3f} mm")


def compute_target2cam(model, data, cam_name):
    """MuJoCo 真值 → OpenCV 坐标系 target→cam 位姿 (回退用)."""
    cid = model.camera(cam_name).id
    bid = model.body("calibration_board_body").id
    R_cw = _cam_rot(cid, data)
    t_cw = data.cam_xpos[cid]
    R_bw = _rot(bid, data)
    t_bw = data.xpos[bid] + R_bw @ np.array([0.0, 0.0, BOARD_Z_OFFSET])
    F = np.diag([1.0, -1.0, -1.0])
    return F @ R_cw.T @ R_bw, F @ R_cw.T @ (t_bw - t_cw)


if __name__ == "__main__":
    main()
