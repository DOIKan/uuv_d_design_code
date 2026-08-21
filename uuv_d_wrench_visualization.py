import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

# ==========================================
# 1. 物理パラメータ・構成設定
# ==========================================
# 7リンクのパラメータ: 長さ[m], 質量[kg], 体積[m^3]
link_lengths = [0.15, 0.05, 0.40, 0.35, 0.40, 0.05, 0.15]
link_masses  = [0.3, 0.3, 0.2, 2.5, 0.3, 0.3, 0.3]
link_volumes = [0.0008, 0.0005, 0.0005, 0.0030, 0.0005, 0.0005, 0.0008] 

# 6関節のパラメータ (順序: P, R, P, P, R, P)
joint_types = ['P', 'R', 'P', 'P', 'R', 'P']
actuator_masses = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] 
joint_limits = [(-135, 135), (-180, 180), (-90, 90), (-90, 90), (-180, 180), (-135, 135)] 

# ★ 変更点：スラスターパラメータ (8個・ベクター配置)
num_thrusters = 8
max_thrust = 15.0   # 各スラスターの最大推力 [N]
sigma = 0.0036      # ドラグモーメント係数
# プロペラの回転方向 (反トルク相殺用)
prop_spin = [1, -1, -1, 1, -1, 1, 1, -1] 

# すべて4番目のリンク(インデックス3)に属する
belonging_links = [3] * num_thrusters 

# 環境定数
rho_water = 1000.0  
g_const = 9.81      

# 回転行列
def rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rot_x(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_rpy(r, p, y):
    return rot_z(y) @ rot_y(p) @ rot_x(r)

# ==========================================
# 2. 運動学 & 物理特性計算コア
# ==========================================
def update_kinematics(joint_angles, base_rpy):
    tmp_link_frames = []
    tmp_joint_positions = []
    tmp_link_centers = []
    
    R = np.eye(3)
    P = np.array([0.0, 0.0, 0.0])
    
    for i in range(7):
        tmp_link_frames.append((R.copy(), P.copy()))
        P_center = P + R @ np.array([link_lengths[i]/2.0, 0, 0])
        tmp_link_centers.append(P_center)
        
        if i < 6:
            P_joint = P + R @ np.array([link_lengths[i], 0, 0])
            tmp_joint_positions.append(P_joint)
            
            theta = joint_angles[i]
            if joint_types[i] == 'P': R_j = rot_y(theta)
            elif joint_types[i] == 'R': R_j = rot_x(theta)
            else: R_j = rot_z(theta)
            R = R @ R_j
            P = P_joint.copy()
            
    R_link4_tmp, P_link4_tmp = tmp_link_frames[3]
    R_base_desired = rot_rpy(base_rpy[0], base_rpy[1], base_rpy[2])
    
    R_global = R_base_desired @ R_link4_tmp.T
    P_global = - R_global @ P_link4_tmp
    
    link_frames = []
    link_centers = []
    joint_positions = []
    
    for i in range(7):
        R_orig, P_orig = tmp_link_frames[i]
        R_new = R_global @ R_orig
        P_new = P_global + R_global @ P_orig
        link_frames.append((R_new, P_new))
        link_centers.append(P_global + R_global @ tmp_link_centers[i])
        
    for i in range(6):
        joint_positions.append(P_global + R_global @ tmp_joint_positions[i])

    total_mass = sum(link_masses) + sum(actuator_masses)
    com_num = np.zeros(3)
    for i in range(7): com_num += link_masses[i] * link_centers[i]
    for j in range(6): com_num += actuator_masses[j] * joint_positions[j]
    r_CoM = com_num / total_mass
    
    total_volume = sum(link_volumes)
    cob_num = np.zeros(3)
    for i in range(7): cob_num += link_volumes[i] * link_centers[i]
    r_CoB = cob_num / total_volume
    
    F_g = np.array([0, 0, -total_mass * g_const])
    F_b = np.array([0, 0, rho_water * g_const * total_volume])
    tau_b = np.cross(r_CoB - r_CoM, F_b)
    W_env = np.concatenate([F_g + F_b, tau_b])
    
    # ★ 変更点：4番目のリンク(胴体)の両端・ベクター配置計算
    thruster_positions = []
    thruster_directions_global = []
    
    R_mount = 0.02  # 取り付けの半径オフセット[m]
    L4 = link_lengths[3]
    
    # Link4のローカル座標系における位置 (始端4つ, 終端4つ)
    loc_positions = [
        np.array([0,  R_mount,  R_mount]),
        np.array([0, -R_mount,  R_mount]),
        np.array([0,  R_mount, -R_mount]),
        np.array([0, -R_mount, -R_mount]),
        np.array([L4,  R_mount,  R_mount]),
        np.array([L4, -R_mount,  R_mount]),
        np.array([L4,  R_mount, -R_mount]),
        np.array([L4, -R_mount, -R_mount])
    ]
    
    # Link4のローカル座標系における推力方向 (ベクター配置: 斜め45度方向など)
    loc_directions = [
        np.array([-1,  1,  1]) / np.sqrt(3),
        np.array([-1, -1,  1]) / np.sqrt(3),
        np.array([-1,  1, -1]) / np.sqrt(3),
        np.array([-1, -1, -1]) / np.sqrt(3),
        np.array([ 1,  1,  1]) / np.sqrt(3),
        np.array([ 1, -1,  1]) / np.sqrt(3),
        np.array([ 1,  1, -1]) / np.sqrt(3),
        np.array([ 1, -1, -1]) / np.sqrt(3)
    ]
    
    for k in range(num_thrusters):
        l_idx = belonging_links[k]
        R_l, P_l = link_frames[l_idx]
        
        pos_global = P_l + R_l @ loc_positions[k]
        dir_global = R_l @ loc_directions[k]
        
        thruster_positions.append(pos_global)
        thruster_directions_global.append(dir_global)
        
    return link_frames, joint_positions, r_CoM, r_CoB, W_env, np.array(thruster_positions), np.array(thruster_directions_global)

def compute_torque_space(r_CoM, thr_pos, thr_dir, W_env):
    A_eq = np.zeros((3, num_thrusters))
    M_matrix = np.zeros((3, num_thrusters))
    bounds = [(-max_thrust, max_thrust) for _ in range(num_thrusters)]
    
    for k in range(num_thrusters):
        p_vec = thr_pos[k] - r_CoM
        F_unit = thr_dir[k]
        
        # モーメントの計算 (位置ベクトル × 力ベクトル + ドラッグトルク)
        M_unit = np.cross(p_vec, F_unit) + sigma * prop_spin[k] * F_unit
        
        A_eq[:, k] = F_unit
        M_matrix[:, k] = M_unit
            
    b_eq = -W_env[:3]
    tau_env = W_env[3:]
    
    # 空間のサンプリング
    n_phi, n_theta = 14, 7
    phi_vals = np.linspace(0, 2*np.pi, n_phi)
    theta_vals = np.linspace(-np.pi/2, np.pi/2, n_theta)
    torque_points = []
    
    for phi in phi_vals:
        for theta in theta_vals:
            n_tau = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
            c = - M_matrix.T @ n_tau
            
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                M_net = M_matrix @ res.x + tau_env
                torque_points.append(M_net)
                
    if len(torque_points) < 4: return None
    torque_points = np.unique(np.round(torque_points, 5), axis=0)
    if len(torque_points) < 4: return None
    
    try:
        return ConvexHull(torque_points)
    except:
        return None

# ==========================================
# 3. GUI & 可視化システム
# ==========================================
fig = plt.figure(figsize=(16, 9))
plt.subplots_adjust(bottom=0.35, left=0.05, right=0.95, wspace=0.2)

ax_robot = fig.add_subplot(121, projection='3d')
ax_torque = fig.add_subplot(122, projection='3d')

# スライダーの配置
sliders_j = []
for i in range(6):
    ax_sj = plt.axes([0.08, 0.28 - i*0.04, 0.35, 0.02])
    lim = joint_limits[i]
    sliders_j.append(Slider(ax_sj, f'Joint {i+1} ({joint_types[i]})', lim[0], lim[1], valinit=0.0, valfmt='%1.1f°'))
    
# ★ 変更点：ベース姿勢用のスライダーのみ配置 (固定チルト用のスライダーは削除)
ax_r = plt.axes([0.55, 0.28, 0.35, 0.02])
slider_roll = Slider(ax_r, 'Base4 Roll', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
ax_p = plt.axes([0.55, 0.24, 0.35, 0.02])
slider_pitch = Slider(ax_p, 'Base4 Pitch', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
ax_y = plt.axes([0.55, 0.20, 0.35, 0.02])
slider_yaw = Slider(ax_y, 'Base4 Yaw', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')

def draw_scene(val=None):
    ax_robot.cla()
    ax_torque.cla()
    
    j_angles = [np.radians(s.val) for s in sliders_j]
    base_rpy = [np.radians(slider_roll.val), np.radians(slider_pitch.val), np.radians(slider_yaw.val)]
    
    link_frames, j_pos, r_CoM, r_CoB, W_env, thr_pos, thr_dir = update_kinematics(j_angles, base_rpy)
    hull = compute_torque_space(r_CoM, thr_pos, thr_dir, W_env)
    
    # --- 左画面: ロボットモデル ---
    for i in range(7):
        R_l, P_l = link_frames[i]
        P_end = P_l + R_l @ np.array([link_lengths[i], 0, 0])
        color = 'darkgreen' if i == 3 else 'navy'
        ax_robot.plot([P_l[0], P_end[0]], [P_l[1], P_end[1]], [P_l[2], P_end[2]], 'o-', color=color, lw=4, ms=6)
    
    if len(j_pos) > 0:
        j_pos = np.array(j_pos)
        ax_robot.scatter(j_pos[:,0], j_pos[:,1], j_pos[:,2], color='darkorange', s=60, label='Actuators')
        
    ax_robot.scatter(r_CoM[0], r_CoM[1], r_CoM[2], color='red', marker='x', s=100, lw=3, label='CoM')
    ax_robot.scatter(r_CoB[0], r_CoB[1], r_CoB[2], color='cyan', marker='o', s=100, edgecolors='b', label='CoB')
    
    # ★ 変更点：スラスター（ベクター配置）の描画
    R_l4, P_l4 = link_frames[3]
    for k in range(num_thrusters):
        pos = thr_pos[k]
        d_vec = thr_dir[k]
        
        # スラスターの推力方向を赤矢印で描画
        ax_robot.quiver(pos[0], pos[1], pos[2], d_vec[0], d_vec[1], d_vec[2], length=0.15, color='crimson', lw=2)
        
        # マウント用の支柱をグレーの点線で描画
        if k < 4:
            mount_base = P_l4
        else:
            mount_base = P_l4 + R_l4 @ np.array([link_lengths[3], 0, 0])
        ax_robot.plot([mount_base[0], pos[0]], [mount_base[1], pos[1]], [mount_base[2], pos[2]], color='gray', lw=1, linestyle='--')
            
    ax_robot.set_title("Underwater Robot Model (8-Vectored on Link 4)")
    ax_robot.set_xlabel("X [m]"), ax_robot.set_ylabel("Y [m]"), ax_robot.set_zlabel("Z [m]")
    ax_robot.set_xlim(-1.5, 1.5), ax_robot.set_ylim(-1.5, 1.5), ax_robot.set_zlim(-1.5, 1.5)
    ax_robot.grid(True)
    ax_robot.legend(loc='upper left')
    
    # --- 右画面: トルク空間 ---
    if hull is not None:
        ax_torque.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2],
                               triangles=hull.simplices, alpha=0.4, color='turquoise', edgecolor='teal', linewidth=0.5)
        ax_torque.scatter(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], s=10, color='darkcyan')
        ax_torque.scatter(0, 0, 0, color='magenta', marker='*', s=150, label='Net Zero Torque')
        ax_torque.legend()
        
        pts = hull.points
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0 + 0.1
        mid_x, mid_y, mid_z = (pts[:,0].max()+pts[:,0].min())*0.5, (pts[:,1].max()+pts[:,1].min())*0.5, (pts[:,2].max()+pts[:,2].min())*0.5
        ax_torque.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_torque.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_torque.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax_torque.text(0.5, 0.5, 0.5, "Hovering Impossible", color='red', ha='center', va='center', fontsize=12, transform=ax_torque.transAxes)
        ax_torque.set_xlim(-1, 1), ax_torque.set_ylim(-1, 1), ax_torque.set_zlim(-1, 1)

    ax_torque.set_title("Minkowski Sum Torque Space (8-Vectored Setup)")
    ax_torque.set_xlabel("Torque Mx [Nm]"), ax_torque.set_ylabel("Torque My [Nm]"), ax_torque.set_zlabel("Torque Mz [Nm]")
    
    fig.canvas.draw_idle()

for s in sliders_j: s.on_changed(draw_scene)
slider_roll.on_changed(draw_scene)
slider_pitch.on_changed(draw_scene)
slider_yaw.on_changed(draw_scene)

draw_scene()
plt.show()