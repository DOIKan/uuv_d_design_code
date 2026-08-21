import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

# ==========================================
# 1. 物理パラメータ・構成設定 (仕様そのまま)
# ==========================================
# 7リンクのパラメータ: 長さ[m], 質量[kg], 体積[m^3]
link_lengths = [0.15, 0.05, 0.40, 0.35, 0.40, 0.05, 0.15]
link_masses  = [0.3, 0.3, 0.2, 2.5, 0.3, 0.3, 0.3]
link_volumes = [0.0008, 0.0005, 0.0005, 0.0030, 0.0005, 0.0005, 0.0008] 

# 6関節のパラメータ (順序: P, R, P, P, R, P)
joint_types = ['P', 'R', 'P', 'P', 'R', 'P']
actuator_masses = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] 
joint_limits = [(-135, 135), (-180, 180), (-90, 90), (-90, 90), (-180, 180), (-135, 135)] 

# スラスターパラメータ (6個)
max_thrust = 15.0   
sigma = 0.0036      
thruster_directions = [1, -1, 1, -1, 1, -1] 

# 各スラスターの属するリンク
belonging_links = [0, 2, 2, 4, 4, 6] 

# 環境定数
rho_water = 1000.0  
g_const = 9.81      

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
    
    # ★ 変更点：重力と浮力を個別のベクトルとして抽出
    F_g = np.array([0, 0, -total_mass * g_const])
    F_b = np.array([0, 0, rho_water * g_const * total_volume])
    
    thruster_positions = []
    loc_positions = [
        np.array([link_lengths[0]/2.0, 0, 0]),  # T1
        np.array([0, 0, 0]),                    # T2
        np.array([link_lengths[2], 0, 0]),      # T3
        np.array([0, 0, 0]),                    # T4
        np.array([link_lengths[4], 0, 0]),      # T5
        np.array([link_lengths[6]/2.0, 0, 0])   # T6
    ]
    
    for k in range(6):
        l_idx = belonging_links[k]
        R_l, P_l = link_frames[l_idx]
        thruster_positions.append(P_l + R_l @ loc_positions[k])
        
    return link_frames, joint_positions, r_CoM, r_CoB, F_g, F_b, np.array(thruster_positions)

# ★ 変更点：エンドエフェクタ先端(EE)を基準とした力空間の計算
def compute_ee_force_space(r_CoM, r_CoB, link_frames, thr_pos, t2_angle, t5_angle, F_g, F_b):
    num_divs = 12
    var_tilt_k = [0, 2, 3, 5] 
    
    num_vars = len(var_tilt_k) * num_divs + 2 
    F_matrix = np.zeros((3, num_vars))
    M_matrix_ee = np.zeros((3, num_vars)) # ★ 重心ではなくエンドエフェクタ回りのモーメント
    A_ub = np.zeros((len(var_tilt_k), num_vars))
    b_ub = np.ones(len(var_tilt_k))
    bounds = []
    
    var_idx = 0
    ub_idx = 0
    
    # エンドエフェクタ（1番目のリンクの始端）の座標を取得
    P_ee = link_frames[0][1]
    
    for k in range(6):
        l_idx = belonging_links[k]
        R_l, _ = link_frames[l_idx]
        
        # モーメントアームの基準をEEに変更
        p_vec_ee = thr_pos[k] - P_ee 
        
        if k in var_tilt_k:
            for i in range(num_divs):
                phi = 2 * np.pi * i / num_divs
                n_local = np.array([0, -np.sin(phi), np.cos(phi)])
                n_global = R_l @ n_local
                
                F_vec = max_thrust * n_global
                # EE回りのモーメントとして計算
                M_vec = np.cross(p_vec_ee, F_vec) + sigma * thruster_directions[k] * F_vec
                
                F_matrix[:, var_idx] = F_vec
                M_matrix_ee[:, var_idx] = M_vec
                A_ub[ub_idx, var_idx] = 1.0 
                bounds.append((0, 1.0))
                var_idx += 1
            ub_idx += 1
        else:
            tilt = t2_angle if k == 1 else t5_angle
            n_local = rot_x(tilt) @ np.array([0, 0, 1])
            n_global = R_l @ n_local
            
            F_unit = n_global
            M_unit = np.cross(p_vec_ee, F_unit) + sigma * thruster_directions[k] * F_unit
            
            F_matrix[:, var_idx] = F_unit
            M_matrix_ee[:, var_idx] = M_unit
            bounds.append((-max_thrust, max_thrust)) 
            var_idx += 1
            
    # ★ EE回りでの環境モーメント（重力・浮力がEEに対して生む回転力）
    tau_env_ee = np.cross(r_CoM - P_ee, F_g) + np.cross(r_CoB - P_ee, F_b)
    
    # 制約条件：EE回りのモーメント合計がゼロになること（姿勢を維持する条件）
    A_eq = M_matrix_ee
    b_eq = -tau_env_ee
    F_env = F_g + F_b
    
    n_phi, n_theta = 14, 7
    phi_vals = np.linspace(0, 2*np.pi, n_phi)
    theta_vals = np.linspace(-np.pi/2, np.pi/2, n_theta)
    force_points = []
    
    for phi in phi_vals:
        for theta in theta_vals:
            n_f = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
            c = - F_matrix.T @ n_f 
            
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                F_net = F_matrix @ res.x + F_env
                force_points.append(F_net)
                
    if len(force_points) < 4: return None
    force_points = np.unique(np.round(force_points, 5), axis=0)
    if len(force_points) < 4: return None
    
    try:
        return ConvexHull(force_points)
    except:
        return None

# ==========================================
# 3. GUI & 可視化システム
# ==========================================
fig = plt.figure(figsize=(16, 9))
plt.subplots_adjust(bottom=0.35, left=0.05, right=0.95, wspace=0.2)

ax_robot = fig.add_subplot(121, projection='3d')
ax_force = fig.add_subplot(122, projection='3d')

sliders_j = []
for i in range(6):
    ax_sj = plt.axes([0.08, 0.28 - i*0.04, 0.35, 0.02])
    lim = joint_limits[i]
    sliders_j.append(Slider(ax_sj, f'Joint {i+1} ({joint_types[i]})', lim[0], lim[1], valinit=0.0, valfmt='%1.1f°'))
    
ax_t2 = plt.axes([0.55, 0.28, 0.35, 0.02])
slider_t2 = Slider(ax_t2, 'Tilt 2 (Fixed)', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
ax_t5 = plt.axes([0.55, 0.24, 0.35, 0.02])
slider_t5 = Slider(ax_t5, 'Tilt 5 (Fixed)', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')

ax_r = plt.axes([0.55, 0.16, 0.35, 0.02])
slider_roll = Slider(ax_r, 'Base4 Roll', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
ax_p = plt.axes([0.55, 0.12, 0.35, 0.02])
slider_pitch = Slider(ax_p, 'Base4 Pitch', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
ax_y = plt.axes([0.55, 0.08, 0.35, 0.02])
slider_yaw = Slider(ax_y, 'Base4 Yaw', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')

def draw_scene(val=None):
    ax_robot.cla()
    ax_force.cla()
    
    j_angles = [np.radians(s.val) for s in sliders_j]
    t2_angle = np.radians(slider_t2.val)
    t5_angle = np.radians(slider_t5.val)
    base_rpy = [np.radians(slider_roll.val), np.radians(slider_pitch.val), np.radians(slider_yaw.val)]
    
    link_frames, j_pos, r_CoM, r_CoB, F_g, F_b, thr_pos = update_kinematics(j_angles, base_rpy)
    hull = compute_ee_force_space(r_CoM, r_CoB, link_frames, thr_pos, t2_angle, t5_angle, F_g, F_b)
    
    # --- 左画面: ロボットモデル ---
    for i in range(7):
        R_l, P_l = link_frames[i]
        P_end = P_l + R_l @ np.array([link_lengths[i], 0, 0])
        color = 'darkgreen' if i == 3 else 'navy'
        ax_robot.plot([P_l[0], P_end[0]], [P_l[1], P_end[1]], [P_l[2], P_end[2]], 'o-', color=color, lw=4, ms=6)
    
    # ★ エンドエフェクタ(1番目リンクの先端)を金色の星マークで強調表示
    P_ee = link_frames[0][1]
    ax_robot.scatter(P_ee[0], P_ee[1], P_ee[2], color='gold', marker='*', s=300, edgecolors='black', label='End-Effector', zorder=5)
    
    if len(j_pos) > 0:
        j_pos = np.array(j_pos)
        ax_robot.scatter(j_pos[:,0], j_pos[:,1], j_pos[:,2], color='darkorange', s=60, label='Actuators')
        
    ax_robot.scatter(r_CoM[0], r_CoM[1], r_CoM[2], color='red', marker='x', s=100, lw=3, label='CoM')
    ax_robot.scatter(r_CoB[0], r_CoB[1], r_CoB[2], color='cyan', marker='o', s=100, edgecolors='b', label='CoB')
    
    var_tilt_k = [0, 2, 3, 5]
    for k in range(6):
        R_l, P_l = link_frames[belonging_links[k]]
        pos = thr_pos[k]
        
        if k in var_tilt_k:
            circle_pts = []
            for i in range(13):
                phi = 2 * np.pi * i / 12
                n_loc = np.array([0, -np.sin(phi), np.cos(phi)])
                n_glob = R_l @ n_loc
                circle_pts.append(pos + n_glob * 0.1)
            circle_pts = np.array(circle_pts)
            ax_robot.plot(circle_pts[:,0], circle_pts[:,1], circle_pts[:,2], color='magenta', alpha=0.6, lw=2)
            
            x_axis = R_l @ np.array([1, 0, 0])
            ax_robot.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], length=0.08, color='gray', lw=1)
        else:
            tilt = t2_angle if k == 1 else t5_angle
            n_loc = rot_x(tilt) @ np.array([0, 0, 1])
            n_glob = R_l @ n_loc
            ax_robot.quiver(pos[0], pos[1], pos[2], n_glob[0], n_glob[1], n_glob[2], length=0.15, color='crimson', lw=2)
            
    ax_robot.set_title("Underwater Robot Model (Base: Link 4)")
    ax_robot.set_xlabel("X [m]"), ax_robot.set_ylabel("Y [m]"), ax_robot.set_zlabel("Z [m]")
    ax_robot.set_xlim(-1.5, 1.5), ax_robot.set_ylim(-1.5, 1.5), ax_robot.set_zlim(-1.5, 1.5)
    ax_robot.grid(True)
    ax_robot.legend(loc='upper left')
    
    # --- 右画面: エンドエフェクタ力空間 ---
    if hull is not None:
        ax_force.plot_trisurf(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2],
                               triangles=hull.simplices, alpha=0.4, color='orange', edgecolor='chocolate', linewidth=0.5)
        ax_force.scatter(hull.points[:, 0], hull.points[:, 1], hull.points[:, 2], s=10, color='darkorange')
        ax_force.scatter(0, 0, 0, color='magenta', marker='*', s=150, label='Net Zero Force')
        ax_force.legend()
        
        pts = hull.points
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0 + 0.1
        mid_x, mid_y, mid_z = (pts[:,0].max()+pts[:,0].min())*0.5, (pts[:,1].max()+pts[:,1].min())*0.5, (pts[:,2].max()+pts[:,2].min())*0.5
        ax_force.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_force.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_force.set_zlim(mid_z - max_range, mid_z + max_range)
    else:
        ax_force.text(0.5, 0.5, 0.5, "EE Force Generation Impossible", color='red', ha='center', va='center', fontsize=12, transform=ax_force.transAxes)
        ax_force.set_xlim(-1, 1), ax_force.set_ylim(-1, 1), ax_force.set_zlim(-1, 1)

    ax_force.set_title("End-Effector Force Space\n(Posture Maintained, Net Torque = 0 around EE)")
    ax_force.set_xlabel("EE Force Fx [N]"), ax_force.set_ylabel("EE Force Fy [N]"), ax_force.set_zlabel("EE Force Fz [N]")
    
    fig.canvas.draw_idle()

for s in sliders_j: s.on_changed(draw_scene)
slider_t2.on_changed(draw_scene)
slider_t5.on_changed(draw_scene)
slider_roll.on_changed(draw_scene)
slider_pitch.on_changed(draw_scene)
slider_yaw.on_changed(draw_scene)

draw_scene()
plt.show()