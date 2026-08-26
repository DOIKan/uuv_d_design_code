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
link_masses  = [0.3, 0.3, 0.2, 2.5, 0.2, 0.3, 0.3]
link_volumes = [0.0008, 0.0005, 0.0005, 0.0030, 0.0005, 0.0005, 0.0008] # 浮力調整用

# 6関節のパラメータ (順序: P, R, P, P, R', 'P')
joint_types = ['P', 'R', 'P', 'P', 'R', 'P']
actuator_masses = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # 各関節のアクチュエータ質量 [kg]
joint_limits = [(-135, 135), (-180, 180), (-90, 90), (-90, 90), (-180, 180), (-135, 135)] # [deg]

# スラスターパラメータ (6個)
max_thrust = 15.0   # 各スラスターの最大推力 [N] (両方向に出力可能)
sigma = 0.0036      # ドラグモーメント係数
thruster_directions = [1, -1, 1, -1, 1, -1] # プロペラ回転方向 (1:CW, -1:CCW)

# 各スラスターの属するリンク
belonging_links = [0, 2, 2, 4, 4, 6] 

# 環境定数
rho_water = 1000.0  # 水の密度 [kg/m^3]
g_const = 9.81      # 重力加速度 [m/s^2]

# 回転行列生成ヘルパー
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
    # Roll(X) -> Pitch(Y) -> Yaw(Z) の順で回転
    return rot_z(y) @ rot_y(p) @ rot_x(r)

# ==========================================
# 2. 運動学 & 物理特性計算コア
# ==========================================
def update_kinematics(joint_angles, tilt_angles, base_rpy):
    # --- Step 1: Link1を原点とした仮の順運動学を計算 ---
    tmp_link_frames = [] # リンク姿勢
    tmp_joint_positions = [] #リンク始点位置
    tmp_link_centers = [] # リンク中心位置
    
    R = np.eye(3)
    P = np.array([0.0, 0.0, 0.0])
    
    for i in range(7):
        tmp_link_frames.append((R.copy(), P.copy())) # リンク始点の姿勢と位置を保存
        P_center = P + R @ np.array([link_lengths[i]/2.0, 0, 0]) # リンク中心位置を計算
        tmp_link_centers.append(P_center) # リンク中心位置をリストに保存
        
        if i < 6:
            P_joint = P + R @ np.array([link_lengths[i], 0, 0])
            tmp_joint_positions.append(P_joint)
            
            theta = joint_angles[i]
            if joint_types[i] == 'P':
                R_j = rot_y(theta)
            elif joint_types[i] == 'R':
                R_j = rot_x(theta)
            else:
                R_j = rot_z(theta)
            R = R @ R_j
            P = P_joint.copy()
            
    # --- Step 2: 第4リンク(インデックス3)を基準位置・姿勢に変換 ---
    R_link4_tmp, P_link4_tmp = tmp_link_frames[3]
    R_base_desired = rot_rpy(base_rpy[0], base_rpy[1], base_rpy[2]) # 目標姿勢の回転行列
    
    # 第4リンクの姿勢を目標姿勢にするためのグローバル変換行列
    R_global = R_base_desired @ R_link4_tmp.T
    # 第4リンクの始点を原点(0,0,0)にするための並進移動
    P_global = - R_global @ P_link4_tmp
    
    # 全ての座標系にグローバル変換を適用
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

    # --- Step 3: 重心 (CoM) と 浮心 (CoB) の計算 ---
    total_mass = sum(link_masses) + sum(actuator_masses)
    com_num = np.zeros(3)
    for i in range(7): com_num += link_masses[i] * link_centers[i] # リンク質量による重心寄与(リンク中心位置)
    for j in range(6): com_num += actuator_masses[j] * joint_positions[j] # アクチュエータ質量による重心寄与(関節位置)
    r_CoM = com_num / total_mass
    
    total_volume = sum(link_volumes)
    cob_num = np.zeros(3)
    for i in range(7): cob_num += link_volumes[i] * link_centers[i] # リンク体積による浮心寄与(リンク中心位置)
    r_CoB = cob_num / total_volume
    
    # 環境レンチ (重力 + 浮力)
    F_g = np.array([0, 0, -total_mass * g_const]) # 重力による力
    F_b = np.array([0, 0, rho_water * g_const * total_volume]) # 浮力による力
    tau_b = np.cross(r_CoB - r_CoM, F_b) # 重力と浮力による復元トルク
    W_env = np.concatenate([F_g + F_b, tau_b]) # 環境レンチ (6次元ベクトル)
    
    # --- Step 4: スラスターの位置と推力方向 ---
    thruster_positions = []
    thruster_dirs = []
    loc_positions = [
        np.array([link_lengths[0]/2.0, 0, 0]),  # T1: L1 center
        np.array([0, 0, 0]),      # T2: L2 end
        np.array([link_lengths[2], 0, 0]),                    # T3: L4 start
        np.array([0, 0, 0]),      # T4: L4 end
        np.array([link_lengths[4], 0, 0]),                    # T5: L6 start
        np.array([link_lengths[6]/2.0, 0, 0])   # T6: L7 center
    ]# スラスターの位置の記述方法
    
    for k in range(6):
        l_idx = belonging_links[k] # スラスターkが属するリンクのインデックス
        R_l, P_l = link_frames[l_idx] # スラスターkが属するリンクの姿勢と位置
        
        thruster_positions.append(P_l + R_l @ loc_positions[k])
        # チルト軸はローカルX軸
        n_local = rot_x(tilt_angles[k]) @ np.array([0, 0, 1]) # ローカルZ軸をチルト角で回転させた方向
        thruster_dirs.append(R_l @ n_local)
        
    return link_frames, joint_positions, r_CoM, r_CoB, W_env, np.array(thruster_positions), np.array(thruster_dirs)

def compute_torque_space(r_CoM, thr_pos, thr_dir, W_env):
    A_eq = np.zeros((3, 6))
    M_thr = np.zeros((3, 6))
    
    for k in range(6):
        n = thr_dir[k]
        p = thr_pos[k] - r_CoM
        A_eq[:, k] = n
        M_thr[:, k] = np.cross(p, n) + sigma * thruster_directions[k] * n
        
    b_eq = -W_env[:3]
    # ★ 両方向推力対応: 境界を -max_thrust から +max_thrust に変更
    bounds = [(-max_thrust, max_thrust) for _ in range(6)]
    tau_env = W_env[3:]
    
    n_phi, n_theta = 14, 7
    phi_vals = np.linspace(0, 2*np.pi, n_phi)
    theta_vals = np.linspace(-np.pi/2, np.pi/2, n_theta)
    torque_points = []
    
    for phi in phi_vals:
        for theta in theta_vals:
            n_tau = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)])
            c = - M_thr.T @ n_tau
            
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                M_net = M_thr @ res.x + tau_env
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
# 下部にグローバル姿勢用のスライダー領域を確保するため bottom を 0.45 に
plt.subplots_adjust(bottom=0.45, left=0.05, right=0.95, wspace=0.2)

ax_robot = fig.add_subplot(121, projection='3d')
ax_torque = fig.add_subplot(122, projection='3d')

# スライダーの配置 (関節角・チルト角)
sliders_j = []
sliders_t = []
for i in range(6):
    ax_sj = plt.axes([0.08, 0.38 - i*0.035, 0.35, 0.02])
    lim = joint_limits[i]
    sliders_j.append(Slider(ax_sj, f'Joint {i+1} ({joint_types[i]})', lim[0], lim[1], valinit=0.0, valfmt='%1.1f°'))
    
    ax_st = plt.axes([0.55, 0.38 - i*0.035, 0.35, 0.02])
    sliders_t.append(Slider(ax_st, f'Tilt {i+1}', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°'))

# スライダーの配置 (第4リンク基準のグローバル姿勢)
ax_r = plt.axes([0.3, 0.12, 0.4, 0.02])
ax_p = plt.axes([0.3, 0.08, 0.4, 0.02])
ax_y = plt.axes([0.3, 0.04, 0.4, 0.02])
slider_roll  = Slider(ax_r, 'Base4 Roll', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
slider_pitch = Slider(ax_p, 'Base4 Pitch', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')
slider_yaw   = Slider(ax_y, 'Base4 Yaw', -180.0, 180.0, valinit=0.0, valfmt='%1.1f°')

def draw_scene(val=None):
    ax_robot.cla()
    ax_torque.cla()
    
    # 角度の取得
    j_angles = [np.radians(s.val) for s in sliders_j]
    t_angles = [np.radians(s.val) for s in sliders_t]
    base_rpy = [np.radians(slider_roll.val), np.radians(slider_pitch.val), np.radians(slider_yaw.val)]
    
    # 計算実行
    link_frames, j_pos, r_CoM, r_CoB, W_env, thr_pos, thr_dir = update_kinematics(j_angles, t_angles, base_rpy)
    hull = compute_torque_space(r_CoM, thr_pos, thr_dir, W_env)
    
    # --- 左画面: ロボットモデルの描画 ---
    for i in range(7):
        R_l, P_l = link_frames[i]
        P_end = P_l + R_l @ np.array([link_lengths[i], 0, 0])
        color = 'darkgreen' if i == 3 else 'navy' # 第4リンクを緑色で強調
        ax_robot.plot([P_l[0], P_end[0]], [P_l[1], P_end[1]], [P_l[2], P_end[2]], 'o-', color=color, lw=4, ms=6)
    
    if len(j_pos) > 0:
        j_pos = np.array(j_pos)
        ax_robot.scatter(j_pos[:,0], j_pos[:,1], j_pos[:,2], color='darkorange', s=60, label='Actuators')
        
    ax_robot.scatter(r_CoM[0], r_CoM[1], r_CoM[2], color='red', marker='x', s=100, lw=3, label='CoM')
    ax_robot.scatter(r_CoB[0], r_CoB[1], r_CoB[2], color='cyan', marker='o', s=100, edgecolors='b', label='CoB')
    
    # スラスター (推力軸)
    for k in range(6):
        ax_robot.quiver(thr_pos[k,0], thr_pos[k,1], thr_pos[k,2], 
                        thr_dir[k,0], thr_dir[k,1], thr_dir[k,2], 
                        length=0.2, color='crimson', lw=2, arrow_length_ratio=0.3)
        ax_robot.text(thr_pos[k,0], thr_pos[k,1], thr_pos[k,2], f' T{k+1}', color='black', fontsize=9)
        
    ax_robot.set_title("Underwater Robot Model (Base: Link 4)")
    ax_robot.set_xlabel("X [m]"), ax_robot.set_ylabel("Y [m]"), ax_robot.set_zlabel("Z [m]")
    ax_robot.set_xlim(-1.5, 1.5), ax_robot.set_ylim(-1.5, 1.5), ax_robot.set_zlim(-1.5, 1.5)
    ax_robot.grid(True)
    ax_robot.legend(loc='upper left')
    
    # --- 右画面: トルク空間の描画 ---
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

    ax_torque.set_title("Feasible Torque Space (Bidirectional Thrust)")
    ax_torque.set_xlabel("Torque Mx [Nm]"), ax_torque.set_ylabel("Torque My [Nm]"), ax_torque.set_zlabel("Torque Mz [Nm]")
    
    fig.canvas.draw_idle()

# スライダーイベントの登録
for s in sliders_j: s.on_changed(draw_scene)
for s in sliders_t: s.on_changed(draw_scene)
slider_roll.on_changed(draw_scene)
slider_pitch.on_changed(draw_scene)
slider_yaw.on_changed(draw_scene)

# 初回描画
draw_scene()
plt.show()