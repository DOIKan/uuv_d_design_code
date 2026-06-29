import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

class DualArmManipulator:
    def __init__(self, base_length, base_radius, left_segments, right_segments):
        self.base_length = base_length
        self.base_radius = base_radius
        self.left_segments = left_segments
        self.right_segments = right_segments
        
        self.axis_dict = {
            'roll': self._rz,
            'pitch': self._rx,
            'yaw': self._ry
        }

    # --- 同次変換行列 ---
    def _rx(self, theta):
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(theta), -np.sin(theta), 0],
                         [0, np.sin(theta), np.cos(theta), 0],
                         [0, 0, 0, 1]])

    def _ry(self, theta):
        return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                         [0, 1, 0, 0],
                         [-np.sin(theta), 0, np.cos(theta), 0],
                         [0, 0, 0, 1]])

    def _rz(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                         [np.sin(theta), np.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def _trans_z(self, d):
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, d],
                         [0, 0, 0, 1]])

    def forward_kinematics_single_arm(self, angles, base_offset_y, segments):
        """片腕の順運動学（指定されたセグメントリストを使用）"""
        positions = []
        
        # 腕の付け根（肩）
        current_transform = np.eye(4)
        current_transform[1, 3] = base_offset_y
        positions.append(current_transform[:3, 3].copy())
        
        for segment, angle in zip(segments, angles):
            rot_matrix = self.axis_dict[segment['axis']](angle)
            current_transform = current_transform @ rot_matrix
            
            trans_matrix = self._trans_z(segment['length'])
            current_transform = current_transform @ trans_matrix
            
            positions.append(current_transform[:3, 3].copy())
            
        return np.array(positions)

    def forward_kinematics(self, left_angles, right_angles):
        """両腕とベースリンクの座標を計算"""
        base_positions = np.array([
            [0, self.base_length/2, 0],  # 左肩
            [0, -self.base_length/2, 0]  # 右肩
        ])
        
        left_positions = self.forward_kinematics_single_arm(left_angles, self.base_length/2, self.left_segments)
        right_positions = self.forward_kinematics_single_arm(right_angles, -self.base_length/2, self.right_segments)
        
        return base_positions, left_positions, right_positions

    def generate_reachability_map(self, num_samples=3000):
        """左右それぞれの到達可能領域（3自由度）"""
        left_ee_positions = np.zeros((num_samples, 3))
        right_ee_positions = np.zeros((num_samples, 3))
        
        for i in range(num_samples):
            left_angles = [np.random.uniform(seg['limits'][0], seg['limits'][1]) for seg in self.left_segments]
            right_angles = [np.random.uniform(seg['limits'][0], seg['limits'][1]) for seg in self.right_segments]
            
            _, l_pos, r_pos = self.forward_kinematics(left_angles, right_angles)
            left_ee_positions[i] = l_pos[-1]
            right_ee_positions[i] = r_pos[-1]
            
        return left_ee_positions, right_ee_positions


def main():
    # --- ロボットのパラメータ設定 ---
    base_length = 0.5
    base_radius = 0.1
    
    # 元の6リンクを左（1〜3番目）と右（4〜6番目）に分割して割り当て
    # ※もし左右を完全対称な同じ長さにしたい場合は、両方に同じリストを渡してください。
    left_segments = [
        {'axis': 'pitch', 'limits': (np.deg2rad(-210), np.deg2rad(30)), 'length': 0.2, 'radius': 0.1},
        {'axis': 'roll', 'limits': (np.deg2rad(-180), np.deg2rad(180)), 'length': 0.2, 'radius': 0.1},
        {'axis': 'pitch', 'limits': (np.deg2rad(-135), np.deg2rad(135)), 'length': 0.25, 'radius': 0.1},
    ]
    
    right_segments = [
        {'axis': 'pitch', 'limits': (np.deg2rad(-30), np.deg2rad(210)), 'length': 0.25, 'radius': 0.10},
        {'axis': 'roll', 'limits': (np.deg2rad(-180), np.deg2rad(180)), 'length': 0.2, 'radius': 0.1},
        {'axis': 'pitch', 'limits': (np.deg2rad(-135), np.deg2rad(135)), 'length': 0.2, 'radius': 0.1},
    ]
    
    robot = DualArmManipulator(base_length, base_radius, left_segments, right_segments)
    
    print("双腕(各3DOF)のリーチアビリティマップを生成中...")
    left_cloud, right_cloud = robot.generate_reachability_map(num_samples=4000)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0.05, 0.35, 0.9, 0.6], projection='3d')
    
    # リーチアビリティマップ
    ax.scatter(left_cloud[:, 0], left_cloud[:, 1], left_cloud[:, 2], c='c', s=2, alpha=0.15, label='Left Map')
    ax.scatter(right_cloud[:, 0], right_cloud[:, 1], right_cloud[:, 2], c='m', s=2, alpha=0.15, label='Right Map')
    
    zero_angles_l = [0.0] * len(left_segments)
    zero_angles_r = [0.0] * len(right_segments)
    base_pos, l_pos, r_pos = robot.forward_kinematics(zero_angles_l, zero_angles_r)
    
    # ベースリンクとベースのEE（原点）
    ax.plot(base_pos[:, 0], base_pos[:, 1], base_pos[:, 2], color='gray', linewidth=base_radius*100, solid_capstyle='round')
    
    # 描画用オブジェクト作成
    left_link_lines, right_link_lines = [], []
    for i in range(len(left_segments)):
        ll, = ax.plot(l_pos[i:i+2, 0], l_pos[i:i+2, 1], l_pos[i:i+2, 2], color='blue', linewidth=left_segments[i]['radius']*100, solid_capstyle='round')
        left_link_lines.append(ll)
    for i in range(len(right_segments)):
        rl, = ax.plot(r_pos[i:i+2, 0], r_pos[i:i+2, 1], r_pos[i:i+2, 2], color='purple', linewidth=right_segments[i]['radius']*100, solid_capstyle='round')
        right_link_lines.append(rl)
        
    left_joint_points, = ax.plot(l_pos[:, 0], l_pos[:, 1], l_pos[:, 2], color='black', marker='o', linestyle='None', zorder=5)
    right_joint_points, = ax.plot(r_pos[:, 0], r_pos[:, 1], r_pos[:, 2], color='black', marker='o', linestyle='None', zorder=5)
    
    left_ee_point, = ax.plot([l_pos[-1, 0]], [l_pos[-1, 1]], [l_pos[-1, 2]], color='cyan', marker='*', markersize=15, linestyle='None', zorder=6)
    right_ee_point, = ax.plot([r_pos[-1, 0]], [r_pos[-1, 1]], [r_pos[-1, 2]], color='magenta', marker='*', markersize=15, linestyle='None', zorder=6)
    
    # グラフ設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3-DOF Dual-Arm Manipulator (Total 7 Links)')
    
    # アスペクト比の調整
    all_points = np.vstack([left_cloud, right_cloud])
    max_range = np.array([all_points[:, 0].max()-all_points[:, 0].min(),
                          all_points[:, 1].max()-all_points[:, 1].min(),
                          all_points[:, 2].max()-all_points[:, 2].min()]).max() / 2.0
    mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.legend(loc='upper right', fontsize='small')

    # --- GUIスライダーの設定 ---
    left_sliders, right_sliders = [], []
    slider_height = 0.03
    margin_bottom = 0.25
    spacing = 0.05
    
    for i, seg in enumerate(left_segments):
        sax_l = fig.add_axes([0.1, margin_bottom - i*spacing, 0.3, slider_height])
        slider_l = Slider(sax_l, f'L-J{i+1}({seg["axis"]})', np.rad2deg(seg['limits'][0]), np.rad2deg(seg['limits'][1]), valinit=0.0)
        left_sliders.append(slider_l)
        
    for i, seg in enumerate(right_segments):
        sax_r = fig.add_axes([0.6, margin_bottom - i*spacing, 0.3, slider_height])
        slider_r = Slider(sax_r, f'R-J{i+1}({seg["axis"]})', np.rad2deg(seg['limits'][0]), np.rad2deg(seg['limits'][1]), valinit=0.0)
        right_sliders.append(slider_r)

    def update(val):
        l_angles = [np.deg2rad(s.val) for s in left_sliders]
        r_angles = [np.deg2rad(s.val) for s in right_sliders]
        
        _, new_l_pos, new_r_pos = robot.forward_kinematics(l_angles, r_angles)
        
        for i in range(len(left_segments)):
            left_link_lines[i].set_data(new_l_pos[i:i+2, 0], new_l_pos[i:i+2, 1])
            left_link_lines[i].set_3d_properties(new_l_pos[i:i+2, 2])
        for i in range(len(right_segments)):
            right_link_lines[i].set_data(new_r_pos[i:i+2, 0], new_r_pos[i:i+2, 1])
            right_link_lines[i].set_3d_properties(new_r_pos[i:i+2, 2])
            
        left_joint_points.set_data(new_l_pos[:, 0], new_l_pos[:, 1])
        left_joint_points.set_3d_properties(new_l_pos[:, 2])
        right_joint_points.set_data(new_r_pos[:, 0], new_r_pos[:, 1])
        right_joint_points.set_3d_properties(new_r_pos[:, 2])
        
        left_ee_point.set_data([new_l_pos[-1, 0]], [new_l_pos[-1, 1]])
        left_ee_point.set_3d_properties([new_l_pos[-1, 2]])
        right_ee_point.set_data([new_r_pos[-1, 0]], [new_r_pos[-1, 1]])
        right_ee_point.set_3d_properties([new_r_pos[-1, 2]])
        
        fig.canvas.draw_idle()

    for s in left_sliders + right_sliders:
        s.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()