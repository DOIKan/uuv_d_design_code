import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

class Manipulator:
    def __init__(self, base_link, segments):
        self.base_link = base_link
        self.segments = segments
        
        self.axis_dict = {
            'roll': self._rz,
            'pitch': self._rx,
            'yaw': self._ry
        }

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

    def forward_kinematics(self, angles):
        positions = []
        
        current_transform = np.eye(4)
        positions.append(current_transform[:3, 3].copy())
        
        current_transform = current_transform @ self._trans_z(self.base_link['length'])
        positions.append(current_transform[:3, 3].copy())
        
        for i, (segment, angle) in enumerate(zip(self.segments, angles)):
            rot_matrix = self.axis_dict[segment['axis']](angle)
            current_transform = current_transform @ rot_matrix
            
            trans_matrix = self._trans_z(segment['length'])
            current_transform = current_transform @ trans_matrix
            
            positions.append(current_transform[:3, 3].copy())
            
        return np.array(positions)

    def generate_reachability_map(self, num_samples=5000):
        end_effector_positions = np.zeros((num_samples, 3))
        for i in range(num_samples):
            random_angles = []
            for seg in self.segments:
                min_angle, max_angle = seg['limits']
                angle = np.random.uniform(min_angle, max_angle)
                random_angles.append(angle)
            
            positions = self.forward_kinematics(random_angles)
            end_effector_positions[i] = positions[-1]
            
        return end_effector_positions

def main():
    base_link = {'length': 0.2, 'radius': 0.1}
    segments = [
        {'axis': 'pitch', 'limits': (np.deg2rad(-135), np.deg2rad(135)), 'length': 0.2, 'radius': 0.1},
        {'axis': 'roll', 'limits': (np.deg2rad(-180), np.deg2rad(180)), 'length': 0.25, 'radius': 0.1},
        {'axis': 'pitch', 'limits': (np.deg2rad(-90), np.deg2rad(90)), 'length': 0.5, 'radius': 0.1},
        {'axis': 'pitch', 'limits': (np.deg2rad(-90), np.deg2rad(90)), 'length': 0.25, 'radius': 0.10},
        {'axis': 'roll', 'limits': (np.deg2rad(-180), np.deg2rad(180)), 'length': 0.2, 'radius': 0.1},
        {'axis': 'pitch', 'limits': (np.deg2rad(-135), np.deg2rad(135)), 'length': 0.2, 'radius': 0.1},
    ]
    
    robot = Manipulator(base_link, segments)
    
    print("リーチアビリティマップを生成中...")
    cloud_points = robot.generate_reachability_map(num_samples=10000)
    
    # 描画ウィンドウの設定（スライダーのスペースを確保するため縦に少し長く、余白を調整）
    fig = plt.figure(figsize=(10, 9))
    # 3Dプロットエリア（下部30%をスライダー用に空ける）
    ax = fig.add_axes([0.05, 0.35, 0.9, 0.6], projection='3d')
    
    # マップのプロット
    ax.scatter(cloud_points[:, 0], cloud_points[:, 1], cloud_points[:, 2], 
               c='c', s=1, alpha=0.1, label='Reachability Map')
    
    # 初期姿勢の計算
    zero_angles = [0.0] * 6
    robot_positions = robot.forward_kinematics(zero_angles)
    xs, ys, zs = robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2]
    
    # --- ロボットの描画オブジェクトを保持する ---
    # （後でGUIから座標データを更新するため、変数に格納しておきます）
    base_line, = ax.plot(xs[0:2], ys[0:2], zs[0:2], color='gray', 
                         linewidth=base_link['radius']*100, solid_capstyle='round', label='Base Link')
    
    link_lines = []
    colors = plt.cm.jet(np.linspace(0, 1, len(segments)))
    for i in range(len(segments)):
        line, = ax.plot(xs[i+1:i+3], ys[i+1:i+3], zs[i+1:i+3], color=colors[i], 
                        linewidth=segments[i]['radius']*100, solid_capstyle='round')
        link_lines.append(line)
        
    # 関節位置の球（点）は更新を容易にするため scatter ではなく plot を使用
    joint_points, = ax.plot(xs[1:], ys[1:], zs[1:], color='black', marker='o', 
                            linestyle='None', markersize=6, zorder=5)
    
    # グラフの見た目を整える
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('6-DOF Manipulator Interactive Simulator')
    
    # アスペクト比の固定
    max_range = np.array([cloud_points[:, 0].max()-cloud_points[:, 0].min(),
                          cloud_points[:, 1].max()-cloud_points[:, 1].min(),
                          cloud_points[:, 2].max()-cloud_points[:, 2].min()]).max() / 2.0
    mid_x = (cloud_points[:, 0].max()+cloud_points[:, 0].min()) * 0.5
    mid_y = (cloud_points[:, 1].max()+cloud_points[:, 1].min()) * 0.5
    mid_z = (cloud_points[:, 2].max()+cloud_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.legend(loc='upper left')

    # --- GUIスライダーの設定 ---
    ax_sliders = []
    sliders = []
    slider_height = 0.03
    margin_bottom = 0.25
    
    for i, seg in enumerate(segments):
        # スライダー用のAxesを作成
        sax = fig.add_axes([0.2, margin_bottom - i*(slider_height + 0.01), 0.6, slider_height])
        min_deg = np.rad2deg(seg['limits'][0])
        max_deg = np.rad2deg(seg['limits'][1])
        
        # スライダーの作成（初期値は0度）
        slider = Slider(sax, f'Joint {i+1} ({seg["axis"]}) [deg]', min_deg, max_deg, valinit=0.0)
        ax_sliders.append(sax)
        sliders.append(slider)

    # スライダーが変更されたときに呼ばれる関数
    def update(val):
        # スライダーの値（度）をラジアンに変換して取得
        current_angles = [np.deg2rad(s.val) for s in sliders]
        
        # 新しい姿勢を計算
        new_positions = robot.forward_kinematics(current_angles)
        nx, ny, nz = new_positions[:, 0], new_positions[:, 1], new_positions[:, 2]
        
        # ベースラインの更新（固定ですが念のため）
        base_line.set_data(nx[0:2], ny[0:2])
        base_line.set_3d_properties(nz[0:2])
        
        # リンクの更新
        for i, line in enumerate(link_lines):
            line.set_data(nx[i+1:i+3], ny[i+1:i+3])
            line.set_3d_properties(nz[i+1:i+3])
            
        # 関節位置（黒い点）の更新
        joint_points.set_data(nx[1:], ny[1:])
        joint_points.set_3d_properties(nz[1:])
        
        # 描画の更新
        fig.canvas.draw_idle()

    # 各スライダーに変更イベントを紐付け
    for s in sliders:
        s.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()