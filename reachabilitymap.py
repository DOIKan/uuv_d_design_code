import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Manipulator:
    def __init__(self, base_link, segments):
        """
        base_link: 土台となるリンクの辞書 {'length': L, 'radius': R}
        segments: 各関節とそれに続くリンクのリスト
                  [{'axis': 'pitch'|'yaw'|'roll', 'limits': (min, max), 'length': L, 'radius': R}, ...]
        """
        self.base_link = base_link
        self.segments = segments
        
        # リンクの方向をZ軸方向と定義
        # Roll: Z軸回り, Pitch: X軸回り, Yaw: Y軸回り
        self.axis_dict = {
            'roll': self._rz,
            'pitch': self._rx,
            'yaw': self._ry
        }

    # --- 同次変換行列のヘルパー関数 ---
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
        """
        与えられた関節角度のリストから、各関節・リンク終端の3D座標を計算する
        """
        positions = []
        
        # ベース（原点）
        current_transform = np.eye(4)
        positions.append(current_transform[:3, 3].copy())
        
        # ベースリンクの先端（第1関節の位置）
        current_transform = current_transform @ self._trans_z(self.base_link['length'])
        positions.append(current_transform[:3, 3].copy())
        
        # 各関節とリンクの計算
        for i, (segment, angle) in enumerate(zip(self.segments, angles)):
            # 1. 関節の回転
            rot_matrix = self.axis_dict[segment['axis']](angle)
            current_transform = current_transform @ rot_matrix
            
            # 2. リンクの並進（長さ分Z軸へ進む）
            trans_matrix = self._trans_z(segment['length'])
            current_transform = current_transform @ trans_matrix
            
            positions.append(current_transform[:3, 3].copy())
            
        return np.array(positions)

    def generate_reachability_map(self, num_samples=5000):
        """
        ランダムに角度を生成し、エンドエフェクタの到達可能座標群を返す
        """
        end_effector_positions = np.zeros((num_samples, 3))
        
        for i in range(num_samples):
            random_angles = []
            for seg in self.segments:
                min_angle, max_angle = seg['limits']
                angle = np.random.uniform(min_angle, max_angle)
                random_angles.append(angle)
            
            positions = self.forward_kinematics(random_angles)
            end_effector_positions[i] = positions[-1] # エンドエフェクタ（終端）の座標
            
        return end_effector_positions

def main():
    # --- ロボットのパラメータ設定 ---
    # 角度はラジアン表記 (np.deg2rad で度数法から変換するとわかりやすいです)
    
    # リンク0 (ベース)
    base_link = {'length': 0.2, 'radius': 0.1}
    
    # 関節1〜6 と リンク1〜6
    segments = [
        # 関節1: Pitch (Z軸が上を向いている状態で左右の首振り)
        {'axis': 'pitch', 'limits': (np.deg2rad(-135), np.deg2rad(135)), 'length': 0.2, 'radius': 0.1},
        # 関節2: Roll (上下の振る舞い)
        {'axis': 'roll', 'limits': (np.deg2rad(-180), np.deg2rad(180)), 'length': 0.25, 'radius': 0.1},
        # 関節3: Pitch
        {'axis': 'pitch', 'limits': (np.deg2rad(-90), np.deg2rad(90)), 'length': 0.5, 'radius': 0.1},
        # 関節4: Pitch (腕のひねり)
        {'axis': 'pitch', 'limits': (np.deg2rad(-90), np.deg2rad(90)), 'length': 0.25, 'radius': 0.10},
        # 関節5: Roll (手首の上下)
        {'axis': 'roll', 'limits': (np.deg2rad(-180), np.deg2rad(180)), 'length': 0.2, 'radius': 0.1},
        # 関節6: Pitch (手首の回転)
        {'axis': 'pitch', 'limits': (np.deg2rad(-135), np.deg2rad(135)), 'length': 0.2, 'radius': 0.1},
    ]
    
    robot = Manipulator(base_link, segments)
    
    # --- 1. リーチアビリティマップの生成 ---
    print("リーチアビリティマップを生成中...")
    cloud_points = robot.generate_reachability_map(num_samples=10000)
    
    # --- 2. ロボットの初期姿勢の生成 ---
    # 例として、すべて0度の状態を計算
    zero_angles = [0.0] * 6
    robot_positions = robot.forward_kinematics(zero_angles)
    
    # --- 3. 可視化 (Matplotlib) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 到達可能領域のプロット (半透明の点群)
    ax.scatter(cloud_points[:, 0], cloud_points[:, 1], cloud_points[:, 2], 
               c='c', s=1, alpha=0.1, label='Reachability Map')
    
    # ロボットのリンクと関節を描画
    # Matplotlibでの3D円柱描画は非常に重いため、太い線（linewidth）でリンクの太さを表現しています
    xs, ys, zs = robot_positions[:, 0], robot_positions[:, 1], robot_positions[:, 2]
    
    # ベースの描画
    ax.plot(xs[0:2], ys[0:2], zs[0:2], color='gray', 
            linewidth=base_link['radius']*100, solid_capstyle='round', label='Base Link')
    
    # 各リンクの描画
    colors = plt.cm.jet(np.linspace(0, 1, len(segments)))
    for i in range(len(segments)):
        ax.plot(xs[i+1:i+3], ys[i+1:i+3], zs[i+1:i+3], color=colors[i], 
                linewidth=segments[i]['radius']*100, solid_capstyle='round')
        # 関節位置に球（点）を描画
        ax.scatter(xs[i+1], ys[i+1], zs[i+1], color='black', s=50, zorder=5)
    
    # グラフの見た目を整える
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('6-DOF Manipulator Reachability Map & Robot Model')
    
    # アスペクト比を揃える（空間が歪まないようにする）
    max_range = np.array([cloud_points[:, 0].max()-cloud_points[:, 0].min(),
                          cloud_points[:, 1].max()-cloud_points[:, 1].min(),
                          cloud_points[:, 2].max()-cloud_points[:, 2].min()]).max() / 2.0
    mid_x = (cloud_points[:, 0].max()+cloud_points[:, 0].min()) * 0.5
    mid_y = (cloud_points[:, 1].max()+cloud_points[:, 1].min()) * 0.5
    mid_z = (cloud_points[:, 2].max()+cloud_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()