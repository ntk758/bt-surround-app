"""
spatial_geometry.py
-------------------
3D Spatial Audio Engine - 数学・空間幾何学・軌道補間・パンニング基盤モジュール。

本モジュールは以下の基盤機能を提供する:
1. Position3D: 直交座標(X, Y, Z)と球面座標(Distance, Azimuth, Elevation)の高精度相互変換
2. 角度正規化・特異点(原点、垂直軸、±180°境界)の厳密なハンドリング
3. 3D軌道補間 (Cartesian LERP, Direction SLERP, WaypointTrajectory)
4. Speaker3D / SpeakerLayout3D (スピーカー幾何学、平面縮退判定、物理音響遅延計算)
5. VBAP2D / VBAP3D (2D/3D Vector Base Amplitude Panning, Triplet selection, Equal-Power正規化)
6. 将来拡張インターフェース (HeadphoneBinauralRenderer, ICrosstalkCanceller, IPerceptualElevationFilter)
"""

from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


def wrap_to_pi(angle_rad: float) -> float:
    """角度(ラジアン)を [-π, π] の範囲に正規化する。"""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def wrap_to_180(angle_deg: float) -> float:
    """角度(度)を [-180.0, 180.0] の範囲に正規化する。"""
    return (angle_deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Position3D:
    """
    3次元空間座標を表すイミュータブルなデータクラス。
    内部単位は meter (距離) および radians (角度) に統一。

    座標系定義 (右手直交座標系):
      X: 左右 (Right: +, Left: -)
      Y: 前後 (Front: +, Rear: -)
      Z: 上下 (Up: +, Down: -)

    球面座標定義:
      Distance: 原点(リスナー)からの距離 [m]
      Azimuth:  水平角 [rad] (正面=0, 右=+π/2, 真後ろ=±π, 左=-π/2)
      Elevation: 仰角 [rad] (水平=0, 真上=+π/2, 真下=-π/2)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self):
        # 非有限値 (NaN, Inf) の厳格なバリデーション
        for val, name in [(self.x, "x"), (self.y, "y"), (self.z, "z")]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Position3D coordinate '{name}' must be finite, got: {val}")

    @classmethod
    def from_spherical(
        cls,
        distance: float,
        azimuth: float,
        elevation: float,
        degrees: bool = False,
    ) -> Position3D:
        """球面座標 (distance, azimuth, elevation) から Position3D を生成する。"""
        if math.isnan(distance) or math.isinf(distance):
            raise ValueError(f"Distance must be finite: {distance}")
        if math.isnan(azimuth) or math.isinf(azimuth):
            raise ValueError(f"Azimuth must be finite: {azimuth}")
        if math.isnan(elevation) or math.isinf(elevation):
            raise ValueError(f"Elevation must be finite: {elevation}")

        if distance < 0.0:
            raise ValueError(f"Distance cannot be negative: {distance}")

        if degrees:
            azimuth = math.radians(azimuth)
            elevation = math.radians(elevation)

        # 仰角を [-π/2, π/2] にクランプ
        elevation = max(-math.pi / 2.0, min(math.pi / 2.0, float(elevation)))
        azimuth = wrap_to_pi(float(azimuth))

        if distance == 0.0:
            return cls(0.0, 0.0, 0.0)

        cos_elev = math.cos(elevation)
        x = distance * cos_elev * math.sin(azimuth)
        y = distance * cos_elev * math.cos(azimuth)
        z = distance * math.sin(elevation)

        # 丸め誤差による微小値 (-0.0 や 1e-16) のクリーンアップ
        if abs(x) < 1e-15: x = 0.0
        if abs(y) < 1e-15: y = 0.0
        if abs(z) < 1e-15: z = 0.0

        return cls(float(x), float(y), float(z))

    def to_spherical(self, degrees: bool = False) -> tuple[float, float, float]:
        """
        直交座標から球面座標 (distance, azimuth, elevation) を計算する。
        特異点(原点、Z軸上)を数値的に安定して処理する。
        """
        dist = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if dist < 1e-12:
            return (0.0, 0.0, 0.0)

        r_xy = math.hypot(self.x, self.y)

        # 仰角 (Elevation)
        elevation = math.atan2(self.z, r_xy)

        # 方位角 (Azimuth)
        if r_xy < 1e-12:
            # 真上または真下の特異点では Azimuth を 0.0 とする
            azimuth = 0.0
        else:
            azimuth = math.atan2(self.x, self.y)

        if degrees:
            return (dist, math.degrees(azimuth), math.degrees(elevation))
        return (dist, azimuth, elevation)

    @property
    def distance(self) -> float:
        """リスナー原点からの距離 [m]"""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def azimuth(self) -> float:
        """方位角 [rad]"""
        return self.to_spherical(degrees=False)[1]

    @property
    def azimuth_deg(self) -> float:
        """方位角 [deg]"""
        return self.to_spherical(degrees=True)[1]

    @property
    def elevation(self) -> float:
        """仰角 [rad]"""
        return self.to_spherical(degrees=False)[2]

    @property
    def elevation_deg(self) -> float:
        """仰角 [deg]"""
        return self.to_spherical(degrees=True)[2]

    def to_array(self) -> np.ndarray:
        """NumPy配列 (3,) [x, y, z] を返す。"""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_unit_vector(self) -> np.ndarray:
        """単位方向ベクトル (3,) を返す。原点の場合は [0, 1, 0] (正面) をフォールバックとして返す。"""
        d = self.distance
        if d < 1e-12:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return np.array([self.x / d, self.y / d, self.z / d], dtype=np.float64)

    def distance_to(self, other: Position3D) -> float:
        """別の点との3次元ユークリッド距離 [m] を計算する。"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # 演算子オーバーロード
    def __add__(self, other: Position3D) -> Position3D:
        return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Position3D) -> Position3D:
        return Position3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Position3D:
        s = float(scalar)
        return Position3D(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, scalar: float) -> Position3D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Position3D:
        s = float(scalar)
        if s == 0.0:
            raise ZeroDivisionError("Division by zero in Position3D")
        return Position3D(self.x / s, self.y / s, self.z / s)


# ==============================================================================
# 3D 軌道補間エンジン (Trajectory Engine)
# ==============================================================================

class ITrajectory(ABC):
    """3D軌道補間の抽象基底インターフェース。"""
    @abstractmethod
    def get_position(self, t: float) -> Position3D:
        """時刻 t [s] における 3D 位置を返す。"""
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """軌道の総所要時間 [s]"""
        pass


class CartesianLerpTrajectory(ITrajectory):
    """
    デカルト座標空間での区分的線形補間 (Linear Interpolation)。
    ※ 数学的に C0 連続性 (位置の連続性) を保証する。接続点での速度ベクトルは C1 連続ではない。
    """
    def __init__(self, start_pos: Position3D, end_pos: Position3D, duration: float):
        if duration <= 0.0:
            raise ValueError(f"Duration must be positive: {duration}")
        self._p0 = start_pos
        self._p1 = end_pos
        self._duration = float(duration)

    @property
    def duration(self) -> float:
        return self._duration

    def get_position(self, t: float) -> Position3D:
        if t <= 0.0:
            return self._p0
        if t >= self._duration:
            return self._p1
        tau = t / self._duration
        return self._p0 * (1.0 - tau) + self._p1 * tau


def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    2つの単位ベクトル v0, v1 間を球面線形補間 (SLERP) する。
    内積が極端な場合(平行、反平行)も安全にハンドリング。
    """
    dot = np.dot(v0, v1)
    dot = max(-1.0, min(1.0, float(dot)))

    # ほぼ平行な場合は線形補間
    if dot > 0.9995:
        res = (1.0 - t) * v0 + t * v1
        norm = np.linalg.norm(res)
        return res / norm if norm > 1e-12 else v0

    # ほぼ真後ろ(反平行)の場合
    if dot < -0.9995:
        # v0と直交する適当な軸を見つける
        ortho = np.array([-v0[1], v0[0], 0.0]) if abs(v0[2]) < 0.9 else np.array([0.0, -v0[2], v0[1]])
        norm_ortho = np.linalg.norm(ortho)
        if norm_ortho < 1e-12:
            ortho = np.array([1.0, 0.0, 0.0])
        else:
            ortho /= norm_ortho
        theta = t * math.pi
        return math.cos(theta) * v0 + math.sin(theta) * ortho

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta_t = theta_0 * t

    s0 = math.sin(theta_0 - theta_t) / sin_theta_0
    s1 = math.sin(theta_t) / sin_theta_0
    res = s0 * v0 + s1 * v1
    norm = np.linalg.norm(res)
    return res / norm if norm > 1e-12 else v0


class DirectionSlerpTrajectory(ITrajectory):
    """
    方向ベクトルは SLERP で補間し、距離は独立して線形補間する軌道。
    頭上通過(Flyover)など、角度の自然な回転運動を滑らかに表現できる。
    ※ 区分内では角速度一定、全体としては C0 連続性 (位置・方向の連続性) を保証する。
    """
    def __init__(self, start_pos: Position3D, end_pos: Position3D, duration: float):
        if duration <= 0.0:
            raise ValueError(f"Duration must be positive: {duration}")
        self._p0 = start_pos
        self._p1 = end_pos
        self._duration = float(duration)
        self._d0 = start_pos.distance
        self._d1 = end_pos.distance
        self._u0 = start_pos.to_unit_vector()
        self._u1 = end_pos.to_unit_vector()

    @property
    def duration(self) -> float:
        return self._duration

    def get_position(self, t: float) -> Position3D:
        if t <= 0.0:
            return self._p0
        if t >= self._duration:
            return self._p1
        tau = t / self._duration
        # 距離の線形補間
        dist = (1.0 - tau) * self._d0 + tau * self._d1
        # 方向ベクトルのSlerp
        u = slerp(self._u0, self._u1, tau)
        return Position3D(dist * u[0], dist * u[1], dist * u[2])


class WaypointTrajectory(ITrajectory):
    """
    複数のキーフレーム (time, Position3D) を順次補間するマルチセグメント軌道。
    Flyover Test (Front -> Above Head -> Rear) 等を定義可能。
    """
    def __init__(
        self,
        waypoints: Sequence[tuple[float, Position3D]],
        use_slerp: bool = True,
    ):
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints required")
        # 時刻順にソート検証
        sorted_wp = sorted(waypoints, key=lambda w: w[0])
        self._segments: list[tuple[float, float, ITrajectory]] = []
        for i in range(len(sorted_wp) - 1):
            t_start, p_start = sorted_wp[i]
            t_end, p_end = sorted_wp[i + 1]
            seg_dur = t_end - t_start
            if seg_dur <= 0.0:
                raise ValueError(f"Waypoint times must be strictly increasing: {t_start} -> {t_end}")
            if use_slerp:
                traj = DirectionSlerpTrajectory(p_start, p_end, seg_dur)
            else:
                traj = CartesianLerpTrajectory(p_start, p_end, seg_dur)
            self._segments.append((t_start, t_end, traj))
        self._total_duration = sorted_wp[-1][0] - sorted_wp[0][0]
        self._t_offset = sorted_wp[0][0]

    @property
    def duration(self) -> float:
        return self._total_duration

    def get_position(self, t: float) -> Position3D:
        local_t = t + self._t_offset
        if local_t <= self._segments[0][0]:
            return self._segments[0][2].get_position(0.0)
        if local_t >= self._segments[-1][1]:
            return self._segments[-1][2].get_position(self._segments[-1][2].duration)

        for t_start, t_end, traj in self._segments:
            if t_start <= local_t <= t_end:
                return traj.get_position(local_t - t_start)
        return self._segments[-1][2].get_position(self._segments[-1][2].duration)


# ==============================================================================
# スピーカー幾何学とVBAPパンニング (Speaker Geometry & VBAP Panning)
# ==============================================================================

@dataclass
class Speaker3D:
    """物理スピーカーの3D空間属性。"""
    name: str
    position: Position3D
    channel_index: int
    is_height: bool = False

    @property
    def unit_vector(self) -> np.ndarray:
        return self.position.to_unit_vector()


class SpeakerLayout3D:
    """スピーカー配置幾何学を管理し、平面縮退や物理遅延を算出するクラス。"""
    def __init__(self, speakers: Sequence[Speaker3D]):
        if not speakers:
            raise ValueError("SpeakerLayout3D must contain at least one speaker")
        self.speakers = list(speakers)
        # 重複スピーカーの検証 (同一点配置の拒否)
        for i in range(len(self.speakers)):
            for j in range(i + 1, len(self.speakers)):
                if self.speakers[i].position.distance_to(self.speakers[j].position) < 1e-4:
                    raise ValueError(
                        f"Duplicate speaker position detected between '{self.speakers[i].name}' "
                        f"and '{self.speakers[j].name}' at {self.speakers[i].position}"
                    )

    def is_planar(self, z_tolerance: float = 0.05) -> bool:
        """すべてのスピーカーが実質的に同一水平面(Zがほぼ一定)にあるか判定する。"""
        if len(self.speakers) < 4:
            return True
        z_vals = [s.position.z for s in self.speakers]
        return (max(z_vals) - min(z_vals)) < z_tolerance

    def calculate_delays_and_gains(
        self,
        source_pos: Position3D,
        speed_of_sound: float = 343.0,
        ref_distance: float = 1.0,
    ) -> dict[str, dict[str, float]]:
        """
        音源から各スピーカーまでの物理到達時間差(ms)と逆二乗距離減衰ゲインを計算。
        最遠スピーカーを遅延基準(0ms)とし、各スピーカーの伝搬遅延補正値を算出。
        """
        distances = {s.name: source_pos.distance_to(s.position) for s in self.speakers}
        max_dist = max(distances.values())
        results = {}
        for s in self.speakers:
            d = distances[s.name]
            # 伝搬遅延差分: 最大距離との差分を遅らせる
            delay_ms = ((max_dist - d) / speed_of_sound) * 1000.0
            # 距離減衰: 基準距離1.0mに対する逆距離則 (最大/最小リミット付き)
            gain = min(3.0, max(0.2, ref_distance / max(0.1, d)))
            results[s.name] = {
                "distance_m": round(d, 3),
                "delay_ms": round(delay_ms, 2),
                "gain": round(gain, 3),
            }
        return results


class VBAP2D:
    """
    水平2Dスピーカーリングに対する Vector Base Amplitude Panning。
    方位角順にソートされた隣接ペア間でゲインを解き、等パワー正規化を行う。
    """
    def __init__(self, speakers: Sequence[Speaker3D]):
        if len(speakers) < 2:
            raise ValueError("VBAP2D requires at least 2 speakers")
        # 水平面方位角順にソート
        self.speakers = sorted(speakers, key=lambda s: s.position.azimuth)
        self.n_spk = len(self.speakers)

    def calculate_gains(self, target_pos: Position3D) -> np.ndarray:
        """
        target_pos に対する全スピーカーのゲイン配列 (n_spk,) を返す。
        等パワー則 (sum(g^2) = 1.0) を満たす。
        """
        target_u = target_pos.to_unit_vector()[:2]  # [x, y]
        target_norm = np.linalg.norm(target_u)
        if target_norm < 1e-12:
            # 原点の場合は全スピーカー均等
            g = np.ones(self.n_spk, dtype=np.float64) / math.sqrt(self.n_spk)
            return g
        target_u /= target_norm

        gains = np.zeros(self.n_spk, dtype=np.float64)

        # 隣接ペアを探索
        for i in range(self.n_spk):
            j = (i + 1) % self.n_spk
            s1 = self.speakers[i]
            s2 = self.speakers[j]
            l1 = s1.unit_vector[:2]
            l2 = s2.unit_vector[:2]

            # 2x2 行列式
            det = l1[0] * l2[1] - l1[1] * l2[0]
            if abs(det) < 1e-6:
                continue

            # L^-1 * p
            g1 = (l2[1] * target_u[0] - l2[0] * target_u[1]) / det
            g2 = (-l1[1] * target_u[0] + l1[0] * target_u[1]) / det

            if g1 >= -1e-5 and g2 >= -1e-5:
                gains[i] = max(0.0, g1)
                gains[j] = max(0.0, g2)
                break

        power = np.sum(gains ** 2)
        if power > 1e-12:
            gains /= math.sqrt(power)
        else:
            # 最近傍スピーカーにフォールバック
            dots = [np.dot(target_u, s.unit_vector[:2]) for s in self.speakers]
            best_idx = int(np.argmax(dots))
            gains[best_idx] = 1.0

        return gains


class VBAP3D:
    """
    3次元 Vector Base Amplitude Panning (Ville Pulkki 1997 原著論文準拠)。

    数学的定式化:
      3台のスピーカーの単位方向ベクトル l_1, l_2, l_3 (各 shape: (3,)) を
      列ベクトルとして並べた 3x3 行列を L_123 とする:
        L_123 = [l_1  l_2  l_3]  (shape: (3, 3), 第0列=l_1, 第1列=l_2, 第2列=l_3)
      音源単位方向ベクトル p (shape: (3,), 列ベクトル) に対し、ゲイン方程式は:
        p = g_1 * l_1 + g_2 * l_2 + g_3 * l_3 = L_123 @ g
      両辺に左から L_123^-1 を乗算することでゲインベクトル g (shape: (3,)) を求める:
        g = inv(L_123) @ p
      コード上の実装:
        g = np.dot(inv_mat, p)  # inv_mat は L_123 の逆行列

    特異性・病態トリプレットの排除:
      1. 行列式 |det(L)| > 1e-3 (同一平面・一直線上のスピーカー群を排除)
      2. 条件数 cond(L) < 100.0 (Near-singular な細長い三角形を排除)
    """
    def __init__(self, layout: SpeakerLayout3D):
        self.layout = layout
        self.is_planar = layout.is_planar()
        self.vbap_2d = VBAP2D(layout.speakers) if self.is_planar or len(layout.speakers) >= 2 else None
        self._triplets: list[tuple[int, int, int, np.ndarray]] = []

        if not self.is_planar and len(layout.speakers) >= 4:
            self._build_triplets()

    def _build_triplets(self):
        """スピーカー組み合わせから、非退化で正の向き・良条件を持つすべてのTripletを事前計算する。"""
        n = len(self.layout.speakers)
        self._triplets = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    l1 = self.layout.speakers[i].unit_vector
                    l2 = self.layout.speakers[j].unit_vector
                    l3 = self.layout.speakers[k].unit_vector
                    mat = np.column_stack([l1, l2, l3])  # 列ベクトルとして配置 (shape: 3, 3)
                    det = np.linalg.det(mat)
                    # 1. 行列式が十分に大きいこと
                    if abs(det) > 1e-3:
                        # 2. 条件数(Condition Number)が十分に小さく良条件であること (Near-singularの排除)
                        cond = np.linalg.cond(mat)
                        if cond < 100.0:
                            inv_mat = np.linalg.inv(mat)
                            self._triplets.append((i, j, k, inv_mat))

    def calculate_gains(self, target_pos: Position3D) -> np.ndarray:
        """
        音源位置 target_pos に対する全スピーカーのゲイン配列を計算。
        平面スピーカー構成の場合は 2D フォールバックへ安全に委譲。
        """
        n_spk = len(self.layout.speakers)
        if self.is_planar or not self._triplets:
            # 平面構成時や有効なTripletがない場合は 2D パンニングへ委譲 (Elevationの物理生成は行わない)
            if self.vbap_2d is None:
                return np.ones(n_spk, dtype=np.float64)  # スピーカー1台のみ
            order = {id(s): i for i, s in enumerate(self.layout.speakers)}
            gains_2d = self.vbap_2d.calculate_gains(target_pos)
            gains = np.zeros(n_spk, dtype=np.float64)
            for spk, g in zip(self.vbap_2d.speakers, gains_2d):
                gains[order[id(spk)]] = g
            return gains

        p = target_pos.to_unit_vector()
        gains = np.zeros(n_spk, dtype=np.float64)
        best_triplet = None
        best_min_g = -1.0
        best_sum_g = math.inf

        # 全組み合わせのTripletは重なり合うため、最初の一致では音源から遠い組を選ぶことがある。
        # 音源を囲む組のうち Σg が最小のもの(球面上で最も小さい三角形)を選ぶ。
        for i, j, k, inv_mat in self._triplets:
            g = np.dot(inv_mat, p)
            min_g = min(g[0], g[1], g[2])
            if min_g >= -1e-5:
                sum_g = g[0] + g[1] + g[2]
                if sum_g < best_sum_g:
                    best_sum_g = sum_g
                    best_triplet = (i, j, k, g)
            elif best_sum_g == math.inf and min_g > best_min_g:
                # 囲む組がない場合の次善策: 負のゲインが最も小さい組
                best_min_g = min_g
                best_triplet = (i, j, k, g)

        if best_triplet is not None:
            i, j, k, g = best_triplet
            gains[i] = max(0.0, float(g[0]))
            gains[j] = max(0.0, float(g[1]))
            gains[k] = max(0.0, float(g[2]))

        power = np.sum(gains ** 2)
        if power > 1e-12:
            gains /= math.sqrt(power)
        else:
            dots = [np.dot(p, s.unit_vector) for s in self.layout.speakers]
            gains[int(np.argmax(dots))] = 1.0

        return gains


# ==============================================================================
# 将来拡張用インターフェース定義 (XTC, Perceptual Elevation, Binaural)
# ==============================================================================

class ICrosstalkCanceller(ABC):
    """スピーカーバイノーラル再生用クロストークキャンセル(XTC)の抽象インターフェース。"""
    @abstractmethod
    def process(self, stereo_input: np.ndarray) -> np.ndarray:
        pass


class PassthroughCrosstalkCanceller(ICrosstalkCanceller):
    """現行Phase用パススルー実装 (信号を無加工で通過)。"""
    def process(self, stereo_input: np.ndarray) -> np.ndarray:
        return stereo_input


class IPerceptualElevationFilter(ABC):
    """水平4chスピーカー用 Perceptual Elevation Enhancement の抽象インターフェース。"""
    @abstractmethod
    def process(self, signal: np.ndarray, elevation_rad: float) -> np.ndarray:
        pass


class ISpatialRenderer(ABC):
    """空間レンダラーの基底インターフェース。"""
    @abstractmethod
    def render(self, input_block: np.ndarray, position: Position3D) -> dict[str, np.ndarray]:
        pass


class HeadphoneBinauralRenderer(ISpatialRenderer):
    """
    ヘッドホン専用バイノーラルレンダラーのスタブインターフェース。
    実SOFA/HRIRデータセット選定後に本格実装予定。
    """
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def render(self, input_block: np.ndarray, position: Position3D) -> dict[str, np.ndarray]:
        # 現段階ではスタブとしてステレオ入力をそのまま返す
        mono = np.mean(input_block, axis=1, keepdims=True) if input_block.ndim > 1 else input_block.reshape(-1, 1)
        return {"binaural_l": mono, "binaural_r": mono}
