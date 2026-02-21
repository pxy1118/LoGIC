from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from stl import mesh


@dataclass(frozen=True)
class TPMSSurfaceMetrics:
    """TPMS 多孔结构比表面积与体积等指标（长度单位：mm）。"""

    surface_area_mm2: float
    solid_volume_mm3: float
    mass_g: float
    porosity: float
    specific_surface_area_mm2_per_g: float
    specific_surface_area_mm2_per_mm3: float

    def format_report(
        self,
        *,
        stl_path: Path,
        material: str,
        bounding_box_mm: Optional[np.ndarray] = None,
    ) -> str:
        """生成对该样本的分析摘要。"""
        header = "=" * 50
        bbox_line = (
            f"包络尺寸:       {bounding_box_mm[0]:.2f} × {bounding_box_mm[1]:.2f} × {bounding_box_mm[2]:.2f} mm"
            if bounding_box_mm is not None
            else None
        )

        lines = [
            "",
            header,
            "✅ TPMS 多孔结构比表面积分析结果",
            header,
            f"STL 文件:       {stl_path.as_posix()}",
            *([bbox_line] if bbox_line else []),
            f"总表面积:       {self.surface_area_mm2:,.1f} mm²",
            f"实体体积:       {self.solid_volume_mm3:,.1f} mm³",
            f"质量 ({material}): {self.mass_g:.4f} g",
            f"孔隙率:         {self.porosity * 100:.2f} %",
            "-" * 50,
            f"质量比表面积:   {self.specific_surface_area_mm2_per_g:.1f} mm²/g",
            f"体积比表面积:   {self.specific_surface_area_mm2_per_mm3:.4f} mm⁻¹",
            header,
        ]

        return "\n".join(lines)


def _ensure_mesh(stl_path: Path) -> mesh.Mesh:
    if not stl_path.exists():
        raise FileNotFoundError(f"文件不存在: {stl_path}")

    try:
        return mesh.Mesh.from_file(stl_path.as_posix(), calculate_normals=False)
    except Exception as exc:  # pragma: no cover - 底层库错误文本即可
        raise RuntimeError(f"无法读取 STL 文件，请确认文件格式正确: {stl_path}") from exc


def _compute_area_and_volume_mm(m: mesh.Mesh) -> Tuple[float, float]:
    vectors = np.asarray(m.vectors, dtype=np.float64)
    if vectors.size == 0:
        raise ValueError("STL 文件不包含任何三角面片。")

    v0 = vectors[:, 0]
    v1 = vectors[:, 1]
    v2 = vectors[:, 2]

    edges1 = v1 - v0
    edges2 = v2 - v0
    cross = np.cross(edges1, edges2)
    area_mm2 = 0.5 * np.linalg.norm(cross, axis=1).sum()

    centroid = vectors.reshape(-1, 3).mean(axis=0)
    v0_centered = v0 - centroid
    v1_centered = v1 - centroid
    v2_centered = v2 - centroid
    volume_terms = np.einsum("ij,ij->i", v0_centered, np.cross(v1_centered, v2_centered))
    volume_mm3 = abs(volume_terms.sum()) / 6.0

    if volume_mm3 <= 0:
        raise ValueError("模型非封闭！体积计算无效，请检查 STL 是否水密。")

    return area_mm2, volume_mm3


def _infer_bounding_box_mm(vectors: np.ndarray) -> np.ndarray:
    min_corner = vectors.reshape(-1, 3).min(axis=0)
    max_corner = vectors.reshape(-1, 3).max(axis=0)
    return max_corner - min_corner


def calculate_tpm_ssa(
    stl_path: str | Path,
    *,
    density_g_per_cm3: float = 4.43,
    expected_size_mm: Optional[float] = 20.0,
) -> tuple[TPMSSurfaceMetrics, Optional[np.ndarray]]:
    """计算 TPMS 多孔结构的比表面积与相关指标，全部单位为 mm / g。"""

    path = Path(stl_path)
    m = _ensure_mesh(path)

    if density_g_per_cm3 <= 0:
        raise ValueError("材料密度必须为正值。")

    area_mm2, volume_mm3 = _compute_area_and_volume_mm(m)

    volume_cm3 = volume_mm3 * 1e-3
    mass_g = volume_cm3 * density_g_per_cm3

    specific_surface_area_mass = area_mm2 / mass_g
    specific_surface_area_volume = area_mm2 / volume_mm3

    vectors = np.asarray(m.vectors)
    if expected_size_mm is None:
        bbox = _infer_bounding_box_mm(vectors)
        total_volume_mm3 = float(np.prod(bbox)) if np.all(bbox > 0) else 0.0
    else:
        bbox = None
        total_volume_mm3 = expected_size_mm ** 3 if expected_size_mm > 0 else 0.0

    porosity = 1 - (volume_mm3 / total_volume_mm3) if total_volume_mm3 > 0 else 0.0

    metrics = TPMSSurfaceMetrics(
        surface_area_mm2=area_mm2,
        solid_volume_mm3=volume_mm3,
        mass_g=mass_g,
        porosity=porosity,
        specific_surface_area_mm2_per_g=specific_surface_area_mass,
        specific_surface_area_mm2_per_mm3=specific_surface_area_volume,
    )
    return metrics, bbox


__all__ = [
    "TPMSSurfaceMetrics",
    "calculate_tpm_ssa",
]
