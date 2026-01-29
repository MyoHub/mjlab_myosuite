"""
Helper functions for converting MuJoCo geometries to trimesh for Viser rendering.

CRITICAL RULE:
MuJoCo mesh geoms already contain correct UVs + textures inside
`mujoco_mesh_to_trimesh`.  NEVER touch mesh.visual for those.

Only planes & procedural primitives get manual UVs + textures.
"""

import mujoco
import numpy as np
import trimesh
import trimesh.visual.material
import viser.transforms as vtf
from mjlab.viewer.viser.conversions import mujoco_mesh_to_trimesh
from mujoco import mjtGeom
from PIL import Image

# ------------------------------------------------------------
#  Texture discovery
# ------------------------------------------------------------


def find_textured_geometries(model):
  """
  Return dict: geom_idx -> (matid, texid)
  """
  textured = {}
  for i in range(model.ngeom):
    matid = model.geom_matid[i] if i < len(model.geom_matid) else -1
    if matid < 0:
      continue

    rgb = int(model.mat_texid[matid, mujoco.mjtTextureRole.mjTEXROLE_RGB])
    rgba = int(model.mat_texid[matid, mujoco.mjtTextureRole.mjTEXROLE_RGBA])
    texid = rgb if rgb >= 0 else rgba
    if texid >= 0:
      textured[i] = (matid, texid)
  return textured


# ------------------------------------------------------------
#  Texture extraction
# ------------------------------------------------------------


def extract_texture_image(model, texid):
  if texid is None or int(texid) < 0:
    return None
  adr = model.tex_adr[texid]
  w = model.tex_width[texid]
  h = model.tex_height[texid]
  c = model.tex_nchannel[texid]

  data = model.tex_data[adr : adr + w * h * c]
  img = data.reshape(h, w, c)

  if c == 3:
    return Image.fromarray(img.astype(np.uint8), "RGB")
  if c == 4:
    return Image.fromarray(img.astype(np.uint8), "RGBA")
  return None


# ------------------------------------------------------------
#  UV helpers
# ------------------------------------------------------------


def make_plane_uv(n):
  """UVs for a quad plane"""
  return np.tile(
    np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), (n // 4 + 1, 1)
  )[:n]


def create_box_with_mujoco_uvs(size=(1, 1, 1)):
  """
  Create a box mesh with 24 vertices (4 per face) to allow different UVs per face.

  This is the primitive cube topology we use for textured dice/box rendering.
  """
  sx, sy, sz = size
  corners = np.array(
    [
      [-sx, -sy, -sz],  # 0
      [sx, -sy, -sz],  # 1
      [sx, sy, -sz],  # 2
      [-sx, sy, -sz],  # 3
      [-sx, -sy, sz],  # 4
      [sx, -sy, sz],  # 5
      [sx, sy, sz],  # 6
      [-sx, sy, sz],  # 7
    ],
    dtype=np.float32,
  )

  face_defs = {
    "F": [4, 5, 6, 7],  # Front (+Z)
    "B": [1, 0, 3, 2],  # Back (-Z)
    "R": [5, 1, 2, 6],  # Right (+X)
    "L": [0, 4, 7, 3],  # Left (-X)
    "U": [3, 2, 6, 7],  # Up (+Y)
    "D": [4, 5, 1, 0],  # Down (-Y)
  }

  vertices: list[np.ndarray] = []
  faces: list[list[int]] = []
  vertex_offset = 0

  for _, corner_indices in face_defs.items():
    face_vertices = corners[corner_indices]
    vertices.extend(face_vertices)
    faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
    faces.append([vertex_offset, vertex_offset + 2, vertex_offset + 3])
    vertex_offset += 4

  return trimesh.Trimesh(
    vertices=np.array(vertices, dtype=np.float32),
    faces=np.array(faces, dtype=np.int64),
    process=False,
  )


# ------------------------------------------------------------
#  Dice box helpers (used by minimal test + play.py)
# ------------------------------------------------------------


_DICE_FACE_DEFS_NAMED: dict[str, list[int]] = {
  "front": [4, 5, 6, 7],  # Z+
  "back": [1, 0, 3, 2],  # Z-
  "right": [5, 1, 2, 6],  # X+
  "left": [0, 4, 7, 3],  # X-
  "top": [7, 6, 2, 3],  # Y+  (matches working equivalence script)
  "bottom": [4, 0, 1, 5],  # Y-
}


def get_uv_mapping_for_dice_file_texture() -> dict[
  str, tuple[float, float, float, float]
]:
  """UV mapping for dice.png on disk: 4×3 T-shaped grid (`..U.LFRB..D.`)."""
  rows, cols = 3, 4
  cw = 1.0 / cols
  ch = 1.0 / rows
  positions: dict[str, tuple[int, int]] = {
    "top": (2, 0),
    "left": (0, 1),
    "front": (1, 1),
    "right": (2, 1),
    "back": (3, 1),
    "bottom": (2, 2),
  }
  uv_mapping: dict[str, tuple[float, float, float, float]] = {}
  for face, (col, row) in positions.items():
    u_min = col * cw
    v_min = row * ch
    u_max = (col + 1) * cw
    v_max = (row + 1) * ch
    uv_mapping[face] = (u_min, v_min, u_max, v_max)
  return uv_mapping


def get_uv_mapping_for_dice_model_texture_vertical_strip() -> dict[
  str, tuple[float, float, float, float]
]:
  """UV mapping for MuJoCo model dice texture: 6×1 vertical strip.

  Order (verified in working equivalence script): L, R, F, B, U, D.
  """
  face_order = ["left", "right", "front", "back", "top", "bottom"]
  uv_mapping: dict[str, tuple[float, float, float, float]] = {}
  for i, face in enumerate(face_order):
    v_min = i / 6.0
    v_max = (i + 1) / 6.0
    uv_mapping[face] = (0.0, v_min, 1.0, v_max)
  return uv_mapping


def choose_dice_uv_mapping(
  texture: Image.Image,
) -> dict[str, tuple[float, float, float, float]]:
  """Pick 4×3 vs 6×1 mapping based on texture aspect ratio."""
  w, h = texture.size
  if h == 0:
    return get_uv_mapping_for_dice_file_texture()
  aspect = w / h
  # file texture: 1024x768 => 1.333..
  if abs(aspect - (4.0 / 3.0)) < 1e-3:
    return get_uv_mapping_for_dice_file_texture()
  # model texture often comes as 256x1536 => 0.1666..
  if abs(aspect - (1.0 / 6.0)) < 1e-3:
    return get_uv_mapping_for_dice_model_texture_vertical_strip()
  # sometimes exported/packed as horizontal 6x1 strip
  if abs(aspect - 6.0) < 1e-3:
    # map across U; same face order
    face_order = ["left", "right", "front", "back", "top", "bottom"]
    uv_mapping: dict[str, tuple[float, float, float, float]] = {}
    for i, face in enumerate(face_order):
      u_min = i / 6.0
      u_max = (i + 1) / 6.0
      uv_mapping[face] = (u_min, 0.0, u_max, 1.0)
    return uv_mapping
  # fallback: keep using 4×3
  return get_uv_mapping_for_dice_file_texture()


def create_textured_box_with_custom_uvs(
  *,
  size: np.ndarray,
  texture: Image.Image,
  uv_mapping: dict[str, tuple[float, float, float, float]],
  position: np.ndarray | None = None,
  rotation: np.ndarray | None = None,
  flip_uv_horizontal: bool = True,
) -> trimesh.Trimesh:
  """Create a textured box with per-face UV rectangles (24 verts total)."""
  sx, sy, sz = size
  vertices = np.array(
    [
      [-sx, -sy, -sz],
      [sx, -sy, -sz],
      [sx, sy, -sz],
      [-sx, sy, -sz],
      [-sx, -sy, sz],
      [sx, -sy, sz],
      [sx, sy, sz],
      [-sx, sy, sz],
    ],
    dtype=np.float32,
  )
  if rotation is not None:
    vertices = (rotation @ vertices.T).T
  if position is not None:
    vertices = vertices + position

  mesh_vertices: list[np.ndarray] = []
  mesh_faces: list[list[int]] = []
  mesh_uvs: list[tuple[float, float]] = []
  vertex_offset = 0

  for face_name, vert_indices in _DICE_FACE_DEFS_NAMED.items():
    face_verts = [vertices[i] for i in vert_indices]
    mesh_vertices.extend(face_verts)
    mesh_faces.append([vertex_offset + 0, vertex_offset + 1, vertex_offset + 2])
    mesh_faces.append([vertex_offset + 0, vertex_offset + 2, vertex_offset + 3])

    u_min, v_min, u_max, v_max = uv_mapping.get(face_name, (0.0, 0.0, 1.0, 1.0))
    if flip_uv_horizontal:
      face_uvs = [
        (u_max, v_max),
        (u_min, v_max),
        (u_min, v_min),
        (u_max, v_min),
      ]
    else:
      face_uvs = [
        (u_min, v_max),
        (u_max, v_max),
        (u_max, v_min),
        (u_min, v_min),
      ]
    mesh_uvs.extend(face_uvs)
    vertex_offset += 4

  mesh = trimesh.Trimesh(
    vertices=np.array(mesh_vertices, dtype=np.float32),
    faces=np.array(mesh_faces, dtype=np.int32),
    process=False,
  )

  material = trimesh.visual.material.SimpleMaterial(
    image=texture.convert("RGBA"), diffuse=[255, 255, 255, 255]
  )
  mesh.visual = trimesh.visual.TextureVisuals(
    uv=np.array(mesh_uvs, dtype=np.float32), material=material
  )
  return mesh


def create_textured_dice_box_mesh(
  mj_model: mujoco.MjModel,
  geom_id: int,
  *,
  bake_transform: bool = False,
) -> trimesh.Trimesh | None:
  """Create a textured dice box mesh from a MuJoCo model geom.

  Uses the model-provided texture and automatically selects 4×3 vs 6×1 UV mapping.
  """
  if mj_model.geom_type[geom_id] != mjtGeom.mjGEOM_BOX:
    return None

  matid = (
    int(mj_model.geom_matid[geom_id]) if geom_id < len(mj_model.geom_matid) else -1
  )
  texid = -1
  if 0 <= matid < mj_model.nmat:
    rgb = int(mj_model.mat_texid[matid, mujoco.mjtTextureRole.mjTEXROLE_RGB])
    rgba_tex = int(mj_model.mat_texid[matid, mujoco.mjtTextureRole.mjTEXROLE_RGBA])
    texid = rgb if rgb >= 0 else rgba_tex

  if texid < 0:
    return None

  img = extract_texture_image(mj_model, texid)
  if img is None:
    return None

  uv_mapping = choose_dice_uv_mapping(img)

  # Optional transform baking.
  pos = mj_model.geom_pos[geom_id] if bake_transform else None
  rot = None
  if bake_transform:
    quat = mj_model.geom_quat[geom_id]
    if not np.allclose(quat, [1, 0, 0, 0]):
      rot_mat = np.zeros((3, 3))
      mujoco.mju_quat2Mat(rot_mat.flatten(), quat)
      rot = rot_mat.reshape(3, 3)
    else:
      rot = np.eye(3)

  return create_textured_box_with_custom_uvs(
    size=mj_model.geom_size[geom_id],
    texture=img,
    uv_mapping=uv_mapping,
    position=pos,
    rotation=rot,
  )


def apply_plane_texture(mesh, model, matid, texid):
  img = extract_texture_image(model, texid)
  if img is None:
    return
  uv = make_plane_uv(len(mesh.vertices))
  rgba = model.mat_rgba[matid] if matid < len(model.mat_rgba) else np.ones(4)
  material = trimesh.visual.material.PBRMaterial(
    baseColorFactor=rgba,
    baseColorTexture=img,
    metallicFactor=0.0,
    roughnessFactor=1.0,
  )
  mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)


# ------------------------------------------------------------
#  Primitive mesh creation
# ------------------------------------------------------------


def create_primitive_mesh(mj_model, geom_id, matid=-1, texid=-1):
  geom_type = mj_model.geom_type[geom_id]
  size = mj_model.geom_size[geom_id]
  rgba = mj_model.geom_rgba[geom_id].copy()
  rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

  if geom_type == mjtGeom.mjGEOM_SPHERE:
    mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif geom_type == mjtGeom.mjGEOM_BOX:
    # Prefer dice-specific textured box when a texture is present.
    if texid is not None and int(texid) >= 0:
      dice_mesh = create_textured_dice_box_mesh(mj_model, geom_id, bake_transform=False)
      if dice_mesh is not None:
        mesh = dice_mesh
      else:
        mesh = create_box_with_mujoco_uvs(size=size)
    else:
      mesh = create_box_with_mujoco_uvs(size=size)

    # Ensure we always have some visuals (texture or vertex colors).
    if (
      getattr(mesh, "visual", None) is None
      or getattr(mesh.visual, "kind", None) is None
    ):
      mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(rgba_uint8, (len(mesh.vertices), 1)),
      )
  elif geom_type == mjtGeom.mjGEOM_CAPSULE:
    mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_CYLINDER:
    mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_PLANE:
    mesh = trimesh.creation.box((20, 20, 0.01))
    if texid >= 0:
      apply_plane_texture(mesh, mj_model, matid, texid)
  elif geom_type == mjtGeom.mjGEOM_ELLIPSOID:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.apply_scale(size)
  else:
    # fallback: simple box
    mesh = trimesh.creation.box(2.0 * size)
    mesh.visual = trimesh.visual.ColorVisuals(
      mesh=mesh, vertex_colors=np.tile(rgba_uint8, (len(mesh.vertices), 1))
    )
  return mesh


# ------------------------------------------------------------
#  Plane construction
# ------------------------------------------------------------


def create_textured_plane(model, geom_idx, matid, texid):
  size = model.geom_size[geom_idx]
  s = max(size[0], size[1]) * 2
  mesh = trimesh.creation.box([s, s, 0.01])
  pos = model.geom_pos[geom_idx]
  quat = model.geom_quat[geom_idx]
  T = np.eye(4)
  T[:3, :3] = vtf.SO3(quat).as_matrix()
  T[:3, 3] = pos
  mesh.apply_transform(T)
  if texid >= 0:
    apply_plane_texture(mesh, model, matid, texid)
  return mesh


# ------------------------------------------------------------
#  Main conversion
# ------------------------------------------------------------


def convert_geometries_to_meshes(model, textured_geom_info=None):
  if textured_geom_info is None:
    textured_geom_info = find_textured_geometries(model)

  meshes = []
  for geom_idx in range(model.ngeom):
    geom_type = model.geom_type[geom_idx]
    matid, texid = textured_geom_info.get(geom_idx, (-1, -1))
    mesh_id = model.geom_dataid[geom_idx]

    # MuJoCo baked mesh
    if mesh_id >= 0 and model.mesh_vertnum[mesh_id] > 0:
      mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)
      pos = model.geom_pos[geom_idx]
      quat = model.geom_quat[geom_idx]
      meshes.append((geom_idx, mesh, pos, quat))
      continue

    # Plane
    if geom_type == mjtGeom.mjGEOM_PLANE:
      mesh = create_textured_plane(model, geom_idx, matid, texid)
      meshes.append((geom_idx, mesh, None, None))
      continue

    # Other primitives

    mesh = create_primitive_mesh(model, geom_idx, matid, texid)
    pos = model.geom_pos[geom_idx]
    quat = model.geom_quat[geom_idx]
    meshes.append((geom_idx, mesh, pos, quat))

  return meshes


def extract_dice_obj_data(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Backwards-compatible helper used by `test.py`.

  Returns a textured dice mesh (baked into world pose) when possible, otherwise a
  colored box mesh (also baked).
  """
  mesh = create_textured_dice_box_mesh(mj_model, geom_id, bake_transform=True)
  if mesh is not None:
    return mesh

  # Fallback: untextured colored box with baked pose.
  mesh = create_box_with_mujoco_uvs(size=mj_model.geom_size[geom_id])
  pos = mj_model.geom_pos[geom_id]
  quat = mj_model.geom_quat[geom_id]
  T = np.eye(4)
  T[:3, :3] = vtf.SO3(quat).as_matrix()
  T[:3, 3] = pos
  mesh.apply_transform(T)

  rgba = mj_model.geom_rgba[geom_id].copy()
  rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
  mesh.visual = trimesh.visual.ColorVisuals(
    mesh=mesh, vertex_colors=np.tile(rgba_uint8, (len(mesh.vertices), 1))
  )
  return mesh
