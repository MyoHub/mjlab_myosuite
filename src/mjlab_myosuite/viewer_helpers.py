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
import trimesh.visual
import trimesh.visual.material
import viser.transforms as vtf
from mjlab.viewer.viser.conversions import mujoco_mesh_to_trimesh
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
#  Texture extraction (for planes only)
# ------------------------------------------------------------


def extract_texture_image(model, texid):
  adr = model.tex_adr[texid]
  w = model.tex_width[texid]
  h = model.tex_height[texid]
  c = model.tex_nchannel[texid]

  data = model.tex_data[adr : adr + w * h * c]
  img = data.reshape(h, w, c)
  img = np.flipud(img)

  if c == 3:
    return Image.fromarray(img.astype(np.uint8), "RGB")
  if c == 4:
    return Image.fromarray(img.astype(np.uint8), "RGBA")

  return None


def make_plane_uv(n):
  """
  UVs for a box plane — works because planes are quads.
  """
  return np.tile(
    np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
    (n // 4 + 1, 1),
  )[:n]


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
#  Plane construction
# ------------------------------------------------------------


def create_textured_plane(model, geom_idx, matid, texid):
  size = model.geom_size[geom_idx]
  s = max(size[0], size[1]) * 2

  mesh = trimesh.creation.box(extents=[s, s, 0.01])

  # apply geom transform directly
  pos = model.geom_pos[geom_idx]
  quat = model.geom_quat[geom_idx]

  T = np.eye(4)
  T[:3, :3] = vtf.SO3(quat).as_matrix()
  T[:3, 3] = pos
  mesh.apply_transform(T)

  apply_plane_texture(mesh, model, matid, texid)
  return mesh


# ------------------------------------------------------------
#  Main conversion function
# ------------------------------------------------------------


def convert_geometries_to_meshes(model, textured_geom_info=None):
  """
  Returns:
      list of (geom_idx, mesh, geom_pos, geom_quat)
      geom_pos / geom_quat = None when already baked into mesh
  """

  if textured_geom_info is None:
    textured_geom_info = find_textured_geometries(model)

  meshes = []

  for geom_idx in range(model.ngeom):
    geom_type = model.geom_type[geom_idx]
    matid, texid = textured_geom_info.get(geom_idx, (-1, -1))

    # -----------------------------------
    #  Mesh geoms  (the important case)
    # -----------------------------------
    mesh_id = model.geom_dataid[geom_idx]
    if mesh_id >= 0 and model.mesh_vertnum[mesh_id] > 0:
      try:
        mesh = mujoco_mesh_to_trimesh(model, geom_idx, verbose=False)

        # DO NOT TOUCH mesh.visual — MuJoCo already gave us UVs + textures

        pos = model.geom_pos[geom_idx]
        quat = model.geom_quat[geom_idx]
        meshes.append((geom_idx, mesh, pos, quat))
        continue
      except Exception:
        pass

    # -----------------------------------
    #  Plane (with or without texture)
    # -----------------------------------
    if geom_type == 0:  # Plane
      if texid >= 0:
        # Plane with texture - create using helper
        try:
          mesh = create_textured_plane(model, geom_idx, matid, texid)
          meshes.append((geom_idx, mesh, None, None))
          continue
        except Exception:
          pass

      # Plane without texture - try mjlab's create_primitive_mesh first
      try:
        from mjlab.viewer.viser.conversions import create_primitive_mesh

        mesh = create_primitive_mesh(model, geom_idx)
        if mesh is not None:
          pos = model.geom_pos[geom_idx]
          quat = model.geom_quat[geom_idx]
          meshes.append((geom_idx, mesh, pos, quat))
          continue
      except Exception:
        pass

      # Fallback: create simple plane mesh without texture
      try:
        size = model.geom_size[geom_idx]
        s = max(size[0], size[1]) * 2
        mesh = trimesh.creation.box(extents=[s, s, 0.01])

        # Apply transform
        pos = model.geom_pos[geom_idx]
        quat = model.geom_quat[geom_idx]
        T = np.eye(4)
        T[:3, :3] = vtf.SO3(quat).as_matrix()
        T[:3, 3] = pos
        mesh.apply_transform(T)

        meshes.append((geom_idx, mesh, None, None))
        continue
      except Exception:
        pass

    # -----------------------------------
    #  Fallback primitive (other types, untextured)
    # -----------------------------------
    try:
      from mjlab.viewer.viser.conversions import create_primitive_mesh

      mesh = create_primitive_mesh(model, geom_idx)
      if mesh is not None:
        pos = model.geom_pos[geom_idx]
        quat = model.geom_quat[geom_idx]
        meshes.append((geom_idx, mesh, pos, quat))
    except Exception:
      pass

  return meshes
