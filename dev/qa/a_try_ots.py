import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from px import *
from scenes._a_helpers import *
from scenes.s2_argue import SCENE as S, CL_X, CD_X

import math
def ots(near, far, face_ang_deg, rho, side_sign, far_off_deg, h, ty, f):
    """камера за плечом near: phi от направления взгляда near; far в far_off_deg от оси кадра."""
    nx, nz = near
    fx_ = math.cos(math.radians(face_ang_deg)) * side_sign
    ux, uz = fx_, -math.sin(math.radians(face_ang_deg))
    E = (nx + rho * ux, nz + rho * uz)
    ang = math.atan2(far[0] - E[0], far[1] - E[1])
    yaw = ang - math.radians(far_off_deg)
    T = (E[0] + 50 * math.sin(yaw), ty, E[1] + 50 * math.cos(yaw))
    return (E[0], h, E[1]), T, f
cands1 = [ots((CD_X, 0), (CL_X, 0), 55, 22, +1, -12, 13.0, 10.5, 300.0),
          ots((CD_X, 0), (CL_X, 0), 50, 24, +1, -10, 13.5, 10.5, 300.0),
          ots((CD_X, 0), (CL_X, 0), 58, 20, +1, -14, 12.5, 10.0, 290.0),
          ots((CD_X, 0), (CL_X, 0), 55, 26, +1, -11, 14.0, 10.5, 320.0)]
cands2 = [ots((CL_X, 0), (CD_X, 0), 55, 32, -1, 10, 12.5, 12.0, 300.0),
          ots((CL_X, 0), (CD_X, 0), 50, 34, -1, 12, 13.0, 12.0, 300.0),
          ots((CL_X, 0), (CD_X, 0), 58, 30, -1, 9, 12.0, 11.5, 290.0),
          ots((CL_X, 0), (CD_X, 0), 55, 36, -1, 11, 13.5, 12.0, 320.0)]
for cc in cands1 + cands2:
    print([round(v, 1) for v in cc[0]], [round(v, 1) for v in cc[1]], cc[2])
frames = []
for (eye, tgt, f) in cands1:
    cam = look_at(eye, tgt, f)
    c = S.base(cam, 0.0)
    items = S.trees_all + S.trophy_items(0.0)
    items.append(S.clawd_item(CL_X, anim='Pointing', frame=20))
    items.append(S.codex_item(CD_X, row=0, col=0, face='back'))
    draw_world(c, cam, items)
    frames.append(c)
for (eye, tgt, f) in cands2:
    cam = look_at(eye, tgt, f)
    c = S.base(cam, 0.0)
    items = S.trees_all + S.trophy_items(0.0)
    items.append(S.codex_item(CD_X, row=5, col=1))
    items.append(S.clawd_item(CL_X, face='back'))
    draw_world(c, cam, items)
    frames.append(c)
rows = [np.concatenate(frames[i:i + 2], 1) for i in range(0, 8, 2)]
big = np.concatenate(rows, 0)
Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).save(sys.argv[1])
