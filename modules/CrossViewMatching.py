import os
import sys
sys.path.append('./')
import cv2
import torch
import numpy as np
from modules.yolox.yolox import Predictor
from modules.alphapose_core.alphapose_core import AlphaPose_Predictor
from modules.vggt.vggt_core import VGGT_Predictor
from pathlib import Path

class MultiViewReIDMatcher:
    """
    多视角 ReID（基于 VGGT 、Alphapose、YOLOX）
    核心逻辑：基于视角迭代式贪心匹配
    """

    def __init__(self,
                 yolox,
                 alphapose,
                 vggt,
                 device=None,
                 verbose=True,
                 max_dist=None,
                 kp_score_th=None,
                 wp_conf_th=None):
        self.verbose = bool(verbose)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ========= 默认算法参数 =========
        self.KP_SCORE_TH = 0.20
        self.WP_CONF_TH  = 0.20
        self.MAX_DIST    = 0.50

        self.GATE_MIN_FACTOR = 0.6
        self.GATE_MAX_FACTOR = 1.6

        self.WSUM_REF = 3.0
        self.K_REF    = 12

        self.R_USE_MATCH  = 0.15   # 是否进行匹配的可靠度阈值
        self.R_TO_SEED    = 0.35   # 是否能够初始化为全局点

        if max_dist is not None:
            self.MAX_DIST = float(max_dist)
        if kp_score_th is not None:
            self.KP_SCORE_TH = float(kp_score_th)
        if wp_conf_th is not None:
            self.WP_CONF_TH = float(wp_conf_th)

        self.GATE_MIN = self.MAX_DIST * self.GATE_MIN_FACTOR
        self.GATE_MAX = self.MAX_DIST * self.GATE_MAX_FACTOR

        self.yolox = yolox
        self.alphapose = alphapose
        self.vggt = vggt

    # ----------------- 工具函数 -----------------
    @staticmethod
    def list_images(root_dir):
        root = Path(root_dir)
        exts = {'.jpg', '.jpeg', '.png'}
        
        results = []

        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                imgs = sorted([p for p in sub.iterdir() if p.suffix.lower() in exts])
                if imgs:
                    results.append(str(imgs[0]))

        return results

    @staticmethod
    def map_xy_orig_to_square(x, y, orig_w, orig_h, oc):
        x0, y0, x1, y1 = oc[:4]
        region_w = max(1.0, x1 - x0)
        region_h = max(1.0, y1 - y0)
        sx = region_w / float(orig_w)
        sy = region_h / float(orig_h)
        x_sq = x0 + float(x) * sx
        y_sq = y0 + float(y) * sy
        return x_sq, y_sq

    @staticmethod
    def bilinear_sample_wp(wp_map, wp_conf, x, y):
        H, W = wp_conf.shape
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))

        x0 = int(np.floor(x)); x1 = min(x0 + 1, W - 1)
        y0 = int(np.floor(y)); y1 = min(y0 + 1, H - 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        p = (wa * wp_map[y0, x0] + wb * wp_map[y0, x1] +
             wc * wp_map[y1, x0] + wd * wp_map[y1, x1])

        c = (wa * wp_conf[y0, x0] + wb * wp_conf[y0, x1] +
             wc * wp_conf[y1, x0] + wd * wp_conf[y1, x1])
        return p, float(c)

    def weighted_3d_aggregate(self, pose_kpts, wp_map, wp_conf, oc, orig_w, orig_h,
                              bbox_conf=1.0, kp_th=0.2, wp_th=0.2):
        pts, wts = [], []
        wsum = 0.0

        for x, y, s in pose_kpts:
            s = float(s)
            if s < kp_th:
                continue

            x_sq, y_sq = self.map_xy_orig_to_square(x, y, orig_w, orig_h, oc)
            p, c = self.bilinear_sample_wp(wp_map, wp_conf, x_sq, y_sq)

            if c < wp_th:
                continue
            if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                continue

            w = s * c * float(bbox_conf)
            pts.append(p)
            wts.append(w)
            wsum += w

        n_valid = len(pts)
        if n_valid == 0 or wsum <= 1e-9:
            return None, 0.0, 0

        pts = np.stack(pts, axis=0)
        wts = np.array(wts, dtype=np.float64)
        wts = wts / (wts.sum() + 1e-12)
        desc = (pts * wts[:, None]).sum(axis=0)
        return desc, float(wsum), int(n_valid)

    def reliability_from_stats(self, wsum, n_valid):
        r1 = min(1.0, max(0.0, float(wsum) / float(self.WSUM_REF)))
        r2 = min(1.0, max(0.0, float(n_valid) / float(self.K_REF)))
        return float(r1 * r2)

    def adaptive_gate(self, r):
        r = float(np.clip(r, 0.0, 1.0))
        return float(self.GATE_MIN + (1.0 - r) * (self.GATE_MAX - self.GATE_MIN))

    @staticmethod
    def color_for_id(idx):
        palette = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 128, 255),
            (255, 128, 0),
            (128, 0, 255),
            (0, 255, 128),
        ]
        return palette[idx % len(palette)]

    def draw_boxes(self, img_bgr, bboxes, pid_to_gid, thickness=2):
        out = img_bgr.copy()
        for pid, box in enumerate(bboxes):
            if pid not in pid_to_gid:
                continue
            gid = pid_to_gid[pid]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(out.shape[1] - 1, x2); y2 = min(out.shape[0] - 1, y2)

            color = self.color_for_id(gid)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(out, f"ID:{gid}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return out
    # 模块处理
    def run_model(self, image_paths):
        """
        对输入的图片列表，依次执行Yolox、Alphapose、VGGT，得到每个视角的检测框、关键点、描述子等信息
         - det_pose_per_view: list of dict, 每个元素对应一个视角，包含 "bboxes" (N,4), "bboxes_conf" (N,), "pose" (N,K,3)
         - orig_imgs: list of 原始图片的BGR numpy数组
         - orig_sizes: list of (W,H) tuple，原始图片的宽高
         - view_counts: list of 每个视角检测到的人数
         - wp: list of VGGT 的全局特征图 (C,H,W)
         - wpc: list of VGGT 的全局特征图置信度 (H,W)
        """
        orig_imgs = []
        orig_sizes = []
        det_pose_per_view = []

        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(f"Failed to read image: {p}")
            H, W = img.shape[:2]
            orig_imgs.append(img)
            orig_sizes.append((W, H))

            det_frame, _ = self.yolox.predict(img, viz=False)
            bboxes = np.asarray(det_frame["bbox"], dtype=np.float32)
            bboxes_conf = np.asarray(det_frame["bboxes_conf"], dtype=np.float32)
            pose = self.alphapose.predict(img, bboxes)  # (N,K,3)

            det_pose_per_view.append({
                "bboxes": bboxes,
                "bboxes_conf": bboxes_conf,
                "pose": pose,
            })

        view_counts = [d["pose"].shape[0] for d in det_pose_per_view]
        if self.verbose:
            print("views:", len(image_paths), "people per view:", view_counts)
        wp, wpc, ocr = self.vggt.predict(image_paths)
        return det_pose_per_view, orig_imgs, orig_sizes, view_counts,wp,wpc,ocr
    
    # 描述子和置信度计算
    def calculate_confidence(self, det_pose_per_view,orig_sizes,view_counts,wp,wpc,ocr):
        """
        根据检测到的关键点和 VGGT 的特征图，计算每个检测到的人的描述子和置信度
        """
        per_view_items = []
        for v in range(len(det_pose_per_view)):
            pose = det_pose_per_view[v]["pose"]
            bconf = det_pose_per_view[v]["bboxes_conf"]
            W0, H0 = orig_sizes[v]
            oc = ocr[v].cpu().numpy()

            items = []
            n_view = view_counts[v]
            for pid in range(n_view):
                desc, wsum, n_valid = self.weighted_3d_aggregate(
                    pose[pid],
                    wp[v],
                    wpc[v],
                    oc=oc,
                    orig_w=W0,
                    orig_h=H0,
                    bbox_conf=float(bconf[pid]) if pid < len(bconf) else 1.0,
                    kp_th=self.KP_SCORE_TH,
                    wp_th=self.WP_CONF_TH,
                )
                r = self.reliability_from_stats(wsum, n_valid) if desc is not None else 0.0
                desc_use = desc if (desc is not None and r >= self.R_USE_MATCH) else None
                gate = self.adaptive_gate(r)

                items.append({
                    "pid": pid,
                    "desc": desc_use,
                    "r": r,
                    "gate": gate,
                    "raw": {
                        "bbox": det_pose_per_view[v]["bboxes"][pid].copy() if pid < len(det_pose_per_view[v]["bboxes"]) else None,
                        "bbox_conf": float(bconf[pid]) if pid < len(bconf) else None,
                        "pose": pose[pid].copy(),
                        "wsum": float(wsum),
                        "n_valid": int(n_valid),
                    }
                })
            per_view_items.append(items)
        return per_view_items
    
    #  贪心匹配：ID与其余视角所有点进行匹配
    def greedy_match_one_seed_to_views(self, seed_gid, remaining_by_view, view_pid_to_gid, globals_desc):
        """
        选中一个ID,在其他视角的剩余点中，寻找最好的匹配关系
        如果存在匹配关系，就匹配，并更新全局描述子的位置（平滑），直到匹配完所有视角
        """
        changed = False

        for v in range(len(remaining_by_view)):
            if seed_gid in set(view_pid_to_gid[v].values()):
                continue

            gdesc = globals_desc[seed_gid]["desc"]
            if gdesc is None:
                continue

            best_pid = None
            best_dist = 1e9
            best_item = None

            # 在这个 view 的 remaining 里找最好的匹配
            for it in remaining_by_view[v]:
                if it["desc"] is None:
                    continue

                dist = float(np.linalg.norm(it["desc"] - gdesc))
                if dist <= it["gate"] and dist < best_dist:
                    best_dist = dist
                    best_pid = it["pid"]
                    best_item = it

            if best_pid is not None:
                # 分配 gid
                view_pid_to_gid[v][best_pid] = seed_gid

                # 更新 global desc (running average)
                g = globals_desc[seed_gid]
                w0 = float(g["w"])
                g["desc"] = (g["desc"] * w0 + best_item["desc"]) / (w0 + 1.0 + 1e-12)
                g["w"] = w0 + 1.0

                # 从 remaining 删除这个 pid
                remaining_by_view[v] = [x for x in remaining_by_view[v] if x["pid"] != best_pid]
                changed = True

                if self.verbose:
                    print(f"  [Match] seed_gid={seed_gid} matched view{v} pid={best_pid} dist={best_dist:.4f}")

        return changed

    # ----------------- 主流程：迭代播种 -----------------
    def iterative_seed_match(self, per_view_items):
        """
        进行ID匹配的函数：
        思路：首先，从所有视角点中挑出置信度高的，作为初始剩余点
        然后，循环执行：
            1) 便利找到第一个有剩余点的视角
            2) 从视角剩余点中挑出置信度最高的，作为新的 seed gid
            3) 用这个 seed gid 去贪心匹配其他视角的剩余点
            4) 重复，直到没有剩余点
        """
        # 定义在循环内，负责挑选点的函数
        def pick_next_seed():
            # 优先选置信度最大的点作为匹配点
            best = None  
            for v in range(S):
                if len(remaining_by_view[v]) == 0:
                    continue
                cand = sorted(remaining_by_view[v], key=lambda x: x["r"], reverse=True)
                chosen = None
                for it in cand:
                    if it["r"] >= self.R_TO_SEED:
                        chosen = it
                        break
                if chosen is None:
                    chosen = cand[0]
                best = (v, chosen)
                break
            return best
        S = len(per_view_items)
        view_pid_to_gid = [dict() for _ in range(S)]

        # 通过置信度筛选每个视角可用点
        remaining_by_view = []
        for v in range(S):
            rem = []
            for it in per_view_items[v]:
                if it["desc"] is None or it["r"] < self.R_USE_MATCH:
                    continue
                rem.append(it)
            remaining_by_view.append(rem)

        globals_desc = []  # [{"desc":..., "w":...}, ...]

        while True:
            seed = pick_next_seed()
            if seed is None:
                break

            seed_view, seed_item = seed
            seed_pid = seed_item["pid"]

            gid = len(globals_desc)
            globals_desc.append({"desc": seed_item["desc"].copy(), "w": 1.0})

            view_pid_to_gid[seed_view][seed_pid] = gid
            remaining_by_view[seed_view] = [x for x in remaining_by_view[seed_view] if x["pid"] != seed_pid]

            if self.verbose:
                print(f"\n[Seed] new gid={gid} from view{seed_view} pid={seed_pid} r={seed_item['r']:.3f}")

            while True:
                changed = self.greedy_match_one_seed_to_views(
                    seed_gid=gid,
                    remaining_by_view=remaining_by_view,
                    view_pid_to_gid=view_pid_to_gid,
                    globals_desc=globals_desc
                )
                if not changed:
                    break
            

        return view_pid_to_gid, globals_desc

    # 主运行函数
    def run(self,
            folder="images2",
            out_dir="output/output_images2/reid_result",
            save_visualization=True):
        """
        输入文件夹路径和输出路径，执行完整的 ReID 流程
        """
        os.makedirs(out_dir, exist_ok=True)

        image_paths = self.list_images(folder)
        S = len(image_paths)
        if S == 0:
            raise RuntimeError(f"No images found in folder: {folder}")
        # ---------- 基础模块运行 ----------
        det_pose_per_view, orig_imgs, orig_sizes, view_counts,wp,wpc,ocr = self.run_model(image_paths)
        
        # ---------- 计算置信度和描述子 ----------
        per_view_items = self.calculate_confidence(det_pose_per_view, orig_sizes, view_counts,wp,wpc,ocr)

        # ---------- prematch 可视化----------
        if save_visualization:
            for v in range(S):
                bboxes = det_pose_per_view[v]["bboxes"]
                n_view = view_counts[v]
                pid_to_gid = {pid: pid for pid in range(n_view)}  # 本地 pid
                vis = self.draw_boxes(orig_imgs[v], bboxes, pid_to_gid, thickness=2)
                out_path = os.path.join(out_dir, f"prematch_view{v}_{os.path.basename(image_paths[v])}")
                cv2.imwrite(out_path, vis)
                if self.verbose:
                    print("Saved (prematch):", out_path)

        # ---------- 匹配ID ----------
        view_pid_to_gid, globals_desc = self.iterative_seed_match(per_view_items)

        # ---------- 匹配可视化 ----------
        if save_visualization:
            for v in range(S):
                bboxes = det_pose_per_view[v]["bboxes"]
                pid_to_gid = view_pid_to_gid[v]
                vis = self.draw_boxes(orig_imgs[v], bboxes, pid_to_gid, thickness=2)
                out_path = os.path.join(out_dir, f"view{v}_{os.path.basename(image_paths[v])}")
                cv2.imwrite(out_path, vis)
                if self.verbose:
                    print("Saved:", out_path)

        # ---------- 7) 输出结果 ----------
        outputs = {}
        for v in range(S):
            outputs[v] = {}
            for it in per_view_items[v]:
                pid = it["pid"]
                if pid not in view_pid_to_gid[v]:
                    continue  
                gid = view_pid_to_gid[v][pid]
                outputs[v][gid] = {
                    "pid": pid,
                    "yolox": {
                        "bbox": it["raw"]["bbox"],
                        "bbox_conf": it["raw"]["bbox_conf"]
                    }
                }

        return outputs


if __name__ == "__main__":
    
    YOLOX_MODEL = r"pretrained/yolox_data/bytetrack_x_mot17.pth.tar"
    YOLOX_THRES = 0.23

    ALPHA_CONFIG = r"modules/alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml"
    ALPHA_CKPT   = r"pretrained/alphapose_data/halpe26_fast_res50_256x192.pth"
    ALPHA_THRES  = 0.1

    VGGT_CKPT = r"pretrained/vggt_data/model.pt"
    RESOLUTION = 518 

    yolox = Predictor(YOLOX_MODEL, YOLOX_THRES)

    alphapose = AlphaPose_Predictor(ALPHA_CONFIG, ALPHA_CKPT, ALPHA_THRES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vggt = VGGT_Predictor(VGGT_CKPT, RESOLUTION, device=device)

    matcher = MultiViewReIDMatcher(verbose=True, yolox=yolox, alphapose=alphapose, vggt=vggt)
    outputs = matcher.run(
        folder="demo/argue10",
        out_dir="output/output_images3/reid_result"
    )


