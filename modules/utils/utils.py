from pathlib import Path
import cv2
import numpy as np
import json
from modules.yolox.yolox import Predictor
from modules.alphapose_core.alphapose_core import AlphaPose_Predictor
from modules.vggt.vggt_core import VGGT_Predictor
from modules.samurai.samurai_core import SAMURAI_core
from tqdm import tqdm

VALID_EXTS = {'.jpg', '.jpeg', '.png'}


# 列出输入目录下所有相机子目录
def list_camera_dirs(images_root: str):
    root = Path(images_root)
    cams = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return cams


# 列出某个相机目录下的所有帧图像
def list_frames(camera_dir: Path):
    frames = sorted([p for p in camera_dir.iterdir() if p.suffix.lower() in VALID_EXTS])
    return frames

# 判断一个 bbox 是否有效
def bbox_valid(b):
    if b is None:
        return False
    if len(b) != 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in b]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return False
    # 太小的框无效
    return (x2 - x1) > 1.0 and (y2 - y1) > 1.0


# 将 bbox 标准化为 float 列表
def to_float_bbox(b):
    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]


# 保存 keypoints 到目标 json 文件
def save_keypoints_json(path: Path, people):
    out = {
        'version': 1.1,
        'people': people,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 以 UTF-8 写文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)


# 中间结果路径默认与keypoints同级别
def get_process_root(output_root: str, process_root: str = ''):
    if process_root:
        return Path(process_root)
    out_path = Path(output_root)

    if out_path.parent.name == 'keypoints':
        return out_path.parent.parent / 'data_process_result'

    return out_path.parent / 'data_process_result'


# 按 gid 在图像上绘制 bbox 和 ID
def draw_bboxes_with_gid(img, gid_to_bbox):
    palette = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255), (255, 128, 0),
        (128, 0, 255), (0, 255, 128),
    ]
    out = img.copy()
    for gid, box in gid_to_bbox.items():
        if not bbox_valid(box):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = palette[int(gid) % len(palette)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f'ID:{gid}', (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


# 模型准备
def build_models(device):
    # YOLOX
    yolox_model = 'pretrained/yolox_data/bytetrack_x_mot17.pth.tar'
    yolox_thres = 0.23
    yolox = Predictor(yolox_model, yolox_thres)

    # AlphaPose 
    alpha_config = 'modules/alphapose_core/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
    alpha_ckpt = 'pretrained/alphapose_data/halpe26_fast_res50_256x192.pth'
    alpha_thres = 0.1
    alphapose = AlphaPose_Predictor(alpha_config, alpha_ckpt, alpha_thres)

    # VGGT
    vggt_ckpt = 'pretrained/vggt_data/model.pt'
    resolution = 518
    vggt = VGGT_Predictor(vggt_ckpt, resolution, device=device)

    # samurai
    sam_ckpt = 'pretrained/samurai_data/sam2.1_hiera_base_plus.pt'
    samurai = SAMURAI_core(sam_ckpt, yolox)

    # 返回全部模型实例
    return yolox, alphapose, vggt, samurai


# 对第一个时间戳进行多视角 ReID 并拿到 gid 对齐后的首帧 bbox
def run_first_timestamp_reid(camera_dirs, matcher):
    # 第一帧的图像路径列表
    first_images = [str(list_frames(cam)[0]) for cam in camera_dirs]

    # 运行基础模型（YOLOX + AlphaPose + VGGT）
    det_pose_per_view, orig_imgs, orig_sizes, view_counts, wp, wpc, ocr = matcher.run_model(first_images)
    per_view_items = matcher.calculate_confidence(det_pose_per_view, orig_sizes, view_counts, wp, wpc, ocr)
    view_pid_to_gid, _ = matcher.iterative_seed_match(per_view_items)
    # 存储每个视角下，不同ID的人对应的框
    view_gid_bbox = []
    # 按视角遍历
    for v in range(len(camera_dirs)):
        gid_bbox = {}
        # 当前视角检测框数组
        bboxes = det_pose_per_view[v]['bboxes']
        for pid, gid in view_pid_to_gid[v].items():
            if pid < len(bboxes):
                gid_bbox[int(gid)] = to_float_bbox(bboxes[pid])
        view_gid_bbox.append(gid_bbox)

    all_gids = sorted({gid for gid_bbox in view_gid_bbox for gid in gid_bbox.keys()})
    return view_gid_bbox, all_gids


# 在每个视角上用 SAMURAI 基于首帧 bbox 做全序列跟踪
def run_samurai_tracking(camera_dirs, view_gid_bbox, samurai):

    tracked = []

    for v, cam_dir in enumerate(camera_dirs):
        print('view %d' %v)
        gid_bbox = view_gid_bbox[v]
        gids_in_view = sorted(gid_bbox.keys())

        if not gids_in_view:
            tracked.append({})
            continue

        # 按ID顺序拼成 SAMURAI 初始框数组
        init_boxes = np.asarray([gid_bbox[g] for g in gids_in_view], dtype=np.float32)
        sam_out = samurai.predict(init_boxes, str(cam_dir), viz=False, save_results=False)

        # 当前视角ID->每帧bbox列表
        gid_tracks = {}
        for idx, gid in enumerate(gids_in_view):
            key = str(idx)
            if key not in sam_out:
                continue
            gid_tracks[gid] = [to_float_bbox(b) for b in sam_out[key]['bbox']]

        tracked.append(gid_tracks)

    return tracked


# 保存首帧多视角匹配可视化
def save_match_visualization(camera_dirs, view_gid_bbox, match_root):
    match_root.mkdir(parents=True, exist_ok=True)
    for v, cam_dir in enumerate(camera_dirs):
        cam_name = cam_dir.name
        first_frame = list_frames(cam_dir)[0]
        img = cv2.imread(str(first_frame))
        if img is None:
            continue
        vis = draw_bboxes_with_gid(img, view_gid_bbox[v])
        out_path = match_root / f'{cam_name}_{first_frame.name}'
        cv2.imwrite(str(out_path), vis)


# 保存每个视角逐帧跟踪可视化
def save_tracking_visualization(camera_dirs, tracked_bboxes, all_gids, tracking_root):
    tracking_root.mkdir(parents=True, exist_ok=True)
    for v, cam_dir in enumerate(camera_dirs):
        cam_name = cam_dir.name
        out_cam_dir = tracking_root / cam_name
        out_cam_dir.mkdir(parents=True, exist_ok=True)
        frames = list_frames(cam_dir)
        gid_tracks = tracked_bboxes[v]
        for f_idx, frame_path in enumerate(tqdm(frames, desc=f'TrackingVis {cam_name}', leave=False)):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            gid_to_bbox = {}
            for gid in all_gids:
                if gid not in gid_tracks:
                    continue
                if f_idx >= len(gid_tracks[gid]):
                    continue
                gid_to_bbox[gid] = gid_tracks[gid][f_idx]
            vis = draw_bboxes_with_gid(img, gid_to_bbox)
            out_path = out_cam_dir / frame_path.name
            cv2.imwrite(str(out_path), vis)


# 逐帧用AlphaPose提取关键点并依靠id写json文件
def run_alphapose_and_save(camera_dirs, tracked_bboxes, all_gids, alphapose, out_root,
                           save_pose_vis=False, pose_root=None):

    max_gid = max(all_gids) if all_gids else -1

    for v, cam_dir in enumerate(camera_dirs):
        cam_name = cam_dir.name
        out_cam_dir = Path(out_root) / cam_name
        out_cam_dir.mkdir(parents=True, exist_ok=True)

        frames = list_frames(cam_dir)
        gid_tracks = tracked_bboxes[v]
        pose_cam_dir = None
        # 若开启 pose 可视化
        if save_pose_vis and pose_root is not None:
            pose_cam_dir = Path(pose_root) / cam_name
            pose_cam_dir.mkdir(parents=True, exist_ok=True)

        for f_idx, frame_path in enumerate(tqdm(frames, desc=f'AlphaPose {cam_name}', leave=False)):
            img = cv2.imread(str(frame_path))

            valid_gids = []
            valid_boxes = []

            for gid in all_gids:
                if gid not in gid_tracks:
                    continue
                if f_idx >= len(gid_tracks[gid]):
                    continue
                box = gid_tracks[gid][f_idx]
                if not bbox_valid(box):
                    continue
                valid_gids.append(gid)
                valid_boxes.append(box)

            people = [None] * (max_gid + 1)

            if valid_boxes:

                boxes_np = np.asarray(valid_boxes, dtype=np.float32)
                poses = alphapose.predict(img, boxes_np)

                for i, gid in enumerate(valid_gids):
                    if i >= poses.shape[0]:
                        break
                    keypoints = poses[i].reshape(-1).tolist()
                    people[gid] = {'pose_keypoints_2d': [float(x) for x in keypoints]}
                pose_vis_img = alphapose.visualize(img.copy(), poses, format='coco17', viz=False) if save_pose_vis else None
            else:
                pose_vis_img = img.copy() if save_pose_vis else None

            out_file = out_cam_dir / f'{frame_path.stem}_keypoints.json'
            save_keypoints_json(out_file, people)
            # 开启可视化保存检测结果
            if save_pose_vis and pose_cam_dir is not None and pose_vis_img is not None:
                gid_to_bbox = {}
                for gid in all_gids:
                    if gid not in gid_tracks:
                        continue
                    if f_idx >= len(gid_tracks[gid]):
                        continue
                    box = gid_tracks[gid][f_idx]
                    if bbox_valid(box):
                        gid_to_bbox[gid] = box
                pose_vis_img = draw_bboxes_with_gid(pose_vis_img, gid_to_bbox)
                cv2.imwrite(str(pose_cam_dir / frame_path.name), pose_vis_img)