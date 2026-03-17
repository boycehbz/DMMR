from modules.vggt.models.vggt import VGGT
from modules.vggt.utils.load_fn import load_and_preprocess_images_square
from modules.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from modules.vggt.utils.geometry import unproject_depth_map_to_point_map
import torch

class VGGT_Predictor:
    def __init__(self, checkpoints, RESOLUTION=512, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.RESOLUTION = RESOLUTION
        self.model = VGGT()
        self.model.load_state_dict(torch.load(checkpoints, map_location=device))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, images_path):
        images_sq, original_coords = load_and_preprocess_images_square(images_path, self.RESOLUTION)
        images_sq = images_sq.to(device=self.device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            images_b = images_sq[None] 
            aggregated_tokens_list, ps_idx = self.model.aggregator(images_b)
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_b.shape[-2:])
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images_b, ps_idx)
            extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()
            intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()
            depth_map = depth_map.squeeze(0).detach().cpu().numpy()
            depth_conf = depth_conf.squeeze(0).detach().cpu().numpy()
            wp = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            wpc = depth_conf
            oc = original_coords
        return wp,wpc,oc
