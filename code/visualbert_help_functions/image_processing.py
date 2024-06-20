from utils import Config
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess

class ImageProcessor:
    def __init__(self, device='cuda'):
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn_cfg.MODEL.DEVICE = device
        self.device = device

        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

        self.frcnn_cfg = frcnn_cfg
        self.image_preprocess = Preprocess(frcnn_cfg)

    def get_visual_embeddings(self, image_path):
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_path)

        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        features = output_dict.get("roi_features").detach().cpu()
        # adding lines to also extract the bounding boxes
        boxes = output_dict.get("boxes").detach().cpu()
        obj_ids = output_dict.get("obj_ids").detach().cpu()
        obj_probs = output_dict.get('obj_probs').detach().cpu()
        attr_ids = output_dict.get('attr_ids').detach().cpu()

        return features, boxes, obj_ids, obj_probs, attr_ids