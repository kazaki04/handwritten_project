from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import os

class VietOCREngine:
    def __init__(self, device='cpu'):
        self.device = device
        self.models = {}
        self._load_models()

    def _load_models(self):
        cfg_base = Cfg.load_config_from_name('vgg_transformer')
        cfg_base['device'] = self.device
        cfg_base['cnn']['pretrained'] = False
        cfg_base['predictor']['beamsearch'] = False
        self.models['vietocr_base'] = Predictor(cfg_base)
        print("Đã tải Model 1: VietOCR")

        weight_path = './models/resnet_handwritten_weight.pth'
        if os.path.exists(weight_path):
            cfg_custom = Cfg.load_config_from_name('resnet_transformer')
            cfg_custom['weights'] = weight_path
            cfg_custom['device'] = self.device
            cfg_custom['cnn']['pretrained'] = False
            cfg_custom['predictor']['beamsearch'] = False
            self.models['custom_resnet'] = Predictor(cfg_custom)
            print("Đã tải Model 2: ResNet")
        else:
            print(f"Chưa tìm thấy model tại {weight_path}. Chỉ dùng model mặc định.")

    def predict(self, image_path: str, model_type: str = 'vietocr_base') -> str:
        try:
            predictor = self.models.get(model_type, self.models['vietocr_base'])
            img = Image.open(image_path)
            return predictor.predict(img)
        except Exception as e:
            return f"Lỗi: {e}"