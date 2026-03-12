import os
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

def train_custom_ocr():
    config = Cfg.load_config_from_name('resnet_transformer')
    
    dataset_params = {
        'name':'hw_vietnamese', 
        'data_root':'./data/',  
        'train_annotation':'train_labels.txt', 
        'valid_annotation':'val_labels.txt'    
    }
    config['dataset'].update(dataset_params)

    config['trainer'].update({
        'batch_size': 32,         
        'print_every': 20,        
        'valid_every': 100,       
        'iters': 10000,          
        'export': './models/resnet_handwritten_weight.pth', 
        'metrics': 10000          
    })

    config['aug']['image_aug'] = True 

    config['optimizer']['lr'] = 0.0001 
    
    config['device'] = 'cuda:0' if os.getenv("USE_CUDA", "true").lower() == "true" else 'cpu'
    
    print("Bắt đầu huấn luyện với kiến trúc ResNet-Transformer nâng cao...")
    trainer = Trainer(config, pretrained=True) 
    trainer.train()
    print("Lưu model tại: ./models/resnet_handwritten_weight.pth")

if __name__ == '__main__':
    train_custom_ocr()