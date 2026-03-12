import os
import json

def convert_to_vietocr_format(base_folder, output_txt_file):
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    
    total_lines = 0
    with open(output_txt_file, 'w', encoding='utf-8') as f_out:
        for root, dirs, files in os.walk(base_folder):
            if 'label.json' in files:
                json_path = os.path.join(root, 'label.json')
                
                with open(json_path, 'r', encoding='utf-8') as f_json:
                    try:
                        data = json.load(f_json)
                        for img_filename, text_label in data.items():
                            parent_dir = os.path.basename(os.path.dirname(root)) 
                            sub_dir = os.path.basename(root)
                            
                            rel_image_path = f"{parent_dir}/{sub_dir}/{img_filename}"
                            clean_text = text_label.replace('\n', ' ').replace('\r', '').strip()
                            
                            f_out.write(f"{rel_image_path}\t{clean_text}\n")
                            total_lines += 1
                            
                    except Exception as e:
                        print(f"Lỗi khi đọc file {json_path}: {e}")

    if total_lines > 0:
        print(f"Đã tạo thành công! Tổng cộng: {total_lines} nhãn.")
        print(f"Đường dẫn file: {output_txt_file}\n")
    else:
        print(f"Cảnh báo: Không tìm thấy dữ liệu nào trong {base_folder}. File txt được tạo ra đang trống rỗng.\n")

if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)

    utils_dir = os.path.dirname(current_script_path)
    src_dir = os.path.dirname(utils_dir)
    project_root = os.path.dirname(src_dir)

    data_dir = os.path.join(project_root, 'data')

    train_data_dir = os.path.join(data_dir, 'train_data')
    test_data_dir = os.path.join(data_dir, 'test_data')
    
    train_out_txt = os.path.join(data_dir, 'train_labels.txt')
    val_out_txt = os.path.join(data_dir, 'val_labels.txt')
    
    print(f"Đang quét thư mục tại: {project_root}...\n")
    
    # Chạy hàm xử lý
    if os.path.exists(train_data_dir):
        print("Đang xử lý tập Train...")
        convert_to_vietocr_format(train_data_dir, train_out_txt)
    else:
        print(f"KHÔNG TÌM THẤY thư mục: {train_data_dir}")
        
    if os.path.exists(test_data_dir):
        print("Đang xử lý tập Test...")
        convert_to_vietocr_format(test_data_dir, val_out_txt)
    else:
        print(f"KHÔNG TÌM THẤY thư mục: {test_data_dir}")