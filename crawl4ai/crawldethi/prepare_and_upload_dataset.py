#!/usr/bin/env python3
"""
Script để clean data và upload lên Hugging Face Dataset
"""

import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from urllib.parse import urlparse
from PIL import Image
import io
import re

# Thư viện Hugging Face
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatasetPreparer:
    def __init__(self, input_file: str = "merged_questions.json"):
        self.input_file = input_file
        self.images_dir = Path("downloaded_images")
        self.images_dir.mkdir(exist_ok=True)
        
        # Regex pattern để loại bỏ số thứ tự câu
        self.question_number_pattern = re.compile(r'^Câu\s+\d+[:.]\s*', re.IGNORECASE)
        
        # Lấy token từ environment variable
        self.hf_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
        if not self.hf_token or self.hf_token == 'YOUR_TOKEN_HERE':
            print("❌ HUGGINGFACE_ACCESS_TOKEN chưa được thiết lập trong file .env")
            print("📝 Hãy tạo file .env với nội dung:")
            print("HUGGINGFACE_ACCESS_TOKEN=your_actual_token_here")
            return
        
        # Đăng nhập Hugging Face
        try:
            login(token=self.hf_token)
            print("✅ Đã đăng nhập Hugging Face thành công")
        except Exception as e:
            print(f"❌ Lỗi đăng nhập Hugging Face: {e}")
            return

    def clean_question_text(self, question: str) -> str:
        """Loại bỏ số thứ tự câu ở đầu câu hỏi"""
        if not question:
            return ""
        
        # Loại bỏ số thứ tự câu (Câu 1:, Câu 2., v.v.)
        cleaned = self.question_number_pattern.sub('', question)
        
        # Loại bỏ khoảng trắng thừa
        return cleaned.strip()

    def has_valid_solution(self, solution) -> bool:
        """Kiểm tra xem solution có hợp lệ không"""
        if solution is None:
            return False
        if isinstance(solution, str) and not solution.strip():
            return False
        return True

    def load_data(self) -> List[Dict]:
        """Đọc dữ liệu từ file JSON"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📊 Đã tải {len(data)} câu hỏi từ {self.input_file}")
            return data
        except Exception as e:
            print(f"❌ Lỗi khi đọc file {self.input_file}: {e}")
            return []

    def download_image(self, image_url: str) -> str:
        """Tải ảnh về và trả về đường dẫn local"""
        if not image_url:
            return None
        
        try:
            # Tạo tên file từ hash của URL
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            parsed_url = urlparse(image_url)
            file_extension = Path(parsed_url.path).suffix or '.jpg'
            filename = f"{url_hash}{file_extension}"
            filepath = self.images_dir / filename
            
            # Kiểm tra nếu file đã tồn tại
            if filepath.exists():
                return str(filepath)
            
            # Tải ảnh
            response = requests.get(image_url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Kiểm tra định dạng ảnh
            try:
                img = Image.open(io.BytesIO(response.content))
                img.verify()  # Kiểm tra ảnh có hợp lệ không
                
                # Lưu ảnh
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"📥 Đã tải ảnh: {filename}")
                return str(filepath)
                
            except Exception as img_error:
                print(f"⚠️ Ảnh không hợp lệ từ URL {image_url}: {img_error}")
                return None
                
        except Exception as e:
            print(f"❌ Lỗi khi tải ảnh từ {image_url}: {e}")
            return None

    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """Clean dữ liệu theo yêu cầu"""
        print("🧹 Bắt đầu clean dữ liệu...")
        
        cleaned_data = []
        images_downloaded = 0
        
        for i, item in enumerate(data):
            # Lấy câu hỏi gốc và clean
            raw_question = item.get('question', '') or ''
            cleaned_question = self.clean_question_text(raw_question)
            
            # Kiểm tra độ dài câu hỏi tối thiểu 20 ký tự (sau khi clean)
            if len(cleaned_question) < 20:
                continue
            
            # Kiểm tra solution có hợp lệ không
            solution = item.get('solution')
            if not self.has_valid_solution(solution):
                continue
            
            # Tạo item mới và loại bỏ trường error
            cleaned_item = {
                'question': cleaned_question,
                'solution': solution.strip() if solution else '',
                'result': item.get('result', '') or '',
                'image_path': None
            }
            
            # Xử lý ảnh nếu có
            image_url = item.get('image_url')
            if image_url:
                image_path = self.download_image(image_url)
                if image_path:
                    cleaned_item['image_path'] = image_path
                    images_downloaded += 1
            
            cleaned_data.append(cleaned_item)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"📊 Đã xử lý {i + 1}/{len(data)} câu hỏi...")
        
        print(f"✅ Hoàn thành clean dữ liệu:")
        print(f"   - Câu hỏi ban đầu: {len(data)}")
        print(f"   - Câu hỏi sau khi clean: {len(cleaned_data)}")
        print(f"   - Đã loại bỏ số thứ tự câu (Câu 1:, Câu 2:, ...)")
        print(f"   - Đã loại bỏ câu hỏi ngắn < 20 ký tự")
        print(f"   - Đã loại bỏ câu không có solution")
        print(f"   - Ảnh đã tải về: {images_downloaded}")
        
        return cleaned_data

    def split_data(self, data: List[Dict], test_size: float = 0.1) -> tuple:
        """Tách dữ liệu thành train và test"""
        import random
        
        # Shuffle data
        random.seed(42)  # Để có kết quả reproducible
        data_copy = data.copy()
        random.shuffle(data_copy)
        
        # Tính số lượng test
        test_count = int(len(data_copy) * test_size)
        train_count = len(data_copy) - test_count
        
        train_data = data_copy[:train_count]
        test_data = data_copy[train_count:]
        
        print(f"📊 Tách dữ liệu:")
        print(f"   - Train: {len(train_data)} câu hỏi ({(1-test_size)*100:.1f}%)")
        print(f"   - Test: {len(test_data)} câu hỏi ({test_size*100:.1f}%)")
        
        return train_data, test_data

    def create_huggingface_dataset(self, train_data: List[Dict], test_data: List[Dict]) -> DatasetDict:
        """Tạo Hugging Face Dataset"""
        print("🔄 Tạo Hugging Face Dataset...")
        
        # Định nghĩa features
        features = Features({
            'question': Value('string'),
            'solution': Value('string'),
            'result': Value('string'),
            'image': HFImage()  # Sẽ tự động load ảnh từ path
        })
        
        # Chuẩn bị dữ liệu cho Dataset
        def prepare_data_for_hf(data_list):
            prepared = []
            for item in data_list:
                hf_item = {
                    'question': item['question'],
                    'solution': item['solution'],
                    'result': item['result'],
                    'image': item['image_path'] if item['image_path'] else None
                }
                prepared.append(hf_item)
            return prepared
        
        train_prepared = prepare_data_for_hf(train_data)
        test_prepared = prepare_data_for_hf(test_data)
        
        # Tạo datasets
        train_dataset = Dataset.from_list(train_prepared, features=features)
        test_dataset = Dataset.from_list(test_prepared, features=features)
        
        # Tạo DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        print("✅ Đã tạo Hugging Face Dataset thành công")
        return dataset_dict

    def upload_to_huggingface(self, dataset: DatasetDict, repo_name: str):
        """Upload dataset lên Hugging Face Hub"""
        print(f"🚀 Đang upload dataset lên Hugging Face: {repo_name}")
        
        try:
            # Upload dataset
            dataset.push_to_hub(
                repo_id=repo_name,
                private=False,  # Đặt True nếu muốn dataset private
                commit_message="Initial dataset upload"
            )
            
            print(f"✅ Đã upload dataset thành công!")
            print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_name}")
            
        except Exception as e:
            print(f"❌ Lỗi khi upload dataset: {e}")

    def save_local_dataset(self, dataset: DatasetDict, output_dir: str = "hf_dataset"):
        """Lưu dataset local để kiểm tra"""
        print(f"💾 Lưu dataset local tại: {output_dir}")
        
        try:
            dataset.save_to_disk(output_dir)
            print(f"✅ Đã lưu dataset local thành công")
        except Exception as e:
            print(f"❌ Lỗi khi lưu dataset local: {e}")

def main():
    """Hàm chính"""
    print("🚀 BẮT ĐẦU CHUẨN BỊ VÀ UPLOAD DATASET LÊN HUGGING FACE")
    print("=" * 60)
    
    # Khởi tạo DatasetPreparer
    preparer = DatasetPreparer()
    
    if not preparer.hf_token or preparer.hf_token == 'YOUR_TOKEN_HERE':
        return
    
    # Bước 1: Load dữ liệu
    data = preparer.load_data()
    if not data:
        return
    
    # Bước 2: Clean dữ liệu
    cleaned_data = preparer.clean_data(data)
    if not cleaned_data:
        print("❌ Không có dữ liệu sau khi clean")
        return
    
    # Bước 3: Tách train/test
    train_data, test_data = preparer.split_data(cleaned_data, test_size=0.1)
    
    # Bước 4: Tạo Hugging Face Dataset
    dataset = preparer.create_huggingface_dataset(train_data, test_data)
    
    # Bước 5: Lưu local để kiểm tra
    preparer.save_local_dataset(dataset)
    
    # Bước 6: Nhập tên repo và upload
    print("\n" + "=" * 60)
    repo_name = input("📝 Nhập tên repository trên Hugging Face (vd: username/dataset-name): ").strip()
    
    if not repo_name:
        print("❌ Tên repository không được để trống")
        return
    
    if "/" not in repo_name:
        print("❌ Tên repository phải có định dạng: username/dataset-name")
        return
    
    # Upload lên Hugging Face
    preparer.upload_to_huggingface(dataset, repo_name)
    
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH TẤT CẢ CÁC BƯỚC!")
    print(f"📊 Dataset info:")
    print(f"   - Train: {len(train_data)} samples")
    print(f"   - Test: {len(test_data)} samples")
    print(f"   - Images: {len([item for item in cleaned_data if item['image_path']])}")

if __name__ == "__main__":
    main() 