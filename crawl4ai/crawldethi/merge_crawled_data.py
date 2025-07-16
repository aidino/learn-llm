#!/usr/bin/env python3
"""
Script để gộp các file crawled_*.json và loại bỏ câu hỏi trùng lặp
"""

import json
import glob
import os
from typing import List, Dict, Set
import hashlib

def load_json_file(file_path: str) -> List[Dict]:
    """Đọc file JSON và trả về dữ liệu"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Đã đọc {len(data)} câu hỏi từ {os.path.basename(file_path)}")
        return data
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {file_path}: {e}")
        return []

def create_question_hash(question: str) -> str:
    """Tạo hash cho câu hỏi để so sánh trùng lặp"""
    # Loại bỏ khoảng trắng thừa và tạo hash
    cleaned_question = " ".join(question.split())
    return hashlib.md5(cleaned_question.encode('utf-8')).hexdigest()

def remove_duplicates(data: List[Dict]) -> List[Dict]:
    """Loại bỏ câu hỏi trùng lặp dựa trên nội dung câu hỏi"""
    seen_questions: Set[str] = set()
    unique_data: List[Dict] = []
    duplicates_count = 0
    
    for item in data:
        question = item.get('question', '')
        question_hash = create_question_hash(question)
        
        if question_hash not in seen_questions:
            seen_questions.add(question_hash)
            unique_data.append(item)
        else:
            duplicates_count += 1
    
    print(f"🔍 Đã loại bỏ {duplicates_count} câu hỏi trùng lặp")
    return unique_data

def merge_crawled_files():
    """Gộp tất cả các file crawled_*.json"""
    # Tìm tất cả các file crawled_*.json
    file_pattern = "crawled_*.json"
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"❌ Không tìm thấy file nào với pattern: {file_pattern}")
        return
    
    print(f"📁 Tìm thấy {len(files)} file crawled_*.json")
    
    # Đọc và gộp dữ liệu từ tất cả các file
    all_data: List[Dict] = []
    total_questions = 0
    
    for file_path in sorted(files):
        data = load_json_file(file_path)
        all_data.extend(data)
        total_questions += len(data)
    
    print(f"📊 Tổng số câu hỏi gốc: {total_questions}")
    
    # Loại bỏ trùng lặp
    print("🔄 Đang loại bỏ câu hỏi trùng lặp...")
    unique_data = remove_duplicates(all_data)
    
    print(f"✨ Số câu hỏi sau khi loại bỏ trùng lặp: {len(unique_data)}")
    
    # Lưu kết quả
    output_file = "merged_questions.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
        print(f"💾 Đã lưu kết quả vào file: {output_file}")
        
        # Tạo file thống kê
        stats = {
            "source_files": len(files),
            "total_original_questions": total_questions,
            "unique_questions": len(unique_data),
            "duplicates_removed": total_questions - len(unique_data),
            "file_list": [os.path.basename(f) for f in sorted(files)]
        }
        
        stats_file = "merge_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"📈 Đã lưu thống kê vào file: {stats_file}")
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")

def main():
    """Hàm chính"""
    print("🚀 Bắt đầu gộp các file crawled_*.json")
    print("=" * 50)
    
    # Kiểm tra thư mục hiện tại
    current_dir = os.getcwd()
    print(f"📂 Thư mục làm việc: {current_dir}")
    
    merge_crawled_files()
    
    print("=" * 50)
    print("✅ Hoàn thành!")

if __name__ == "__main__":
    main() 