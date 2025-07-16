#!/usr/bin/env python3
"""
Script để phân tích dữ liệu đã gộp từ merged_questions.json
"""

import json
import re
from collections import Counter
from typing import Dict, List

def analyze_merged_data():
    """Phân tích dữ liệu đã gộp"""
    
    # Đọc file merged_questions.json
    try:
        with open('merged_questions.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📊 Đã tải {len(data)} câu hỏi từ merged_questions.json")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return
    
    # Thống kê cơ bản
    print("\n" + "="*50)
    print("📈 THỐNG KÊ TỔNG QUAN")
    print("="*50)
    
    total_questions = len(data)
    questions_with_images = sum(1 for item in data if item.get('image_url') is not None)
    questions_with_errors = sum(1 for item in data if item.get('error', False))
    
    print(f"📝 Tổng số câu hỏi: {total_questions}")
    print(f"🖼️  Câu hỏi có hình ảnh: {questions_with_images}")
    print(f"❌ Câu hỏi có lỗi: {questions_with_errors}")
    print(f"✅ Câu hỏi không có lỗi: {total_questions - questions_with_errors}")
    
    # Phân tích độ dài câu hỏi
    print("\n" + "="*50)
    print("📏 PHÂN TÍCH ĐỘ DÀI CÂU HỎI")
    print("="*50)
    
    question_lengths = [len(item.get('question', '')) for item in data]
    avg_length = sum(question_lengths) / len(question_lengths)
    min_length = min(question_lengths)
    max_length = max(question_lengths)
    
    print(f"📊 Độ dài trung bình: {avg_length:.1f} ký tự")
    print(f"📉 Độ dài ngắn nhất: {min_length} ký tự")
    print(f"📈 Độ dài dài nhất: {max_length} ký tự")
    
    # Phân tích các từ khóa toán học
    print("\n" + "="*50)
    print("🔢 PHÂN TÍCH CHỦ ĐỀ TOÁN HỌC")
    print("="*50)
    
    math_keywords = {
        'phương trình': 0,
        'giải': 0,
        'tính': 0,
        'biểu thức': 0,
        'hình học': 0,
        'diện tích': 0,
        'chu vi': 0,
        'phân số': 0,
        'số nguyên': 0,
        'đạo hàm': 0,
        'tích phân': 0,
        'ma trận': 0,
        'vector': 0,
        'logarit': 0
    }
    
    for item in data:
        question = item.get('question', '').lower()
        for keyword in math_keywords:
            if keyword in question:
                math_keywords[keyword] += 1
    
    # Sắp xếp theo số lượng
    sorted_keywords = sorted(math_keywords.items(), key=lambda x: x[1], reverse=True)
    
    for keyword, count in sorted_keywords:
        if count > 0:
            percentage = (count / total_questions) * 100
            print(f"🔤 {keyword.capitalize()}: {count} câu ({percentage:.1f}%)")
    
    # Phân tích cấu trúc câu hỏi
    print("\n" + "="*50)
    print("📝 PHÂN TÍCH CẤU TRÚC CÂU HỎI")
    print("="*50)
    
    question_types = {
        'Câu 1': 0,
        'Câu 2': 0,
        'Câu 3': 0,
        'Câu 4': 0,
        'Câu 5': 0,
        'Khác': 0
    }
    
    for item in data:
        question = item.get('question', '')
        found_type = False
        for q_type in question_types:
            if q_type != 'Khác' and question.startswith(q_type):
                question_types[q_type] += 1
                found_type = True
                break
        if not found_type:
            question_types['Khác'] += 1
    
    for q_type, count in question_types.items():
        if count > 0:
            percentage = (count / total_questions) * 100
            print(f"📋 {q_type}: {count} câu ({percentage:.1f}%)")
    
    # Phân tích LaTeX/MathJax
    print("\n" + "="*50)
    print("📐 PHÂN TÍCH CÔNG THỨC TOÁN HỌC")
    print("="*50)
    
    latex_patterns = 0
    for item in data:
        question = item.get('question', '') or ''
        solution = item.get('solution', '') or ''
        if '\\(' in question or '\\[' in question or '\\(' in solution or '\\[' in solution:
            latex_patterns += 1
    
    percentage_latex = (latex_patterns / total_questions) * 100
    print(f"📊 Câu hỏi có công thức LaTeX: {latex_patterns} ({percentage_latex:.1f}%)")
    
    # Tìm câu hỏi ngắn nhất và dài nhất
    print("\n" + "="*50)
    print("🔍 MẪU CÂU HỎI")
    print("="*50)
    
    valid_questions = [item for item in data if item.get('question')]
    if valid_questions:
        shortest_question = min(valid_questions, key=lambda x: len(x.get('question', '')))
        longest_question = max(valid_questions, key=lambda x: len(x.get('question', '')))
        
        print(f"📝 Câu hỏi ngắn nhất ({len(shortest_question['question'])} ký tự):")
        print(f"   {shortest_question['question'][:100]}...")
        
        print(f"\n📝 Câu hỏi dài nhất ({len(longest_question['question'])} ký tự):")
        print(f"   {longest_question['question'][:100]}...")
    else:
        print("❌ Không tìm thấy câu hỏi hợp lệ")

def main():
    """Hàm chính"""
    print("🔍 BẮT ĐẦU PHÂN TÍCH DỮ LIỆU ĐÃ GỘP")
    analyze_merged_data()
    print("\n✅ HOÀN THÀNH PHÂN TÍCH!")

if __name__ == "__main__":
    main() 