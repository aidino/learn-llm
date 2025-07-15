#!/usr/bin/env python3
"""
Script để parse PDF bằng docling và trích xuất câu hỏi, phương pháp, lời giải và đáp án
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import docling components
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_docling_converter():
    """Thiết lập DocumentConverter với các options phù hợp"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return doc_converter

def parse_pdf(file_path: str) -> str:
    """Parse PDF và trả về nội dung dưới dạng markdown"""
    logger.info(f"Bắt đầu parse PDF: {file_path}")
    
    converter = setup_docling_converter()
    result = converter.convert(file_path)
    
    # Xuất nội dung dưới dạng markdown
    markdown_content = result.document.export_to_markdown()
    
    logger.info("Hoàn thành parse PDF")
    return markdown_content

def extract_questions_data(content: str) -> List[Dict[str, str]]:
    """
    Phân tích nội dung để trích xuất câu hỏi, phương pháp, lời giải và đáp án
    """
    questions_data = []
    
    # Patterns để nhận dạng các phần khác nhau
    # Thường thì đề thi sẽ có các pattern như:
    # Câu 1: ... 
    # A. ... B. ... C. ... D. ...
    # Hoặc các pattern khác
    
    # Tách nội dung thành các dòng
    lines = content.split('\n')
    
    current_question = {}
    current_section = ""
    question_text = ""
    options = []
    method_text = ""
    solution_text = ""
    answer_text = ""
    
    # Patterns để nhận dạng
    question_pattern = re.compile(r'^(Câu|Question|Bài)\s*(\d+)[:\.]?\s*(.*)', re.IGNORECASE)
    option_pattern = re.compile(r'^([A-D])[:\.\)]\s*(.*)', re.IGNORECASE)
    method_pattern = re.compile(r'(phương pháp|method|cách làm|giải|approach)', re.IGNORECASE)
    solution_pattern = re.compile(r'(lời giải|solution|giải chi tiết|detailed solution)', re.IGNORECASE)
    answer_pattern = re.compile(r'(đáp án|answer|kết quả|result)[:\.]?\s*([A-D])', re.IGNORECASE)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Kiểm tra nếu là câu hỏi mới
        question_match = question_pattern.match(line)
        if question_match:
            # Lưu câu hỏi trước đó nếu có
            if current_question and question_text:
                current_question["Câu hỏi"] = question_text.strip()
                current_question["Phương pháp"] = method_text.strip()
                current_question["Lời giải"] = solution_text.strip()
                current_question["Đáp án"] = answer_text.strip()
                questions_data.append(current_question)
            
            # Bắt đầu câu hỏi mới
            current_question = {}
            question_text = question_match.group(3) if question_match.group(3) else ""
            options = []
            method_text = ""
            solution_text = ""
            answer_text = ""
            current_section = "question"
            continue
        
        # Kiểm tra nếu là option (A, B, C, D)
        option_match = option_pattern.match(line)
        if option_match and current_section == "question":
            option_letter = option_match.group(1)
            option_content = option_match.group(2)
            options.append(f"{option_letter}. {option_content}")
            question_text += f"\n{option_letter}. {option_content}"
            continue
        
        # Kiểm tra nếu là phương pháp
        if method_pattern.search(line):
            current_section = "method"
            method_text += line + "\n"
            continue
        
        # Kiểm tra nếu là lời giải
        if solution_pattern.search(line):
            current_section = "solution"
            solution_text += line + "\n"
            continue
        
        # Kiểm tra nếu là đáp án
        answer_match = answer_pattern.search(line)
        if answer_match:
            current_section = "answer"
            answer_text = answer_match.group(2)
            continue
        
        # Thêm vào section hiện tại
        if current_section == "question":
            question_text += " " + line
        elif current_section == "method":
            method_text += line + "\n"
        elif current_section == "solution":
            solution_text += line + "\n"
        elif current_section == "answer":
            answer_text += " " + line
    
    # Lưu câu hỏi cuối cùng
    if current_question and question_text:
        current_question["Câu hỏi"] = question_text.strip()
        current_question["Phương pháp"] = method_text.strip()
        current_question["Lời giải"] = solution_text.strip()
        current_question["Đáp án"] = answer_text.strip()
        questions_data.append(current_question)
    
    # Nếu không tìm thấy câu hỏi theo pattern, thử phương pháp khác
    if not questions_data:
        # Phương pháp backup: tìm kiếm các đoạn văn bản có thể là câu hỏi
        questions_data = extract_questions_fallback(content)
    
    return questions_data

def extract_questions_fallback(content: str) -> List[Dict[str, str]]:
    """
    Phương pháp dự phòng để trích xuất câu hỏi khi pattern chính không hoạt động
    """
    logger.warning("Sử dụng phương pháp dự phòng để trích xuất câu hỏi")
    
    # Tách content thành các đoạn
    paragraphs = content.split('\n\n')
    questions_data = []
    
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 20:  # Bỏ qua đoạn quá ngắn
            continue
            
        # Tạo một câu hỏi generic
        question_data = {
            "Câu hỏi": paragraph.strip(),
            "Phương pháp": "Cần phân tích thêm để xác định phương pháp",
            "Lời giải": "Cần phân tích thêm để xác định lời giải chi tiết",
            "Đáp án": "Cần phân tích thêm để xác định đáp án"
        }
        questions_data.append(question_data)
        
        # Giới hạn số lượng để tránh quá nhiều dữ liệu không chính xác
        if len(questions_data) >= 10:
            break
    
    return questions_data

def save_to_json(data: List[Dict[str, str]], output_file: str):
    """Lưu dữ liệu ra file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Đã lưu {len(data)} câu hỏi vào file: {output_file}")

def main():
    """Hàm chính"""
    input_file = "dethi_sample.pdf"
    output_file = "questions_data.json"
    
    if not Path(input_file).exists():
        logger.error(f"Không tìm thấy file: {input_file}")
        return
    
    try:
        # Parse PDF
        content = parse_pdf(input_file)
        
        # Lưu nội dung markdown để kiểm tra
        with open("parsed_content.md", 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Đã lưu nội dung đã parse vào file: parsed_content.md")
        
        # Trích xuất dữ liệu câu hỏi
        questions_data = extract_questions_data(content)
        
        if not questions_data:
            logger.warning("Không tìm thấy câu hỏi nào. Vui lòng kiểm tra lại file PDF hoặc cải thiện pattern nhận dạng.")
            return
        
        # Lưu ra file JSON
        save_to_json(questions_data, output_file)
        
        # In một vài câu hỏi mẫu
        logger.info(f"Đã trích xuất được {len(questions_data)} câu hỏi.")
        if questions_data:
            logger.info("Câu hỏi mẫu:")
            sample_question = questions_data[0]
            for key, value in sample_question.items():
                logger.info(f"{key}: {value[:100]}...")
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý: {str(e)}")
        raise

if __name__ == "__main__":
    main() 