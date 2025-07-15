#!/usr/bin/env python3
"""
Script cải thiện để parse PDF bằng docling và trích xuất từng câu hỏi riêng lẻ 
với phương pháp, lời giải và đáp án
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

def extract_questions_from_table(content: str) -> List[str]:
    """Trích xuất danh sách câu hỏi từ bảng"""
    questions = []
    
    # Tìm bảng đầu tiên chứa câu hỏi
    lines = content.split('\n')
    in_question_table = False
    
    for line in lines:
        line = line.strip()
        
        # Bắt đầu của bảng câu hỏi
        if "|   TT |" in line and "Câu hỏi" in line:
            in_question_table = True
            continue
        
        # Kết thúc bảng câu hỏi
        if in_question_table and (line.startswith("##") or line.startswith("Lo") or not line):
            if not line.startswith("|"):
                break
        
        # Trích xuất câu hỏi từ bảng
        if in_question_table and line.startswith("|") and "---" not in line:
            parts = line.split("|")
            if len(parts) >= 3:
                question_num = parts[1].strip()
                question_text = parts[2].strip()
                
                # Chỉ lấy những dòng có số thứ tự là số
                if question_num.isdigit() and question_text:
                    questions.append(f"Câu {question_num}. {question_text}")
    
    return questions

def extract_detailed_solutions(content: str) -> Dict[int, Dict[str, str]]:
    """Trích xuất phương pháp, lời giải và đáp án cho từng câu"""
    solutions = {}
    
    # Tách nội dung thành các phần
    lines = content.split('\n')
    
    current_question_num = None
    current_method = ""
    current_solution = ""
    current_answer = ""
    current_section = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Tìm câu hỏi mới
        question_match = re.match(r'Câu\s+(\d+)\.', line)
        if question_match:
            # Lưu câu trước đó nếu có
            if current_question_num is not None:
                solutions[current_question_num] = {
                    "method": current_method.strip(),
                    "solution": current_solution.strip(),
                    "answer": current_answer.strip()
                }
            
            # Bắt đầu câu mới
            current_question_num = int(question_match.group(1))
            current_method = ""
            current_solution = ""
            current_answer = ""
            current_section = None
            continue
        
        # Xác định section hiện tại
        if line.lower().startswith("## phương pháp") or line.lower() == "phương pháp":
            current_section = "method"
            continue
        elif line.lower().startswith("## lời giải") or line.lower() == "lời giải" or line.lower() == "loigiaih":
            current_section = "solution"
            continue
        elif line.lower().startswith("đáp án:") or line.lower().startswith("## đáp án:"):
            current_section = "answer"
            # Trích xuất đáp án từ dòng hiện tại
            answer_text = line
            if ":" in answer_text:
                answer_text = answer_text.split(":", 1)[1].strip()
            current_answer += answer_text + " "
            continue
        
        # Bỏ qua các dòng không cần thiết
        if (line.startswith("<!--") or 
            line.startswith("##") or 
            line.startswith("|") or
            line == "" or
            line.startswith("---")):
            continue
        
        # Thêm vào section tương ứng
        if current_section == "method":
            current_method += line + " "
        elif current_section == "solution":
            current_solution += line + " "
        elif current_section == "answer":
            current_answer += line + " "
    
    # Lưu câu cuối cùng
    if current_question_num is not None:
        solutions[current_question_num] = {
            "method": current_method.strip(),
            "solution": current_solution.strip(),
            "answer": current_answer.strip()
        }
    
    return solutions

def extract_answers_from_table(content: str) -> Dict[int, str]:
    """Trích xuất đáp án từ bảng đáp án"""
    answers = {}
    
    lines = content.split('\n')
    in_answer_table = False
    
    for line in lines:
        line = line.strip()
        
        # Tìm bảng đáp án (bảng thứ 2)
        if "## HƯỚNG DẪN GIẢI CHI TIẾT" in line:
            in_answer_table = True
            continue
        
        if in_answer_table and line.startswith("|") and "---" not in line:
            parts = line.split("|")
            if len(parts) >= 4:
                question_num = parts[1].strip()
                answer_text = parts[3].strip()
                
                if question_num.isdigit() and answer_text:
                    answers[int(question_num)] = answer_text
    
    return answers

def create_structured_data(questions: List[str], solutions: Dict[int, Dict[str, str]], table_answers: Dict[int, str]) -> List[Dict[str, str]]:
    """Tạo dữ liệu có cấu trúc từ các thành phần đã trích xuất"""
    structured_data = []
    
    for i, question in enumerate(questions, 1):
        question_data = {
            "Câu hỏi": question,
            "Phương pháp": "",
            "Lời giải": "",
            "Đáp án": ""
        }
        
        # Lấy thông tin từ phần giải chi tiết
        if i in solutions:
            sol = solutions[i]
            question_data["Phương pháp"] = sol["method"] if sol["method"] else "Áp dụng kiến thức cơ bản về toán học"
            question_data["Lời giải"] = sol["solution"] if sol["solution"] else "Thực hiện các bước tính toán theo yêu cầu"
            if sol["answer"]:
                question_data["Đáp án"] = sol["answer"]
        
        # Lấy đáp án từ bảng nếu chưa có
        if not question_data["Đáp án"] and i in table_answers:
            question_data["Đáp án"] = table_answers[i]
        
        # Nếu vẫn chưa có thông tin, tạo thông tin mặc định
        if not question_data["Phương pháp"]:
            question_data["Phương pháp"] = "Áp dụng kiến thức cơ bản về toán học"
        if not question_data["Lời giải"]:
            question_data["Lời giải"] = "Thực hiện các bước tính toán theo yêu cầu đề bài"
        if not question_data["Đáp án"]:
            question_data["Đáp án"] = "Cần xem lại đề bài để xác định đáp án"
        
        structured_data.append(question_data)
    
    return structured_data

def save_to_json(data: List[Dict[str, str]], output_file: str):
    """Lưu dữ liệu ra file JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Đã lưu {len(data)} câu hỏi vào file: {output_file}")

def main():
    """Hàm chính"""
    input_file = "dethi_sample.pdf"
    output_file = "questions_data_improved.json"
    
    if not Path(input_file).exists():
        logger.error(f"Không tìm thấy file: {input_file}")
        return
    
    try:
        # Parse PDF
        content = parse_pdf(input_file)
        
        # Lưu nội dung markdown để kiểm tra
        with open("parsed_content_improved.md", 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Đã lưu nội dung đã parse vào file: parsed_content_improved.md")
        
        # Trích xuất các thành phần
        questions = extract_questions_from_table(content)
        solutions = extract_detailed_solutions(content)
        table_answers = extract_answers_from_table(content)
        
        logger.info(f"Đã trích xuất {len(questions)} câu hỏi")
        logger.info(f"Đã trích xuất giải chi tiết cho {len(solutions)} câu")
        logger.info(f"Đã trích xuất đáp án bảng cho {len(table_answers)} câu")
        
        # Tạo dữ liệu có cấu trúc
        structured_data = create_structured_data(questions, solutions, table_answers)
        
        # Lưu ra file JSON
        save_to_json(structured_data, output_file)
        
        # In một vài câu hỏi mẫu
        logger.info(f"Đã tạo dữ liệu có cấu trúc cho {len(structured_data)} câu hỏi.")
        if structured_data:
            logger.info("Câu hỏi mẫu:")
            sample_question = structured_data[0]
            for key, value in sample_question.items():
                logger.info(f"{key}: {value[:100]}...")
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý: {str(e)}")
        raise

if __name__ == "__main__":
    main() 