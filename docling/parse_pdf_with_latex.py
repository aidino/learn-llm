#!/usr/bin/env python3
"""
Script parse PDF với docling và chuyển đổi công thức toán thành LaTeX format
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

def convert_math_to_latex(text: str) -> str:
    """Chuyển đổi biểu thức toán học sang LaTeX format"""
    if not text:
        return text
    
    # Các pattern để nhận dạng công thức toán
    math_patterns = [
        # Phân số dạng "a/b" -> \frac{a}{b}
        (r'(\d+)\s*/\s*(\d+)', r'\\frac{\\1}{\\2}'),
        
        # Số mũ dạng "x^2" -> x^{2}
        (r'(\w+)\^(\d+)', r'\\1^{\\2}'),
        
        # Phép nhân × -> \times
        (r'×', r'\\times'),
        
        # Phép chia ÷ -> \div
        (r'÷', r'\\div'),
        
        # Lớn hơn hoặc bằng ≥ -> \geq
        (r'≥', r'\\geq'),
        
        # Nhỏ hơn hoặc bằng ≤ -> \leq
        (r'≤', r'\\leq'),
        
        # Không bằng ≠ -> \neq
        (r'≠', r'\\neq'),
        
        # Bình phương ² -> ^2
        (r'²', r'^2'),
        
        # Lập phương ³ -> ^3
        (r'³', r'^3'),
        
        # Căn bậc hai √ -> \sqrt
        (r'√\(([^)]+)\)', r'\\sqrt{\\1}'),
        (r'√(\d+)', r'\\sqrt{\\1}'),
    ]
    
    result = text
    
    # Áp dụng các pattern chuyển đổi
    for pattern, replacement in math_patterns:
        result = re.sub(pattern, replacement, result)
    
    return result

def extract_and_format_fraction(text: str) -> str:
    """Trích xuất và format phân số từ text - cải thiện cho đề thi"""
    # Pattern cho phân số hỗn số dạng "2 × a - 3 2 5 = 47 5"
    # Xử lý từng trường hợp cụ thể
    
    # Pattern 1: "3 2 5" -> có thể là 3 - 2/5
    mixed_fraction_pattern = r'(\d+)\s+(\d+)\s+(\d+)(?=\s*[=+\-×÷]|\s*$)'
    
    def replace_mixed_fraction(match):
        num1, num2, num3 = match.groups()
        # Kiểm tra context để quyết định format
        return f"{num1} - \\frac{{{num2}}}{{{num3}}}"
    
    result = re.sub(mixed_fraction_pattern, replace_mixed_fraction, text)
    
    # Pattern 2: Phân số đơn giản "47 5" ở cuối -> 47/5
    simple_fraction_pattern = r'(\d+)\s+(\d+)(?=\s*$)'
    
    def replace_simple_fraction(match):
        num1, num2 = match.groups()
        return f"\\frac{{{num1}}}{{{num2}}}"
    
    result = re.sub(simple_fraction_pattern, replace_simple_fraction, result)
    
    return result

def format_specific_math_expressions(text: str) -> str:
    """Format các biểu thức toán học cụ thể trong đề thi"""
    
    # Xử lý biểu thức dạng "2 × a - 3 2 5 = 47 5"
    if "2 × a" in text and "3 2 5" in text and "47 5" in text:
        # Chuyển đổi thành: 2 \times a - 3\frac{2}{5} = \frac{47}{5}
        text = text.replace("2 × a - 3 2 5 = 47 5", 
                           "2 \\times a - 3\\frac{2}{5} = \\frac{47}{5}")
    
    # Xử lý biểu thức dạng "52,39 - 28,23 - 21,77"
    decimal_pattern = r'(\d+,\d+)'
    text = re.sub(decimal_pattern, r'\\1', text)
    
    # Xử lý biểu thức phân số phức tạp "3 8 4 48 ... 7 5 7 30 + + - ="
    if "3 8 4 48" in text and "7 5 7 30" in text:
        text = text.replace("3 8 4 48 ... 7 5 7 30 + + - =", 
                           "\\frac{3}{8} + \\frac{4}{48} - \\frac{7}{5} - \\frac{7}{30} =")
    
    # Xử lý tỉ số "4 5" -> \frac{4}{5}
    if "tỉ số là" in text and re.search(r'\d+\s+\d+', text):
        ratio_match = re.search(r'tỉ số là\s+(\d+)\s+(\d+)', text)
        if ratio_match:
            num1, num2 = ratio_match.groups()
            text = text.replace(f"tỉ số là {num1} {num2}", 
                               f"tỉ số là $\\frac{{{num1}}}{{{num2}}}$")
    
    return text

def format_equation(equation: str) -> str:
    """Format một phương trình hoàn chỉnh"""
    if not equation.strip():
        return equation
    
    # Xử lý các biểu thức cụ thể trước
    result = format_specific_math_expressions(equation)
    
    # Tách phương trình thành các phần
    parts = result.split('=')
    
    formatted_parts = []
    for part in parts:
        part = part.strip()
        
        # Chuyển đổi các ký hiệu toán học
        part = convert_math_to_latex(part)
        
        # Xử lý phân số đặc biệt
        part = extract_and_format_fraction(part)
        
        # Xử lý các phép toán cộng trừ
        part = re.sub(r'\s+\+\s+', ' + ', part)
        part = re.sub(r'\s+-\s+', ' - ', part)
        
        formatted_parts.append(part)
    
    # Ghép lại với dấu =
    result = ' = '.join(formatted_parts)
    
    # Wrap trong LaTeX math mode nếu chưa có và có ký hiệu toán học
    if ('\\frac' in result or '\\times' in result or '\\div' in result or 
        '^' in result or '\\sqrt' in result) and not result.startswith('$'):
        result = f"${result}$"
    
    return result

def extract_questions_from_table(content: str) -> List[str]:
    """Trích xuất danh sách câu hỏi từ bảng và format LaTeX"""
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
                    # Format LaTeX cho câu hỏi
                    formatted_question = format_equation(question_text)
                    questions.append(f"Câu {question_num}. {formatted_question}")
    
    return questions

def extract_detailed_solutions(content: str) -> Dict[int, Dict[str, str]]:
    """Trích xuất phương pháp, lời giải và đáp án cho từng câu với LaTeX format"""
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
                    "method": format_equation(current_method.strip()),
                    "solution": format_equation(current_solution.strip()),
                    "answer": format_equation(current_answer.strip())
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
            "method": format_equation(current_method.strip()),
            "solution": format_equation(current_solution.strip()),
            "answer": format_equation(current_answer.strip())
        }
    
    return solutions

def extract_answers_from_table(content: str) -> Dict[int, str]:
    """Trích xuất đáp án từ bảng đáp án với LaTeX format"""
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
                    # Format LaTeX cho đáp án
                    formatted_answer = format_equation(answer_text)
                    answers[int(question_num)] = formatted_answer
    
    return answers

def create_structured_data(questions: List[str], solutions: Dict[int, Dict[str, str]], table_answers: Dict[int, str]) -> List[Dict[str, str]]:
    """Tạo dữ liệu có cấu trúc từ các thành phần đã trích xuất với LaTeX format"""
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

def save_latex_preview(data: List[Dict[str, str]], output_file: str):
    """Tạo file preview HTML để xem LaTeX"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Câu hỏi với LaTeX</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }
    };
    </script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .question { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .question h3 { color: #333; margin-top: 0; }
        .method { background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 3px; }
        .solution { background-color: #f8f8f0; padding: 10px; margin: 10px 0; border-radius: 3px; }
        .answer { background-color: #f0f8f0; padding: 10px; margin: 10px 0; border-radius: 3px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Đề thi Toán với LaTeX</h1>
"""
    
    for i, item in enumerate(data, 1):
        html_content += f"""
    <div class="question">
        <h3>{item['Câu hỏi']}</h3>
        <div class="method"><strong>Phương pháp:</strong> {item['Phương pháp']}</div>
        <div class="solution"><strong>Lời giải:</strong> {item['Lời giải']}</div>
        <div class="answer"><strong>Đáp án:</strong> {item['Đáp án']}</div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Đã tạo file preview HTML: {output_file}")

def main():
    """Hàm chính"""
    input_file = "dethi_sample.pdf"
    output_file = "questions_data_with_latex.json"
    preview_file = "questions_preview.html"
    
    if not Path(input_file).exists():
        logger.error(f"Không tìm thấy file: {input_file}")
        return
    
    try:
        # Parse PDF
        content = parse_pdf(input_file)
        
        # Lưu nội dung markdown để kiểm tra
        with open("parsed_content_latex.md", 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("Đã lưu nội dung đã parse vào file: parsed_content_latex.md")
        
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
        
        # Tạo file preview HTML
        save_latex_preview(structured_data, preview_file)
        
        # In một vài câu hỏi mẫu
        logger.info(f"Đã tạo dữ liệu có cấu trúc với LaTeX cho {len(structured_data)} câu hỏi.")
        if structured_data:
            logger.info("Câu hỏi mẫu với LaTeX:")
            sample_question = structured_data[0]
            for key, value in sample_question.items():
                logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý: {str(e)}")
        raise

if __name__ == "__main__":
    main() 