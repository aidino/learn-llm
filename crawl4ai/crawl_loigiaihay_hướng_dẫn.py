#!/usr/bin/env python3
"""
Crawl phần HƯỚNG DẪN GIẢI CHI TIẾT từ trang loigiaihay.com
Sử dụng thư viện crawl4ai để lấy nội dung từ phần tử có ID 'sub-question-2'
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy


async def crawl_huong_dan_giai_chi_tiet(url: str):
    """
    Crawl phần HƯỚNG DẪN GIẢI CHI TIẾT từ trang loigiaihay.com
    
    Args:
        url: URL của trang đề thi cần crawl
        
    Returns:
        dict: Dữ liệu đã crawl được
    """
    
    # Cấu hình để crawl toàn bộ trang trước để kiểm tra cấu trúc
    initial_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["script", "style", "nav", "footer"],
        word_count_threshold=5
    )
    
    print(f"🔍 Đang kiểm tra cấu trúc trang: {url}")
    
    async with AsyncWebCrawler() as crawler:
        # Crawl toàn bộ trang trước để kiểm tra
        result = await crawler.arun(url=url, config=initial_config)
        
        if not result.success:
            print(f"❌ Lỗi khi crawl trang: {result.error_message}")
            return None
        
        # Lưu HTML để kiểm tra cấu trúc
        html_output_path = Path("crawl4ai/output/loigiaihay_full_structure.html")
        html_output_path.parent.mkdir(exist_ok=True)
        
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(result.cleaned_html)
        
        print(f"💾 Đã lưu cấu trúc HTML vào: {html_output_path}")
        
        # Kiểm tra xem có phần tử với ID 'sub-question-2' không
        if 'id="sub-question-2"' in result.cleaned_html:
            print("✅ Tìm thấy phần tử với ID 'sub-question-2'")
        else:
            print("⚠️  Không tìm thấy phần tử với ID 'sub-question-2', đang tìm các pattern khác...")
            
            # Tìm kiếm các pattern có thể liên quan
            patterns_to_check = [
                'HƯỚNG DẪN GIẢI CHI TIẾT',
                'sub-question',
                'box-question',
                'hướng dẫn giải',
                'giải chi tiết'
            ]
            
            found_patterns = []
            for pattern in patterns_to_check:
                if pattern.lower() in result.cleaned_html.lower():
                    found_patterns.append(pattern)
            
            if found_patterns:
                print(f"🔍 Tìm thấy các pattern: {', '.join(found_patterns)}")
            else:
                print("❌ Không tìm thấy pattern nào liên quan")
    
    return result


async def crawl_specific_huong_dan_giai(url: str):
    """
    Crawl cụ thể phần HƯỚNG DẪN GIẢI CHI TIẾT
    
    Args:
        url: URL của trang đề thi
        
    Returns:
        dict: Nội dung phần hướng dẫn giải
    """
    
    # Schema để extract nội dung HƯỚNG DẪN GIẢI CHI TIẾT
    huong_dan_schema = {
        "name": "Hướng Dẫn Giải Chi Tiết",
        "baseSelector": "#sub-question-2, .box-question[id='sub-question-2']",
        "fields": [
            {
                "name": "tieu_de",
                "selector": "h2, h3, .title, .heading",
                "type": "text"
            },
            {
                "name": "noi_dung_giai",
                "selector": ".content, .solution, .answer, p, div",
                "type": "text"
            },
            {
                "name": "cac_buoc_giai", 
                "selector": "ol li, ul li, .step",
                "type": "text"
            },
            {
                "name": "cong_thuc",
                "selector": ".formula, .math, .equation",
                "type": "text" 
            },
            {
                "name": "hinh_anh",
                "selector": "img",
                "type": "attribute",
                "attribute": "src"
            }
        ]
    }
    
    # Cấu hình crawl với CSS selector cụ thể
    config = CrawlerRunConfig(
        css_selector="#sub-question-2",  # Chỉ lấy phần tử này
        extraction_strategy=JsonCssExtractionStrategy(huong_dan_schema),
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["script", "style"],
        word_count_threshold=3
    )
    
    print(f"🎯 Đang crawl phần HƯỚNG DẪN GIẢI CHI TIẾT từ: {url}")
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        
        if not result.success:
            print(f"❌ Lỗi khi crawl: {result.error_message}")
            return None
        
        # Parse kết quả JSON
        try:
            extracted_data = json.loads(result.extracted_content) if result.extracted_content else []
            
            if extracted_data:
                print(f"✅ Đã extract được {len(extracted_data)} phần tử")
                
                # Lưu kết quả vào file JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(f"crawl4ai/output/huong_dan_giai_chi_tiet_{timestamp}.json")
                output_file.parent.mkdir(exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Đã lưu kết quả vào: {output_file}")
                
                # Hiển thị nội dung đã extract
                for i, item in enumerate(extracted_data, 1):
                    print(f"\n📝 Phần {i}:")
                    for key, value in item.items():
                        if value and value.strip():
                            print(f"  {key}: {value[:200]}{'...' if len(str(value)) > 200 else ''}")
                
                return extracted_data
            else:
                print("⚠️  Không extract được dữ liệu, thử với selector khác...")
                return await crawl_fallback_method(url)
                
        except json.JSONDecodeError as e:
            print(f"❌ Lỗi parse JSON: {e}")
            print(f"Raw content: {result.extracted_content[:500]}...")
            return None


async def crawl_fallback_method(url: str):
    """
    Phương pháp dự phòng khi không tìm thấy ID cụ thể
    """
    print("🔄 Đang thử phương pháp dự phòng...")
    
    # Schema mở rộng để tìm các pattern khác
    fallback_schema = {
        "name": "Hướng Dẫn Giải - Fallback",
        "baseSelector": ".box-question, .solution-box, [class*='question'], [class*='answer'], [class*='solution']",
        "fields": [
            {
                "name": "tieu_de",
                "selector": "h1, h2, h3, h4, .title",
                "type": "text"
            },
            {
                "name": "noi_dung",
                "selector": "p, div, .content",
                "type": "text"
            }
        ]
    }
    
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(fallback_schema),
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        
        if result.success and result.extracted_content:
            try:
                data = json.loads(result.extracted_content)
                
                # Lọc chỉ lấy những phần có chứa "hướng dẫn" hoặc "giải"
                filtered_data = []
                for item in data:
                    content = str(item).lower()
                    if any(keyword in content for keyword in ['hướng dẫn', 'giải', 'chi tiết', 'bài giải']):
                        filtered_data.append(item)
                
                if filtered_data:
                    print(f"✅ Tìm thấy {len(filtered_data)} phần có liên quan đến hướng dẫn giải")
                    return filtered_data
                
            except json.JSONDecodeError:
                pass
    
    return None


async def main():
    """Hàm chính để chạy crawler"""
    url = "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html"
    
    print("🚀 Bắt đầu crawl trang loigiaihay.com")
    print(f"🔗 URL: {url}")
    print("=" * 60)
    
    # Bước 1: Kiểm tra cấu trúc trang
    await crawl_huong_dan_giai_chi_tiet(url)
    
    print("\n" + "=" * 60)
    
    # Bước 2: Crawl cụ thể phần HƯỚNG DẪN GIẢI CHI TIẾT
    result = await crawl_specific_huong_dan_giai(url)
    
    if result:
        print("\n✅ Hoàn thành crawl thành công!")
    else:
        print("\n❌ Không thể crawl được nội dung mong muốn")
    
    return result


if __name__ == "__main__":
    # Chạy crawler
    asyncio.run(main()) 