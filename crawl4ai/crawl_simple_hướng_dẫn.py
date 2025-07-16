#!/usr/bin/env python3
"""
Script đơn giản để crawl phần HƯỚNG DẪN GIẢI CHI TIẾT từ loigiaihay.com
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


async def crawl_huong_dan_giai_simple(url: str):
    """
    Crawl đơn giản phần HƯỚNG DẪN GIẢI CHI TIẾT
    
    Args:
        url: URL trang cần crawl
        
    Returns:
        str: Nội dung text đã crawl được
    """
    
    # Cấu hình crawl với CSS selector cho phần tử có ID sub-question-2
    config = CrawlerRunConfig(
        css_selector='div.box-question[id="sub-question-2"]',  # Selector cụ thể
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["script", "style", "nav", "footer", "header"],
        word_count_threshold=3,
        exclude_external_links=True,
        exclude_external_images=True
    )
    
    print(f"🔍 Đang crawl: {url}")
    print("🎯 Tìm kiếm phần HƯỚNG DẪN GIẢI CHI TIẾT...")
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        
        if not result.success:
            print(f"❌ Lỗi: {result.error_message}")
            return None
        
        # Kiểm tra nếu có nội dung
        if result.cleaned_html:
            print("✅ Tìm thấy nội dung!")
            
            # Lưu kết quả
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Lưu HTML
            html_file = Path(f"crawl4ai/output/huong_dan_giai_html_{timestamp}.html")
            html_file.parent.mkdir(exist_ok=True)
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(result.cleaned_html)
            
            # Lưu Markdown
            md_file = Path(f"crawl4ai/output/huong_dan_giai_md_{timestamp}.md")
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(result.markdown)
            
            print(f"💾 Đã lưu HTML: {html_file}")
            print(f"💾 Đã lưu Markdown: {md_file}")
            
            # Hiển thị một phần nội dung
            print("\n📖 Nội dung đã crawl (Markdown):")
            print("-" * 50)
            print(result.markdown[:1000] + "..." if len(result.markdown) > 1000 else result.markdown)
            print("-" * 50)
            
            return {
                'html': result.cleaned_html,
                'markdown': result.markdown,
                'url': result.url,
                'timestamp': timestamp
            }
        else:
            print("⚠️  Không tìm thấy nội dung với selector này")
            return await crawl_fallback_simple(url)


async def crawl_fallback_simple(url: str):
    """
    Phương pháp dự phòng - crawl tất cả các box-question
    """
    print("🔄 Thử phương pháp dự phòng...")
    
    config = CrawlerRunConfig(
        css_selector='.box-question',  # Tất cả box-question
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["script", "style", "nav", "footer"],
        word_count_threshold=5
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        
        if result.success and result.markdown:
            # Tìm phần có chứa "HƯỚNG DẪN GIẢI"
            lines = result.markdown.split('\n')
            huong_dan_section = []
            capture = False
            
            for line in lines:
                if 'HƯỚNG DẪN GIẢI' in line.upper():
                    capture = True
                    huong_dan_section.append(line)
                elif capture:
                    if line.strip() and not line.startswith('#'):
                        huong_dan_section.append(line)
                    elif line.startswith('##') and capture:
                        break  # Kết thúc section này
            
            if huong_dan_section:
                content = '\n'.join(huong_dan_section)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(f"crawl4ai/output/huong_dan_fallback_{timestamp}.md")
                output_file.parent.mkdir(exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Tìm thấy section HƯỚNG DẪN GIẢI!")
                print(f"💾 Đã lưu: {output_file}")
                print(f"\n📖 Nội dung:")
                print("-" * 50)
                print(content[:800] + "..." if len(content) > 800 else content)
                print("-" * 50)
                
                return {
                    'content': content,
                    'method': 'fallback',
                    'timestamp': timestamp
                }
        
        print("❌ Không tìm thấy nội dung HƯỚNG DẪN GIẢI")
        return None


def main():
    """Hàm chính"""
    # URL mặc định
    url = "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html"
    
    print("🚀 CRAWL4AI - HƯỚNG DẪN GIẢI CHI TIẾT")
    print("=" * 60)
    print(f"URL: {url}")
    print("=" * 60)
    
    # Chạy crawler
    result = asyncio.run(crawl_huong_dan_giai_simple(url))
    
    if result:
        print("\n✅ THÀNH CÔNG!")
        print("📁 Kiểm tra thư mục crawl4ai/output/ để xem kết quả")
    else:
        print("\n❌ KHÔNG THÀNH CÔNG!")
        print("💡 Hãy kiểm tra URL hoặc cấu trúc trang web")


if __name__ == "__main__":
    main() 