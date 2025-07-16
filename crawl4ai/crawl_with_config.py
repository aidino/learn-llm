#!/usr/bin/env python3
"""
Script crawl HƯỚNG DẪN GIẢI CHI TIẾT với cấu hình từ file JSON
Sử dụng file config_crawl.json để tùy chỉnh các tham số
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


def load_config(config_file="crawl4ai/config_crawl.json"):
    """Đọc file cấu hình JSON"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file cấu hình: {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi đọc file cấu hình: {e}")
        return None


async def crawl_with_config(url: str, config: dict):
    """
    Crawl với cấu hình từ file JSON
    
    Args:
        url: URL cần crawl
        config: Dictionary cấu hình từ JSON
    """
    
    # Lấy cấu hình crawl
    crawl_conf = config.get("crawl_config", {})
    selectors = config.get("selectors", {})
    output_conf = config.get("output", {})
    
    # Tạo CrawlerRunConfig
    cache_mode = getattr(CacheMode, crawl_conf.get("cache_mode", "BYPASS"))
    
    crawler_config = CrawlerRunConfig(
        css_selector=selectors.get("primary"),
        cache_mode=cache_mode,
        excluded_tags=crawl_conf.get("excluded_tags", []),
        word_count_threshold=crawl_conf.get("word_count_threshold", 3),
        exclude_external_links=crawl_conf.get("exclude_external_links", True),
        exclude_external_images=crawl_conf.get("exclude_external_images", True)
    )
    
    print(f"🔍 Đang crawl: {url}")
    print(f"🎯 Primary selector: {selectors.get('primary')}")
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=crawler_config)
        
        if not result.success:
            print(f"❌ Lỗi: {result.error_message}")
            return None
        
        if result.cleaned_html:
            print("✅ Tìm thấy nội dung với primary selector!")
            return await save_results(result, config, "primary")
        else:
            print("⚠️  Không tìm thấy nội dung với primary selector")
            return await crawl_fallback(url, config)


async def crawl_fallback(url: str, config: dict):
    """Phương pháp dự phòng"""
    print("🔄 Thử fallback selector...")
    
    selectors = config.get("selectors", {})
    crawl_conf = config.get("crawl_config", {})
    
    cache_mode = getattr(CacheMode, crawl_conf.get("cache_mode", "BYPASS"))
    
    fallback_config = CrawlerRunConfig(
        css_selector=selectors.get("fallback"),
        cache_mode=cache_mode,
        excluded_tags=crawl_conf.get("excluded_tags", []),
        word_count_threshold=crawl_conf.get("word_count_threshold", 5)
    )
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=fallback_config)
        
        if result.success and result.markdown:
            # Tìm kiếm keywords trong nội dung
            keywords = config.get("keywords", {}).get("target", [])
            content = result.markdown
            
            # Tìm section chứa HƯỚNG DẪN GIẢI
            lines = content.split('\n')
            found_section = []
            capture = False
            
            for line in lines:
                if any(keyword.upper() in line.upper() for keyword in keywords):
                    capture = True
                    found_section.append(line)
                elif capture:
                    if line.strip() and not line.startswith('#'):
                        found_section.append(line)
                    elif line.startswith('##') and capture:
                        break
            
            if found_section:
                print("✅ Tìm thấy section HƯỚNG DẪN GIẢI!")
                
                # Tạo result object cho fallback
                class FallbackResult:
                    def __init__(self, content, url):
                        self.markdown = '\n'.join(content)
                        self.cleaned_html = f"<div>{self.markdown}</div>"
                        self.url = url
                        self.success = True
                
                fallback_result = FallbackResult(found_section, url)
                return await save_results(fallback_result, config, "fallback")
        
        print("❌ Không tìm thấy nội dung với fallback method")
        return None


async def save_results(result, config: dict, method: str):
    """Lưu kết quả theo cấu hình"""
    output_conf = config.get("output", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(output_conf.get("directory", "crawl4ai/output"))
    output_dir.mkdir(exist_ok=True)
    
    prefix = output_conf.get("prefix", "huong_dan_giai")
    formats = output_conf.get("formats", ["markdown"])
    
    results = {
        'method': method,
        'timestamp': timestamp,
        'files': []
    }
    
    # Lưu theo format được chỉ định
    if "html" in formats:
        html_file = output_dir / f"{prefix}_html_{method}_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(result.cleaned_html)
        results['files'].append(str(html_file))
        print(f"💾 Đã lưu HTML: {html_file}")
    
    if "markdown" in formats:
        md_file = output_dir / f"{prefix}_md_{method}_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(result.markdown)
        results['files'].append(str(md_file))
        print(f"💾 Đã lưu Markdown: {md_file}")
    
    if "json" in formats:
        json_file = output_dir / f"{prefix}_info_{method}_{timestamp}.json"
        info = {
            'url': result.url,
            'method': method,
            'timestamp': timestamp,
            'content_length': len(result.markdown),
            'html_length': len(result.cleaned_html)
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        results['files'].append(str(json_file))
        print(f"💾 Đã lưu JSON: {json_file}")
    
    # Hiển thị preview
    print(f"\n📖 Preview nội dung:")
    print("-" * 50)
    preview = result.markdown[:1000] + "..." if len(result.markdown) > 1000 else result.markdown
    print(preview)
    print("-" * 50)
    
    return results


def main():
    """Hàm chính"""
    print("🚀 CRAWL4AI - HƯỚNG DẪN GIẢI CHI TIẾT (Config Mode)")
    print("=" * 60)
    
    # Đọc cấu hình
    config = load_config()
    if not config:
        print("❌ Không thể đọc được file cấu hình!")
        return
    
    # Chọn URL
    urls = config.get("urls", {})
    if not urls:
        print("❌ Không tìm thấy URL nào trong cấu hình!")
        return
    
    print("📋 Các URL có sẵn:")
    for key, url in urls.items():
        print(f"  {key}: {url}")
    
    # Sử dụng URL đầu tiên làm mặc định
    url_key = list(urls.keys())[0]
    url = urls[url_key]
    
    print(f"\n🔗 Sử dụng URL: {url_key}")
    print(f"🌐 {url}")
    print("=" * 60)
    
    # Chạy crawler
    result = asyncio.run(crawl_with_config(url, config))
    
    if result:
        print(f"\n✅ THÀNH CÔNG!")
        print(f"📁 Phương pháp: {result['method']}")
        print(f"📂 Files đã tạo: {len(result['files'])}")
        for file_path in result['files']:
            print(f"   📄 {file_path}")
    else:
        print(f"\n❌ KHÔNG THÀNH CÔNG!")
        print("💡 Hãy kiểm tra URL hoặc cập nhật cấu hình selectors")


if __name__ == "__main__":
    main() 