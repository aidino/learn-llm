#!/usr/bin/env python3
"""
Script crawl HƯỚNG DẪN GIẢI CHI TIẾT với cấu hình từ file YAML
Sử dụng file config_crawl.yaml để tùy chỉnh các tham số
"""

import asyncio
import json
import sys
import yaml
from datetime import datetime
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


def load_config(config_file="crawl4ai/config_crawl.yaml"):
    """Đọc file cấu hình YAML"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"✅ Đã đọc file cấu hình: {config_file}")
            return config
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file cấu hình: {config_file}")
        print("💡 Hãy tạo file config_crawl.yaml hoặc kiểm tra đường dẫn")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Lỗi đọc file YAML: {e}")
        return None
    except ImportError:
        print("❌ Thiếu thư viện PyYAML. Cài đặt bằng: pip install pyyaml")
        return None


async def crawl_with_config(url: str, config: dict):
    """
    Crawl với cấu hình từ file YAML
    
    Args:
        url: URL cần crawl
        config: Dictionary cấu hình từ YAML
    """
    
    # Lấy cấu hình crawl
    crawl_conf = config.get("crawl_config", {})
    selectors = config.get("selectors", {})
    output_conf = config.get("output", {})
    advanced = config.get("advanced", {})
    
    # Tạo CrawlerRunConfig với nhiều tùy chọn hơn
    cache_mode = getattr(CacheMode, crawl_conf.get("cache_mode", "BYPASS"))
    
    crawler_config = CrawlerRunConfig(
        css_selector=selectors.get("id_selector", selectors.get("primary")),  # Thử id_selector trước
        cache_mode=cache_mode,
        excluded_tags=crawl_conf.get("excluded_tags", []),
        word_count_threshold=crawl_conf.get("word_count_threshold", 3),
        exclude_external_links=crawl_conf.get("exclude_external_links", True),
        exclude_external_images=crawl_conf.get("exclude_external_images", True),
        exclude_social_media_links=crawl_conf.get("exclude_social_media_links", True)
    )
    
    print(f"🔍 Đang crawl: {url}")
    print(f"🎯 Primary selector: {selectors.get('id_selector', selectors.get('primary'))}")
    
    # Retry mechanism
    max_retries = advanced.get("max_retries", 3)
    delay = advanced.get("delay_between_requests", 1)
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"🔄 Thử lại lần {attempt + 1}/{max_retries}...")
                await asyncio.sleep(delay)
            
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=crawler_config)
                
                if not result.success:
                    print(f"⚠️  Attempt {attempt + 1} failed: {result.error_message}")
                    continue
                
                if result.cleaned_html and result.cleaned_html.strip():
                    print("✅ Tìm thấy nội dung với primary selector!")
                    return await save_results(result, config, "primary")
                else:
                    print("⚠️  Không tìm thấy nội dung với primary selector")
                    if attempt == max_retries - 1:  # Chỉ thử fallback ở lần cuối
                        return await crawl_fallback(url, config)
                    
        except Exception as e:
            print(f"❌ Lỗi attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return await crawl_fallback(url, config)
    
    return None


async def crawl_fallback(url: str, config: dict):
    """Phương pháp dự phong với nhiều strategies"""
    print("🔄 Thử các phương pháp dự phòng...")
    
    selectors = config.get("selectors", {})
    crawl_conf = config.get("crawl_config", {})
    
    # Thử các selector khác nhau theo thứ tự
    fallback_selectors = [
        ("fallback", selectors.get("fallback")),
        ("alternative", selectors.get("alternative")),
        ("primary_without_id", 'div.box-question'),
        ("any_box", '.box, .question, .answer, .solution')
    ]
    
    cache_mode = getattr(CacheMode, crawl_conf.get("cache_mode", "BYPASS"))
    
    for method_name, selector in fallback_selectors:
        if not selector:
            continue
            
        print(f"🎯 Thử {method_name} selector: {selector}")
        
        fallback_config = CrawlerRunConfig(
            css_selector=selector,
            cache_mode=cache_mode,
            excluded_tags=crawl_conf.get("excluded_tags", []),
            word_count_threshold=crawl_conf.get("word_count_threshold", 5)
        )
        
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=fallback_config)
                
                if result.success and result.markdown:
                    # Tìm kiếm keywords trong nội dung
                    keywords = config.get("keywords", {}).get("target", [])
                    content = result.markdown
                    
                    # Tìm section chứa HƯỚNG DẪN GIẢI
                    found_section = extract_target_section(content, keywords)
                    
                    if found_section:
                        print(f"✅ Tìm thấy section HƯỚNG DẪN GIẢI với {method_name}!")
                        
                        # Tạo result object cho fallback
                        fallback_result = create_fallback_result(found_section, url)
                        return await save_results(fallback_result, config, f"fallback_{method_name}")
                        
        except Exception as e:
            print(f"⚠️  Lỗi với {method_name}: {str(e)}")
            continue
    
    print("❌ Không tìm thấy nội dung với bất kỳ phương pháp nào")
    return None


def extract_target_section(content: str, keywords: list):
    """Trích xuất section chứa keywords target"""
    lines = content.split('\n')
    found_section = []
    capture = False
    
    for line in lines:
        line_upper = line.upper()
        if any(keyword.upper() in line_upper for keyword in keywords):
            capture = True
            found_section.append(line)
        elif capture:
            if line.strip():
                # Tiếp tục capture nếu không phải header mới
                if not (line.startswith('##') and len(found_section) > 10):
                    found_section.append(line)
                else:
                    break  # Kết thúc section
            else:
                found_section.append(line)  # Giữ lại dòng trống
    
    return found_section if found_section else None


def create_fallback_result(content_lines: list, url: str):
    """Tạo result object cho fallback method"""
    class FallbackResult:
        def __init__(self, content, url):
            self.markdown = '\n'.join(content)
            self.cleaned_html = f"<div class='huong-dan-giai'>{self.markdown}</div>"
            self.url = url
            self.success = True
    
    return FallbackResult(content_lines, url)


async def save_results(result, config: dict, method: str):
    """Lưu kết quả theo cấu hình YAML"""
    output_conf = config.get("output", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(output_conf.get("directory", "crawl4ai/output"))
    output_dir.mkdir(exist_ok=True)
    
    # Tạo thư mục theo ngày nếu được cấu hình
    if output_conf.get("create_date_folder", False):
        date_folder = datetime.now().strftime("%Y-%m-%d")
        output_dir = output_dir / date_folder
        output_dir.mkdir(exist_ok=True)
    
    prefix = output_conf.get("prefix", "huong_dan_giai")
    formats = output_conf.get("formats", ["markdown"])
    include_timestamp = output_conf.get("include_timestamp", True)
    
    timestamp_suffix = f"_{timestamp}" if include_timestamp else ""
    
    results = {
        'method': method,
        'timestamp': timestamp,
        'files': [],
        'stats': {
            'content_length': len(result.markdown),
            'html_length': len(result.cleaned_html),
            'url': result.url
        }
    }
    
    # Lưu theo format được chỉ định
    if "html" in formats:
        html_file = output_dir / f"{prefix}_html_{method}{timestamp_suffix}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(result.cleaned_html)
        results['files'].append(str(html_file))
        print(f"💾 Đã lưu HTML: {html_file}")
    
    if "markdown" in formats:
        md_file = output_dir / f"{prefix}_md_{method}{timestamp_suffix}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(result.markdown)
        results['files'].append(str(md_file))
        print(f"💾 Đã lưu Markdown: {md_file}")
    
    if "json" in formats:
        json_file = output_dir / f"{prefix}_info_{method}{timestamp_suffix}.json"
        info = {
            'url': result.url,
            'method': method,
            'timestamp': timestamp,
            'content_length': len(result.markdown),
            'html_length': len(result.cleaned_html),
            'file_paths': results['files']
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


def select_url_interactive(urls: dict):
    """Cho phép người dùng chọn URL interactively"""
    if len(urls) == 1:
        url_key = list(urls.keys())[0]
        return url_key, urls[url_key]
    
    print("📋 Các URL có sẵn:")
    url_list = list(urls.items())
    for i, (key, url) in enumerate(url_list, 1):
        print(f"  {i}. {key}")
        print(f"     {url[:80]}...")
    
    while True:
        try:
            choice = input(f"\n🔢 Chọn URL (1-{len(url_list)}) hoặc Enter để dùng mặc định: ").strip()
            
            if not choice:  # Default choice
                url_key = url_list[0][0]
                return url_key, urls[url_key]
            
            idx = int(choice) - 1
            if 0 <= idx < len(url_list):
                url_key = url_list[idx][0]
                return url_key, urls[url_key]
            else:
                print(f"❌ Vui lòng chọn số từ 1 đến {len(url_list)}")
                
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ")
        except KeyboardInterrupt:
            print("\n👋 Thoát chương trình")
            sys.exit(0)


def main():
    """Hàm chính"""
    print("🚀 CRAWL4AI - HƯỚNG DẪN GIẢI CHI TIẾT (YAML Config)")
    print("=" * 60)
    
    # Đọc cấu hình
    config = load_config()
    if not config:
        print("❌ Không thể đọc được file cấu hình!")
        return
    
    # Hiển thị thông tin cấu hình
    print(f"⚙️  Log level: {config.get('advanced', {}).get('log_level', 'INFO')}")
    print(f"📁 Output directory: {config.get('output', {}).get('directory', 'crawl4ai/output')}")
    print(f"📄 Output formats: {', '.join(config.get('output', {}).get('formats', ['markdown']))}")
    
    # Chọn URL
    urls = config.get("urls", {})
    if not urls:
        print("❌ Không tìm thấy URL nào trong cấu hình!")
        return
    
    url_key, url = select_url_interactive(urls)
    
    print(f"\n🔗 Sử dụng URL: {url_key}")
    print(f"🌐 {url}")
    print("=" * 60)
    
    # Chạy crawler
    result = asyncio.run(crawl_with_config(url, config))
    
    if result:
        print(f"\n✅ THÀNH CÔNG!")
        print(f"📁 Phương pháp: {result['method']}")
        print(f"📊 Thống kê:")
        print(f"   📝 Độ dài nội dung: {result['stats']['content_length']} ký tự")
        print(f"   📄 Số files tạo: {len(result['files'])}")
        print(f"📂 Files đã tạo:")
        for file_path in result['files']:
            print(f"   📄 {file_path}")
    else:
        print(f"\n❌ KHÔNG THÀNH CÔNG!")
        print("💡 Hãy kiểm tra:")
        print("   - URL có đúng không")
        print("   - Kết nối internet")
        print("   - Cập nhật selectors trong config YAML")


if __name__ == "__main__":
    main() 