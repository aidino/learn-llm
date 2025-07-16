#!/usr/bin/env python3
"""
Demo script để test parallel crawling với 2-3 URLs
"""

import asyncio
import sys
from pathlib import Path

# Import script chính
try:
    from crawl_parallel_yaml import load_config, crawl_all_urls_parallel, save_parallel_results
except ImportError:
    print("❌ Không thể import script chính. Hãy đảm bảo file crawl_parallel_yaml.py tồn tại")
    sys.exit(1)


def create_demo_config():
    """Tạo config demo với 2-3 URLs để test"""
    return {
        "urls": [
            "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-cau-giay-nam-2023-a142098.html",
            "https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-truong-vinschool-nam-2023-a142088.html"
        ],
        "parallel_config": {
            "max_concurrent_workers": 2,
            "per_url_timeout": 30,
            "batch_delay": 1,
            "continue_on_error": True,
            "show_progress": True
        },
        "selectors": {
            "primary": 'div.box-question[id="sub-question-2"]',
            "fallback": ".box-question",
            "alternative": ".solution-box, [class*='question'], [class*='answer'], [class*='solution']",
            "id_selector": "#sub-question-2"
        },
        "crawl_config": {
            "excluded_tags": ["script", "style", "nav", "footer", "header", "aside", "noscript"],
            "word_count_threshold": 3,
            "exclude_external_links": True,
            "exclude_external_images": True,
            "exclude_social_media_links": True,
            "cache_mode": "BYPASS",
            "timeout": 30,
            "user_agent": "Mozilla/5.0 (compatible; Crawl4AI/0.6.3; Educational Purpose)"
        },
        "output": {
            "directory": "crawl4ai/output",
            "prefix": "demo_parallel",
            "formats": ["html", "markdown", "json"],
            "include_timestamp": True,
            "create_date_folder": False,
            "create_summary": True
        },
        "keywords": {
            "target": [
                "HƯỚNG DẪN GIẢI CHI TIẾT",
                "hướng dẫn giải",
                "giải chi tiết",
                "bài giải",
                "cách giải",
                "lời giải"
            ],
            "exclude": [
                "quảng cáo",
                "advertisement",
                "sidebar",
                "footer",
                "navigation",
                "menu",
                "banner",
                "popup"
            ]
        },
        "advanced": {
            "max_retries": 2,
            "delay_between_requests": 1,
            "use_proxy": False,
            "proxy_settings": {
                "http": "",
                "https": ""
            },
            "log_level": "INFO",
            "save_raw_html": False,
            "verify_ssl": True
        }
    }


async def demo_parallel_crawl():
    """Demo chạy parallel crawling"""
    print("🧪 DEMO PARALLEL CRAWLING")
    print("=" * 50)
    
    # Sử dụng config demo
    config = create_demo_config()
    urls = config["urls"]
    
    print(f"📋 Demo với {len(urls)} URLs:")
    for i, url in enumerate(urls, 1):
        print(f"  {i}. {url[:80]}...")
    
    print(f"\n⚙️  Config:")
    print(f"   Max workers: {config['parallel_config']['max_concurrent_workers']}")
    print(f"   Timeout per URL: {config['parallel_config']['per_url_timeout']}s")
    print(f"   Output formats: {', '.join(config['output']['formats'])}")
    
    print("\n" + "=" * 50)
    
    try:
        # Chạy parallel crawling
        results = await crawl_all_urls_parallel(urls, config)
        
        # Lưu kết quả
        summary = await save_parallel_results(results, config)
        
        print(f"\n✅ DEMO HOÀN THÀNH!")
        print(f"📊 Kết quả:")
        print(f"   🎯 URLs thành công: {summary['successful_urls']}/{summary['total_urls']}")
        print(f"   📄 Files đã tạo: {len(summary['files_created'])}")
        
        # Hiển thị thông tin chi tiết
        for i, result in enumerate(summary['results']):
            status = "✅" if result['success'] else "❌"
            if result['success']:
                print(f"   {status} URL {i+1}: {result['method']} ({result['duration']:.1f}s, {result['content_length']} chars)")
            else:
                print(f"   {status} URL {i+1}: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Lỗi trong demo: {str(e)}")
        return False


def main():
    """Hàm chính"""
    print("🚀 Demo Script - Parallel Crawling Test")
    
    # Kiểm tra dependencies
    try:
        import yaml
        import crawl4ai
        print("✅ Dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Cài đặt: pip install pyyaml crawl4ai")
        return
    
    # Tạo output directory nếu chưa có
    output_dir = Path("crawl4ai/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Chạy demo
    try:
        success = asyncio.run(demo_parallel_crawl())
        if success:
            print("\n🎉 Demo thành công! Kiểm tra thư mục crawl4ai/output để xem kết quả")
        else:
            print("\n😞 Demo không thành công")
    except KeyboardInterrupt:
        print("\n⏹️  Demo đã bị dừng")
    except Exception as e:
        print(f"\n💥 Lỗi không mong muốn: {str(e)}")


if __name__ == "__main__":
    main() 