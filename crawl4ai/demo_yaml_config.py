#!/usr/bin/env python3
"""
Demo script cho CRAWL4AI với YAML config
Cho phép người dùng nhập URL tùy chỉnh hoặc chọn từ danh sách có sẵn
"""

import asyncio
import sys
import yaml
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


def add_custom_url_to_config(config_file="crawl4ai/config_crawl.yaml"):
    """Thêm URL tùy chỉnh vào config"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file config: {config_file}")
        return None

    custom_url = input("🌐 Nhập URL muốn crawl: ").strip()
    if not custom_url:
        print("❌ URL không được để trống!")
        return None
    
    if not custom_url.startswith(('http://', 'https://')):
        custom_url = 'https://' + custom_url
    
    # Thêm vào config
    custom_name = f"custom_{len(config.get('urls', {})) + 1}"
    if 'urls' not in config:
        config['urls'] = {}
    
    config['urls'][custom_name] = custom_url
    
    # Ghi lại file config
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        print(f"✅ Đã thêm URL mới: {custom_name}")
        return config
    except Exception as e:
        print(f"❌ Lỗi ghi file config: {e}")
        return None


async def demo_crawl(url: str):
    """Demo crawl đơn giản"""
    print(f"\n🚀 Bắt đầu demo crawl")
    print(f"🔗 URL: {url}")
    print("=" * 60)
    
    # Cấu hình đơn giản
    config = CrawlerRunConfig(
        css_selector="#sub-question-2",
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["script", "style", "nav", "footer"],
        word_count_threshold=3,
        exclude_external_links=True
    )
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config)
            
            if not result.success:
                print(f"❌ Crawl thất bại: {result.error_message}")
                return None
            
            if result.cleaned_html and "HƯỚNG DẪN GIẢI" in result.markdown.upper():
                print("✅ Tìm thấy HƯỚNG DẪN GIẢI CHI TIẾT!")
                
                # Lưu kết quả nhanh
                output_dir = Path("crawl4ai/output")
                output_dir.mkdir(exist_ok=True)
                
                demo_file = output_dir / "demo_result.md"
                with open(demo_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Demo Crawl Result\n\n")
                    f.write(f"**URL:** {url}\n\n")
                    f.write(f"**Thời gian:** {asyncio.get_event_loop().time()}\n\n")
                    f.write("---\n\n")
                    f.write(result.markdown)
                
                print(f"💾 Đã lưu kết quả demo: {demo_file}")
                
                # Hiển thị preview
                lines = result.markdown.split('\n')
                preview_lines = []
                found_huong_dan = False
                
                for line in lines:
                    if 'HƯỚNG DẪN GIẢI' in line.upper():
                        found_huong_dan = True
                    
                    if found_huong_dan:
                        preview_lines.append(line)
                        if len(preview_lines) >= 20:  # Giới hạn 20 dòng
                            break
                
                print(f"\n📖 Preview (20 dòng đầu):")
                print("-" * 50)
                print('\n'.join(preview_lines))
                print("-" * 50)
                
                return demo_file
            else:
                print("⚠️  Không tìm thấy phần HƯỚNG DẪN GIẢI CHI TIẾT")
                print("💡 Có thể trang này không có nội dung như mong đợi")
                return None
                
    except Exception as e:
        print(f"❌ Lỗi trong quá trình crawl: {e}")
        return None


def main():
    """Hàm chính cho demo"""
    print("🎯 DEMO CRAWL4AI - YAML CONFIG")
    print("=" * 50)
    
    while True:
        print("\n📋 Lựa chọn:")
        print("1. Sử dụng URL từ config có sẵn")
        print("2. Thêm URL tùy chỉnh và crawl")
        print("3. Nhập URL trực tiếp (không lưu config)")
        print("4. Thoát")
        
        choice = input("\n🔢 Chọn (1-4): ").strip()
        
        if choice == '1':
            # Sử dụng script YAML chính
            print("\n🔄 Chuyển đến script YAML chính...")
            import subprocess
            subprocess.run([sys.executable, "crawl4ai/crawl_with_config_yaml.py"])
            
        elif choice == '2':
            # Thêm URL mới vào config
            config = add_custom_url_to_config()
            if config:
                print("\n🔄 Chạy lại script với config mới...")
                import subprocess
                subprocess.run([sys.executable, "crawl4ai/crawl_with_config_yaml.py"])
                
        elif choice == '3':
            # Demo crawl trực tiếp
            url = input("\n🌐 Nhập URL: ").strip()
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                
                result = asyncio.run(demo_crawl(url))
                if result:
                    print(f"\n✅ Demo hoàn thành! Kết quả lưu tại: {result}")
                else:
                    print("\n❌ Demo không thành công")
            
        elif choice == '4':
            print("\n👋 Tạm biệt!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Thoát chương trình!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}") 