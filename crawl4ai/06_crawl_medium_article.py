import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
import json
from datetime import datetime
import os

async def crawl_medium_article():
    """
    Crawl content từ Medium article về "A Real-time Retrieval System for RAG on Social Media Data"
    """
    
    # URL cần crawl
    url = "https://medium.com/decodingml/a-real-time-retrieval-system-for-rag-on-social-media-data-9cc01d50a2a0"
    
    # Cấu hình browser
    browser_config = BrowserConfig(
        headless=True,  # Chạy ẩn browser
        verbose=True,   # Hiển thị thông tin debug
        viewport_width=1280,
        viewport_height=720
    )
    
    # Cấu hình crawler
    crawler_config = CrawlerRunConfig(
        # Content processing
        word_count_threshold=10,  # Lọc các đoạn text có ít hơn 10 từ
        excluded_tags=['nav', 'header', 'footer', 'aside'],  # Loại bỏ các thẻ không cần thiết
        exclude_external_links=True,  # Loại bỏ link ngoài
        
        # Cache control
        cache_mode=CacheMode.BYPASS,  # Bỏ qua cache để lấy content mới
        
        # Media handling
        exclude_external_images=False,  # Giữ lại ảnh từ external
        wait_for_images=True,  # Đợi ảnh load xong
        
        # Screenshot
        screenshot=True,  # Chụp screenshot trang
        
        # Xử lý content động
        process_iframes=True,
        remove_overlay_elements=True
    )
    
    print(f"🚀 Bắt đầu crawl Medium article...")
    print(f"📝 URL: {url}")
    print("-" * 80)
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Thực hiện crawl
            result = await crawler.arun(
                url=url,
                config=crawler_config
            )
            
            if result.success:
                print("✅ Crawl thành công!")
                
                # Thông tin cơ bản
                print(f"\n📊 THÔNG TIN CƠ BẢN:")
                print(f"   - URL: {result.url}")
                print(f"   - Status Code: {result.status_code}")
                print(f"   - Độ dài HTML: {len(result.html):,} ký tự")
                print(f"   - Độ dài Markdown: {len(result.markdown):,} ký tự")
                print(f"   - Thời gian crawl: {result.response_headers.get('date', 'N/A')}")
                
                # In một phần content để xem trước
                print(f"\n📝 CONTENT PREVIEW (500 ký tự đầu):")
                print("-" * 50)
                print(result.markdown[:500])
                print("-" * 50)
                
                # Thông tin về links
                internal_links = result.links.get("internal", [])
                external_links = result.links.get("external", [])
                print(f"\n🔗 LINKS:")
                print(f"   - Internal links: {len(internal_links)}")
                print(f"   - External links: {len(external_links)}")
                
                # Thông tin về media
                images = result.media.get("images", [])
                print(f"\n🖼️ MEDIA:")
                print(f"   - Tổng số ảnh: {len(images)}")
                
                if images:
                    print("   - Top 3 ảnh:")
                    for i, img in enumerate(images[:3]):
                        print(f"     {i+1}. {img['src']}")
                        print(f"        Alt: {img.get('alt', 'N/A')}")
                
                # Lưu kết quả vào files
                await save_results(result, url)
                
                return result
                
            else:
                print(f"❌ Crawl thất bại: {result.error_message}")
                return None
                
    except Exception as e:
        print(f"❌ Lỗi khi crawl: {str(e)}")
        return None

async def save_results(result, url):
    """
    Lưu kết quả crawl vào các file khác nhau
    """
    # Tạo thư mục output nếu chưa có
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"medium_article_{timestamp}"
    
    print(f"\n💾 ĐANG LƯU KẾT QUẢ...")
    
    # 1. Lưu raw HTML
    html_file = os.path.join(output_dir, f"{base_filename}_raw.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(result.html)
    print(f"   ✅ Raw HTML: {html_file}")
    
    # 2. Lưu cleaned HTML 
    if result.cleaned_html:
        cleaned_html_file = os.path.join(output_dir, f"{base_filename}_cleaned.html")
        with open(cleaned_html_file, 'w', encoding='utf-8') as f:
            f.write(result.cleaned_html)
        print(f"   ✅ Cleaned HTML: {cleaned_html_file}")
    
    # 3. Lưu Markdown
    markdown_file = os.path.join(output_dir, f"{base_filename}.md")
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(result.markdown)
    print(f"   ✅ Markdown: {markdown_file}")
    
    # 4. Lưu thông tin metadata dưới dạng JSON
    metadata = {
        "url": result.url,
        "status_code": result.status_code,
        "crawl_timestamp": timestamp,
        "html_length": len(result.html),
        "markdown_length": len(result.markdown),
        "internal_links_count": len(result.links.get("internal", [])),
        "external_links_count": len(result.links.get("external", [])),
        "images_count": len(result.media.get("images", [])),
        "response_headers": dict(result.response_headers) if result.response_headers else {}
    }
    
    metadata_file = os.path.join(output_dir, f"{base_filename}_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Metadata: {metadata_file}")
    
    # 5. Lưu links
    links_data = {
        "internal_links": result.links.get("internal", []),
        "external_links": result.links.get("external", [])
    }
    
    links_file = os.path.join(output_dir, f"{base_filename}_links.json")
    with open(links_file, 'w', encoding='utf-8') as f:
        json.dump(links_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Links: {links_file}")
    
    # 6. Lưu thông tin images
    if result.media.get("images"):
        images_file = os.path.join(output_dir, f"{base_filename}_images.json")
        with open(images_file, 'w', encoding='utf-8') as f:
            json.dump(result.media["images"], f, indent=2, ensure_ascii=False)
        print(f"   ✅ Images: {images_file}")
    
    # 7. Lưu screenshot nếu có
    if result.screenshot:
        screenshot_file = os.path.join(output_dir, f"{base_filename}_screenshot.png")
        try:
            with open(screenshot_file, 'wb') as f:
                # Kiểm tra xem screenshot là bytes hay string
                if isinstance(result.screenshot, str):
                    # Nếu là string (base64), decode trước
                    import base64
                    f.write(base64.b64decode(result.screenshot))
                else:
                    # Nếu đã là bytes, ghi trực tiếp
                    f.write(result.screenshot)
            print(f"   ✅ Screenshot: {screenshot_file}")
        except Exception as e:
            print(f"   ⚠️ Không thể lưu screenshot: {str(e)}")

async def main():
    """
    Hàm main để chạy crawler
    """
    print("🤖 CRAWL4AI - MEDIUM ARTICLE CRAWLER")
    print("=" * 50)
    
    result = await crawl_medium_article()
    
    if result:
        print(f"\n🎉 HOÀN THÀNH!")
        print("📁 Tất cả file đã được lưu trong thư mục 'output/'")
        print("\n💡 Bạn có thể:")
        print("   - Đọc file .md để xem content đã được format")
        print("   - Xem file .html để xem raw content")
        print("   - Kiểm tra file .json để xem metadata và links")
    else:
        print("\n😞 Crawl không thành công. Vui lòng kiểm tra lại URL hoặc kết nối mạng.")

if __name__ == "__main__":
    asyncio.run(main()) 