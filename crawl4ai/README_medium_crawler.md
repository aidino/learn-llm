# Medium Article Crawler với Crawl4AI

Script Python này sử dụng thư viện **crawl4ai** phiên bản mới nhất để crawl content từ bài viết Medium về "A Real-time Retrieval System for RAG on Social Media Data".

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Internet connection

### Cài đặt dependencies

```bash
# Cài đặt crawl4ai
pip install crawl4ai

# Cài đặt Playwright browsers (cần thiết cho crawl4ai)
playwright install
```

Hoặc nếu bạn đang sử dụng UV:

```bash
uv add crawl4ai
playwright install
```

## 📁 Cấu trúc file

```
crawl4ai/
├── 06_crawl_medium_article.py    # Script chính
├── README_medium_crawler.md      # File này
└── output/                       # Thư mục lưu kết quả (tự động tạo)
    ├── medium_article_[timestamp]_raw.html
    ├── medium_article_[timestamp]_cleaned.html
    ├── medium_article_[timestamp].md
    ├── medium_article_[timestamp]_metadata.json
    ├── medium_article_[timestamp]_links.json
    ├── medium_article_[timestamp]_images.json
    └── medium_article_[timestamp]_screenshot.png
```

## 🎯 Cách sử dụng

### Chạy script cơ bản

```bash
cd crawl4ai
python 06_crawl_medium_article.py
```

### Output được tạo

Script sẽ tạo các file sau trong thư mục `output/`:

1. **Raw HTML** (`*_raw.html`): HTML gốc từ trang web
2. **Cleaned HTML** (`*_cleaned.html`): HTML đã được làm sạch
3. **Markdown** (`*.md`): Content đã convert sang định dạng Markdown
4. **Metadata** (`*_metadata.json`): Thông tin meta về quá trình crawl
5. **Links** (`*_links.json`): Danh sách tất cả links (internal/external)
6. **Images** (`*_images.json`): Thông tin về các ảnh trong bài viết
7. **Screenshot** (`*_screenshot.png`): Screenshot của trang web

## ⚙️ Cấu hình

### Browser Config
```python
browser_config = BrowserConfig(
    headless=True,      # Chạy ẩn browser
    verbose=True,       # Hiển thị log debug
    viewport_width=1280,
    viewport_height=720
)
```

### Crawler Config
```python
crawler_config = CrawlerRunConfig(
    word_count_threshold=10,        # Lọc text < 10 từ
    excluded_tags=['nav', 'header', 'footer', 'aside'],  # Loại bỏ thẻ không cần
    exclude_external_links=True,    # Loại bỏ link ngoài
    cache_mode=CacheMode.BYPASS,    # Bỏ qua cache
    screenshot=True,                # Chụp screenshot
    wait_for_images=True,          # Đợi ảnh load
    process_iframes=True,          # Xử lý iframe
    remove_overlay_elements=True   # Xóa popup/overlay
)
```

## 📊 Thông tin Output

Khi chạy thành công, script sẽ hiển thị:

```
🤖 CRAWL4AI - MEDIUM ARTICLE CRAWLER
==================================================
🚀 Bắt đầu crawl Medium article...
📝 URL: https://medium.com/decodingml/a-real-time-retrieval-system-for-rag-on-social-media-data-9cc01d50a2a0
--------------------------------------------------------------------------------
✅ Crawl thành công!

📊 THÔNG TIN CƠ BẢN:
   - URL: https://medium.com/decodingml/...
   - Status Code: 200
   - Độ dài HTML: 125,340 ký tự
   - Độ dài Markdown: 8,520 ký tự
   - Thời gian crawl: Mon, 23 Dec 2024 10:30:45 GMT

📝 CONTENT PREVIEW (500 ký tự đầu):
--------------------------------------------------
# A Real-time Retrieval System for RAG on Social Media Data

In the era of information overload, building effective...
--------------------------------------------------

🔗 LINKS:
   - Internal links: 12
   - External links: 0

🖼️ MEDIA:
   - Tổng số ảnh: 8
   - Top 3 ảnh:
     1. https://miro.medium.com/v2/resize:fit:1400/1*abc123.png
        Alt: System Architecture Diagram
     2. https://miro.medium.com/v2/resize:fit:1200/1*def456.png
        Alt: RAG Pipeline Flow

💾 ĐANG LƯU KẾT QUẢ...
   ✅ Raw HTML: output/medium_article_20241223_103045_raw.html
   ✅ Cleaned HTML: output/medium_article_20241223_103045_cleaned.html
   ✅ Markdown: output/medium_article_20241223_103045.md
   ✅ Metadata: output/medium_article_20241223_103045_metadata.json
   ✅ Links: output/medium_article_20241223_103045_links.json
   ✅ Images: output/medium_article_20241223_103045_images.json
   ✅ Screenshot: output/medium_article_20241223_103045_screenshot.png

🎉 HOÀN THÀNH!
```

## 🔧 Tùy chỉnh

### Crawl URL khác

Để crawl URL khác, thay đổi biến `url` trong hàm `crawl_medium_article()`:

```python
url = "https://your-new-url.com"
```

### Thay đổi cấu hình crawl

Bạn có thể tùy chỉnh các tham số trong `CrawlerRunConfig`:

- `word_count_threshold`: Ngưỡng số từ tối thiểu
- `excluded_tags`: Danh sách thẻ HTML cần loại bỏ
- `exclude_external_links`: Có loại bỏ link ngoài không
- `screenshot`: Có chụp screenshot không
- `wait_for_images`: Có đợi ảnh load không

### Thêm xử lý JavaScript

Nếu trang cần tương tác JavaScript:

```python
js_code = [
    "window.scrollTo(0, document.body.scrollHeight);",  # Scroll xuống cuối
    "document.querySelector('.load-more').click();"     # Click nút load more
]

result = await crawler.arun(
    url=url,
    config=crawler_config,
    js_code=js_code
)
```

## 🐛 Troubleshooting

### Lỗi "playwright not found"
```bash
playwright install
```

### Lỗi "Permission denied" khi tạo file
Đảm bảo có quyền ghi trong thư mục hiện tại hoặc chạy với `sudo` (không khuyến khích).

### Lỗi timeout hoặc connection
- Kiểm tra kết nối internet
- Thử tăng timeout trong `BrowserConfig`
- Kiểm tra xem URL có accessible không

### Lỗi "Module not found"
```bash
pip install crawl4ai
```

## 📚 Tài liệu tham khảo

- [Crawl4AI Documentation](https://github.com/unclecode/crawl4ai)
- [Crawl4AI Examples](https://github.com/unclecode/crawl4ai/tree/main/docs/examples)
- [Medium Article Link](https://medium.com/decodingml/a-real-time-retrieval-system-for-rag-on-social-media-data-9cc01d50a2a0)

## 🤝 Đóng góp

Nếu bạn gặp lỗi hoặc muốn cải thiện script, hãy tạo issue hoặc pull request!

---

**Lưu ý**: Script này sử dụng API mới nhất của crawl4ai với `AsyncWebCrawler` và các config objects. Đảm bảo bạn đang sử dụng phiên bản crawl4ai mới nhất để tránh lỗi compatibility. 