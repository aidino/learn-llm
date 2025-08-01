import os
import asyncio
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai import LLMExtractionStrategy
from dotenv import load_dotenv

_=load_dotenv()

# URL_FILE = 'dethi_lop6_toan.depth2.url.txt'
URL_FILE = 'sample_link.txt'


def write_output_to_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

class Exam(BaseModel):
    question: str = Field(..., description="Câu hỏi chính của bài thi, thường bắt đầu bằng 'Câu ...'. Câu hỏi có thể chứa các công thức latex")
    image_url: Optional[str] = Field(None, description="URL hình ảnh minh họa, trong trường hợp câu hỏi cần hình vẽ để mô tả")
    solution: str = Field(..., description="Hướng dẫn giải, trong hướng dẫn có thể có các công thức latex")
    result: Optional[str] = Field(None, description="Đáp án của những câu hỏi trắc nghiệm")

async def main():
    try:
        with open(URL_FILE, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print(f"Lỗi: File {URL_FILE} trống hoặc không tồn tại.")
            return
        print(f"🔎 Tìm thấy {len(urls)} link trong file {URL_FILE}.")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{URL_FILE}'. Vui lòng tạo file này và thêm các link vào.")
        return
    # 1. Define the LLM extraction strategy
    # gemini_token = os.getenv("GEMINI_API_KEY")
    
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(provider="openrouter/qwen/qwen3-8b:free", api_token=os.getenv("OPEN_ROUTER")),
        schema=Exam.model_json_schema(), # Or use model_json_schema()
        extraction_type="schema",
        instruction="""
        Bạn là một chuyên gia trích xuất dữ liệu web. Nhiệm vụ của bạn là đọc nội dung của một đề thi được cung cấp và trích xuất TOÀN BỘ các câu hỏi có trong đó.
        - Hãy trích xuất tất cả các câu hỏi, bắt đầu từ 'Câu 1' cho đến câu cuối cùng.
        - Với mỗi câu hỏi, lấy đầy đủ nội dung câu hỏi, hình ảnh minh họa (nếu có), lời giải chi tiết và đáp án cuối cùng.
        - Các công thức toán học trong câu hỏi và lời giải PHẢI được định dạng bằng LaTeX.
        - Bỏ qua tất cả các nội dung không phải là câu hỏi như: lời giới thiệu đầu trang, các bình luận, quảng cáo, hoặc các link liên quan ở cuối trang.
        - Định dạng kết quả đầu ra theo đúng cấu trúc JSON schema đã được cung cấp.
        """,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="fit_markdown",   # or "html", "fit_markdown"
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        css_selector="#sub-question-2",
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # 4. Let's say we want to crawl a single page
        results = []
        file_index=1
        for index, url in enumerate(urls):
            result = await crawler.arun(
                url=url,
                config=crawl_config,
            )

            if result.success:
                # 5. The extracted content is presumably JSON
                data = json.loads(result.extracted_content)
                results.extend(data)
                # print(f"đã thêm {len(data)} bài, tổng số hiện nay là: {len(results)}")
                # print("Extracted items:", data)
                # write_output_to_file('extracted.json', data)
                
                # 6. Show usage stats
                # llm_strategy.show_usage()  # prints token usage
            else:
                print("Error:", result.error_message)
            
            print(f"crawed {index+1}/{len(urls)}")
            if (index+1)%5 == 0:
                write_output_to_file(f"crawled/openrouter_crawled_p{file_index}.json", results)
                results=[]
                file_index +=1
            
            
            
        write_output_to_file('crawled/openrouter_crawled_last.json', results)

if __name__ == "__main__":
    asyncio.run(main())
