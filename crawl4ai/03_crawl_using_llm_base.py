import os
import json
import asyncio
from typing import Dict
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, BrowserConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class DeThiMonToan(BaseModel):
    school_name: str = Field(..., description="Tên của trường cung cấp đề thi")
    url: str = Field(..., description="Đường dẫn đến đề thi")
    
def write_output_to_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)
        
async def extract_structured_data_using_llm(
    provider: str, api_token: str = None, extra_headers: Dict[str, str] = None
):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    if api_token is None and provider != "ollama":
        print(f"API token is required for {provider}. Skipping this example.")
        return

    browser_config = BrowserConfig(headless=True)

    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 2000}
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            llm_config = LLMConfig(provider=provider,api_token=api_token),
            schema=DeThiMonToan.model_json_schema(),
            extraction_type="schema",
            instruction="""
            Từ nội dung được crawl, extract toàn bộ thông tin về đề thi môn toán vào lớp 6 của các trường.""",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-c1387.html", config=crawler_config
        )
        print(result.extracted_content)
        write_output_to_file('dethilop6.txt', result.extracted_content)
        

if __name__ == "__main__":

    asyncio.run(
        extract_structured_data_using_llm(
            provider="openai/gpt-4o-mini", api_token=os.getenv("OPENAI_API_KEY")
        )
    )

