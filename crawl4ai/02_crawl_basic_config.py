import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )
    
    # IMPORTANT: By default cache mode is set to CacheMode.ENABLED. So to have fresh content, you need to set it to CacheMode.BYPASS
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.vietjack.com/tai-lieu-mon-toan/de-thi-vao-lop-6-toan-truong-thcs-trong-diem-bn-2025.jsp",
            config=run_config
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())