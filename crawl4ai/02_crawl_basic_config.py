import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

def write_output_to_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)

async def main():
    browser_config = BrowserConfig(
        headless=True,
        verbose=True
    )
    run_config = CrawlerRunConfig(
        # Cache control
        cache_mode=CacheMode.ENABLED,  # Use cache if available,
        # Content filtering
        word_count_threshold=10,        # Minimum words per content block
        excluded_tags=['form', 'header'],
        exclude_external_links=True,    # Remove external links
        # Content processing
        remove_overlay_elements=True,   # Remove popups/modals
        process_iframes=True           # Process iframe content
    )
    
    # IMPORTANT: By default cache mode is set to CacheMode.ENABLED. So to have fresh content, you need to set it to CacheMode.BYPASS
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://loigiaihay.com/de-thi-vao-lop-6-mon-toan-c1387.html",
            config=run_config
        )
        # Different content formats
        # print(result.html)         # Raw HTML
        write_output_to_file("output/raw_html.html", result.html)
        
        # print(result.cleaned_html) # Cleaned HTML
        write_output_to_file("output/cleaned_html.html", result.cleaned_html)
        
        # print(result.markdown.raw_markdown) # Raw markdown from cleaned html
        write_output_to_file("output/raw_markdown.md", result.markdown.raw_markdown)
        
        # print(result.markdown.fit_markdown) # Most relevant content in markdown
        write_output_to_file("output/fit_markdown.md", result.markdown.fit_markdown)
        
        # Check success status
        print(f"Status: {result.success}")      # True if crawl succeeded
        print(f"Status code: {result.status_code}")  # HTTP status code (e.g., 200, 404)        
        # Access extracted media and links
        
        
        print("="*20, "MEDIA: ", "="*20)
        image_urls = [image.get("src","") for image in result.media.get("images", [])]
        print(image_urls)
        write_output_to_file("output/image_urls.txt", "\n".join(image_urls))
        
        print("="*20, "LINK: ", "="*20)
        result_links = [link.get("href","") for link in result.links.get("internal", [])]
        print(result.links)        # Dictionary of internal and external links
        write_output_to_file("output/result_links.txt", "\n".join(result_links))
        

if __name__ == "__main__":
    asyncio.run(main())