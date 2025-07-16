import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import JsonXPathExtractionStrategy

async def extract_idiom_xpath():

    # 2. Define the JSON schema (XPath version) idioms-list
    schema = {
        "name": "Idioms via XPath",
        "baseSelector": "div.idioms-list",
        "fields": [
            {
                "name": "idiom-title",
                "selector": "h2.idiom-title",
                "type": "text"
            },
            {
                "name": "idiom-meaning",
                "selector": "h2.idiom-meaning",
                "type": "text"
            },
            {
                "name": "idiom-examples",
                "selector": "h2.idiom-examples",
                "type": "text"
            }
        ]
    }

    # 3. Place the strategy in the CrawlerRunConfig
    config = CrawlerRunConfig(
        extraction_strategy=JsonXPathExtractionStrategy(schema, verbose=True)
    )

    # 4. Use raw:// scheme to pass dummy_html directly
    raw_url = "https://www.phrases.org.uk/idioms/whipper-snapper.html"

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=raw_url,
            config=config
        )
        
        print("RESULT: ")
        print("res")

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        data = json.loads(result.extracted_content)
        print("DATA:")
        print(data)

asyncio.run(extract_idiom_xpath())
