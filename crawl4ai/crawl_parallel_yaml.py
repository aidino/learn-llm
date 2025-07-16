#!/usr/bin/env python3
"""
Script crawl HƯỚNG DẪN GIẢI CHI TIẾT với crawling song song
Crawl tất cả URLs từ config YAML đồng thời để tối ưu hiệu suất
"""

import asyncio
import json
import sys
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


@dataclass
class CrawlResult:
    """Kết quả crawl cho một URL"""
    url: str
    success: bool
    method: str
    content: str
    html: str
    error: Optional[str] = None
    duration: float = 0.0
    content_length: int = 0


def load_config(config_file="crawl4ai/config_crawl.yaml") -> Optional[Dict]:
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


async def crawl_single_url(url: str, config: Dict, url_index: int) -> CrawlResult:
    """
    Crawl một URL duy nhất với retry mechanism
    
    Args:
        url: URL cần crawl
        config: Dictionary cấu hình từ YAML
        url_index: Index của URL trong danh sách (để tracking)
    
    Returns:
        CrawlResult: Kết quả crawl
    """
    start_time = time.time()
    
    # Lấy cấu hình
    crawl_conf = config.get("crawl_config", {})
    selectors = config.get("selectors", {})
    advanced = config.get("advanced", {})
    parallel_conf = config.get("parallel_config", {})
    
    print(f"🔍 [{url_index+1}] Đang crawl: {url[:80]}...")
    
    # Tạo CrawlerRunConfig
    cache_mode = getattr(CacheMode, crawl_conf.get("cache_mode", "BYPASS"))
    
    crawler_config = CrawlerRunConfig(
        css_selector=selectors.get("id_selector", selectors.get("primary")),
        cache_mode=cache_mode,
        excluded_tags=crawl_conf.get("excluded_tags", []),
        word_count_threshold=crawl_conf.get("word_count_threshold", 3),
        exclude_external_links=crawl_conf.get("exclude_external_links", True),
        exclude_external_images=crawl_conf.get("exclude_external_images", True),
        exclude_social_media_links=crawl_conf.get("exclude_social_media_links", True)
    )
    
    # Retry mechanism
    max_retries = advanced.get("max_retries", 3)
    delay = advanced.get("delay_between_requests", 1)
    timeout = parallel_conf.get("per_url_timeout", 45)
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"🔄 [{url_index+1}] Thử lại lần {attempt + 1}/{max_retries}...")
                await asyncio.sleep(delay)
            
            # Sử dụng timeout cho mỗi URL
            async with AsyncWebCrawler() as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=crawler_config),
                    timeout=timeout
                )
                
                if not result.success:
                    print(f"⚠️  [{url_index+1}] Attempt {attempt + 1} failed: {result.error_message}")
                    continue
                
                if result.cleaned_html and result.cleaned_html.strip():
                    duration = time.time() - start_time
                    print(f"✅ [{url_index+1}] Thành công với primary selector! ({duration:.1f}s)")
                    
                    return CrawlResult(
                        url=url,
                        success=True,
                        method="primary",
                        content=result.markdown or "",
                        html=result.cleaned_html,
                        duration=duration,
                        content_length=len(result.markdown or "")
                    )
                else:
                    print(f"⚠️  [{url_index+1}] Không tìm thấy nội dung với primary selector")
                    if attempt == max_retries - 1:  # Chỉ thử fallback ở lần cuối
                        return await crawl_fallback_single(url, config, url_index, start_time)
                    
        except asyncio.TimeoutError:
            print(f"⏰ [{url_index+1}] Timeout sau {timeout}s (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return await crawl_fallback_single(url, config, url_index, start_time)
        except Exception as e:
            print(f"❌ [{url_index+1}] Lỗi attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return await crawl_fallback_single(url, config, url_index, start_time)
    
    # Nếu tất cả attempts đều thất bại
    duration = time.time() - start_time
    return CrawlResult(
        url=url,
        success=False,
        method="failed",
        content="",
        html="",
        error="Tất cả attempts đều thất bại",
        duration=duration
    )


async def crawl_fallback_single(url: str, config: Dict, url_index: int, start_time: float) -> CrawlResult:
    """Phương pháp dự phòng cho một URL"""
    print(f"🔄 [{url_index+1}] Thử các phương pháp dự phòng...")
    
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
            
        print(f"🎯 [{url_index+1}] Thử {method_name} selector: {selector}")
        
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
                    found_section = extract_target_section(result.markdown, keywords)
                    
                    if found_section:
                        duration = time.time() - start_time
                        print(f"✅ [{url_index+1}] Tìm thấy section HƯỚNG DẪN GIẢI với {method_name}! ({duration:.1f}s)")
                        
                        return CrawlResult(
                            url=url,
                            success=True,
                            method=f"fallback_{method_name}",
                            content='\n'.join(found_section),
                            html=f"<div class='huong-dan-giai'>{'\n'.join(found_section)}</div>",
                            duration=duration,
                            content_length=len('\n'.join(found_section))
                        )
                        
        except Exception as e:
            print(f"⚠️  [{url_index+1}] Lỗi với {method_name}: {str(e)}")
            continue
    
    duration = time.time() - start_time
    return CrawlResult(
        url=url,
        success=False,
        method="fallback_failed",
        content="",
        html="",
        error="Không tìm thấy nội dung với bất kỳ phương pháp nào",
        duration=duration
    )


def extract_target_section(content: str, keywords: list) -> Optional[List[str]]:
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


async def crawl_all_urls_parallel(urls: List[str], config: Dict) -> List[CrawlResult]:
    """
    Crawl tất cả URLs song song với giới hạn concurrent workers
    
    Args:
        urls: Danh sách URLs cần crawl
        config: Dictionary cấu hình từ YAML
    
    Returns:
        List[CrawlResult]: Danh sách kết quả cho tất cả URLs
    """
    parallel_conf = config.get("parallel_config", {})
    max_workers = parallel_conf.get("max_concurrent_workers", 3)
    batch_delay = parallel_conf.get("batch_delay", 2)
    show_progress = parallel_conf.get("show_progress", True)
    
    print(f"🚀 Bắt đầu crawl {len(urls)} URLs với {max_workers} workers song song")
    
    # Tạo semaphore để giới hạn số lượng concurrent workers
    semaphore = asyncio.Semaphore(max_workers)
    
    async def crawl_with_semaphore(url: str, url_index: int) -> CrawlResult:
        """Wrapper để sử dụng semaphore"""
        async with semaphore:
            return await crawl_single_url(url, config, url_index)
    
    # Tạo tasks cho tất cả URLs
    start_time = time.time()
    tasks = [
        crawl_with_semaphore(url, i) 
        for i, url in enumerate(urls)
    ]
    
    # Chạy tất cả tasks song song với progress tracking
    if show_progress:
        print(f"⏳ Đang xử lý {len(tasks)} URLs...")
        
    # Sử dụng asyncio.gather để chạy song song
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Xử lý kết quả và exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ [{i+1}] Exception: {str(result)}")
            final_results.append(CrawlResult(
                url=urls[i],
                success=False,
                method="exception",
                content="",
                html="",
                error=str(result),
                duration=0.0
            ))
        else:
            final_results.append(result)
    
    total_duration = time.time() - start_time
    
    # Thống kê tổng hợp
    successful = sum(1 for r in final_results if r.success)
    failed = len(final_results) - successful
    
    print(f"\n📊 TỔNG KẾT CRAWLING:")
    print(f"⏱️  Tổng thời gian: {total_duration:.1f}s")
    print(f"✅ Thành công: {successful}/{len(urls)} URLs")
    print(f"❌ Thất bại: {failed}/{len(urls)} URLs")
    print(f"⚡ Tốc độ trung bình: {len(urls)/total_duration:.1f} URLs/giây")
    
    return final_results


async def save_parallel_results(results: List[CrawlResult], config: Dict) -> Dict:
    """Lưu kết quả crawl song song"""
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
    create_summary = output_conf.get("create_summary", True)
    
    timestamp_suffix = f"_{timestamp}" if include_timestamp else ""
    
    summary_data = {
        'timestamp': timestamp,
        'total_urls': len(results),
        'successful_urls': sum(1 for r in results if r.success),
        'failed_urls': sum(1 for r in results if not r.success),
        'results': [],
        'files_created': []
    }
    
    # Lưu từng URL result
    for i, result in enumerate(results):
        if not result.success:
            print(f"⚠️  [{i+1}] Bỏ qua URL thất bại: {result.url}")
            summary_data['results'].append({
                'url': result.url,
                'success': False,
                'error': result.error,
                'method': result.method,
                'duration': result.duration
            })
            continue
        
        url_safe_name = f"url_{i+1:02d}"
        
        result_info = {
            'url': result.url,
            'success': True,
            'method': result.method,
            'duration': result.duration,
            'content_length': result.content_length,
            'files': []
        }
        
        # Lưu theo format được chỉ định
        if "html" in formats:
            html_file = output_dir / f"{prefix}_{url_safe_name}_html{timestamp_suffix}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(result.html)
            result_info['files'].append(str(html_file))
            summary_data['files_created'].append(str(html_file))
            print(f"💾 [{i+1}] Đã lưu HTML: {html_file}")
        
        if "markdown" in formats:
            md_file = output_dir / f"{prefix}_{url_safe_name}_md{timestamp_suffix}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(f"# URL: {result.url}\n\n")
                f.write(f"Method: {result.method}\n")
                f.write(f"Duration: {result.duration:.1f}s\n\n")
                f.write("---\n\n")
                f.write(result.content)
            result_info['files'].append(str(md_file))
            summary_data['files_created'].append(str(md_file))
            print(f"💾 [{i+1}] Đã lưu Markdown: {md_file}")
        
        if "json" in formats:
            json_file = output_dir / f"{prefix}_{url_safe_name}_info{timestamp_suffix}.json"
            info = {
                'url': result.url,
                'method': result.method,
                'duration': result.duration,
                'content_length': result.content_length,
                'timestamp': timestamp,
                'success': True
            }
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            result_info['files'].append(str(json_file))
            summary_data['files_created'].append(str(json_file))
            print(f"💾 [{i+1}] Đã lưu JSON: {json_file}")
        
        summary_data['results'].append(result_info)
    
    # Tạo summary file nếu được yêu cầu
    if create_summary:
        summary_file = output_dir / f"{prefix}_summary{timestamp_suffix}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"📋 Đã tạo summary file: {summary_file}")
        summary_data['files_created'].append(str(summary_file))
    
    # Hiển thị preview của results thành công
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"\n📖 Preview một số kết quả thành công:")
        print("-" * 80)
        for i, result in enumerate(successful_results[:3]):  # Chỉ hiển thị 3 đầu tiên
            preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
            print(f"[{i+1}] URL: {result.url}")
            print(f"    Method: {result.method} | Duration: {result.duration:.1f}s")
            print(f"    Content: {preview}")
            print("-" * 80)
    
    return summary_data


def main():
    """Hàm chính"""
    print("🚀 CRAWL4AI - CRAWLING SONG SONG HƯỚNG DẪN GIẢI CHI TIẾT")
    print("=" * 80)
    
    # Đọc cấu hình
    config = load_config()
    if not config:
        print("❌ Không thể đọc được file cấu hình!")
        return
    
    # Lấy danh sách URLs
    urls = config.get("urls", [])
    if not urls:
        print("❌ Không tìm thấy URLs nào trong cấu hình!")
        return
    
    if not isinstance(urls, list):
        print("❌ URLs phải là dạng list trong config YAML!")
        print("💡 Format đúng:")
        print("urls:")
        print("  - 'https://example1.com'")
        print("  - 'https://example2.com'")
        return
    
    # Hiển thị thông tin cấu hình
    parallel_conf = config.get("parallel_config", {})
    output_conf = config.get("output", {})
    
    print(f"📋 Số lượng URLs: {len(urls)}")
    print(f"⚙️  Max concurrent workers: {parallel_conf.get('max_concurrent_workers', 3)}")
    print(f"⏰ Timeout per URL: {parallel_conf.get('per_url_timeout', 45)}s")
    print(f"📁 Output directory: {output_conf.get('directory', 'crawl4ai/output')}")
    print(f"📄 Output formats: {', '.join(output_conf.get('formats', ['markdown']))}")
    print("=" * 80)
    
    # Hiển thị danh sách URLs
    print("🔗 Danh sách URLs sẽ được crawl:")
    for i, url in enumerate(urls, 1):
        print(f"  {i:2d}. {url}")
    
    # Xác nhận trước khi crawl
    try:
        response = input(f"\n❓ Bạn có muốn crawl {len(urls)} URLs này không? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("👋 Thoát chương trình")
            return
    except KeyboardInterrupt:
        print("\n👋 Thoát chương trình")
        return
    
    print("=" * 80)
    
    # Chạy crawler song song
    try:
        results = asyncio.run(crawl_all_urls_parallel(urls, config))
        summary = asyncio.run(save_parallel_results(results, config))
        
        print(f"\n✅ HOÀN THÀNH!")
        print(f"📊 Thống kê cuối cùng:")
        print(f"   🎯 URLs thành công: {summary['successful_urls']}/{summary['total_urls']}")
        print(f"   📄 Files đã tạo: {len(summary['files_created'])}")
        print(f"   📂 Thư mục output: {output_conf.get('directory', 'crawl4ai/output')}")
        
        if summary['successful_urls'] == 0:
            print("\n⚠️  Không có URL nào thành công!")
            print("💡 Hãy kiểm tra:")
            print("   - Kết nối internet")
            print("   - URLs có đúng không")
            print("   - Cập nhật selectors trong config YAML")
    
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng crawling theo yêu cầu")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {str(e)}")


if __name__ == "__main__":
    main() 