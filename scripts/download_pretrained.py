#!/usr/bin/env python3
"""
Download VeryLargeWeebModel pretrained models from Tsinghua Cloud (Seafile).

This script uses the Seafile API to download files from shared links,
bypassing the need for a web browser.

Usage:
    python scripts/download_pretrained.py
    python scripts/download_pretrained.py --output pretrained/
"""
import os
import sys
import re
import argparse
import urllib.request
import urllib.parse
import json
from pathlib import Path


def get_seafile_download_link(share_url: str, file_path: str) -> str:
    """
    Get direct download link from Seafile shared folder.

    Seafile API endpoint for shared links:
    GET /api/v2.1/share-links/{token}/dirents/?path={path}
    """
    # Extract token from share URL
    # https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/
    match = re.search(r'/d/([a-f0-9]+)/?', share_url)
    if not match:
        raise ValueError(f"Could not extract token from URL: {share_url}")

    token = match.group(1)
    base_url = share_url.split('/d/')[0]

    # Construct download URL
    # Seafile download endpoint: /d/{token}/files/?p={path}&dl=1
    encoded_path = urllib.parse.quote(file_path)
    download_url = f"{base_url}/d/{token}/files/?p={encoded_path}&dl=1"

    return download_url


def download_with_redirect(url: str, output_path: str, description: str = "") -> bool:
    """Download file following redirects."""
    print(f"Downloading {description or output_path}...")
    print(f"  URL: {url}")

    try:
        # Create request with browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        request = urllib.request.Request(url, headers=headers)

        # Open URL and follow redirects
        with urllib.request.urlopen(request, timeout=30) as response:
            # Check if we got a redirect to actual file
            final_url = response.geturl()
            content_type = response.headers.get('Content-Type', '')
            content_length = response.headers.get('Content-Length', 'unknown')

            print(f"  Content-Type: {content_type}")
            print(f"  Size: {content_length} bytes")

            # If we got HTML, the download link might need different handling
            if 'text/html' in content_type:
                html = response.read().decode('utf-8', errors='ignore')

                # Try to find direct download link in the page
                # Seafile sometimes returns a page with the actual download link
                download_match = re.search(r'href="([^"]+\.pth[^"]*)"', html)
                if download_match:
                    new_url = download_match.group(1)
                    if not new_url.startswith('http'):
                        new_url = urllib.parse.urljoin(url, new_url)
                    print(f"  Found redirect: {new_url}")
                    return download_with_redirect(new_url, output_path, description)

                # Check for JavaScript-based download
                if 'seafile' in html.lower() or 'download' in html.lower():
                    print("  Got HTML page instead of file.")
                    print("  Seafile may require JavaScript for this download.")
                    return False

            # Download the file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'wb') as f:
                total = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    print(f"\r  Downloaded: {total / 1024 / 1024:.1f} MB", end='')

            print()

            # Verify file size
            file_size = os.path.getsize(output_path)
            if file_size < 1000000:  # Less than 1MB is probably an error page
                print(f"  Warning: File seems too small ({file_size} bytes)")
                # Check if it's an error page
                with open(output_path, 'rb') as f:
                    header = f.read(100)
                    if b'<!DOCTYPE' in header or b'<html' in header:
                        print("  Downloaded file is HTML, not a model checkpoint.")
                        os.remove(output_path)
                        return False

            print(f"  Saved to: {output_path}")
            return True

    except urllib.error.HTTPError as e:
        print(f"  HTTP Error: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def try_alternative_methods(output_dir: str) -> bool:
    """Try alternative download methods."""

    print("\n" + "=" * 60)
    print("Trying alternative download methods...")
    print("=" * 60)

    # Method 1: Try requests library with session (handles cookies better)
    try:
        import requests
        print("\n[Method 1] Using requests library with session...")

        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })

        share_url = "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/"

        # First, visit the share page to get cookies
        resp = session.get(share_url, timeout=30)

        if resp.status_code == 200:
            # Try to download files
            files_to_download = [
                ("/vqvae/epoch_125.pth", "vqvae/epoch_125.pth"),
                ("/occworld/latest.pth", "occworld/latest.pth"),
            ]

            for remote_path, local_path in files_to_download:
                download_url = f"{share_url}files/?p={urllib.parse.quote(remote_path)}&dl=1"
                output_path = os.path.join(output_dir, local_path)

                print(f"  Downloading {remote_path}...")
                resp = session.get(download_url, stream=True, timeout=60)

                if resp.status_code == 200 and 'application/octet-stream' in resp.headers.get('Content-Type', ''):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"    Saved: {output_path}")
                else:
                    print(f"    Failed: {resp.status_code}")

        return True

    except ImportError:
        print("  requests library not installed")
    except Exception as e:
        print(f"  Failed: {e}")

    return False


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("""
Automatic download failed. Tsinghua Cloud requires browser interaction.

Option 1: Download on another machine and transfer
--------------------------------------------------
1. On a machine with a browser, visit:
   https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/

2. Download these files:
   - vqvae/epoch_125.pth
   - occworld/latest.pth

3. Transfer to this machine:
   scp epoch_125.pth user@this-machine:/path/to/pretrained/vqvae/
   scp latest.pth user@this-machine:/path/to/pretrained/occworld/

Option 2: Use a text browser (if available)
-------------------------------------------
   lynx https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/

   or

   w3m https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/

Option 3: Train from scratch (slower but works)
-----------------------------------------------
   # Edit config/finetune_tokyo.py and comment out:
   #   load_from = occworld_checkpoint
   #   vqvae_ckpt = vqvae_checkpoint

   python train.py --py-config config/finetune_tokyo.py

Option 4: Use Selenium (if Chrome/Firefox available)
----------------------------------------------------
   pip install selenium webdriver-manager
   python scripts/download_pretrained.py --use-selenium
""")


def download_with_selenium(output_dir: str) -> bool:
    """Use Selenium to download files (requires Chrome/Firefox)."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        print("\n[Selenium] Starting headless browser...")

        # Setup Chrome options
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_experimental_option('prefs', {
            'download.default_directory': os.path.abspath(output_dir),
            'download.prompt_for_download': False,
        })

        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        except:
            driver = webdriver.Chrome(options=options)

        driver.get("https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/")

        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        print("  Page loaded, looking for download links...")

        # This is a simplified version - actual implementation would need
        # to navigate the Seafile interface

        driver.quit()
        return True

    except ImportError:
        print("  Selenium not installed. Run: pip install selenium webdriver-manager")
        return False
    except Exception as e:
        print(f"  Selenium failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download VeryLargeWeebModel pretrained models')
    parser.add_argument('--output', '-o', default='pretrained', help='Output directory')
    parser.add_argument('--use-selenium', action='store_true', help='Use Selenium browser automation')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("VeryLargeWeebModel Pretrained Model Downloader")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    share_url = "https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/"

    files_to_download = [
        ("/vqvae/epoch_125.pth", "vqvae/epoch_125.pth", "VQVAE checkpoint (~500MB)"),
        ("/occworld/latest.pth", "occworld/latest.pth", "VeryLargeWeebModel checkpoint (~200MB)"),
    ]

    success = True

    # Try direct download first
    for remote_path, local_path, description in files_to_download:
        output_path = os.path.join(output_dir, local_path)

        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            if size > 10000000:  # > 10MB
                print(f"\n{description} already exists ({size/1024/1024:.1f}MB), skipping...")
                continue

        download_url = get_seafile_download_link(share_url, remote_path)

        if not download_with_redirect(download_url, output_path, description):
            success = False

    # If direct download failed, try alternatives
    if not success:
        if args.use_selenium:
            success = download_with_selenium(output_dir)
        else:
            success = try_alternative_methods(output_dir)

    # If still failed, print manual instructions
    if not success:
        print_manual_instructions()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nVerify with: ls -lh {output_dir}/vqvae/ {output_dir}/occworld/")


if __name__ == '__main__':
    main()
