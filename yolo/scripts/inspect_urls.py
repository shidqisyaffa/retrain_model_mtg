import urllib.request
import re
import json

urls = [
    'https://universe.roboflow.com/keerthana-lxkqi/knife-altum',
    'https://universe.roboflow.com/nust-5tnkk/knife-detection-v5urd',
    'https://universe.roboflow.com/school-pwipo/knife-detection-lpiw1',
    'https://universe.roboflow.com/sailbot-5c9lp/face-detection-4dzpn',
    'https://universe.roboflow.com/objectdetection-dxeuw/face-detection-z1x9s'
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

for url in urls:
    print(f"\n{'='*70}")
    print(f"URL: {url}")
    print(f"{'='*70}")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as response:
            html = response.read().decode('utf-8', errors='ignore')
            title = re.search(r'<title>(.*?)</title>', html)
            title_text = title.group(1).strip() if title else 'No Title'
            print(f"Title: {title_text}")

            # Look for JSON-LD or Next.js page data
            next_data = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
            if next_data:
                try:
                    data = json.loads(next_data.group(1))
                    page_props = data.get("props", {}).get("pageProps", {})
                    project = page_props.get("project", {}) or page_props.get("dataset", {}) or {}
                    
                    name = project.get("name") or project.get("id")
                    classes = project.get("classes", []) or project.get("annotationClasses", [])
                    images_count = project.get("images") or project.get("totalImages")
                    splits = project.get("splits", {})
                    versions = project.get("versions", [])
                    
                    print(f"Project Name   : {name}")
                    print(f"Classes        : {classes}")
                    print(f"Total Images   : {images_count}")
                    if versions:
                        latest = versions[0]
                        v_classes = latest.get("classes", [])
                        v_images = latest.get("images")
                        print(f"Latest Version : v{latest.get('id')} ({v_images} images, classes: {v_classes})")
                except Exception as je:
                    print(f"Error parsing NEXT_DATA: {je}")
            else:
                # Fallback regex
                classes = re.findall(r'"class(?:es)?":\s*\[([^\]]+)\]', html)
                print(f"Fallback classes match: {classes[:2]}")
    except urllib.error.HTTPError as he:
        print(f"HTTP Error {he.code}: {he.reason}")
    except Exception as e:
        print(f"Error: {e}")
