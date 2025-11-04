#!/usr/bin/env python3
"""
Quick script to download and prepare training data.
"""

import urllib.request
import os

def download_sample_data():
    """Download Alice in Wonderland as sample data."""

    # Create data directory
    os.makedirs("data", exist_ok=True)

    print("Downloading Alice in Wonderland from Project Gutenberg...")
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    output_path = "data/training_text.txt"

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"[OK] Downloaded to {output_path}")

        # Clean the file
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Remove Project Gutenberg header/footer
        if "*** START OF" in text:
            text = text.split("*** START OF")[1]
        if "*** END OF" in text:
            text = text.split("*** END OF")[0]

        # Save cleaned version
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())

        # Show stats
        print(f"[OK] Cleaned text")
        print(f"  - Characters: {len(text):,}")
        print(f"  - Words: {len(text.split()):,}")
        print(f"  - Size: {len(text)/1024:.2f} KB")

        return True

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    # Change to backend directory if running from project root
    if os.path.basename(os.getcwd()) != "backend":
        if os.path.exists("backend"):
            os.chdir("backend")
    
    download_sample_data()
