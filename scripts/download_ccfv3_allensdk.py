"""Backward-compatible wrapper for the old downloader name.

AllenSDK is no longer required.  This wrapper delegates to the direct Allen
Institute downloader so existing commands keep working.
"""
from download_ccfv3 import main

if __name__ == "__main__":
    main()
