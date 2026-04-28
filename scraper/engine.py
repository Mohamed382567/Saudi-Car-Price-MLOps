import asyncio
import random
import requests
import time
import os
from urllib.parse import urlparse # Added to extract the domain automatically
from playwright.async_api import async_playwright
from database.db_manager import insert_stage1_data, update_car_details, get_links_to_scrape
from scraper.parser import parse_car_details, extract_price_from_text # Importing our logic

async def run_phase1(pages=4):
    """ Phase 1: Focuses only on navigating and fetching links/text. """
    print(f"🚀 Scraping search pages...")
    
    # Retrieve the base URL from environment variables (GitHub Secrets)
    base_url = os.getenv("SCRAPE_URL")
    
    # Automatically extract the domain from the secret URL (e.g., https://syarah.com)
    parsed_uri = urlparse(base_url)
    domain = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
    
    all_data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for p_num in range(1, pages + 1):
    
            # Construct the paginated URL
            url = f"{base_url}&page={p_num}"
            await page.goto(url, wait_until="domcontentloaded")
            
            # Scroll to trigger lazy loading of cards
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            await asyncio.sleep(2)
            
            # Locate all car post cards
            cards = await page.query_selector_all('a[id^="posts-card-body-"]')
            for card in cards:
                text = await card.inner_text()
                link = await card.get_attribute("href")
                
                # Use the parser function to extract the price from raw text
                price = extract_price_from_text(text)
                all_data.append({
                    "Full_Text": text.replace('\n', ' '),
                    # Dynamically prepend the domain to relative links
                    "Link": f"{domain}{link}" if link.startswith('/') else link,
                    "Price": price
                })
        await browser.close()
    
    # Save the gathered links and basic info to the database
    insert_stage1_data(all_data)

def run_phase2():
    """ Phase 2: Focuses only on the loop and HTTP requests for deep details. """
    links = get_links_to_scrape()
    print(f"🔍 Processing {len(links)} links...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for url in links:
        try:
            # Fetch the individual car page
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                # Call the parser to handle the HTML/JSON logic for car specifications
                details = parse_car_details(response.text)
                if details:
                    update_car_details(url, details)
            
            # Random sleep to avoid being blocked by the server
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            print(f"⚠️ Network error on {url}: {e}")
