import asyncio
import random
import requests
import time
from playwright.async_api import async_playwright
from database.db_manager import insert_stage1_data, update_car_details, get_links_to_scrape
from scraper.parser import parse_car_details, extract_price_from_text # Importing our logic

async def run_phase1(pages=4):
    """ Phase 1: Focuses only on navigating and fetching links/text. """
    print(f"🚀 Scraping search pages...")
    all_data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for p_num in range(1, pages + 1):
            url = f"https://syarah.com/en/filters?make_id=4%2C60%2C38%2C58%2C69%2C37%2C51%2C67%2C78%2C35%2C33%2C55%2C5%2C48%2C15%2C53%2C74%2C20%2C19&condition_id=0%2C1&page={p_num}"
            await page.goto(url, wait_until="domcontentloaded")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
            await asyncio.sleep(2)
            
            cards = await page.query_selector_all('a[id^="posts-card-body-"]')
            for card in cards:
                text = await card.inner_text()
                link = await card.get_attribute("href")
                # Use the parser function for the price
                price = extract_price_from_text(text)
                all_data.append({
                    "Full_Text": text.replace('\n', ' '),
                    "Link": f"https://syarah.com{link}" if link.startswith('/') else link,
                    "Price": price
                })
        await browser.close()
    insert_stage1_data(all_data)

def run_phase2():
    """ Phase 2: Focuses only on the loop and HTTP requests. """
    links = get_links_to_scrape()
    print(f"🔍 Processing {len(links)} links...")
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for url in links:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                # Call the parser to handle the HTML/JSON logic
                details = parse_car_details(response.text)
                if details:
                    update_car_details(url, details)
            
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            print(f"⚠️ Network error on {url}: {e}")