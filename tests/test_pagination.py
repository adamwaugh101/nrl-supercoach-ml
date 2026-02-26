# %%
import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://www.nrlsupercoachstats.com/stats.php?year=2025', wait_until='networkidle')
    
    page.eval_on_selector(
        'select.ui-pg-selbox',
        """select => {
            select.value = '200';
            select.dispatchEvent(new Event('change', { bubbles: true }));
        }"""
    )
    time.sleep(1)
    
    total = page.inner_text('#sp_1_list1')
    print(f'Total pages element: "{total}"')
    
    browser.close()