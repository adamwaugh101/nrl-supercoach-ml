# %%
import sys
import time
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

# %%
BASE_URL = "https://www.nrlsupercoachstats.com/stats.php"
START_YEAR = 2015
END_YEAR = 2025
OUTPUT_DIR = Path("data/bronze")
SCRAPE_DELAY = 6

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
def get_column_names(page) -> list[str]:
    """Extract column names from the jqGrid th id attributes, stripping the 'list1_' prefix."""
    try:
        names = page.eval_on_selector_all(
            "#gbox_list1 thead th[id]",
            "elements => elements.map(el => el.id.replace('list1_', ''))"
        )
        logger.info(f"Extracted {len(names)} column names")
        return names
    except Exception as e:
        logger.warning(f"Could not extract column names: {e}")
        return []


# %%
def set_page_size(page):
    """Set the jqGrid page size to 200 to avoid too many paginations."""
    try:
        page.eval_on_selector(
            "select.ui-pg-selbox",
            """select => {
                select.value = '200';
                select.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        time.sleep(1)
        logger.info("Page size set to 200")
    except Exception as e:
        logger.warning(f"Could not set page size: {e}")


# %%
def get_total_pages(page) -> int:
    """Get the total number of pages from the jqGrid pagination info."""
    try:
        total = page.eval_on_selector(
            "#sp_1_list1_pager",
            "el => parseInt(el.textContent.trim())"
        )
        return total
    except Exception as e:
        logger.warning(f"Could not get total pages: {e}")
        return 1


# %%
def scrape_current_page(page) -> list[list[str]]:
    """Scrape all rows from the currently visible jqGrid page."""
    rows = []
    try:
        page.wait_for_selector("#gbox_list1 tbody tr", timeout=8000)
        table_rows = page.query_selector_all("#gbox_list1 tbody tr")

        for row in table_rows:
            cells = row.query_selector_all("td")
            if not cells:
                continue
            cell_values = [c.inner_text().strip() for c in cells]
            if not any(cell_values):
                continue
            rows.append(cell_values)

    except Exception as e:
        logger.warning(f"Failed to scrape page: {e}")

    return rows


# %%
def click_next_page(page) -> bool:
    """Click the next page button. Returns False if the button is disabled."""
    try:
        next_btn = page.query_selector("#next_list1_pager")
        if not next_btn:
            return False

        # jqGrid marks disabled buttons with ui-state-disabled
        is_disabled = next_btn.get_attribute("class") or ""
        if "ui-state-disabled" in is_disabled:
            return False

        next_btn.click()
        time.sleep(SCRAPE_DELAY)
        return True

    except Exception as e:
        logger.warning(f"Could not click next page: {e}")
        return False


# %%
def scrape_year(year: int) -> pd.DataFrame:
    """Scrape all pages for a given year and return as a DataFrame."""
    url = f"{BASE_URL}?year={year}"
    all_rows = []
    col_names = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        logger.info(f"Loading {url}")
        page.goto(url, wait_until="networkidle")

        set_page_size(page)
        col_names = get_column_names(page)
        total_pages = get_total_pages(page)
        logger.info(f"Total pages: {total_pages}")

        current_page = 1
        while True:
            logger.info(f"[{year}] Scraping page {current_page}/{total_pages}")
            rows = scrape_current_page(page)
            all_rows.extend(rows)

            if current_page >= total_pages:
                break

            has_next = click_next_page(page)
            if not has_next:
                logger.info(f"[{year}] No more pages at page {current_page}")
                break

            current_page += 1

        browser.close()

    if not all_rows:
        logger.warning(f"No data scraped for year {year}")
        return pd.DataFrame()

    cell_df = pd.DataFrame(all_rows)

    if col_names and len(col_names) == len(cell_df.columns):
        cell_df.columns = col_names
    else:
        logger.warning(f"Column mismatch ({len(col_names)} names vs {len(cell_df.columns)} cols) — using numbered columns")
        cell_df.columns = [f"col_{i}" for i in range(len(cell_df.columns))]

    cell_df.insert(0, "year", year)

    return cell_df


# %%
def run_scrape(start_year: int = START_YEAR, end_year: int = END_YEAR):
    """Run the full historical scrape across all years."""
    for year in range(start_year, end_year + 1):
        output_path = OUTPUT_DIR / f"stats_{year}.parquet"

        if output_path.exists():
            logger.info(f"Skipping {year} — already scraped at {output_path}")
            continue

        logger.info(f"Starting scrape for {year}")
        df = scrape_year(year)

        if not df.empty:
            df.to_parquet(output_path, index=False)
            logger.success(f"Saved {len(df)} rows to {output_path}")
        else:
            logger.warning(f"No data for {year}, skipping save")


# %%
run_scrape(start_year=2026, end_year=2026)