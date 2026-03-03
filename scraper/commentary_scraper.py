# %%
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright

# %%
URLS = [
    "https://scplaybook.com.au/blog/2026/02/24/nrl-round-1-team-lists-vegas-squads-nrl-supercoach-analysis",
    "https://scplaybook.com.au/blog/2026/02/22/nrl-supercoach-cheapie-analysis-forward-bargain-buys",
    "https://scplaybook.com.au/blog/2026/02/20/nrl-supercoach-cheapie-analysis-backline-bargain-buys",
]

OUTPUT_DIR = Path("data/raw/commentary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
def slugify(url: str) -> str:
    """Turn a URL into a safe filename."""
    slug = re.sub(r"https?://[^/]+/", "", url)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-")
    return slug[:80]

# %%
async def scrape_article(page, url: str) -> dict:
    """Navigate to URL and extract the article body text."""
    print(f"Scraping: {url}")
    await page.goto(url, wait_until="networkidle", timeout=30000)

    # Wait for article content to render
    await page.wait_for_selector("article, main, .post-content, h1", timeout=10000)

    # Extract title
    title = await page.title()

    # Pull all paragraph text from the article body
    # SC Playbook renders content in <p> tags inside the main article area
    paragraphs = await page.eval_on_selector_all(
        "article p, main p, .entry-content p, .post-content p",
        "els => els.map(el => el.innerText.trim()).filter(t => t.length > 0)"
    )

    # Fallback: grab all visible <p> tags if selectors above miss
    if not paragraphs:
        paragraphs = await page.eval_on_selector_all(
            "p",
            "els => els.map(el => el.innerText.trim()).filter(t => t.length > 30)"
        )

    body_text = "\n\n".join(paragraphs)

    return {
        "url": url,
        "title": title,
        "scraped_at": datetime.utcnow().isoformat(),
        "body": body_text,
    }

# %%
async def scrape_all(urls: list[str]) -> list[dict]:
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Set a realistic user agent to avoid bot detection
        await page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

        for url in urls:
            try:
                result = await scrape_article(page, url)
                results.append(result)

                # Save individual file per article
                slug = slugify(url)
                out_path = OUTPUT_DIR / f"{slug}.json"
                out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                print(f"  Saved: {out_path} ({len(result['body'])} chars)")

            except Exception as e:
                print(f"  ERROR scraping {url}: {e}")
                results.append({"url": url, "error": str(e)})

        await browser.close()

    return results

# %%
results = asyncio.run(scrape_all(URLS))

print(f"\nDone. {len([r for r in results if 'body' in r])} / {len(URLS)} articles scraped.")