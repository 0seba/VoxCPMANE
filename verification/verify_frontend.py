import os
from playwright.sync_api import sync_playwright

def verify_frontend():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Absolute path to index.html
        cwd = os.getcwd()
        html_path = os.path.join(cwd, "src/voxcpmane/frontend/index.html")
        page.goto(f"file://{html_path}")

        # Verify Generate tab is active by default
        # Using specific selectors to ensure visibility
        page.wait_for_selector("#tab-content-generate")
        # Check that it's NOT hidden (Tailwind 'hidden' class)
        # We can check visibility using Playwright's is_visible()
        if not page.is_visible("#tab-content-generate"):
             print("Error: Generate tab content should be visible")

        if page.is_visible("#tab-content-create"):
             print("Error: Create tab content should be hidden")

        # Screenshot Generate Tab
        page.screenshot(path="verification/tab_generate.png")
        print("Captured Generate Tab")

        # Click Create Voice Tab
        page.click("#tab-btn-create")

        # Verify Create tab is active
        page.wait_for_selector("#tab-content-create")
        if not page.is_visible("#tab-content-create"):
             print("Error: Create tab content should be visible after click")

        if page.is_visible("#tab-content-generate"):
             print("Error: Generate tab content should be hidden after switch")

        # Screenshot Create Tab
        page.screenshot(path="verification/tab_create.png")
        print("Captured Create Tab")

        browser.close()

if __name__ == "__main__":
    verify_frontend()
