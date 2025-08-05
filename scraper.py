import csv, random, time, os, traceback
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

START_URL  = "https://www.tcdb.com/ViewCard.cfm/sid/482758/cid/27980633?PageIndex=1"
CSV_OUT    = "cards.csv"
MAX_PAGES  = 200

# ---------- setup Chrome options ----------
opts = uc.ChromeOptions()
opts.add_argument("--disable-blink-features=AutomationControlled")

# ---------- initialize driver ----------
driver = None
records = []
page = 0

try:
    driver = uc.Chrome(
        version_main=137,
        use_subprocess=True,
        options=opts,
        browser_executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    )
    driver.get(START_URL)
    print("Loaded:", driver.title)

    while page < MAX_PAGES:
        try:
            WebDriverWait(driver, 12).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.easyzoom"))
            )
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            h4 = soup.find("h4", class_="site")
            if not h4:
                print("⚠️ Card header not found")
                break

            text_parts = h4.get_text().split(" - ")
            if len(text_parts) < 3:
                print("⚠️ Card data format unexpected:", h4.get_text())
                break

            num    = text_parts[0].replace("#", "").strip()
            player = text_parts[1].strip()
            team   = text_parts[2].strip()

            img_div = soup.find("div", class_="easyzoom")
            img_url = "https://www.tcdb.com" + img_div.find("a")["href"] if img_div else "N/A"

            records.append({
                "player_name": player,
                "card_set": "Topps 2025",
                "card_number": num,
                "team": team,
                "front_image_url": img_url
            })
            print(f"✅ Scraped {player} – card #{num}")
            page += 1

            try:
                next_btn = driver.find_element(
                    By.XPATH, "//a[.//button[contains(@class,'btn-primary')] and contains(.,'Next')]"
                )
                next_btn.click()
                time.sleep(random.uniform(2, 4))
            except Exception:
                print("⏹️ No Next button; stopping.")
                break

        except Exception as e:
            print("❌ Error during scraping page:", e)
            traceback.print_exc()
            break

except Exception as e:
    print("❌ Failed to start browser or load initial page:", e)
    traceback.print_exc()

finally:
    if driver:
        driver.quit()

    if records:
        base_name = "cards"
        ext = ".csv"
        counter = 1
        while True:
            file_name = f"{base_name}_{counter}{ext}"
            if not os.path.exists(file_name):
                break
            counter += 1

        try:
            with open(file_name, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["player_name", "card_set", "card_number", "team", "front_image_url"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            print(f"\n✅ Saved: {file_name}")
        except Exception as e:
            print("❌ Failed to write CSV:", e)
    else:
        print("\n⚠️ No records to save.")
