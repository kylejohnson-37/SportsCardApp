import csv, time, random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# ---------- CONFIG ----------
START_URL = "https://www.tcdb.com/ViewCard.cfm/sid/482758/cid/27980431?PageIndex=1"
CARD_SET = "Topps 2025"
CSV_OUT = "cards.csv"
MAX_PAGES = 5
MIN_DELAY = 2.0
MAX_DELAY = 4.0
# ----------------------------

records = []

def parse_card_html(html):
    soup = BeautifulSoup(html, "html.parser")

    h4_site = soup.find("h4", class_="site")
    card_no = player = team = None
    if h4_site:
        txt_parts = h4_site.get_text(strip=True).split("-")
        if txt_parts:
            card_no = txt_parts[0].replace("#", "").strip()
        a_tags = h4_site.find_all("a")
        if len(a_tags) >= 2:
            player = a_tags[0].text.strip()
            team = a_tags[1].text.strip()

    front_img_div = soup.find("div", class_="easyzoom")
    front_img_url = None
    if front_img_div:
        a_tag = front_img_div.find("a")
        if a_tag and "href" in a_tag.attrs:
            front_img_url = "https://www.tcdb.com" + a_tag["href"]

    return {
        "player_name": player,
        "card_set": CARD_SET,
        "card_number": card_no,
        "team": team,
        "front_image_url": front_img_url,
    }

# ---------- Main Loop: one browser per card ----------
next_url = START_URL

for i in range(MAX_PAGES):
    print(f"\n🔁 Launching browser for card {i + 1}")
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless=new")  # remove if you want to see browser
    options.add_argument("start-maximized")

    driver = uc.Chrome(options=options, use_subprocess=True)
    driver.get(next_url)
    time.sleep(2.5)

    html = driver.page_source
    card_data = parse_card_html(html)
    records.append(card_data)
    print(f"✅ Scraped: {card_data['player_name']} (#{card_data['card_number']})")

    # Get the next card's href
    try:
        next_a = driver.find_element(By.XPATH, "//a[.//button[contains(text(),'Next')]]")
        next_href = next_a.get_attribute("href")
        if not next_href:
            print("⛔ No 'Next' link found — end of set?")
            break
        next_url = next_href
        print(f"➡️  Next URL: {next_url}")
    except Exception as e:
        print(f"⚠️ Failed to find next URL: {e}")
        break

    driver.quit()
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

# ---------- Save CSV ----------
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "player_name", "card_set", "card_number", "team", "front_image_url"
    ])
    writer.writeheader()
    writer.writerows(records)

print(f"\n✅ Done! {len(records)} cards saved to {CSV_OUT}")
