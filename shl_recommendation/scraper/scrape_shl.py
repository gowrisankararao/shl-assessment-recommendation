from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os
import re

os.makedirs("shl_recommendation/data", exist_ok=True)


options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0")

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/?start={}"

all_links = set()
start = 0

print("Collecting product links...\n")

while True:
    url = CATALOG_URL.format(start)
    print("Opening:", url)

    driver.get(url)
    time.sleep(3)

    links = driver.find_elements(By.XPATH, "//a[contains(@href,'product-catalog/view')]")

    previous_count = len(all_links)

    for link in links:
        href = link.get_attribute("href")

        if href and "product-catalog/view" in href:
            if "job-solution" not in href.lower():
                all_links.add(href)

    print("Collected so far:", len(all_links))

    if len(all_links) == previous_count:
        print("Pagination finished.")
        break

    start += 12

print("\nTotal links collected:", len(all_links))

data = []
print("\nScraping detail pages...\n")

for idx, link in enumerate(all_links):
    print(f"{idx+1}/{len(all_links)}")

    try:
        driver.get(link)

        wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        time.sleep(2)

        name = driver.find_element(By.TAG_NAME, "h1").text.strip()

     
        description = ""
        try:
            meta_desc = driver.find_element(By.XPATH, "//meta[@name='description']")
            description = meta_desc.get_attribute("content").strip()
        except:
            description = ""

        page_text = driver.page_source

       
        duration = 0

       
        match = re.search(r"max\s*(\d+)", page_text, re.IGNORECASE)

        if not match:
            match = re.search(r"(\d+)\s*minutes", page_text, re.IGNORECASE)

       
        if not match:
            match = re.search(r"Completion Time.*?(\d+)", page_text, re.IGNORECASE)

        if match:
            duration = int(match.group(1))
        else:
            duration = 0

    
      
        adaptive_support = "Yes" if "adaptive" in page_text.lower() else "No"

        
        remote_support = "Yes" if "remote" in page_text.lower() else "No"

       
        test_type = []

        try:
            badges = driver.find_elements(By.XPATH, "//span[contains(@class,'test-type')]")
            for badge in badges:
                text = badge.text.strip()
                if text:
                    test_type.append(text)
        except:
            pass

        # Fallback category detection
        categories = [
            "Knowledge & Skills",
            "Personality & Behaviour",
            "Ability & Aptitude",
            "Competencies"
        ]

        for category in categories:
            if category.lower() in page_text.lower():
                if category not in test_type:
                    test_type.append(category)

        if not test_type:
            test_type = ["Not specified"]

        data.append({
            "name": name,
            "url": link,
            "description": description,
            "duration": duration,
            "adaptive_support": adaptive_support,
            "remote_support": remote_support,
            "test_type": ", ".join(test_type),
            "combined_text": f"{name}. {description}"
        })

        time.sleep(1.5)

    except Exception as e:
        print("Error scraping:", link)
        continue

driver.quit()

df = pd.DataFrame(data)
df.drop_duplicates(subset=["url"], inplace=True)

print("\nFinal unique assessments:", len(df))

df.to_csv("shl_recommendation/data/shl_catalog.csv", index=False)

print("\nScraping complete. CSV saved successfully!")
