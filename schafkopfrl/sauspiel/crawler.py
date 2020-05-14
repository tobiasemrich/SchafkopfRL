from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# setup
from schafkopfrl.sauspiel.game_transcript import GameTranscript

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_experimental_option("detach", True)
driver = webdriver.Chrome("../../chromedriver.exe", options=options)
driver.get("http://www.sauspiel.de")

#login
elem = driver.find_element_by_id("ontop_login")
elem.clear()
elem.send_keys("Tobiaz")
elem = driver.find_element_by_id("login_inline_password")
elem.clear()
elem.send_keys("-Template123")
elem.send_keys(Keys.RETURN)

#view game
driver.get('https://www.sauspiel.de/spiele/1002670697-wenz-ohne-2-von-delpiero1109')

gt = GameTranscript()
gt.parse(driver)
print(gt)

print(gt.toJSON())





