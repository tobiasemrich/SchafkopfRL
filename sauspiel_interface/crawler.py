from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from game_transcript import GameTranscript

import credentials
import json

'''
To run the crawler you will need 
- the chromedriver installed
- become Vereinsmitglied at Sauspiel to get access to all games
'''

def format_json(obj, indent=0):
    indent_str = ' ' * indent
    if isinstance(obj, dict):
        if not obj:
            return '{}'
        lines = ['{']
        for i, (k, v) in enumerate(obj.items()):
            comma = ',' if i < len(obj) - 1 else ''
            lines.append(f'{indent_str}  "{k}": {format_json(v, indent + 2)}{comma}')
        lines.append(f'{indent_str}}}')
        return '\n'.join(lines)
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        items = [format_json(item, indent) for item in obj]
        return '[' + ','.join(items) + ']'
    else:
        return json.dumps(obj)

def crawl():
  options = webdriver.ChromeOptions()
  prefs = {"profile.managed_default_content_settings.images": 2}
  options.add_experimental_option("prefs", prefs)
  options.add_argument("--headless=new")
  options.add_argument("--no-sandbox")
  options.add_argument("--disable-dev-shm-usage")
  options.add_argument("--window-size=1920,1080")
  #options.add_argument("--start-maximized")
  #options.add_experimental_option("detach", True)
  service = Service(executable_path="/usr/bin/chromedriver")
  driver = webdriver.Chrome(service=service, options=options)
  driver.get("http://www.sauspiel.de")

  wait = WebDriverWait(driver, 5)

  #login
  login_input = wait.until(EC.element_to_be_clickable((By.ID, "ontop_login")))
  login_input.clear()
  login_input.send_keys(credentials.username)

  password_input = wait.until(EC.element_to_be_clickable((By.ID, "login_inline_password")))
  password_input.clear()
  password_input.send_keys(credentials.password)
  password_input.send_keys(Keys.RETURN)

  #setup database
  games = []

  normal_games = 0

  for i in range (1000000000, 1000001000):
    print(i)
    #get game
    driver.get('https://www.sauspiel.de/spiele/'+str(i))

    gt = GameTranscript()
    try:
      gt.fast_parse(driver)
    except Exception as e:
      print("could not parse game")
      # print(e)
      

    if len(gt.sonderregeln) == 0:
      games.append(gt.toJSON())
      normal_games += 1
  

  # Save games to JSON file
  if games:
    with open('data/crawled_games.json', 'w') as f:
      f.write(format_json(games))
  print("found "+str(normal_games) + " normal games")




if __name__ == '__main__':
  crawl()