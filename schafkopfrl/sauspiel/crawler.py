from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from sqlitedict import SqliteDict

# setup
from schafkopfrl.sauspiel.game_transcript import GameTranscript

def crawl():
  options = webdriver.ChromeOptions()
  prefs = {"profile.managed_default_content_settings.images": 2}
  options.add_experimental_option("prefs", prefs)
  #options.add_argument("--start-maximized")
  #options.add_experimental_option("detach", True)
  driver = webdriver.Chrome("../../chromedriver.exe", options=options)
  driver.get("http://www.sauspiel.de")

  #login
  elem = driver.find_element_by_id("ontop_login")
  elem.clear()
  elem.send_keys("xxx")
  elem = driver.find_element_by_id("login_inline_password")
  elem.clear()
  elem.send_keys("xxx")
  elem.send_keys(Keys.RETURN)

  #setup database
  games = SqliteDict('games.sqlite', autocommit=True)

  normal_games = 0

  for i in range (330000000, 1000000000):
    print(i)
    #view game
    driver.get('https://www.sauspiel.de/spiele/'+str(i))

    gt = GameTranscript()
    try:
      gt.fast_parse(driver)
    except:
      print("could not parse game")
      continue

    if len(gt.sonderregeln) == 0:
      normal_games += 1
      print("found "+str(normal_games) + " normal games")

    games[i] = gt
    #games[i] = driver.page_source

  #for key, value in games.iteritems():
  #  print (key, value)


  #print(gt)

  #print(gt.toJSON())



if __name__ == '__main__':
  #import cProfile

  #pr = cProfile.Profile()
  #pr.enable()
  crawl()
  #pr.disable()
  # after your program ends
  #pr.print_stats(sort="tottime")

