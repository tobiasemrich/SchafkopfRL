from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from schafkopfrl.gamestate import GameState
from schafkopfrl.models.actor_critic_lstm import ActorCriticNetworkLSTM
import torch

#create player
from schafkopfrl.players.rl_player import RlPlayer

policy = ActorCriticNetworkLSTM()
policy.load_state_dict(torch.load("../policies/10850000.pt"))
policy.to(device='cuda')
rl_player = RlPlayer(0, policy)

# setup
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

# click spielen
driver.get('https://www.sauspiel.de/schafkopf/spielen?r=1')

#open table
myElem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'table-list-item__btn-icon')))
driver.find_element_by_class_name("table-list-item__btn-icon").click()

import time

print("you have now 10 seconds to join a table")
time.sleep(10)

total_reward = 0
total_games = 0
#game started
while True:
  # get own hand
  hand_element = WebDriverWait(driver, 500).until(EC.presence_of_element_located((By.CLASS_NAME, 'hand_p0')))
  hand_cards_element = WebDriverWait(hand_element, 500).until(EC.presence_of_element_located((By.CLASS_NAME, 'hand__cards')))
  WebDriverWait(hand_cards_element, 500).until(lambda driver: len(driver.find_elements_by_css_selector("*"))==8)
  card_elements = hand_cards_element.find_elements_by_css_selector("*")
  print(str(card_elements))
  cards_abrev = [card_element.get_attribute("class")[-2:] for card_element in card_elements]
  print(str(cards_abrev))

  translation_color = {"S":0, "H":1, "G":2, "E":3}
  translation_number = {"7":0, "8":1, "9":2, "U":3, "O":4, "K":5, "X":6, "A":7}

  inv_translation_color = {v: k for k, v in translation_color.items()}
  inv_translation_number = {v: k for k, v in translation_number.items()}

  cards = [[translation_color[c[0]], translation_number[c[1]]] for c in cards_abrev]

  rl_player.take_cards(cards)


  # get first player
  first_player = None
  for p in range(4):
    participant_box = driver.find_element_by_xpath("/html/body/div[1]/div[9]/div["+str(2+p)+"]")
    try:
      announcement = participant_box.find_element_by_class_name("participant-box__announcement")
      print("player-"+str(p) + " "+announcement.text)
      first_player = p
    except NoSuchElementException:
      pass

  #create game state
  game_state = GameState((first_player-1)%4)

  #call game

  called_game = rl_player.call_game_type(game_state)
  print("Called Game: "+str(called_game))
  called_game = called_game[0]

  #game selection
  #TODO: If only one game is possible then don't click on it
  table_view_element = driver.find_element_by_class_name("table-view")

  game_selector_element = WebDriverWait(table_view_element, 500).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class='game-selector g-highlight'], div[class='game-selector g-hidden']")))
  print("my turn")

  if "g-hidden" not in game_selector_element.get_attribute("class"):

    weiter_btn = table_view_element.find_element_by_class_name("game-selector__btn_Weiter")

    #Sauspiel
    if called_game[1] == 0:
      sauspiel_select_btn = table_view_element.find_element_by_class_name("game-selector__btn_Sauspiel")
      #if disabled then weiter
      if "game-selector__btn_disabled" in sauspiel_select_btn.get_attribute("class"):
        weiter_btn.click()
      else:
        sauspiel_select_btn.click()
        sauspiel_select_element = table_view_element.find_element_by_class_name("game-selector_suit-Sauspiel")
        time.sleep(2)
        if called_game[0] == 0:
          sauspiel_select_element.find_element_by_class_name("game-selector__btn_suit-S").click()
        if called_game[0] == 2:
          sauspiel_select_element.find_element_by_class_name("game-selector__btn_suit-G").click()
        if called_game[0] == 3:
          sauspiel_select_element.find_element_by_class_name("game-selector__btn_suit-E").click()

    #Solo
    elif called_game[1] == 2:
      solo_select_btn = table_view_element.find_element_by_class_name("game-selector__btn_Solo")
      if "game-selector__btn_disabled" in solo_select_btn.get_attribute("class"):
        weiter_btn.click()
      else:
        solo_select_btn.click()
        solo_select_element = table_view_element.find_element_by_class_name("game-selector_suit-Solo")
        time.sleep(2)
        if called_game[0] == 0:
          solo_select_element.find_element_by_class_name("game-selector__btn_suit-S").click()
        if called_game[0] == 1:
          solo_select_element.find_element_by_class_name("game-selector__btn_suit-H").click()
        if called_game[0] == 2:
          solo_select_element.find_element_by_class_name("game-selector__btn_suit-G").click()
        if called_game[0] == 3:
          solo_select_element.find_element_by_class_name("game-selector__btn_suit-E").click()

    #Wenz
    elif called_game[1] == 1:
      wenz_select_btn = table_view_element.find_element_by_class_name("game-selector__btn_Wenz")
      if "game-selector__btn_disabled" in wenz_select_btn.get_attribute("class"):
        weiter_btn.click()
      else:
        wenz_select_btn.click()

    #Weiter
    else:
      weiter_btn.click()

  # get game type
  # find out who is playing what
  game_player = None
  played_game = None
  alle_weiter = False
  while played_game == None:
    time.sleep(1)
    for p in range(4):
      participant_box = driver.find_element_by_xpath("/html/body/div[1]/div[9]/div[" + str(2 + p) + "]")
      try:
        announcement = participant_box.find_element_by_class_name("participant-box__announcement")
        print("player-" + str(p) + " " + announcement.text)
        if announcement.text == "Auf die Kugel":
          game_player = p
          played_game = [0, 0]
        if announcement.text == "Auf die Blaue":
          game_player = p
          played_game = [2, 0]
        if announcement.text == "Auf die Alte":
          game_player = p
          played_game = [3, 0]
        if announcement.text == "An Wenz":
          game_player = p
          played_game = [None, 1]
        if announcement.text == "A Schellen-Solo":
          game_player = p
          played_game = [0, 2]
        if announcement.text == "A Herz-Solo":
          game_player = p
          played_game = [1, 2]
        if announcement.text == "A Gras-Solo":
          game_player = p
          played_game = [1, 2]
        if announcement.text == "A Eichel-Solo":
          game_player = p
          played_game = [1, 2]
        if announcement.text == "Kommt raus..." and p!= first_player:
          alle_weiter = True
      except Exception:
        pass
  if alle_weiter:
    continue
  print("played game: "+str(played_game))
  game_state.game_type = played_game
  game_state.game_player = game_player

  #trick reader
  tricks_elements = table_view_element.find_element_by_class_name("tricks__els")
  for t in range(8):
    game_state.trick_number = t

    for c in range(4):
      current_player_id = (first_player+c)%4
      if current_player_id == 0:
        selected_card = rl_player.select_card(game_state)
        abbrev_selected_card = inv_translation_color[selected_card[0][0]] + inv_translation_number[selected_card[0][1]]
        print("Player 0 plays: " + abbrev_selected_card + "("+str(selected_card[1])+")")
        time.sleep(1)
        card_element = table_view_element.find_element_by_class_name("card_"+abbrev_selected_card)
        driver.execute_script("arguments[0].click();", card_element)
        game_state.player_plays_card(current_player_id, selected_card[0], selected_card[1])

      else:
        trick_element = WebDriverWait(tricks_elements, 120).until(
          EC.presence_of_element_located((By.XPATH, './/div[' + str(t + 1) + ']/div')))
        trick_card = WebDriverWait(trick_element, 120).until(EC.presence_of_element_located((By.XPATH, './/div['+str(c+1)+']')))
        card_abrev = trick_card.get_attribute("class")[-2:]
        print("Player " + str(current_player_id)+" plays: " + card_abrev)
        played_card = [translation_color[card_abrev[0]], translation_number[card_abrev[1]]]
        print(str(played_card))
        game_state.player_plays_card(current_player_id, played_card, 1)

      if c == 3:
        first_player = game_state.trick_owner[t]
        print("player "+ str(first_player) + " took trick " + str(t))


  total_games +=1
  total_reward += game_state.get_rewards()[0]
  print("Game-"+str(total_games) +" : "+str(game_state.get_rewards()[0])+ " --- Total="+str(total_reward)+" --- per_game="+str(total_reward/total_games))



