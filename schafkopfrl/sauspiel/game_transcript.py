import json


class GameTranscript:

  card_dict = {
    "Schellen-Sieben" : [0,0],
    "Schellen-Acht" : [0, 1],
    "Schellen-Neun" : [0,2],
    "Schellen-Unter" : [0,3],
    "Der Runde": [0, 4],
    "Schellen-König" : [0, 5],
    "Schellen-Zehn" : [0,6],
    "Die Hundsgfickte" : [0,7],
    "Herz-Sieben": [1, 0],
    "Herz-Acht": [1, 1],
    "Herz-Neun": [1, 2],
    "Herz-Unter": [1, 3],
    "Der Rote": [1, 4],
    "Herz-König": [1, 5],
    "Herz-Zehn": [1, 6],
    "Herz-Sau": [1, 7],
    "Gras-Sieben": [2, 0],
    "Gras-Acht": [2, 1],
    "Gras-Neun": [2, 2],
    "Gras-Unter": [2, 3],
    "Der Blaue": [2, 4],
    "Gras-König": [2, 5],
    "Gras-Zehn": [2, 6],
    "Die Blaue": [2, 7],
    "Eichel-Sieben": [3, 0],
    "Eichel-Acht": [3, 1],
    "Eichel-Neun": [3, 2],
    "Eichel-Unter": [3, 3],
    "Der Alte": [3, 4],
    "Eichel-König": [3, 5],
    "Eichel-Zehn": [3, 6],
    "Die Alte": [3, 7]
  }


  def __init__(self):
    self.id = None

    self.sonderregeln = None
    self.klopfer = None
    self.kontra = None

    self.player_dict = {}
    self.player_hands = {i:[] for i in range(4)}

    self.bidding_round = []

    self.course_of_game = [[[None, None] for x in range(4)] for y in range(8)]

    self.trick_owner = [None for x in range(8)]

    self.game_player = None


  def parse(self, driver):

    # get id
    self.id = driver.find_element_by_xpath("/html/body/div[1]/div/div[1]/div/h2").text.split(",")[0].split("#")[1]

    # get participant names
    self.player_dict[driver.find_element_by_xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[1]/div[1]/a/div[2]").text] = 0
    self.player_dict[driver.find_element_by_xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[2]/div[1]/a/div[2]").text] = 1
    self.player_dict[driver.find_element_by_xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[3]/div[1]/a/div[2]").text] = 2
    self.player_dict[driver.find_element_by_xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[4]/div[1]/a/div[2]").text] = 3


    # get game information
    game_table_elem = driver.find_element_by_class_name("game-result-table")
    self.sonderregeln = game_table_elem.find_element_by_xpath(".//tbody/tr[2]/td").text
    self.klopfer = game_table_elem.find_element_by_xpath(".//tbody/tr[6]/td").text
    self.kontra = game_table_elem.find_element_by_xpath(".//tbody/tr[7]/td").text

    # get starting hands
    starting_hands_elements = driver.find_element_by_class_name("card-rows").find_elements_by_class_name(
      "game-protocol-item")
    for i, starting_hand_elem in enumerate(starting_hands_elements):
      cards_elements = starting_hand_elem.find_element_by_class_name(
        "game-protocol-cards").find_elements_by_css_selector("*")
      for card_elem in cards_elements:
        if "highlight" in card_elem.get_attribute("class").split(" "):
          self.player_hands[i].insert(0, self.card_dict[card_elem.text])
        else:
          self.player_hands[i].append(self.card_dict[card_elem.text])

    # get bidding round
    messages_elements = driver.find_element_by_xpath("/html/body/div[1]/div/section/div[3]/div[2]").find_elements_by_css_selector("*")
    for message_elem in messages_elements:
      self.bidding_round.append(message_elem.text)

    self.game_player = self.player_dict[self.bidding_round[-1].split(" ")[0]]

    # get tricks
    for trick in range(8):
      self.trick_owner[trick] = self.player_dict[driver.find_element_by_xpath(
        "/html/body/div[1]/div/section/div[" + str(4 + trick) + "]/div[2]/div[1]/div[1]/a/div[2]").text]

      for card in range(4):
        card_name = driver.find_element_by_xpath("/html/body/div[1]/div/section/div["+str(4+trick)+"]/div[2]/div[2]/div["+str(1+card)+"]/span").text
        self.course_of_game[trick][card] = GameTranscript.card_dict[card_name]

  def __str__(self):
    res = ""
    res += str(self.sonderregeln) + "\n"
    res += str(self.klopfer) + "\n"
    res += str(self.kontra) + "\n"

    res += str(self.player_dict) + "\n"
    res += str(self.player_hands) + "\n"
    res += str(self.bidding_round) + "\n"
    res += str(self.course_of_game) + "\n"
    res += str(self.trick_owner) + "\n"
    return res


  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__,sort_keys=False, indent=4)