import json

from lxml import etree
from io import StringIO


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

    self.tarif = None

    self.sonderregeln = []
    self.klopfer = []
    self.kontra = []

    self.player_dict = {}
    self.player_hands = {i:[] for i in range(4)}

    self.bidding_round = []

    self.course_of_game = [[[None, None] for x in range(4)] for y in range(8)]

    self.trick_owner = [None for x in range(8)]




  def fast_parse(self, driver):

    parser = etree.HTMLParser(remove_blank_text=True)
    tree = etree.parse(StringIO(driver.page_source), parser)

    # get id
    self.id = tree.xpath("/html/body/div[1]/div/div[1]/div/h2")[0].text.split(",")[0].split("#")[1]

    # get participant names
    self.player_dict[tree.xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[1]/div[1]/a/div[2]")[0].text.strip()] = 0
    self.player_dict[tree.xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[2]/div[1]/a/div[2]")[0].text.strip()] = 1
    self.player_dict[tree.xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[3]/div[1]/a/div[2]")[0].text.strip()] = 2
    self.player_dict[tree.xpath(
      "/html/body/div[1]/div/div[5]/div[1]/div/div[4]/div[1]/a/div[2]")[0].text.strip()] = 3

    # get game information
    self.tarif = tree.xpath("/html/body/div[1]/div/div[5]/div[2]/table/tbody/tr[1]/td")[0].text.strip()
    #game_table_elem = tree.xpath("//table[@class='game-result-table']")
    sr = tree.xpath("/html/body/div[1]/div/div[5]/div[2]/table/tbody/tr[2]/td")[0]
    if sr.text.strip() != "-":
      for im in sr:
        self.sonderregeln.append(im.get("title"))
    kl = tree.xpath("/html/body/div[1]/div/div[5]/div[2]/table/tbody/tr[6]/td")[0]
    if kl.text.strip() != "-":
       self.klopfer = [a.text for a in kl]

    con = tree.xpath("/html/body/div[1]/div/div[5]/div[2]/table/tbody/tr[7]/td")[0]
    if con.text.strip() != "-":
      self.kontra = [a.text for a in con]


    # get starting hands
    for p in range(4):
      row = tree.xpath("/html/body/div[1]/div/section/div[2]/div[2]/div["+str(p+1)+"]")[0]
      for card in row[1]:
        if "highlight" in card.get('class').split(" "):
          self.player_hands[p].insert(0, self.card_dict[card.text])
        else:
          self.player_hands[p].append(self.card_dict[card.text])

    # get bidding round
    messages_elements = tree.xpath("/html/body/div[1]/div/section/div[3]/div[2]")[0]
    for message_elem in messages_elements:
      self.bidding_round.append(' '.join(message_elem.text.split()))


    # get tricks
    tricks = 8
    if "Kurze Karte" in self.sonderregeln:
      tricks = 6
    if len(self.bidding_round) == 4:  # all said weiter
      tricks = 0
    for trick in range(tricks):
      self.trick_owner[trick] = self.player_dict[tree.xpath(
        "/html/body/div[1]/div/section/div[" + str(4 + trick) + "]/div[2]/div[1]/div[1]/a/div[2]")[0].text.strip()]

      for card in range(4):
        card_name = tree.xpath(
          "/html/body/div[1]/div/section/div[" + str(4 + trick) + "]/div[2]/div[2]/div[" + str(
            1 + card) + "]/span")[0].text.strip()
        self.course_of_game[trick][card] = GameTranscript.card_dict[card_name]


  def toJSON(self):
    return {
      "id": self.id,
      "tarif": self.tarif,
      "sonderregeln": self.sonderregeln,
      "klopfer": self.klopfer,
      "kontra": self.kontra,
      "player_dict": self.player_dict,
      "player_hands": self.player_hands,
      "bidding_round": self.bidding_round,
      "course_of_game": self.course_of_game,
      "trick_owner": self.trick_owner
    }
