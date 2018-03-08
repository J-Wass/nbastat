from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import re
import os

#run web driver headlessly
os.environ['MOZ_HEADLESS'] = '1'
binary = FirefoxBinary('/usr/bin/firefox')
driver = webdriver.Firefox(firefox_binary=binary)
advanced_csv = 'Rk,Player,Pos,Age,Tm,G,MP,PER,TS%,3PAr,FTr,ORB%,DRB%,TRB%,AST%,STL%,BLK%,TOV%,USG%,,OWS,DWS,WS,WS/48,,OBPM,DBPM,BPM,VORP,Year\n'
per_game_csv = 'Rk,Player,Pos,Age▼,Tm,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PS/G,Year\n'

#iterate all advanced stats
for year in range(1950,2018):
    driver.get('https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_advanced.html')
    driver.execute_script('document.querySelector("#advanced_stats_toggle_partial_table").click()')
    driver.execute_script('document.querySelector("#all_advanced_stats > div:nth-child(1) > div:nth-child(3) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(4) > button:nth-child(1)").click()')
    csv = driver.find_element_by_css_selector("#csv_advanced_stats").text.split('\n',1)[-1]

    #add year to each row, clean up formatting
    csv = csv.replace('\n', ',' + str(year) + '\n')
    csv +=  ',' + str(year) + '\n'
    advanced_csv += csv

#iterate all per_game stats
for year in range(1950,2018):
    driver.get('https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_per_game.html')
    driver.execute_script('document.querySelector("#per_game_stats_toggle_partial_table").click()')
    driver.execute_script('document.querySelector("#all_per_game_stats > div:nth-child(1) > div:nth-child(3) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(4) > button:nth-child(1)").click()')
    csv = driver.find_element_by_css_selector("#csv_per_game_stats").text.split('\n',1)[-1]

    #add year to each row, clean up formatting
    csv = csv.replace('\n', ',' + str(year) + '\n')
    csv +=  ',' + str(year) + '\n'
    per_game_csv += csv

#write to csv
stats = open("advanced_stats.csv", "w")
stats.write(advanced_csv)
stats.close()
stats = open("per_game_stats.csv", "w")
stats.write(per_game_csv)
stats.close()
