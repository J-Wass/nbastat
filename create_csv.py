from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import re
import os

#run web driver headlessly
os.environ['MOZ_HEADLESS'] = '1'
binary = FirefoxBinary('/usr/bin/firefox')
driver = webdriver.Firefox(firefox_binary=binary)
total_csv = 'Rk,Player,Pos,Age,Tm,G,MP,PER,TS%,3PAr,FTr,ORB%,DRB%,TRB%,AST%,STL%,BLK%,TOV%,USG%,,OWS,DWS,WS,WS/48,,OBPM,DBPM,BPM,VORP,Year\n'

#iterate all recorded seasons
for year in range(1950,2018):
    driver.get('https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_advanced.html')
    driver.execute_script('document.querySelector("#advanced_stats_toggle_partial_table").click()')
    driver.execute_script('document.querySelector("#all_advanced_stats > div:nth-child(1) > div:nth-child(3) > ul:nth-child(1) > li:nth-child(1) > div:nth-child(2) > ul:nth-child(1) > li:nth-child(4) > button:nth-child(1)").click()')
    csv = driver.find_element_by_css_selector("#csv_advanced_stats").text.split('\n',1)[-1]

    #add year to each row, clean up formatting
    csv = csv.replace('\n', ',' + str(year) + '\n')
    csv +=  ',' + str(year) + '\n'
    total_csv += csv

#write to csv
stats = open("player_stats.csv", "w")
stats.write(total_csv)
stats.close()
