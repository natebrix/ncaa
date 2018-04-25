###############################################################################
# Ken Pomeroy

import requests
import pandas as pd
from string import digits
from bs4 import BeautifulSoup
from utils import *

kp_cols = ['rank', 'name', 'conference', 'win_loss', 'pyth',
           'adj_o', 'adj_o_rank', 'adj_d', 'adj_d_rank', 'adj_t',
           'adj_t_rank', 'luck', 'luck_rank', 'sos_pyth', 'sos_pyth_rank',
           'sos_opp_o', 'sos_opp_o_rank', 'sos_opp_d', 'sos_opp_d_rank', 'ncsos_pyth',
           'ncsos_pyth_rank']
kp_cols_keep = ['name_spelling', 'name', 'season', 'pyth',
                'adj_o', 'adj_d', 'adj_t',
                'luck', 'sos_pyth',
                'sos_opp_o', 'sos_opp_d', 'ncsos_pyth']
kp_ratings = kp_cols_keep[3:]
remove_digits = str.maketrans('', '', digits)


def convert_kenpom_name(x):
    return x.translate(remove_digits).strip().lower()


def get_kenpom_data(year):
    print("Fetching KenPom data for year %s" % year)
    html = requests.get('http://kenpom.com/index.php?y=%d' % year, verify=False)
    soup = BeautifulSoup(html.text, "lxml")
    teams = []
    table = soup.find(id="ratings-table")
    for team in table.findAll("tbody")[0].findAll("tr"):
        stats = [td.text for td in team.findAll('td')]
        if len(stats) > 0:
            teams.append(list(map(str.strip, stats)))
    teams = pd.DataFrame(data=teams, columns=kp_cols)
    teams['season'] = year
    teams['name_spelling'] = teams.name.apply(convert_kenpom_name)
    teams = teams[kp_cols_keep]
    for c in kp_ratings:
        teams[c] = pd.to_numeric(teams[c])
    return teams


def create_kenpom(name='2018/kenpom.csv', current_season=2018, strict=False):
    t_map = pd.read_csv('2018/teamspellings.csv', encoding = "ISO-8859-1")
    t_all = []
    for year in range(2003, current_season + 1):
        t = get_kenpom_data(year)
        teams = pd.merge(t, t_map, left_on='name_spelling', right_on='TeamNameSpelling')
        if strict and t.shape[0] != teams.shape[0]:
            missing = set(t['name_spelling'].unique()) - set(t_map['TeamNameSpelling'].unique())
            print(missing)
            raise ValueError('Some records were lost in determining team ids. Check for new or renamed institutions.')
        t_all.append(teams)
    k = pd.concat(t_all)
    k.rename(columns={'TeamID': 'team'}, inplace=True)
    k = k[['season', 'team'] + kp_ratings]
    k = k.sort_values(by=['season', 'team'])

    for c in k.columns:
        if c[-4:] == 'rank':
            k[c] = transform_rank_to_continuous(k[c])

    k.to_csv(name, index=False)
