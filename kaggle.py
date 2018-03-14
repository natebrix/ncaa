from utils import *
import numpy as np
import pandas as pd

col_id = 'ID'
col_pred = 'Pred'

def read_submission(name):
    g = pd.read_csv(name)
    s = pd.DataFrame()
    s['home'] = g[col_id].apply(lambda x: int(x.split('_')[1]))
    s['away'] = g[col_id].apply(lambda x: int(x.split('_')[2]))
    s['season'] = g[col_id].apply(lambda x: int(x.split('_')[0]))
    s['neutral'] = 1
    s[col_pred] = g[col_pred]
    return s

def write_submission(s, name, expected_seasons=4):
    print('Writing submission for %d seasons.' % expected_seasons)
    seasons = s['season'].unique()
    if len(seasons) < expected_seasons:
        print('Invalid submission: some seasons are missing: %s' % seasons)
    s[col_id] = s.season.map(str) + '_' + s.home.map(str) + '_' + s.away.map(str)
    submission = s[[col_id, col_pred]]
    submission.to_csv(name, index=False)


def read_kaggle_season(ncaaCsv, year=None, ncaa=False):
    g = pd.read_csv(ncaaCsv)
    if year is not None:
        g = g.query('Season==%d' % year)
    if ncaa:
        g = pd.concat([group.tail(63) for name, group in g.groupby('Season')])  # no play-ins
    g2 = pd.DataFrame()
    g2['neutral'] = (g.Wloc == 'N')
    g2['ncaa'] = ncaa
    g2['season'] = g['Season']
    g2['home'] = np.where(g.Wloc == 'A', g.Lteam, g.Wteam)
    g2['homesc'] = np.where(g.Wloc == 'A', g.Lscore, g.Wscore)
    g2['away'] = np.where(g.Wloc == 'A', g.Wteam, g.Lteam)
    g2['awaysc'] = np.where(g.Wloc == 'A', g.Wscore, g.Lscore)
    g2['daynum'] = g.Daynum.copy()
    return g2


# Read kaggle games but with individual records by team-game, recording
# the opponent's information in columns beginning with opp_
def read_kaggle_by_opp(ncaa_csv):
    g_w = pd.read_csv(ncaa_csv)
    g_w.rename(columns={'Season': 'season'}, inplace=True)
    g_w['Ww'] = 1
    g_w['Lw'] = 0
    g_l = g_w.copy()
    g_w.rename(columns=strip_and_rename(g_w, 'W', 'L'), inplace=True)
    g_l.rename(columns=strip_and_rename(g_l, 'L', 'W'), inplace=True)
    return g_w.append(g_l)[[c for c in g_w.columns if c[0] != 'W' and c[0] != 'L']]


def get_submission_data(name):
    s = append_submission_complement(read_submission(name), [col_pred])  # submissions start with lower numbered teams - want all
    games = read_kaggle_season('2018/NCAAtourneydetailedresults.csv', ncaa=True)
    games['y'] = np.where(games.homesc > games.awaysc, 1.0, 0.0)
    return pd.merge(games, s, left_on=game_keys, right_on=game_keys, how='inner')


# Read kaggle games and aggregate stats by team-season.
def read_kaggle_by_team(ncaa_csv):
    g = read_kaggle_by_opp(ncaa_csv)
    keep = ['to', 'fga', 'fta', 'or', 'dr', 'opp_or', 'opp_dr', 'fgm3', 'fgm', 'ftm', 'w']
    return group_game_stats_by_team_season(g, keep)
