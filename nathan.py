import operator
import numpy as np
import numpy.linalg as LA
import pandas as pd
from kaggle import *
from utils import *

###############################################################################
# My own metric

detailed_results = '2018/regularseasondetailedresults.csv'

def process_exceptions(games):
    return games


def compute_raw_ratings(games, div1, sensitivity, recencyAdj, use_rh=True):
    seasonLen = 154  # kaggle
    raw_ratings = []
    for index, g in games.iterrows():
        # omit exhibitions and non-D1 games
        if (g.home in div1 or g.away in div1) and not g.ncaa:
            diff = g.homesc - g.awaysc
            if g.neutral:  # neutral site
                diff -= home_factor
            daysFromEnd = seasonLen - g.daynum
            diff_adj = sensitivity + recencyAdj * (seasonLen - daysFromEnd) / seasonLen
            rating = win_probability(diff * diff_adj) if use_rh else diff * diff_adj
            if np.isnan(rating):
                raise ValueError('Obtained NaN for rating for game %s' % g)
            raw_ratings.append([g.home, g.away, rating, g.homesc, g.awaysc, g.daynum])
    return raw_ratings


def get_name_to_index(raw_ratings):
    teams = set()
    for [h, a, rh, hsc, asc, date] in raw_ratings:
        teams.add(h)
        teams.add(a)
    teamlist = sorted(list(teams))
    nameidx = {x: i for i, x in enumerate(teamlist)}
    return teamlist, nameidx


def compute_adjacency(raw_ratings, teamlist, nameidx):
    adj = np.zeros([len(teamlist), len(teamlist)])
    for [h, a, rh, hsc, asc, date] in raw_ratings:
        rh = min(0.99, max(0.01, rh))
        adj[nameidx[h], nameidx[a]] += rh
        adj[nameidx[a], nameidx[h]] += 1 - rh
    return adj


# Eigenvector power iteration
def eigenvalue_power(b0, a, iterCount):
    b = b0
    for i in range(iterCount):
        ab = np.dot(a, b)
    b = ab / LA.norm(ab)
    return b


def all_games(rr, team):
    return [r[0:5] for r in rr if r[0] == team or r[1] == team]


def average_rating(rr, team):
    return np.average(
        [(r[0] == team) * r[2] + (1 - (r[0] == team)) * (1 - r[2]) for r in rr if r[0] == team or r[1] == team])


def write_nathan_rating(s, name):
    s.to_csv(name, index=False)


def make_nathan_rating(teamCount=10, iter=20, sensitivity=2.5, recencyAdj=2.0, use_rh=True, stdev=0, log=True):
    all_games = read_kaggle_season(detailed_results)
    print('Making eigenvalue rating for seasons: %s' % (all_games['season'].unique()))
    dfs = []
    for year in all_games.season.unique():
        games = all_games.query('season==%d' % year)
        process_exceptions(games)
        div1 = set(games.home) | set(games.away)
        raw_ratings = compute_raw_ratings(games, div1, sensitivity, recencyAdj, use_rh=use_rh)
        teamlist, nameidx = get_name_to_index(raw_ratings)
        adj = compute_adjacency(raw_ratings, teamlist, nameidx)
        b0 = np.array([average_rating(raw_ratings, t) for t in teamlist])
        b = eigenvalue_power(b0, adj, iter)

        # Sort teams and ratings in descending order
        namescore = {team: b[i] for i, team in enumerate(teamlist)}
        sorted_ns = sorted(namescore.items(), key=operator.itemgetter(1), reverse=True)
        df_year = pd.DataFrame({'team': teamlist})
        df_year['season'] = year
        df_year['rating'] = df_year['team'].map(namescore)
        dfs.append(df_year)

        final = [[t, r] for [t, r] in sorted_ns]

        if log:
            print(year)
            for i in range(teamCount):
                print(final[i])

    df = pd.concat(dfs, ignore_index=True)
    return df
