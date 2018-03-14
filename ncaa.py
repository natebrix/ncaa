# NCAA basketball ratings
# Nathan Brixius, 3/11/2018
#
# This is essentially an ensemble model based on a few things:
# 1) nathan.py: My own eigenvalue centrality rating based on an idea from:
#      http://blog.biophysengr.net/2012/03/eigenbracket-2012-using-graph-theory-to.html
#    combined with the Sokol 2010 win probability model described here:
#      http://netprophetblog.blogspot.com/2011/06/logistic-regressionmarkov-chain-lrmc.html
# 2) kenpom.py: Public Ken Pomeroy data
# 3) Other ratings available from Kaggle
# 4) Dean Oliver's "four factors"
#
# These are all mixed together in a logistic regression.
#
# There are many other things to try, but I don't have time.
#
#
# General workflow:
#   1) go to Kaggle and get the necessary files. Put them in a new year directory.
#   2) Fix up all of the filenames in the .py to point to the right place. Refactor this code next year.
#   3) Rebuild the 'nathan' ratings
#   4) Rebuild the KenPom ratings
#   5) Try to run the code below (make_submission)

import numpy as np
import math
import pandas as pd
from kenpom import *
from kaggle import *

from sklearn.linear_model import LogisticRegression


massey = '2018/MasseyOrdinals_2018_133_only_43Systems.csv'
nathan = '2018/nathan.csv'

###############################################################################

def get_seeds():
    seeds = pd.read_csv('2018/NCAATourneySeeds.csv')
    seeds.rename(columns={'Season': 'season', 'Seed': 'seed_full', 'TeamID': 'team'}, inplace=True)
    seeds['seed'] = seeds['seed_full'].apply(lambda s: int(s.translate({ord(c): None for c in 'abWXYZ'})))
    return seeds


# todo I need a way to integrate 'round' into the model.
# If I train on tourney results then it could be useful somehow.
# I can fill in round for a submission template.

# I am sure there is some amazing way to do this with groupby and apply
# or whatever. It's easier for me to write a for-loop.
def write_rounds(games):
    games['round'] = -1
    j_season = games.columns.get_loc('season')
    j_round = games.columns.get_loc('round')
    games.sort_values(by=['season', 'daynum'], ascending=False, inplace=True)
    current_season = -1
    current_round = 0
    teams_in_round = 1
    k = 0
    for i in range(games.shape[0]):
        if current_season != games.iloc[i, j_season]:
            current_season = games.iloc[i, j_season]
            current_round = 0
            teams_in_round = 1
            k = 0
        games.iloc[i, j_round] = current_round
        k += 1
        if k >= teams_in_round:
            current_round += 1
            teams_in_round *= 2
            k = 0
    return games


###############################################################################
# Distances

def haversine(l1, l2):
    rl1 = math.radians(90 - l1[0])
    rl2 = math.radians(90 - l2[0])
    t1 = math.cos(rl1) * math.cos(rl2)
    t2 = math.sin(rl1) * math.sin(rl2)
    t3 = math.cos(math.radians(l1[1] - l2[1]))
    # print(t1, t2, t3, t1+t2*t3)
    return math.acos(min(max(t1 + t2 * t3, -1.0), 1.0)) * 6371


dist_scale = 0.3  # 0.3
dist_target = 100.0  # 100.0


def indexed_distance(s):
    # todo ???
    return s.apply(lambda d: max(0.0, 1.0 - dist_scale * d * d / (dist_target * dist_target)))


def make_distance_columns(g, prefix):
    g['%s_dist' % prefix] = g.apply(
        lambda r: haversine((r['lat'], r['lng']), (r['lat_%s' % prefix], r['lng_%s' % prefix])), axis=1)
    g['%s_dist_idx' % prefix] = indexed_distance(g['%s_dist' % prefix])


def get_tourney_distances():
    tg = pd.read_csv('2018/tourneygeog.csv')
    tg['home'] = tg.apply(lambda r: min(r['wteam'], r['lteam']), axis=1)  # by convention
    tg['away'] = tg.apply(lambda r: max(r['wteam'], r['lteam']), axis=1)  # by convention
    t = pd.read_csv('2018/teamgeog.csv')
    g0 = pd.merge(tg, t, left_on=['home'], right_on=['team_id'], how='inner', suffixes=('', '_home'))
    g = pd.merge(g0, t, left_on=['away'], right_on=['team_id'], how='inner', suffixes=('', '_away'))
    make_distance_columns(g, 'home')
    make_distance_columns(g, 'away')
    g['home_factor'] = g['home_dist_idx'] - g['away_dist_idx']
    return g[
        ['season', 'daynum', 'home', 'away', 'home_dist', 'away_dist', 'home_dist_idx', 'away_dist_idx', 'home_factor']]


def score_submission(name):
    r = get_submission_data(name)
    result = get_submission_stats(r['y'], r[col_pred])
    for season, group in r.groupby('season'):
        print(season, get_submission_stats(group['y'], group[col_pred]))
    return result


def write_submission_with_names(s, name):
    teams = pd.read_csv('2018/teams.csv')
    teams.rename(columns={'TeamID': 'team', 'TeamName': 'name'}, inplace=True)
    s2 = merge_home_away(s, teams, keys=[])
    s2 = s2[['name_home', 'name_away', col_pred, 'team_home', 'team_away']]
    s2.to_csv(name, index=False)


def get_upsets(name):
    r = get_submission_data(name)
    pc = predicted_correctly(r['y'], r[col_pred])
    upsets = r[np.logical_not(pc)]
    teams = pd.read_csv('2018/teams.csv')
    u1 = pd.merge(upsets, teams, left_on='home', right_on='Team_Id', how='inner')
    u = pd.merge(u1, teams, left_on='away', right_on='Team_Id', how='inner', suffixes=('_home', '_away'))
    return u \
        .sort_values(by=['season', 'daynum', 'home']) \
        .drop(['neutral_x', 'neutral_y', 'y', 'Team_Id_home', 'Team_Id_away'], axis=1)


##################

def read_four_factors(ncaa_csv='2018/regularseasondetailedresults.csv'):
    print("Reading detailed results from %s" % ncaa_csv)
    g = read_kaggle_by_team(ncaa_csv)
    g['efg'] = (g['fgm'] + 0.5 * g['fgm3']) / g['fga']
    g['top'] = g['to'] / (g['fga'] + 0.44 * g['fta'] + g['to'])
    g['orp'] = g['or'] / (g['or'] + g['opp_dr'])
    g['drp'] = g['dr'] / (g['dr'] + g['opp_or'])
    g['ftf'] = g['ftm'] / g['fta']
    print("Calculated four factors for seasons %s" % g['season'].unique())
    return g


def read_massey():
    print("Reading Massey ratings from %s" % massey)
    g = pd.read_csv(massey)
    g.rename(columns={'Season': 'season', 'RankingDayNum': 'rating_day_num', 'SystemName': 'sys_name', \
                      'TeamID': 'team', 'OrdinalRank': 'orank'}, inplace=True)
    print("Read seasons %s" % g['season'].unique())

    # keep_ratings = ['CPR'] #, 'WLK'] #, 'DOL'] #, 'CPA', 'DCI', 'COL', 'BOB']
    keep_ratings = ['WLK']  # , 'WLK'] #, 'DOL'] #, 'CPA', 'DCI', 'COL', 'BOB']
    # keep_ratings = ['CPR', 'WLK'] #, 'DOL'] #, 'CPA', 'DCI', 'COL', 'BOB']
    # keep_ratings = ['CPR', 'WLK', 'DOL', 'CPA', 'DCI', 'COL', 'BOB']
    # keep_ratings = ['CPA']
    print("Keeping the following ratings: %s" % keep_ratings)
    m = g.query('rating_day_num==133 and season>=2003')
    m = m.pivot_table(index=['team', 'season'], columns='sys_name', values='orank').reset_index()
    ren = {c: "rating_" + c.lower() for c in keep_ratings}
    m.rename(columns=ren, inplace=True)
    for c in ren.values():
        m[c] = transform_rank_to_continuous(m[c])
    print("Retained seasons %s" % m['season'].unique())
    return m[['team', 'season'] + list(ren.values())].fillna(351 / 1.0)  # todo team count


def read_nathan():
    print("Reading Nathan ratings from %s" % nathan)
    g = pd.read_csv(nathan)
    print("Read seasons %s" % g['season'].unique())
    return g


def read_kenpom(in_file='2018/kenpom.csv'):
    print("Reading KenPom ratings from %s" % in_file)
    g = pd.read_csv(in_file)
    print("Read seasons %s" % g['season'].unique())
    return g


def make_team_features():
    m = read_massey()
    r = read_nathan()
    r1 = pd.merge(m, r, left_on=['team', 'season'], right_on=['team', 'season'], how='inner')
    k = read_kenpom()
    r2 = pd.merge(r1, k, left_on=['team', 'season'], right_on=['team', 'season'], how='inner')
    ff = read_four_factors()
    ratings = pd.merge(r2, ff, left_on=['team', 'season'], right_on=['team', 'season'], how='inner')
    print('Created team features for the following seasons: %s' % list(ratings['season'].unique()))
    return normalize(ratings)


def make_training_games(season_file='2018/regularseasondetailedresults.csv'):
    games = read_kaggle_season(season_file)
    print("Read %d games from '%s'." % (games.shape[0], season_file))
    games['home_factor'] = np.where(games.neutral, 0.0, home_factor)
    games2 = games.copy()
    games2['home'] = games['away']
    games2['homesc'] = games['awaysc']
    games2['away'] = games['home']
    games2['awaysc'] = games['homesc']
    games2['home_factor'] = np.where(games2.neutral, 0.0, -home_factor)
    g = pd.concat([games, games2]).sort_values(by=['season', 'daynum', 'home'])
    return g


def make_tourney_games(sub_file):
    games = read_submission(sub_file)
    games['daynum'] = 154.0  # todo
    #t = get_tourney_distances() # note: I don't have time to make team locations this year
    #t = t[['season', 'home', 'away', 'home_factor']]
    #g2 = pd.merge(games, t, left_on=['season', 'home', 'away'], right_on=['season', 'home', 'away'])
    g2 = None
    if not g2 or g2.shape[0] == 0:
        print('Assuming no home court advantage for tourney games since no tourney locations provided.')
        games['home_factor'] = 0
        return games
    return g2


# note I am leaving out 'time'
def retained_features(X):
    # with kenpom, four factors actually makes it worse
    # f = ['efg', 'top', 'orp', 'drp', 'ftf'] # the four factors stuff doesn't help much
    f = []
    ratings = [c for c in X.columns if c[:6] == 'rating'] + suffix(kp_ratings, 'home') + suffix(kp_ratings, 'away')
    years = [c for c in X.columns if c[:5] == 'season_']
    return game_keys + ratings + years + ['y', 'home_factor'] + suffix(f, 'home') + suffix(f, 'away')


def make_categorical_dummies(X):
    X = pd.concat([X, pd.get_dummies(X['season'], prefix='season')], axis=1)
    dummies = []
    for y in years:
        c_y = 'season_%d' % y
        if c_y not in X.columns:
            X[c_y] = 0
            dummies.append(c_y)
    if len(dummies) > 0:
        print('Added %d season dummies: %s' % (len(dummies), dummies))
    return X


def check_feature_matrix(ncaa, X_ncaa, check_complete=True):
    # todo check me for NAN
    if check_complete and X_ncaa.shape[0] != ncaa.shape[0]:
        missing = set(list(ncaa['home'].unique()) + list(ncaa['away'].unique())) - set(
            list(X_ncaa['home'].unique()) + list(X_ncaa['away'].unique()))
        raise ValueError("Feature matrix does not have enough rows %d/%d. These teams are missing: %s" % (
        X_ncaa.shape[0], ncaa.shape[0], missing))


def make_features(games, team_features, make_y=True, binary_y=False, ignore_year=None):
    print('Making features for %d games for %d teams and %d features.' % \
          (games.shape[0], team_features.shape[0], team_features.shape[1]))

    if ignore_year is not None:
        print('Ignoring year %d.' % ignore_year)
        games = games.query('season != %d' % ignore_year)
        team_features = team_features.query('season != %d' % ignore_year)

    if games.shape[0] == 0:
        raise ValueError("No games provided.")
    X = merge_home_away(games, team_features)
    if X.shape[0] == 0:
        raise ValueError("Failed to create X - are you missing data for a season?")
    y = None
    if make_y:
        if binary_y:
            y = np.where(X.homesc > X.awaysc, 1, 0)
        else:
            diff = X.homesc - X.awaysc
            y = diff.apply(lambda x: win_probability(4.0 * x))
    X['y'] = y
    w = None  # pow(0.95, 156.0 - np.array(X.daynum)) #X.daynum #/ 156.0
    X = make_categorical_dummies(X)
    X = X[retained_features(X)].copy()
    y = X['y'].copy()
    del X['y']
    return X, y, w


def get_estimator():
    est = LogisticRegression()
    return est


def get_partitions_by(X, y, X_ncaa, by='season'):
    for k in X_ncaa[by].unique():
        X_k = X.query('%s==%s' % (by, k))
        X_ncaa_k = X_ncaa.query('%s==%s' % (by, k))
        y_k = y[X_k.index]
        yield X_k, y_k, X_ncaa_k


# Use results from other tourneys as a predictor for this one.
def get_partitions_other(X, y, X_ncaa, by='season'):
    for k in X_ncaa[by].unique():
        X_k = X.query('%s!=%s' % (by, k))
        X_ncaa_k = X_ncaa.query('%s==%s' % (by, k))
        y_k = y[X_k.index]
        yield X_k, y_k, X_ncaa_k


def estimate(X, y, X_ncaa, predict, get_partitions=None, clip=True):
    if get_partitions is None:
        y_fit = predict(X, y, X_ncaa)
    else:
        y_fit = predict(X, y, X_ncaa)  # to ensure all values set no matter partition
        for X_k, y_k, X_ncaa_k in get_partitions(X, y, X_ncaa):
            y_f_k = predict(X_k, y_k, X_ncaa_k)
            print(X_k.shape, len(y_f_k), len(X_k.index), len(y_fit))
            y_fit[X_ncaa_k.index] = y_f_k
    if clip:
        delta = 0.05  # a little aggressive, e.g. missed MSU-MTSU in 2016
        y_fit[y_fit <= delta] = 0.001
        y_fit[y_fit >= 1 - delta] = 0.999
        # even more aggressive...
        #y_fit[y_fit <= delta*2] = delta
        #y_fit[y_fit >= 1 - delta*2] = 1-delta
    return y_fit


default_upset_tol = 0.07

def create_upsets(s2, pred_tol=default_upset_tol, seed_tol=2):
    seeds = get_seeds()
    submission = merge_home_away(s2, seeds)
    j_pred = submission.columns.get_loc(col_pred)
    j_seed_home = submission.columns.get_loc('seed_home')
    j_seed_away = submission.columns.get_loc('seed_away')
    mods = []
    for i in range(submission.shape[0]):
        pred = submission.iloc[i, j_pred]
        seed_diff = submission.iloc[i, j_seed_home] - submission.iloc[i, j_seed_away]
        # induce home upset: home is higher seed and probability almost 0.5
        if pred >= 0.5 - pred_tol and pred <= 0.5 and seed_diff >= seed_tol:
            submission.iloc[i, j_pred] = 0.501
            mods.append(i)
        # induce away upset: away is higher seed and probability just over 0.5
        if pred <= 0.5 + pred_tol and pred >= 0.5 and -seed_diff >= seed_tol:
            submission.iloc[i, j_pred] = 0.499
            mods.append(i)
    print('Adjusted to create %d upsets.' % len(mods))
    return submission


def replace_pred(s, X, y):
    s2 = s
    del s2[col_pred]
    X[col_pred] = y
    s2 = pd.merge(s2, X, left_on=game_keys, right_on=game_keys, how='inner')
    s2.reset_index(inplace=True)
    return s2


# make a submission based on logistic regression
def make_submission(ratings=None, phase=1, season_file='2018/regularseasondetailedresults.csv'):
    if ratings is None:
        ratings = make_team_features()

    if phase == 1:
        ignore_year = current_season
        sub_file = '2018/samplesubmissionStage1.csv'
    else:
        ignore_year = None
        sub_file = '2018/samplesubmissionStage2.csv'

    games = make_training_games(season_file)
    X, y, w = make_features(games, ratings, binary_y=True, ignore_year=ignore_year)
    check_feature_matrix(games, X, check_complete=False)
    ncaa = make_tourney_games(sub_file)
    ncaa_tourneys = len(ncaa['season'].unique())

    X_ncaa, y_ncaa, w_ncaa = make_features(ncaa, ratings, make_y=False, ignore_year=ignore_year)
    check_feature_matrix(ncaa, X_ncaa)

    est = get_estimator()
    predict = lambda X, y, X_ncaa: est.fit(X, y).predict_proba(X_ncaa)[:, 1]
    y_fit = estimate(X, y, X_ncaa, predict=predict)
    s = replace_pred(ncaa, X_ncaa, y_fit)
    s = create_upsets(s, pred_tol=default_upset_tol)
    write_submission(s, 'output/submission.csv', ncaa_tourneys)
    write_submission_with_names(s, 'output/submission_names.csv')
    if phase == 1:
        return score_submission('output/submission.csv')
    else:
        return ()
