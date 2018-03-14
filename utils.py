import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

# This is my magic constant for home court advantage, in points.
# I have no idea why it is this value - I just grabbed what I used in 2017. In past years I did something smarter.
home_factor = 1.0
current_season = 2018
years = list(range(2003, current_season+1)) # this is correct: if current year is 2018, we are looking at 2017 season.
y_cols = ['season_%d' % y for y in years]

massey_date = '%m/%d/%Y'
wolfe_date = '%d-%b-%y'
game_keys = ['home', 'away', 'season']
epoch_count = 200  # 200


def complement(s, c):
    return 1.0 - s[c]


def suffix(f, s):
    return list(map(lambda x: x + '_' + s, f))


def strip_and_rename(g, t, opp_t):
    me = {c: c[1:] for c in g.columns if c[0] == t}
    opp = {c: 'opp_' + c[1:] for c in g.columns if c[0] == opp_t}
    return {**me, **opp}


def append_submission_complement(s, comp_cols, comp_func=complement):
    s2 = s.copy()
    s2['home'] = s['away']
    s2['away'] = s['home']
    for c in comp_cols:
        s2[c] = comp_func(s, c)
    return pd.concat([s, s2])


def group_game_stats_by_team_season(g, keep):
    g_by_team = g.groupby(['team', 'season'])
    g = g_by_team[keep].sum().reset_index()
    for c in g.columns:
        if c != 'season' and c != 'team':
            g[c] = g[c].astype('float')
    return g


def transform_rank_to_continuous(s):
    return 100 - 4 * np.log(s + 1) - s / 22.0  # recommended by Sonos


def normalize(r):
    r2 = r.set_index(['team', 'season'])
    r3 = (r2 - r2.mean()) / r2.std()
    return r3.reset_index()


def normalize_scale(r):
    r2 = r.set_index(['team', 'season'])
    r3 = (r2 - r2.mean()) / (r2.max() - r2.min())
    return r3.reset_index()


def predicted_correctly(y_true, y_pred):
    return np.logical_or(np.logical_and(y_pred >= 0.5, y_true >= 0.99), np.logical_and(y_pred < 0.5, y_true < 0.01))


def get_submission_stats(y_true, y_pred):
    wins = np.count_nonzero(predicted_correctly(y_true, y_pred))
    loss = log_loss(y_true, y_pred, labels=[0.0, 1.0])
    brier = brier_score_loss(y_true, y_pred)
    return 1.0 * wins / len(y_pred), loss, wins, brier


def merge_home_away(ncaa, ff, keys=['season']):
    t1 = pd.merge(ncaa, ff, left_on=['home'] + keys, right_on=['team'] + keys, how='inner')
    t2 = pd.merge(t1, ff, left_on=['away'] + keys, right_on=['team'] + keys, how='inner', suffixes=('_home', '_away'))
    return t2


# CDF of standard normal distribution.
# Thanks John D. Cook:
# http://www.johndcook.com/python_phi.html
def phi(x):
    import math
    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x) / math.sqrt(2.0)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return 0.5 * (1.0 + sign * y)


# The Win Probability model
def win_probability(x):
    return phi(0.0189 * x - 0.0756)
