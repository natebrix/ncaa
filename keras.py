#from keras.models import Sequential
#from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge, Activation
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.advanced_activations import PReLU, ELU
#from keras.layers.normalization import BatchNormalization

###############################################################################
# Keras
# def create_keras_model(num_teams, num_features):
#   print('Creating keras model with %d teams and %d features.' % (num_teams, num_features))
#   model = Sequential()
#   model.add(Dense(12, input_dim=num_features, init='uniform'))
#   model.add(BatchNormalization())
#   model.add(Dense(1))
#   model.add(Activation('sigmoid'))
#   model.compile(loss='binary_crossentropy', optimizer="sgd")
#   return model
#
#
# def estimate_keras(model, X, y, X_ncaa, w=None):
#   index_count = len(game_keys)
#   model.fit(X.values[:, index_count:], y, batch_size=1024, sample_weight=w,
#     nb_epoch=epoch_count, validation_split=0.1, verbose=2)
#   y_fit = model.predict(X_ncaa.values[:, index_count:], batch_size=1024, verbose=2)[:, 0]
#   return y_fit
#
#
# def make_keras(ratings=None, sub_file='sample_submission_0.csv', season_file='regularseasondetailedresults.csv'):
#   if ratings is None:
#     ratings = make_team_features()
#   games = make_training_games(season_file)
#   num_teams = games.shape[0]
#   index_count = len(game_keys)
#
#   X, y, w = make_features(games, ratings, binary_y=True)
#   check_feature_matrix(games, X, check_complete=False)
#   ncaa = make_tourney_games(sub_file)
#
#   X_ncaa, y_ncaa, w_ncaa = make_features(ncaa, ratings, make_y=False)
#   check_feature_matrix(ncaa, X_ncaa)
#
#   num_features = X.shape[1] - index_count
#   model = create_keras_model(num_teams, num_features)
#   predict = lambda X, y, X_ncaa: estimate_keras(model, X, y, X_ncaa, w=w)
#   #y_fit = estimate(X, y, X_ncaa, get_partitions=get_partitions_by, predict=predict)
#   y_fit = estimate(X, y, X_ncaa, predict=predict)
#
#   s = replace_pred(ncaa, X_ncaa, y_fit)
#   write_submission(s, 'submission.csv')
#   write_submission_with_names(s, 'submission_names.csv')
#   return score_submission('submission.csv')
