import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import TimeSeriesSplit, train_test_split


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values('date').unique()
        days = sorted(unique_dates, reverse=True)

        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates.date > days[train_start])
                              & (dates.date <= days[train_end])].index
            test_idx = dates[(dates.date > days[test_start])
                             & (dates.date <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def plot_preds_scatter(df, ticker=None):
        if ticker is not None:
            idx = pd.IndexSlice
            df = df.loc[idx[ticker, :], :]
        j = sns.jointplot(x='predicted', y='actuals',
                          robust=True, ci=None,
                          line_kws={'lw': 1, 'color': 'k'},
                          scatter_kws={'s': 1},
                          data=df,
                          kind='reg')
        j.ax_joint.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        j.ax_joint.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        j.ax_joint.set_xlabel('Predicted')
        j.ax_joint.set_ylabel('Actuals')

def plot_time_series_split(df, k):

    df_floats = df.astype('float64')
    tss = TimeSeriesSplit(k)
    indices = []
    test_id = []
    df_floats = df_floats.values.tolist()

    for train, test in tss.split(df):
        indices.append(train[-1])
        test_id.append(test[-1])

    fig, axes = plt.subplots(k, figsize=(20, 20))

    for i in range(k):
        start = indices[i] + 1
        end = test_id[i] + 1
        axes[i].plot(range(start), df_floats[:start], 'b')
        axes[i].plot(range(start, end), df_floats[start:end], 'r')
        axes[i].plot(range(end, len(df)), df_floats[end:], 'k')

    plt.show()


def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# fit model
	model = model_fit(train, cfg)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = model_predict(model, history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	print(' > %.3f' % error)
	return error