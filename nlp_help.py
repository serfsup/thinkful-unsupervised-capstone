import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from bokeh.io import output_notebook
from bokeh.layouts import row
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap


class LsaPlotting:
    def __init__(self, X_train_lsa: np.ndarray, y_train: pd.Series,
                 X_eval_lsa: np.ndarray, y_eval: pd.Series):
        """
        Takes a X_train and X_eval as numpy ndarrays, and y_train and y_eval
        as pd.Series. Creates class attributes based upon these parameters
        """
        self.X_train_lsa = X_train_lsa
        self.y_train = y_train
        self.X_eval_lsa = X_eval_lsa
        self.y_eval = y_eval
        assert len(X_train_lsa) == len(y_train)
        assert len(X_eval_lsa) == len(y_eval)
        self.categories = set(y_train)
        self.colors = sns.color_palette()

    def compare_plot_2d(self, train_title: str, eval_title: str,
                        comp_0_label: str = 'Component 0',
                        comp_1_label: str = 'Component 1'):
        """
        Takes a train_title, an eval_title, a comp_0_label (X-axis) and a
        comp_1_label (Y-axis)and returns a matplotlib figuer with two
        scatterplots based off the data based into the instance of the class.
        A plot of train on the top and a plot of the eval on the bottom.
        """
        pairs = {p: c for p, c in zip(self.categories, self.colors)}
        fig, ax = plt.subplots(2, figsize=(10, 15))
        sns.scatterplot(self.X_train_lsa[:, 0], self.X_train_lsa[:, 1],
                        hue=self.y_train, ax=ax[0], palette=pairs)
        ax[0].set_xlabel(comp_0_label)
        ax[0].set_ylabel(comp_1_label)
        ax[0].set_title(train_title)
        ax[0].legend(loc=2)

        sns.scatterplot(self.X_eval_lsa[:, 0], self.X_eval_lsa[:, 1],
                        hue=self.y_eval, ax=ax[1], palette=pairs)
        ax[1].set_xlabel(comp_0_label)
        ax[1].set_ylabel(comp_1_label)
        ax[1].set_title(eval_title)
        ax[1].legend(loc=2)

    def compare_plot_3d(self, train_title: str, eval_title: str,
                        comp_0_label: str = 'Component 0',
                        comp_1_label: str = 'Component 1',
                        comp_2_label: str = 'Component 2'):
        """
        Takes a train_title and an eval_title as strings uses them as titles to
        matplotlib 3d plots created by using class attributes.
        """
        pairs = {p: c for p, c in zip(self.categories, self.colors)}
        _train_df = pd.DataFrame(self.X_train_lsa)
        _train_df['target'] = self.y_train.values

        _temp_df = pd.DataFrame(self.X_eval_lsa)
        _temp_df['target'] = self.y_eval.values

        fig = plt.figure(figsize=(12, 15))
        ax1 = fig.add_subplot(211, projection='3d')

        for p, c in pairs.items():
            data = _train_df[_train_df.target == p]
            xs = data.loc[:, 0]
            ys = data.loc[:, 1]
            zs = data.loc[:, 2]
            ax1.scatter(xs, ys, zs, cmap='tab10', label=p, s=80)

        ax1.set_title(train_title)
        ax1.set_xlabel(comp_0_label)
        ax1.set_ylabel(comp_1_label)
        ax1.set_zlabel(comp_2_label)
        ax1.legend(loc=2)

        ax2 = fig.add_subplot(212, projection='3d')

        for p, c in pairs.items():
            evl = _temp_df[_temp_df.target == p]
            xs = evl.loc[:, 0]
            ys = evl.loc[:, 1]
            zs = evl.loc[:, 2]
            ax2.scatter(xs, ys, zs, cmap='tab10', label=p, s=80)

        ax2.set_title(eval_title)
        ax2.set_xlabel(comp_0_label)
        ax2.set_ylabel(comp_1_label)
        ax2.set_zlabel(comp_2_label)
        ax2.legend(loc=2)

    def compare_plot_interactive(self, X_train: pd.Series, X_eval: pd.Series):
        """
        Takes the self.X_train_lsa object (np.ndarray) and converts in into
        a df, makes column names based on the number of parameters adds a
        column for the lemmas from the corresponding series as well as the
        target. A colormap and plotly tooltips are created as constants. Plots
        of the first two componets are plotted.
        """
        _train = pd.DataFrame(self.X_train_lsa)
        col_names = [f'comp_{i}' for i in range(_train.shape[1])]
        _train.columns = col_names
        _train['lemmas'] = X_train.values
        _train['president'] = self.y_train.values

        _eval = pd.DataFrame(self.X_eval_lsa)
        _eval.columns = col_names
        _eval['lemmas'] = X_eval.values
        _eval['president'] = self.y_eval.values

        # Constants between plots
        hover = HoverTool(tooltips=[('President', '@president'),
                                    ('lemmas', '@lemmas'),
                                    ('Components 0 1',
                                     ('@comp_1, @comp_2'))])

        plot_cmap = factor_cmap('president', palette=Category10[10],
                                factors=_train.president.unique())

        # Train plot
        source_train = ColumnDataSource(_train)
        p_train = figure(x_axis_label='Compenent 0',
                         title='LSA on tfidf train data',
                         y_axis_label='Component 1',
                         tools=[hover, 'pan', 'wheel_zoom'])
        p_train.circle('comp_0', 'comp_1', source=source_train,
                       legend='president', fill_color=plot_cmap,
                       size=8, alpha=.5)

        p_train.legend.location = 'top_left'

        # Eval plot
        source_eval = ColumnDataSource(_eval)
        p_eval = figure(x_axis_label='Component 0', y_axis_label='Component 1',
                        tools=[hover, 'pan', 'wheel_zoom'],
                        title='LSA on tfidf eval data')
        p_eval.circle('comp_0', 'comp_1', source=source_eval,
                      legend='president',
                      fill_color=plot_cmap, size=8, alpha=.5)
        p_eval.legend.location = 'top_left'

        layout = row(p_train, p_eval)
        output_notebook()
        show(layout)


def eval_clusters(model: KMeans, X_train: pd.DataFrame, X_eval: pd.DataFrame,
                  X_holdout: pd.DataFrame):
    """
    Takes an instance of a KMeans model, train, eval, and holdout df's and
    predicts their labels and creates a df column to hold the labels. Prints
    aggregations of the train and eval by cluster. Also prints the 10 most
    importand features in each cluster as well as a set containing the common
    words between the test and the eval.
    """
    # create labels from the clustered model
    train_labels = model.predict(X_train)
    eval_labels = model.predict(X_eval)
    holdout_labels = model.predict(X_holdout)

    # add the cluster labels to the respective df
    X_train['clusters'] = train_labels
    X_eval['clusters'] = eval_labels
    X_holdout['clusters'] = holdout_labels

    # aggregate the train and eval by cluster and print
    print('Train set aggregated: \n', X_train.groupby(['clusters']).mean())
    print('\nEval set aggregated: \n', X_eval.groupby(['clusters']).mean())

    # Loop over the train and eval dfs filtered by cluster and eval simularity
    for i in range(model.n_clusters):
        data_train = X_train[X_train.clusters == i]
        data_eval = X_eval[X_eval.clusters == i]
        # cond statment to make sure cluster is not part of set of top values.
        print(f'Cluster {i}:')
        if i < 1:
            start, stop = 0, 10
        else:
            start, stop = 1, 11
        train = data_train.mean().sort_values(
            ascending=False)[start:stop].copy()
        print(train, '\n')
        evl = data_eval.mean().sort_values(ascending=False)[start:stop].copy()
        print(evl, '\n')
        overlap = {*train.index}.intersection({*evl.index})
        print(
            f'There are {len(overlap)} features which are in both training',
            'and eval.')
        if len(overlap) > 0:
            print(f'These features are {overlap}', '\n')


def make_kmeans_clusters(train_data, start: int = 2, stop: int = 11,
                         step: int = 1) -> None:
    models: List = []
    names: List = []
    silhouettes: List = []

    # convert train_data to a df
    if type(train_data) == np.ndarray:
        col_names = [f'comp_{i}' for i in range(train_data.shape[1])]
        train_data = pd.DataFrame(train_data, columns=col_names)

    for cluster in range(start, stop, step):
        models.append(('KMeans', KMeans(n_clusters=cluster, init='k-means++',
                                        random_state=15)))

        models.append(('MiniKatch', MiniBatchKMeans(n_clusters=cluster,
                                                    init='random',
                                                    batch_size=500)))

    for name, model in models:
        names.append(name)
        model.fit(train_data)
        labels = model.labels_
        print(model)
        if len(set(labels)) > 1:
            y_pred = model.fit_predict(train_data)
            silhouette = silhouette_score(train_data, labels,
                                          metric='euclidean')
            silhouettes.append(silhouette)
            if silhouette > 0:
                print(f'Clusters {model.n_clusters}\t silhoutte \
                    {silhouette}\n')
                print(name, pd.crosstab(y_pred, labels), '\n')


def fit_and_predict(model, params: Dict,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_eval: pd.DataFrame,
                    y_eval: pd.DataFrame):
    """
    Takes an instantiated sklearn model, training data (X_train, y_train),
    and performs cross-validation and then prints the mean of the cross-
    validation accuracies. Returns fitted grid-searched model.
    """
    assert len(X_train) == len(y_train)
    assert len(X_eval) == len(y_eval)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=15)
    pipe = Pipeline(steps=[('model', model)])
    clf = GridSearchCV(pipe, cv=skf, param_grid=params, n_jobs=10, iid=True)
    clf.fit(X_train, y_train)
    print('The mean cross_val accuracy on train is',
          f'{clf.cv_results_["mean_test_score"]}.')
    print('The std of the cross_val accuracy is',
          f'{clf.cv_results_["std_test_score"]}.')
    y_pred = clf.predict(X_eval)
    print(classification_report(y_eval, y_pred))
    print(confusion_matrix(y_eval, y_pred))
    return clf


def kmeans_centroid_plot(model: KMeans,
                         train_df: pd.DataFrame, eval_df: pd.DataFrame,
                         X_train: pd.Series, y_train: pd.Series,
                         X_eval: pd.Series, y_eval: pd.Series,
                         train_title: str, eval_title: str,
                         columns: List[str]) -> None:
    cluster_centers = model.cluster_centers_[:, :2]

    train_df['cluster'] = model.labels_.astype('str')
    train_df['president'] = y_train.values
    train_df['lemmas'] = X_train.values

    eval_df['cluster'] = model.predict(eval_df[columns]).astype('str')
    eval_df['president'] = y_eval.values
    eval_df['lemmas'] = X_eval.values

    hover = HoverTool(tooltips=[('Cluster', '@cluster'),
                                ('President', '@president'),
                                ('lemmas', '@lemmas')])

    cluster_cmap = factor_cmap('cluster',
                               palette=Category10[cluster_centers.shape[0]],
                               factors=eval_df.cluster.unique().astype('str'))

    source_train = ColumnDataSource(train_df)
    source_eval = ColumnDataSource(eval_df)

    fig_params = {'x_axis_label': 'Component_0', 'y_axis_label': 'Component 1',
                  'tools': [hover, 'pan', 'wheel_zoom']}

    circle_x_params = {'x': cluster_centers[:, 0], 'y': cluster_centers[:, 1],
                       'size': 30, 'alpha': .9, 'color': 'firebrick'}

    circle_params = {'x': 'comp_0', 'y': 'comp_1', 'legend': 'cluster',
                     'fill_color': cluster_cmap, 'size': 10, 'alpha': .3}

    p_train_cluster = figure(title=train_title, **fig_params)
    p_train_cluster.circle_x(**circle_x_params)
    p_train_cluster.circle(source=source_train, **circle_params)
    p_train_cluster.legend.location = 'top_left'

    p_eval_cluster = figure(title=eval_title, **fig_params)
    p_eval_cluster.circle_x(**circle_x_params)
    p_eval_cluster.circle(source=source_eval, **circle_params)
    p_eval_cluster.legend.location = 'top_left'

    layout = row(p_train_cluster, p_eval_cluster)
    show(layout)
