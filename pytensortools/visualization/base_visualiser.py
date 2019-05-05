from abc import ABC, abstractmethod
from .. import visualization
from ..evaluation.base_evaluator import BaseEvaluator
import pytensor
import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib as mpl


mpl.rcParams['font.family'] = 'PT Sans'

def create_visualiser(visualiser_params, summary):
    Visualiser = getattr(visualization, visualiser_params['type'])
    kwargs = visualiser_params.get('arguments', {})
    return Visualiser(summary=summary, **kwargs)


def create_visualisers(visualisers_params, summary):
    visualisers = []
    for visualiser_params in visualisers_params:
        visualisers.append(create_visualiser(visualiser_params, summary))
    return visualisers


class BaseVisualiser(BaseEvaluator):
    figsize = (5.91, 3.8)
    _name = 'visualisation'
    def __init__(self, summary, filename=None, figsize=None):
        self.summary = summary
        self.DecomposerType = getattr(pytensor.decomposition, summary['model_type'])
        self.DecompositionType = self.DecomposerType.DecompositionType

        if figsize is not None:
            self.figsize = figsize
        
        if filename is not None:
            self._name = filename

    def __call__(self, data_reader, h5):
        return self._visualise(data_reader, h5)

    @abstractmethod
    def _visualise(self, data_reader, h5):
        pass

    def create_figure(self, *args, **kwargs):
        return plt.figure(*args, figsize=self.figsize, **kwargs)


class FactorLinePlotter(BaseVisualiser):
    _name = 'factor_lineplot'
    def __init__(self, summary, modes, normalise=True, labels=None, show_legend=True, filename=None, figsize=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.modes = modes
        self.figsize = (self.figsize[0]*len(modes), self.figsize[1])
        self.labels = labels
        self.show_legend = show_legend
        self.normalise = normalise

    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor_matrices = self.load_final_checkpoint(h5)

        num_cols = len(self.modes)
        for i, mode in enumerate(self.modes):
            ax = fig.add_subplot(1, num_cols, i+1)
            factor = factor_matrices[mode]
            
            if self.normalise:
                factor = factor/np.linalg.norm(factor, axis=0, keepdims=True)

            ax.plot(factor)
            ax.set_title(f'Mode {mode}')
            if (data_reader.mode_names is not None) and (len(data_reader.mode_names) > mode):
                ax.set_title(data_reader.mode_names[mode])
            if self.labels is not None:
                ax.set_xlabel(self.labels[i])
            
            if self.show_legend:
                letter = string.ascii_lowercase[mode]
                ax.legend([f'{letter}{i}' for i in range(factor.shape[1])], loc='upper right')

        return fig


class FactorScatterPlotter(BaseVisualiser):
    """Note: only works for two classes"""
    _name = 'factor_scatterplot'
    def __init__(self, summary, mode, normalise=True, common_axis=True, label=None, legend=None, filename=None, figsize=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.mode = mode
        self.normalise = normalise
        self.label = label
        self.legend = legend
        self.common_axis = common_axis
        self.figsize = (self.figsize[0]*summary['model_rank']*0.7, self.figsize[1])

    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor = self.load_final_checkpoint(h5)[self.mode]
        rank = factor.shape[1]

        if self.normalise:
            factor = factor/np.linalg.norm(factor, axis=0, keepdims=True)

        self.figsize = (self.figsize[0]*rank, self.figsize[1])

        x_values = np.arange(factor.shape[0])

        classes = data_reader.classes

        assert len(set(classes)) == 2

        different_classes = np.unique(classes)
        class1 = different_classes[0]
        class2 = different_classes[1]

        for r in range(rank):
            ax = fig.add_subplot(1, rank, r+1)

            ax.scatter(x_values[classes==class1], factor[classes==class1, r], color='tomato')
            ax.scatter(x_values[classes==class2], factor[classes==class2, r], color='darkslateblue')
            
            if (data_reader.mode_names is not None) and len(data_reader.mode_names) > self.mode:
                ax.set_xlabel(data_reader.mode_names[self.mode])
            if self.label is not None:
                ax.set_xlabel(self.label)
            
            if r == 0:
                ax.set_ylabel('Factor')
            elif self.common_axis:
                ax.set_yticks([])
            
            ax.set_title(f'Component {r}')

            if self.common_axis:
                fmin = factor.min()
                fmax = factor.max()
                df = fmax - fmin
                ax.set_ylim(fmin - 0.01*df, fmax + 0.01*df)

            if self.legend is not None: 
                ax.legend(self.legend)

        return fig


class LogPlotter(BaseVisualiser):
    _name = 'logplot'
    def __init__(self, summary, logger_name, log_name=None, filename=None, figsize=None,):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.logger_name = logger_name
        self.log_name = log_name
    
    def _visualise(self, data_reader, h5):
        its = h5[f'{self.logger_name}/iterations'][...]
        values = h5[f'{self.logger_name}/values'][...]

        fig = self.create_figure()
        ax = fig.add_subplot(111)
        ax.plot(its, values)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title(self.log_name)

        return fig
