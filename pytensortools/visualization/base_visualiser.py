from abc import ABC, abstractmethod
from .. import visualization
from ..evaluation.base_evaluator import BaseEvaluator
import pytensor
import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib as mpl

from plottools.fMRI.tile_plots import create_fmri_factor_plot
import plottools
from scipy.stats import ttest_ind

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

    def _visualise_mode(self, data_reader, factor_matrices, ax, mode, label=None,):
        factor = factor_matrices[mode]
        
        if self.normalise:
            factor = factor/np.linalg.norm(factor, axis=0, keepdims=True)

        ax.plot(factor)
        ax.set_title(f'Mode {mode + 1}')
        if (data_reader.mode_names is not None) and (len(data_reader.mode_names) > mode):
            ax.set_title(data_reader.mode_names[mode])

        ax.set_xlabel(label)
        
        if self.show_legend:
            letter = string.ascii_lowercase[mode]
            ax.legend([f'{letter}{i}' for i in range(factor.shape[1])], loc='upper right')


    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor_matrices = self.load_final_checkpoint(h5)

        num_cols = len(self.modes)
        for i, mode in enumerate(self.modes):
            ax = fig.add_subplot(1, num_cols, i+1)

            label = None
            if self.labels is not None:
                label = self.labels[i]

            self._visualise_mode(data_reader, factor_matrices, ax, mode, label=label)

        return fig

class ClassLinePlotter(BaseVisualiser):
    _name = 'ClassLinePlotter'

    def __init__(self, summary, mode, class_name, filename=None, figsize=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.mode = mode
        self.class_name = class_name
    
    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        ax = fig.add_subplot(111)
        factor_matrices = self.load_final_checkpoint(h5)

        ax.plot(factor_matrices[self.mode])
        ylim = ax.get_ylim()

        classes = data_reader.classes[self.mode][self.class_name].squeeze()
        unique_classes = np.unique(classes)
        class_id = {c: i for i, c in enumerate(np.unique(classes))}
        classes = np.array([class_id[c] for c in classes])

        diff = classes[1:] - classes[:-1]
        for i, di in enumerate(diff):
            if di != 0:
                ax.plot([i + 0.5, i + 0.5], ylim, 'r')
        
        ax.set_ylim(ylim)
        return fig


# TODO: BaseSingleComponentPlotter
class SingleComponentLinePlotter(BaseVisualiser):
    _name = "single_factor_lineplot"
    def __init__(self, summary, mode, normalise=True, common_axis=True, label=None, filename=None, figsize=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.mode = mode
        self.normalise = normalise
        self.label = label
        self.common_axis = common_axis
        self.figsize = (self.figsize[0]*summary['model_rank']*0.7, self.figsize[1])
    
    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor = self.load_final_checkpoint(h5)[self.mode]
        rank = factor.shape[1]

        if self.normalise:
            factor = factor/np.linalg.norm(factor, axis=0, keepdims=True)

        self.figsize = (self.figsize[0]*rank, self.figsize[1])


        for r in range(rank):
            ax = fig.add_subplot(1, rank, r+1)

            ax.plot(factor[:, r])
            
            if (data_reader.mode_names is not None) and len(data_reader.mode_names) > self.mode:
                ax.set_xlabel(data_reader.mode_names[self.mode])
            if self.label is not None:
                ax.set_xlabel(self.label)
            
            if r == 0:
                ax.set_ylabel('Factor')
            
            ax.set_title(f'Component {r + 1}')

        return fig

class FactorScatterPlotter(BaseVisualiser):
    """Note: only works for two classes"""
    _name = 'factor_scatterplot'
    def __init__(self, summary, mode, class_name, normalise=True, common_axis=True, label=None, legend=None, include_p_value=False, filename=None, figsize=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.mode = mode
        self.normalise = normalise
        self.label = label
        self.legend = legend
        self.common_axis = common_axis
        self.class_name = class_name
        self.figsize = (self.figsize[0]*summary['model_rank']*0.7, self.figsize[1])
        self.include_p_value = include_p_value

    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor = self.load_final_checkpoint(h5)[self.mode]
        rank = factor.shape[1]

        if self.normalise:
            factor = factor/np.linalg.norm(factor, axis=0, keepdims=True)

        self.figsize = (self.figsize[0]*rank, self.figsize[1])

        x_values = np.arange(factor.shape[0])

        classes = data_reader.classes[self.mode][self.class_name].squeeze()


            
        if self.include_p_value:
            assert len(set(classes)) == 2
            indices = [[i for i, c in enumerate(classes) if c == class_] for class_ in set(classes)]
            p_values = tuple(ttest_ind(factor[indices[0]], factor[indices[1]], equal_var=False).pvalue)
        #assert len(set(classes)) == 2

        different_classes = np.unique(classes)
        #class1 = different_classes[0]
        #class2 = different_classes[1]

        for r in range(rank):
            ax = fig.add_subplot(1, rank, r+1)

            for c in different_classes:
                ax.scatter(x_values[classes==c], factor[classes==c, r], label=c)
            
            #ax.scatter(x_values[classes==class1], factor[classes==class1, r], color='tomato')
            #ax.scatter(x_values[classes==class2], factor[classes==class2, r], color='darkslateblue')
            
            if (data_reader.mode_names is not None) and len(data_reader.mode_names) > self.mode:
                ax.set_xlabel(data_reader.mode_names[self.mode])
            if self.label is not None:
                ax.set_xlabel(self.label)
            
            if r == 0:
                ax.set_ylabel('Factor')
            elif self.common_axis:
                ax.set_yticks([])
            
            if self.include_p_value:
                assert len(set(classes)) == 2
                ax.set_title(f'Component {r + 1}, p-value: {p_values[r]:5.2e}')
            else:
                ax.set_title(f'Component {r + 1}')


            if self.common_axis:
                fmin = factor.min()
                fmax = factor.max()
                df = fmax - fmin
                ax.set_ylim(fmin - 0.01*df, fmax + 0.01*df)

            if self.legend is not None: 
                ax.legend(self.legend)
            else:
                ax.legend()

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

class FactorfMRIImage(BaseVisualiser):
    def __init__(
        self, 
        summary, 
        mode, 
        mask_path,
        template_path,
        filename=None, 
        figsize=None, 
        tile_plot_kwargs=None
        ):

        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.mode = mode
        self.tile_plot_kwargs = tile_plot_kwargs
        self.mask_path = mask_path
        self.template_path = template_path
        if figsize is None:
            figsize = (self.figsize[0]*summary['model_rank']*0.7, self.figsize[1])
        self.figsize = figsize

        if tile_plot_kwargs is None:
            tile_plot_kwargs = {}
        self.tile_plot_kwargs = tile_plot_kwargs

    def _visualise(self, data_reader, h5):
        factor = self.load_final_checkpoint(h5)[self.mode]

        mask = plottools.fMRI.base.load_mask(self.mask_path)
        template = plottools.fMRI.base.load_template(self.template_path)

        fig, axes = plt.subplots(1, self.summary['model_rank'], figsize=self.figsize)
        for i, ax in enumerate(axes):
            ax.set_title(f'Component {i + 1}')
            fmri_factor = plottools.fMRI.base.get_fMRI_images(factor[:, i], mask, axis=0)
            create_fmri_factor_plot(fmri_factor, template, ax=ax, **self.tile_plot_kwargs)

        if (self.tile_plot_kwargs is not None) and ('threshold' in self.tile_plot_kwargs):
            fig.suptitle(f'Threshold = {self.tile_plot_kwargs["threshold"]}')
        return fig

class EvolvingFactorfMRIImage(FactorfMRIImage):
    def __init__(
        self, 
        summary, 
        mode, 
        mask_path,
        template_path,
        filename=None, 
        figsize=None, 
        tile_plot_kwargs=None,
        component=0
        ):

        super().__init__(summary=summary,
                         mode=mode,
                         mask_path=mask_path,
                         template_path=template_path,
                         filename=filename,
                         figsize=figsize,
                         tile_plot_kwargs=tile_plot_kwargs)

        self.component = component

    def _visualise(self, data_reader, h5):
        
        factor = self.load_final_checkpoint(h5)[self.mode]

        mask = plottools.fMRI.base.load_mask(self.mask_path)
        template = plottools.fMRI.base.load_template(self.template_path)
        fmri_factor = plottools.fMRI.base.get_fMRI_images(factor[:, self.component], mask, axis=0)
        fig = plottools.fMRI.base.create_fmri_evolving_factor_plot(
            fmri_factor, template, **self.tile_plot_kwargs
        )[0]

        return fig

class ResidualHistogram(BaseVisualiser):
    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor_matrices = self.load_final_checkpoint(h5)
        tensor = data_reader.tensor
        #TODO: will not work for parafac2
        predicted_tensor = self.DecomposerType(factor_matrices).construct_tensor()

        residuals = tensor.ravel() - predicted_tensor.ravel()

        ax = fig.add_subplot(111)
        ax.hist(residuals)
        return fig

class LeverageScatterPlot(BaseVisualiser):

    def __init__(self, summary, mode, filename=None, figsize=None, annotation=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.mode = mode
        self.annotation = annotation
        self.figsize = (self.figsize[0]*summary['model_rank']*0.7, self.figsize[1])

    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor = self.load_final_checkpoint(h5)[self.mode]
        decomposition = self.load_final_checkpoint(h5)
        rank = decomposition.rank

        

        leverage_scores = pytensor.metrics.leverages(factor)
        predicted_tensor = decomposition.construct_tensor()

        residuals = np.sum((predicted_tensor-data_reader.tensor)**2, (1,2))
        ax = fig.add_subplot(111)
        ax.plot(residuals, leverage_scores)

        if self.annotation is not None:
            labels = data_reader.classes[self.annotation]
            for label, l, r in zip(labels, leverage_scores, residuals):
                ax.annotate(label, [l, r])

        return fig


class EvolvingComponentMatrixMap(BaseVisualiser):

    def __init__(self, summary, class_name, filename=None, figsize=None):
        super().__init__(summary=summary, filename=filename, figsize=figsize)
        self.class_name = class_name
        self.figsize = (self.figsize[0]*summary['model_rank']*0.7, self.figsize[1])
    
    def _visualise(self, data_reader, h5):
        fig = self.create_figure()
        factor = np.array(self.load_final_checkpoint(h5)[1])

        classes = data_reader.classes[2][self.class_name].squeeze()

        unique_classes = np.unique(classes)

        fig, axes = plt.subplots(1, self.summary['model_rank'], figsize=self.figsize)
        for i, ax in enumerate(axes):
            ax.set_title(f'Component {i + 1}')

            component = factor[..., i]
            sorted_component = np.empty_like(component)

            offset = 0
            for c in unique_classes:
                class_idx = classes==c
                sorted_component[offset:offset+sum(class_idx)] = component[class_idx]
                offset += sum(class_idx)

            ax.imshow(sorted_component) 

        return fig