from abc import ABC, abstractmethod, abstractproperty
import os
from itertools import product
from pathlib import Path
from subprocess import Popen
from tempfile import TemporaryDirectory
import warnings

import h5py
import numpy as np
from hdf5storage import savemat as savemat_73
from scipy.io import loadmat, savemat
from scipy.stats import ttest_ind

# from . import preprocessor

# en liste med classes for hver mode?
# En liste med dictionaries med labelinfo som f.eks. site for hver mode?
# - kanskje vi kan putte on off som labels her også?



def _to_string_list(iterable):
    """Convert an iterable to a 1D numpy string array.
    
    Used in savemat
    """
    return np.squeeze(np.array(iterable)).astype(str).astype(object)


class ClassID:
    """Translate class ID to MATLAB style class ids.

    Input: 
        Dict that maps label name to label values
    Output:
        Dict that maps label name to list of unique values (to look up label value)
        and array of numbers instead of label values. The number array starts at
        one to be MATLAB friendly.
    """
    def __init__(self, class_dict):
        self.class_dict = class_dict

    def __getitem__(self, item):
        if item not in self.class_dict:
            raise KeyError(
                f"{item} is not the name of a class. Cannot create ClassID array."
            )

        classes = np.squeeze(self.class_dict[item])
        unique_values = np.unique(classes)

        name_to_id = {name: i + 1 for i, name in enumerate(np.squeeze(unique_values))}
        # id_to_name = [name for name in name_to_id.values()]

        id_array = np.array([name_to_id[i] for i in classes])

        return unique_values, id_array


class ClassIDs:
    def __init__(self, class_dicts):
        self.class_dicts = class_dicts

    def __getitem__(self, item):
        return ClassID(self.class_dicts[item])


class BaseDataReader(ABC):
    """

    Attributes
    ----------
    tensor : np.ndarray
    classes : List[Dict[str, np.ndarray]]
        List with class dictionaries, one dict per mode. Keys are class names and values are
        numpy arrays denoting which class of the corresponding fiber in the data tensor.
    mode_names : List[str]
        List of names of the modes in the tensor
    """

    @abstractmethod
    def __init__(self, mode_names=None):
        self._tensor = None
        self._classes = None
        self.mode_names = mode_names

    @property
    def tensor(self):
        return self._tensor

    @property
    def classes(self):
        return self._classes

    @property
    def class_ids(self):
        return ClassIDs(self.classes)

    def to_matlab(self, label_names, outfile):
        """Convert dataset to PLS toolbox style dataset.

        Parameters
        ----------
        label_names : List
            List of lists. Each inner list contains the names of the classes that should
            be regarded as labels in the MATLAB dataset.
        outfile : str
            Location of the saved file.
        """
        if len(label_names) < len(self.tensor.shape):
            label_names = label_names + [[]] * (
                len(self.tensor.shape) - len(label_names)
            )
        # Divide self.classes in labels and classes
        labels = []
        classes = []
        class_names = []
        class_ids = []
        for mode, mode_label_names in enumerate(label_names):
            labels_ = self.classes[mode]
            classes_ = self.class_ids[mode]

            mode_class_names = list(
                set(self.classes[mode].keys()) - set(mode_label_names)
            )
            mode_label_names = label_names[mode]
            class_names.append(mode_class_names)

            labels.append(
                [
                    _to_string_list(labels_[label_name])
                    for label_name in mode_label_names
                ]
            )
            classes.append([classes_[class_name][1] for class_name in mode_class_names])
            class_ids.append(
                [
                    _to_string_list(labels_[class_name])
                    for class_name in mode_class_names
                ]
            )

        # Save matlab file
        tensor_matfile = {"tensor": self.tensor}

        def to_cell_array(arr):
            return np.array([np.array(i, dtype=object) for i in arr], dtype=object)

        metadata_matfile = {
            "mode_titles": np.array(self.mode_names, dtype=object),
            "class_names": np.array(class_names, dtype=object),
            "classes": np.array(classes),
            "label_names": np.array(label_names, dtype=object),
            "labels": np.array(labels, dtype=object),
            "class_ids": np.array(class_ids, dtype=object),
        }
        print(metadata_matfile)

        with TemporaryDirectory() as tempdir:
            tensorfile = Path(tempdir) / "tensor.mat"
            metafile = Path(tempdir) / "meta.mat"
            savemat_73(str(tensorfile), tensor_matfile)
            savemat(metafile, metadata_matfile)

            matlab_script = f'load("{tensorfile}");load("{metafile}");outfile="{outfile}";{MATLAB_CREATE_DATASET}'
            matlab_script = matlab_script.replace("\n", "")

            p = Popen(["matlab", "-nosplash", "-nodesktop", "-r", matlab_script])
            print(p.communicate())
            print(f"Stored file in {outfile}")


class NumpyDataReader(BaseDataReader):
    def __init__(self, tensor, classes, mode_names=None):
        super().__init__(mode_names=mode_names)
        self._tensor = tensor
        self._classes = classes

class MatlabDataReader(BaseDataReader):
    def __init__(self, file_path, tensor_name, classes=None, mode_names=None):
        """Example:
            dataset = MatlabDataReader('./data.mat', 'data', [{}, {'schizophrenia': 'classes'}, {}], ['voxel', 'patient', 'time'])
        """
        super().__init__(mode_names=mode_names)
        self.file_path = file_path
        self._tensor = np.array(
            loadmat(file_path, variable_names=[tensor_name])[tensor_name]
        )

        if classes is not None:
            self._classes = [{} for _ in self._tensor.shape]
            for class_dict, mode_classes in zip(self._classes, classes):
                for name, varname in mode_classes.items():
                    class_dict[name] = np.array(
                        loadmat(file_path, variable_names=[varname])[varname]
                    )


class HDF5DataReader(BaseDataReader):
    def _load_h5_dataset(self, file_path, dataset):
        with h5py.File(file_path, "r") as h5:
            return h5[dataset][...]

    def _load_data_tensor(self, file_path, tensor_name):
        return self._load_h5_dataset(file_path, tensor_name)

    def _load_class(self, file_path, class_name):
        return self._load_h5_dataset(file_path, class_name)

    def _load_meta_data(self, file_path, classes):
        _classes = [{} for _ in self._tensor.shape]
        for class_dict, mode_classes in zip(_classes, classes):
            for name, varname in mode_classes.items():
                class_dict[name] = self._load_class(file_path, varname)
        return _classes

    def __init__(
        self,
        file_path,
        tensor_name,
        meta_info_path=None,
        classes=None,
        mode_names=None,
        transpose=True,
    ):
        """classes: list of dicts, class name maps to variable name in the h5 file.
        """
        super().__init__(mode_names=mode_names)
        self.file_path = file_path

        if meta_info_path is None:
            meta_info_path = file_path

        self.meta_info_path = meta_info_path

        self._tensor = self._load_data_tensor(self.file_path, tensor_name)
        self.transpose = transpose
        if self.transpose:
            self._tensor = self._tensor.T

        if classes is not None:
            self._classes = self._load_meta_data(self.meta_info_path, classes)


def load_strrefs(h5, strref):
    strdata = h5[strref][:].T
    if len(strdata.shape) != 2:
        strdata = strdata[np.newaxis]
    return [
        ''.join(chr(c) for c in strdatum) for strdatum in strdata
    ]


def load_strref(h5, strref):
    return load_strrefs(h5, strref)[0]


class PLSDataReader(BaseDataReader):
    def _get_modenames(self, file_path):
        with h5py.File(file_path, 'r') as h5:
            refs = h5['data/title'][0]
            return [load_strref(h5, ref) for ref in refs]
    
    def _load_tensor(self, file_path):
        with h5py.File(file_path, 'r') as h5:
            return h5['data/data'][:]

    def _load_classes(self, file_path):
        with h5py.File(file_path, 'r') as h5:
            label_refs = h5['data/label'][:]
            labeldata = [
                [
                    [None for _ in range(label_refs.shape[2])]
                    for _ in range(label_refs.shape[1])
                ] for _ in range(label_refs.shape[0])
            ]

            # Iterate over all possible indices for label_refs
            for i, j, k in product(*map(range, label_refs.shape)):
                labeldata[i][j][k] = load_strrefs(h5, label_refs[i, j, k])
        
        labels = [{} for _ in enumerate(labeldata[0][0])]
        for label in labeldata:
            for mode, (label_content, label_name) in enumerate(zip(*label)):
                if label_name[0] == '\x00\x00':
                    continue
                labels[mode][label_name[0]] = label_content
        return labels
        
    def __init__(self, file_path):
        self.file_path = file_path

        mode_names = self._get_modenames(file_path)
        super().__init__(mode_names)

        self._tensor = self._load_tensor(file_path)
        self._classes = self._load_classes(file_path)

        



# Used to convert to MATLAB style dataset.
if 'MATLAB_TOOLBOX_PATH' in os.environ:
    MATLAB_TOOLBOX_PATH = str(os.environ['MATLAB_TOOLBOX_PATH'])
else:
    MATLAB_TOOLBOX_PATH = "../../matlab/toolboxes/"
    warnings.warn(
        f"MATLAB_TOOLBOX_PATH system variable is not set. Using {MATLAB_TOOLBOX_PATH} instead",
        RuntimeWarning
    )


MATLAB_CREATE_DATASET = f"""
disp('Adding to path');
addpath(genpath('{MATLAB_TOOLBOX_PATH}'));
disp('Added to path');
data = dataset(tensor);

for i = 1:length(mode_titles)
    data.title{{i}} = mode_titles{{i}};
end;
disp('Setting labels');
for i = 1:length(mode_titles)
    disp('mode');
    disp(i);

    labels_ = labels{{i}};
    disp('labels');
    if size(labels_, 1) == 1
        data.label{{i, 1}} = cellstr(labels_);
    elseif size(labels_, 1) > 1
        for j = 1:size(labels_, 1)
            data.label{{i,j}} = cellstr(labels_(j, :));
        end;
    end;

    disp('labelnames');
    label_names_ = label_names{{i}};
    if size(label_names_, 1) == 1
        data.labelname{{i,1}} = cellstr(label_names_);
    elseif size(label_names_, 1) > 1
        for j = 1:size(label_names_, 1)
            data.labelname{{i,j}} = cellstr(label_names_(j, :));
        end;
    end;

    disp('classes');
    classes_ = classes{{i}};
    class_names_ = class_names{{i}};
    class_ids_ = class_ids{{i}};
    if size(classes_, 1) == 1
        disp('singleton');
        data.class{{i, 1}} = squeeze(double(classes_));
        data.classname{{i, 1}} = squeeze(class_names_);
        data.classid{{i, 1}} = cellstr(class_ids_);
    elseif size(classes_, 1) > 1
        disp('multiple');
        for j = 1:size(classes_, 1)
            disp('class');
            data.class{{i,j}} = double(classes_(j, 1:end));
            disp('classname');
            data.classname{{i,j}} = class_names_(j, 1:end);
            disp('classid');
            data.classid{{i,j}} = cellstr(class_ids_(j, 1:end));
        end;
    end;
end;

save(outfile, 'data', '-v7.3');
disp(size(tensor));
exit
"""
