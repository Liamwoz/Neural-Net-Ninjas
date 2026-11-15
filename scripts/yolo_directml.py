"""Experimental DirectML bridge for the pinned Ultralytics detection trainer.

The model runs on DirectML; detection loss and validation run on CPU because
DirectML lacks operators required by YOLO26. Patches are scoped to training.
"""
from contextlib import contextmanager

import torch
import torch_directml
import ultralytics
import ultralytics.engine.trainer as trainer_module
from ultralytics.nn.tasks import BaseModel


def select_amd_device():
    devices = [(i, torch_directml.device_name(i).rstrip('\x00'))
               for i in range(torch_directml.device_count())]
    matches = [(i, name) for i, name in devices if '6800' in name.lower()]
    if len(matches) != 1:
        raise RuntimeError(f'Could not uniquely identify the RX 6800 XT. DirectML devices: {devices}')
    index, name = matches[0]
    device = torch_directml.device(index)
    print(f'DirectML training device: {name} ({device})')
    return device


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()  # Keep autograd connected to GPU predictions.
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _cpu_loss(model, batch, preds=None):
    if getattr(model, 'criterion', None) is None:
        model.criterion = model.init_criterion()
        for criterion in (model.criterion.one2many, model.criterion.one2one):
            criterion.device = torch.device('cpu')
            criterion.stride = criterion.stride.cpu()
            criterion.proj = criterion.proj.cpu()
            criterion.bbox_loss = criterion.bbox_loss.cpu()
            if criterion.class_weights is not None:
                criterion.class_weights = criterion.class_weights.cpu()
    if preds is None:
        preds = model.forward(batch['img'])
    return model.criterion(_cpu_tree(preds), _cpu_tree(batch))


@contextmanager
def directml_training(device):
    """Temporarily adapt the pinned trainer; use device='cpu', amp=False in train()."""
    if ultralytics.__version__ != '8.4.212' or torch.__version__.split('+')[0] != '2.4.1':
        raise RuntimeError('DirectML bridge requires ultralytics 8.4.212 and PyTorch 2.4.1; install requirements-directml.txt.')
    cls = trainer_module.BaseTrainer
    original = (trainer_module.select_device, cls._get_memory, cls._clear_memory, cls.validate, BaseModel.loss)
    select, get_memory, clear_memory, validate, loss = original

    def cpu_validate(self):
        self.ema.ema.cpu()
        self.device = torch.device('cpu')
        try:
            return validate(self)
        finally:
            self.device = device
            self.ema.ema.to(device)
            # Moving the EMA replaces its tensors. Ultralytics caches references
            # for updates, so rebuild them on the next optimizer step.
            self.ema._pairs = None

    trainer_module.select_device = lambda value: device if value == 'cpu' else select(value)
    cls._get_memory = lambda self, fraction=False: 0
    cls._clear_memory = lambda self, threshold=None: None
    cls.validate = cpu_validate
    BaseModel.loss = _cpu_loss
    try:
        yield
    finally:
        trainer_module.select_device, cls._get_memory, cls._clear_memory, cls.validate, BaseModel.loss = original
