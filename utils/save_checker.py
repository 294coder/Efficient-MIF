# author: Zihan
# date: 2023/11/12

from dataclasses import dataclass
import numpy as np
from pathlib import Path
import shutil
from functools import partial

from utils.log_utils import easy_logger

@dataclass()
class BestMetricSaveChecker:
    _best_metric: float
    metric_name: str
    check_order: str
    
    def __init__(self, metric_name: str, check_order: str=None):
        self.metric_name = metric_name
        assert check_order in ['up', 'down', None]
        if check_order is None: check_order = self._default_setting()
        else: self.check_order = check_order
        if check_order == 'up': default_best_metric_val = -np.Inf
        else: default_best_metric_val = np.Inf

        self._best_metric = default_best_metric_val
        self._check_fn = (lambda new, old: new > old) if check_order=='up' else \
                         (lambda new, old: new <= old)
                         
    def _default_setting(self):
        metric_name = self.metric_name.lower()
        _default_dict = {
            'psnr': 'up',
            'ssim': 'up',
            'sam': 'down',
            'ergas': 'down',
            'cc': 'up',
            'scc': 'up',
        }
        if check_order := _default_dict.get(metric_name):
            if check_order is None:
                raise ValueError(f'No default setting for metric {metric_name}, ' +
                                    'you should provide @check_order manually')
            else:
                return check_order
        
    def __call__(self, val_metrics: dict[str, float], *args):
        assert self.metric_name in val_metrics.keys(), \
            f'@metric_name {self.metric_name} should in @val_metrics, but got {val_metrics}'
        new_val = val_metrics[self.metric_name]
        prev_val = self._best_metric
        
        _save = self._check_fn(new_val, prev_val)
        if _save: self._best_metric = new_val
        
        return _save
    
    @property
    def best_metric(self):
        return self._best_metric
    
    
# TODO: need test
class SavedWeightsNumMonitor:
    def __init__(self,
                 saved_weight_path: str,
                 glob_pattern: str='ep*/',
                 sort_saved_fn: callable=None,
                 kept_num: int=4,
                 rm_verbose: bool=False):
        self.saved_weight_path = Path(saved_weight_path)
        self.glob_pattern = glob_pattern
        self.sort_saved_fn = sort_saved_fn
        self.kept_num = kept_num
        self.rm_verbose = rm_verbose
        if rm_verbose:
            self.logger = easy_logger()
    
    @property
    def saved_weight_path(self):
        return self.saved_weight_path.glob(self.glob_pattern)
        
    @property
    def saved_num(self):
        # excluse .pth in the base dir
        return len(list(self.saved_weight_path))
    
    def rm_for_keep_k(self):
        if self.saved_num > self.kept_num:
            saved_name = list(map(lambda x: x.name, self.saved_weight_path))
            sort_fn = self.sort_saved_fn if self.sort_saved_fn is not None else \
                      partial(sorted, key=lambda x: int(x))
            sorted_saved_name = sort_fn(saved_name)
            rm_saved_name = sorted_saved_name[self.kept_num:]
            
            for p in rm_saved_name:
                shutil.rmtree(self.saved_weight_path / p)
                if self.rm_verbose: 
                    self.logger.print(f'remove {p}')
                    
    def __call__(self):
        self.rm_for_keep_k()
    
    
if __name__ == '__main__':
    checker = BestMetricSaveChecker('sam', 'down')
    
    val_d1 = {'sam': 2.3, 'psnr': 10, 'ssim':0.8}
    val_d2 = {'sam': 2.4, 'psnr': 12, 'ssim':0.9}
    
    print(checker.best_metric)
    print(checker.metric_name)
    
    print(checker(val_d1))
    print(checker.best_metric)
    
    print(checker(val_d2))
    print(checker.best_metric)