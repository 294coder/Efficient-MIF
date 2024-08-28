# author: Zihan
# date: 2023/11/12

from dataclasses import dataclass
import numpy as np
from pathlib import Path
import shutil
from functools import partial

import sys
sys.path.append('./')

from utils.log_utils import easy_logger
from utils.misc import is_none

logger = easy_logger(func_name='save_checker')

@dataclass()
class BestMetricSaveChecker:
    _best_metric: "float | str"
    metric_name: "str | list[str]"
    check_order: str
    
    def __init__(self,
                 metric_name: "str | list[str] | dict[str, float]"=None,
                 check_order: str=None,
                 *,
                 avg_metrics_weights: dict[str, float]=None):
        # overide the avg_metrics_weights if it is None
        if is_none(avg_metrics_weights):
            if isinstance(metric_name, (list, tuple)):
                self.metric_name = metric_name
                avg_metrics_weights = {m: 1. for m in metric_name}
            elif isinstance(metric_name, dict):
                self.metric_name = list(metric_name.keys())
                avg_metrics_weights = metric_name
            elif isinstance(metric_name, str):
                self.metric_name = metric_name
                avg_metrics_weights = {metric_name: 1.}
            else:
                raise ValueError(f'@metric_name should be a str or a list of str or a dict of str->float but got {metric_name}')
        else:
            self.metric_name = metric_name
                
        if is_none(metric_name):
            logger.warning(f'No metric name provided, and SaveChecker will alway return True')
        
        assert check_order in ['up', 'down', None]
        if check_order is None: 
            check_order = self._default_setting()
        self.check_order = check_order
        
        self.avg_metrics_weights = avg_metrics_weights
        self.avg_metrics = isinstance(self.metric_name, list) and avg_metrics_weights is not None
        logger.info(f'SaveChecker initialized with metric_name={self.metric_name}, ',
                    f'check_order={check_order}, avg_metrics_weights={avg_metrics_weights}')
        
        if isinstance(metric_name, list):
            assert not is_none(avg_metrics_weights), \
                f'@avg_metrics_weights should not be None when @metric_name is a list'
            for m in metric_name:
                assert m in avg_metrics_weights.keys(), \
                    f'@avg_metrics_weights should have key {m} but got {avg_metrics_weights}'
        
        if check_order == 'up': 
            default_best_metric_val = -np.Inf
        elif check_order == 'down': 
            default_best_metric_val = np.Inf
        else:
            default_best_metric_val = 'none'

        self._best_metric = default_best_metric_val
        self._check_fn = (lambda new, old: new > old) if check_order=='up' else \
                         (lambda new, old: new <= old)
                         
    def _default_setting(self):
        metric_name = self.metric_name.lower()
        if is_none(metric_name):
            return 'none'
        
        _default_dict = {
            'psnr': 'up',
            'ssim': 'up',
            'sam': 'down',
            'ergas': 'down',
            'cc': 'up',
            'scc': 'up',
            'sd': 'up',
            'en': 'up',
            'sf': 'up',
            'vif': 'up',
            'mse': 'down',
            'ag': 'up',
            'mi': 'up',
        }
        if check_order := _default_dict.get(metric_name):
            if check_order is None:
                raise ValueError(f'No default setting for metric {metric_name}, ' +
                                    'you should provide @check_order manually')
            else:
                return check_order
        
    def __call__(self, val_metrics: dict[str, float], *args):
        if is_none(self.metric_name):
            return True
        
        if self.avg_metrics:
            for m in self.metric_name:
                assert m in val_metrics.keys(), f'@val_metrics should have key {m} but got {val_metrics}'
        else:
            assert self.metric_name in val_metrics.keys(), f'@val_metrics should have key {self.metric_name} but got {val_metrics}'
        
        if not self.avg_metrics:
            new_val = val_metrics[self.metric_name]
        else:
            new_val = np.sum([val_metrics[k] * self.avg_metrics_weights[k] for k in self.metric_name])
        
        prev_val = self._best_metric
        _save = self._check_fn(new_val, prev_val)
        if _save: 
            self._best_metric = new_val
        
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
    checker = BestMetricSaveChecker({"psnr": 0.1, "ssim": 0.2}, 'up')
    
    val_d1 = {'sam': 2.3, 'psnr': 10, 'ssim':0.8}
    val_d2 = {'sam': 2.4, 'psnr': 12, 'ssim':0.9}
    
    print(checker.best_metric)
    print(checker.metric_name)
    
    print(checker(val_d1))
    print(checker.best_metric)
    
    print(checker(val_d2))
    print(checker.best_metric)