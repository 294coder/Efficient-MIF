import os
import warnings
import warnings
warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")

os.environ["MPLCONFIGDIR"] = ".cache/.hypothesis/"
os.environ["HF_HOME"] = ".cache/transformers"
os.environ["MPLCONFIGDIR"] = ".cache/matplotlib"

from dataclasses import dataclass
import sys
sys.path.append('./')

from utils._metric_legacy import *
from utils.metric import *
from utils.metric_VIF import *
from utils.print_helper import *
from utils.log_utils import *
from utils.log_utils import TrainStatusLogger as TrainProcessTracker
from utils.misc import *
from utils.misc import recursive_search_dict2namespace as convert_config_dict
from utils.load_params import *
from utils.optim_utils import *
from utils.network_utils import *
from utils.visualize import *
from utils.inference_helper_func import *
from utils.loss_utils import *
from utils.save_checker import *
from utils.train_utils import *
from utils.progress_utils import *

config_load = yaml_load


### basic constants
@dataclass
class BasicConstants:
    TRAIN: bool = True
    

if __name__ == "__main__":
    # import torch

    # a = torch.randn(3, 3, 256, 256)
    # b = torch.randn(3, 3, 256, 256)
    # analysis = AnalysisPanAcc()
    # d = analysis(a, b)
    # print(analysis.print_str())

    # d = analysis(a, b)
    # print(analysis.print_str())
    
    print(BasicConstants.TRAIN)
