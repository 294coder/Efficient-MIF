import os
import warnings
warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cache_dir = os.path.join(root_dir, 'cache')

os.environ["MPLCONFIGDIR"] = os.path.join(cache_dir, "matplotlib")
os.environ["HF_HOME"] = os.path.join(cache_dir, "hf_home")
os.environ["MPLCONFIGDIR"] = os.path.join(cache_dir, "matplotlib")

from dataclasses import dataclass
import sys
sys.path.append('./')

from ._metric_legacy import *
from .metric_sharpening import *
from .metric_VIF import *
from .log_utils import *
from .misc import *
from .misc import recursive_search_dict2namespace as convert_config_dict
from .load_params import *
from .optim_utils import *
from .network_utils import *
from .visualize import *
from .inference_helper_func import *
from .loss_utils import *
from .save_checker import *
from .train_utils import *
from .progress_utils import *
from .model_perf_utils import *
from .ema_utils import *

config_load = yaml_load





