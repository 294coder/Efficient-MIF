import os
import json
import logging
from datetime import datetime
from functools import partial
import signal
from typing import Any, Dict, List, Optional, Union, Protocol
from collections.abc import Sequence, Iterable
from contextlib import contextmanager

import loguru
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
import shortuuid

from utils.misc import NameSpace, default, is_main_process
from utils.visualize import get_spectral_image_ready

import time
from rich.console import Console
from rich.logging import RichHandler
from importlib import reload

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = int(sec%60)
    return h,m,s

class TimeFilter(logging.Filter):

    def filter(self, record):
        try:
          start = self.start
        except AttributeError:
          start = self.start = time.time()

        time_elapsed = get_time(time.time() - start)

        record.relative = "{0}:{1:02d}:{2:02d}".format(*time_elapsed)

        # self.last = record.relativeCreated/1000.0
        return True


def save2json_file(d: dict, path: str, mode: str = "w", indent: int = 4):
    assert path.endswith(".json"), "@path should end with .json"
    with open(path, mode) as f:
        json.dump(d, f, indent=indent)
    print(f"save json file in {path}")


def loss_dict2str(
    loss_dict: "dict[int, float | torch.Tensor]", 
    world_size: int = 1, 
    round_fp: int = 6
) -> str:
    log_str = ""
    for k, v in loss_dict.items():
        log_str += f"{k}: {v / world_size:.{round_fp}f} "
    return log_str


## decrepted
class TrainStatusLogger(object):
    def __init__(
        self, id="None", path="./train_status/status.pt", resume=False, args=None
    ):
        """
        track training status as a context manager
        :param id: run id, which is defined in parser args
        :param path: pkl file's path
        :param resume: resume training. if you want to resume last training run, set id=None.
                       if you want to specify one run to resume, pass a specified id.
        """
        self.time_now = datetime.now()
        self.path = path
        self.id = id
        self.resume = resume
        self.status = {
            "id": id,
            "status": "running",
            "time_stamp": str(self.time_now.timestamp()),
            "time": self.time_now.strftime("%Y-%m-%d, %H:%M:%S"),
            "args": args,
        }
        self._base_path = os.path.dirname(self.path)
        if not os.path.exists(self.path):
            if not os.path.exists(self._base_path):
                os.mkdir(self._base_path)
            self.status_all = [self.status]
        else:
            self.status_all = self.load_train_status()
            self.check_unique_id()
            if resume:
                if id == "None":
                    self.status, _ = self.find_last_untrained_status()
                else:
                    self.status = self.find_run_by_id(id)
                    # print('warning: argument @id is not equal to the resume id which will be ignored')
            else:
                self.status_all.append(self.status)
                
        # ======= handle the KeyboardInterrupt signal =======
        def handler(*args):
            print('catch signal: KeyboardInterrupt')
            print('EXITTING...')
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, handler)

    @staticmethod
    def _check_status_legal(status):
        assert status in ("running", "done", "break")

    def load_train_status(self):
        if os.path.getsize(self.path) > 0:
            # with open(self.path, "rb") as f:
            #     l = pickle.load(f)
            l = torch.load(self.path)
        else:
            raise EOFError("file is empty, you should delete it")
        print("load previous train status")
        return l

    def save_train_status(self):
        # with open(self.path, "wb") as f:
        #     pickle.dump(self.status_all, f)
        torch.save(self.status_all, self.path)
        print("save all train status")

    def update_train_status(self, status):
        self._check_status_legal(status)
        self.status["status"] = status

    def find_last_untrained_status(self):
        f_sort = lambda d: d["time_stamp"] if d["status"] == "break" else "0"
        last_status = sorted(self.status_all, key=f_sort)[-1]
        return last_status, last_status["id"]

    def find_run_by_id(self, id):
        s = self._find_id(id)
        if s["status"] != "break":
            return s
        raise AttributeError(
            f"no id: {id} in not an existing run or has already been done"
        )

    def _find_id(self, id):
        for s in self.status_all:
            if s["id"] == id:
                return s

    def print_status_by_id(self, id):
        s = self._find_id(id)
        for k, v in s:
            if isinstance(v, NameSpace):
                print(v)
            else:
                print(f"{k}: {v}")

    def check_unique_id(self):
        ids = []
        for d in self.status_all:
            ids.append(d["id"])
        assert len(ids) == len(np.unique(ids)), "exist id conflict"
        assert not self.status["id"] in ids or self.resume, (
            "id conflicts, check your run id "
            "or delete all tracker pkl file. "
            f"the pkl file can be found in {self.path}"
        )

    def __enter__(self):
        nbreak = 0
        ndone = 0
        for d in self.status_all:
            s = d["status"]
            if s == "done":
                ndone += 1
            elif s == "break":
                nbreak += 1

        print("=" * 20, "Log Train Process", "=" * 20, sep="")
        print(f"all runs: {ndone} run(s) done, {nbreak} run(s) break")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f'traceback: {exc_tb}')
        if exc_type is not None or exc_val is not None:
            print("=" * 20, "Find Error Happen", "=" * 20, sep="")
            print(f"catch error type: {exc_type}, error value: {exc_val}")
            self.update_train_status("break")
        else:
            print("=" * 20, "Training End", "=" * 20, sep="")
            self.update_train_status("done")
            
        # only save in main process
            if is_main_process():
                self.save_train_status()

    def __repr__(self):
        def dict_str(d):
            s = "id: {:<10}  status: {:<7}  time_stamp: {:<20}  time: {:<20}".format(
                d["id"], d["status"], d["time_stamp"], d["time"]
            )
            return s

        repr = ""
        for d in self.status_all:
            repr += dict_str(d) + "\n"
        return repr
    
def generate_id(length: int = 8) -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


def easy_logger(level='INFO', format_str: str=None, func_name: str=None):
    reload(logging)
    
    if format_str is not None and func_name is not None:
            format_str = "%(func_name)s: " + format_str
    elif format_str is None and func_name is not None:
        format_str = "%(func_name)s: %(message)s"
    else:
        format_str = "%(message)s"
        
    file_name = os.environ.get('LOG_FILE', None)
    if file_name:
        console = Console(file=open(file_name, 'a+'))
    else:
        console = None
        
    class FuncNameFilter(logging.Filter):
        def filter(self, record):
            record.func_name = func_name
            return True
    
    rich_handler = RichHandler(console=console, show_path=False, level=level, markup=True)
    rich_handler.setFormatter(logging.Formatter(format_str, datefmt='%X'))
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(rich_handler)
    
    class ProtocalLogger(Protocol):
        @classmethod
        def print(ctx, *msg, sep=" ", level: str | int="INFO"):
            msg = sep.join(map(str, msg))
            if isinstance(level, str):
                level = eval(f'logging.{level.upper()}')
            logger.log(level, msg, extra={"markup": True})
            
        @classmethod
        def debug(ctx, msg):
            pass
        
        @classmethod
        def info(ctx, msg):
            pass
        
        @classmethod
        def warning(ctx, msg):
            pass
        
        @classmethod
        def error(ctx, *msg, sep=" ", raise_error: bool=False, error_type=None):
            msg = sep.join(map(str, msg))
            ctx.print(msg, level='ERROR')
            if raise_error:
                if error_type is not None:
                    raise error_type(msg)
                
                raise RuntimeError(msg)
                
    # add filter to add func_name to log
    if func_name is not None:
        logger.addFilter(FuncNameFilter())
    
    # logger signature
    logger: ProtocalLogger
    
    # add attributes to logger
    logger.print = ProtocalLogger.print
    logger.debug = partial(ProtocalLogger.print, level='DEBUG')
    logger.info = partial(ProtocalLogger.print, level='INFO')
    logger.warning = partial(ProtocalLogger.print, level='WARNING')
    logger.error = ProtocalLogger.error
    
    logger._console = rich_handler.console
    
    return logger


from loguru import logger, _logger
class LoguruLogger:
    _logger = logger
    console = None
    handler = []
    
    _first_import = True
    _default_file_format = "<green>[{time:MM-DD HH:mm:ss}]</green> <level>[{level}] {message}</level>"
    _default_console_format = "[{time:HH:mm:ss}] <level>[{level}] {message}</level>"
    
    @classmethod
    def logger(cls,
               sink=None,
               format=None,
               filter=None,
               **kwargs) -> "_logger.Logger":
        reload(loguru)
        
        if cls._first_import:
            cls._logger.remove()  # the first time import
            cls.console = Console(color_system=None)
            cls._logger.add(
                default(sink, lambda x: cls.console.print(x)),
                colorize=True,
                format=default(format, cls._default_console_format),
                **kwargs
            )
            
            cls._first_import = False
        
        else:
            if sink is not None:
                handler = cls._logger.add(sink, format=default(format, cls._default_file_format), filter=filter, **kwargs)
                
                cls.handler.append(handler)
                
        return cls._logger
    
    @classmethod
    def add(cls, *args, **kwargs):
        handler = cls._logger.add(*args, **kwargs)
        cls.handler.append(handler)
        
    @classmethod
    def remove_all(cls):
        for h in cls.handler:
            cls._logger.remove(h)
        cls.handler = []
        
    @classmethod
    def remove_id(cls, id):
        cls._logger.remove(id)
        
    @classmethod
    def bind(cls, *args, **kwargs):
        return cls._logger.bind(*args, **kwargs)


@contextmanager
def catch_any_error():
    try:
        logger = LoguruLogger.logger()
        yield logger
    except Exception as e:
        logger.error(f"catch error: {e}", raise_error=True)
        logger.exception(e)
    finally:
        LoguruLogger.remove_all()
        

def get_logger(
    base_path: str = None,
    name: str = None,
    args=None,
    std_level=logging.INFO,
    file_level: Union[tuple, int] = (logging.DEBUG,),
    file_handler_names: Union[tuple, str] = ("debug",),
    file_mode: str = "w",
    show_pid: bool = False,
    method_dataset_as_prepos=True,
):
    """
    get logger to log
    :param base_path: such like './log/'
    :param name: logger name such as 'train_epoch_300'
    :param std_level: stream level
    :param file_level: file level
    :param file_handler_names:
    :param file_mode: 'a' append, 'w' write
    :param show_pid: show thread id
    :return: logger and List[handlers]
    """
    assert name is not None, "@param name should not be None"
    reload(logging)
    
    # assert base_path is not None, "@param base_path should not be None"
    if not show_pid:
        # format_str = "[%(asctime)s - %(funcName)s]-%(levelname)s: %(message)s"
        # format_str = "(%(relative)s) %(message)s"
        format_str = "%(message)s"
    else:
        # format_str = (
        #     "[%(asctime)s - %(funcName)s - pid: %(thread)d]-%(levelname)s: %(message)s"
        # )
        format_str = "pid: %(thread)d) %(message)s"
    rich_handler = RichHandler(show_path=False, markup=True)
    rich_handler.setFormatter(logging.Formatter(format_str, datefmt='%X'))

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(rich_handler)

    hdls = []

    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setLevel(std_level)
    # hdls.append(stream_handler)
    # logger.addHandler(stream_handler)

    if base_path is not None:
        assert len(file_handler_names) == len(file_level), \
            "@param file_handler_names and @param file_level should be list and equal length"
        for n, level in zip(file_handler_names, file_level):
            if method_dataset_as_prepos:
                file_log_dir = os.path.join(base_path, args.full_arch, args.dataset, name)
            else:
                file_log_dir = os.path.join(base_path, name)
            if not os.path.exists(file_log_dir):
                os.makedirs(file_log_dir)
                print(f"logging: make log file [{os.path.abspath(file_log_dir)}]")
            file_log_path = os.path.join(file_log_dir, n + ".log")
            # file_handler = logging.FileHandler(file_log_path, mode=file_mode)
            file_console = Console(file=open(file_log_path, 'w'))
            file_handler = RichHandler(console=file_console, show_path=False, markup=True)
            # formatter = logging.Formatter(
            #     "[%(asctime)s - %(name)s] - %(levelname)s: %(message)s"
            # )
            # file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            hdls.append(file_handler)
            logger.addHandler(file_handler)
    else:
        file_console = None
        file_handler = None
        file_log_dir = None
        file_log_path = None
            
    for handler in logger.handlers:
        handler.addFilter(TimeFilter())
        
    def log_print(*msg, sep=" ", level="INFO", dist=False, proc_id=None):
        if dist or is_main_process():
            if isinstance(level, str):
                level_int = eval(f"logging.{level}")
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            msgs += sep.join(map(str, msg))
            logger.log(level=level_int, msg=msgs)
    
    # register some attrs
    logger.print = log_print
    logger._console = rich_handler.console
    logger._file_console = file_console
    logger._file_path = file_log_path

    return logger, hdls, file_log_dir


# class WandbLogger:
#     def __init__(
#         self,
#         project_name,
#         run_name=None,
#         save_path=None,
#         config=None,
#         resume=None,
#         **kwargs,
#     ):
#         self.run = wandb.init(
#             project=project_name,
#             name=run_name,
#             dir=save_path,
#             config=config,
#             resume=resume,
#             settings=wandb.Settings(start_method="fork"),
#             **kwargs,
#         )

#     def watch(self, type, network, graph=False, freq=20):
#         # type(str) One of "gradients", "parameters", "all", or None
#         wandb.watch(network, log=type, log_graph=graph, log_freq=freq)

#     def log_image(self, tensor_or_numpy, img_name, *args):
#         # e.g. cols = ['id','pred', 'gt', 'res']
#         # tensor_or_numpy should be tensor [B, C, H, W] or numpy array [H, W, C]
#         # C can only be 1, 3 or 4
#         # warning: you should control B for clear visualization, suggest 2<B<7
#         if isinstance(tensor_or_numpy, torch.Tensor):
#             x = tensor_or_numpy.cpu().numpy()
#         elif isinstance(tensor_or_numpy, np.ndarray):
#             x = tensor_or_numpy
#         else:
#             raise NotImplementedError

#         assert x.shape[-1] in [1, 3, 4]
#         img_log = [wandb.Image(xi) for xi in x] if x.ndim == 4 else [wandb.Image(x)]
#         self.run.log({img_name: img_log})

#     def log_table(self, table_data, cols, table_name, *args):
#         # for example
#         # my_data = [
#         #     [0, wandb.Image("img_0.jpg"), 0, 0],
#         #     [1, wandb.Image("img_1.jpg"), 8, 0],
#         #     [2, wandb.Image("img_2.jpg"), 7, 1],
#         #     [3, wandb.Image("img_3.jpg"), 1, 1]
#         # ]
#         #
#         # create a wandb.Table() with corresponding columns
#         # columns =["id", "image", "prediction", "truth"]
#         # test_table = wandb.Table(data=my_data, columns=columns)

#         # add_data
#         # add row: table.add_data("3a", "3b", "3c")
#         # add col: table.add_column(name="col_name", data=col_data)

#         wandb.log({table_name: wandb.Table(data=table_data, columns=cols)})

#     def log_curve(self, value=None, name=None, d=None, *args):
#         if d is None:
#             wandb.log({name: value})
#         else:
#             wandb.log(d)


class NoneLogger:
    def __init__(self, *args, **kwargs):
        class NoneWriter:
            def __init__(self) -> None:
                pass
            
            def close(self, *args, **kwargs):
                pass
            
        self.writer = NoneWriter()
        self.logger, *_ = get_logger(name=kwargs.pop('name'))
        
        # add time and run_id
        args = kwargs['cfg']
        args.logger_config.name = (
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            + "_"
            + args.logger_config.name
        )
        args.logger_config.name += "_" + args.run_id + f"_{args.comment}"
        name = args.logger_config.name
        
        self.log_file_dir = os.path.join(args.logger_config.base_path,
                                         args.full_arch, args.dataset, name)
        
    @property    
    def console(self):
        return self.logger._console
    
    @property
    def file_console(self):
        return self.file_console._file_console
    
    def watch(self, *args, **kwargs):
        pass
    
    def log_image(self, *args, **kwargs):
        pass
    
    def log_images(self, *args, **kwargs):
        pass
    
    def log_curve(self, *args, **kwargs):
        pass
    
    def log_curves(self, *args, **kwargs):
        pass
    
    def log_network(self, *args, **kwargs):
        pass
    
    def print(self, *msg, level="INFO", dist=False, proc_id=None):
        if dist or is_main_process():
            if isinstance(level, str):
                level_int = eval(f"logging.{level}")
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)
            
    def info(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.INFO
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)
            
    def debug(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.DEBUG
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)
            
    def warning(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.WARNING
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)
            
    def error(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.ERROR
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.logger.log(level=level_int, msg=msgs)


class TensorboardLogger:
    def __init__(
        self,
        args=None,
        tsb_logdir=None,
        comment=None,
        file_stream_log=True,
        config_file_mv="./configs",
        config_file_type="yaml",
        method_dataset_as_prepos=False
    ):
        """

        Args:
            args: config args from main.py
            tsb_logdir: tensorboard log dir
            comment:
            file_stream_log: file stream dir
            config_file_mv: where arch_config.yaml dir at
        """
        self.grad_dict = {}
        # if not os.path.exists(logdir):
        #     os.mkdir(logdir)
        self.hooks = {}
        self.watch_type = "None"

        self.freq = 10
        
        # add time and run_id
        args.logger_config.name = (
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            + "_"
            + args.logger_config.name
        )
        args.logger_config.name += "_" + args.run_id + f"_{args.comment}"

        if file_stream_log:
            self.file_logger, self.file_hdls, self.log_file_dir = get_logger(
                **args.logger_config.to_dict(), 
                args=args,
                method_dataset_as_prepos=method_dataset_as_prepos
            )
            config_cp_path = os.path.join(self.log_file_dir, "config.json")
            save2json_file(args.to_dict(), config_cp_path)
            # shutil.copy2(os.path.join(config_file_mv, f'{args.arch}_config.{config_file_type}'), self.log_file_dir)
            self.print(
                f"\nmove config file to {os.path.abspath(self.log_file_dir)}", level="INFO"
            )
            
        self.writer = SummaryWriter(default(tsb_logdir, self.log_file_dir), comment)
            
        # set file path to os.environ
        os.environ['LOG_FILE'] = self.logger_file_path
        self.file_logger.debug(f'set os environ `LOG_FILE`={self.logger_file_path}')
    
    @property    
    def console(self) -> Console:
        return self.file_logger._console
    
    @property
    def logger_file_path(self) -> str:
        return self.file_logger._file_path
    
    @property
    def file_console(self):
        return self.file_console._file_console
            
    def check_tensor_float(self, x):
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)
        
        return x

    @is_main_process
    def watch(self, network: nn.Module, watch_type: str, freq: int):
        assert watch_type in (
            "all",
            "grad",
            "None",
        ), "@watch_type should only be all, grad or None"
        if watch_type == "None":
            return
        self.watch_type = watch_type
        self.freq = freq

        def _hook(grad, name):
            self.grad_dict[name] = grad

        for n, p in network.named_parameters():
            hook = partial(_hook, name=n)
            self.hooks[n] = hook
            p.register_hook(hook)

    @is_main_process
    def log_curve(self, x, name, epoch):
        self.writer.add_scalar(name, self.check_tensor_float(x), epoch)

    @is_main_process
    def log_curves(self, x_dict: Dict, epoch):
        # for example:
        # for i in range(100):
        #     writer.add_scalars('run_14h', {'xsinx': i * np.sin(i / r),
        #                                    'xcosx': i * np.cos(i / r),
        #                                    'tanx': np.tan(i / r)}, i)

        # self.writer.add_scalars(main_name, x_dict, epoch)
        for k, v in x_dict.items():
            self.writer.add_scalar(k, self.check_tensor_float(v), epoch)
    
    @is_main_process
    def log_image(self, image, name, epoch):
        if image.ndim == 3:
            assert image.shape[0] <= 3, (
                f"the number of image channel "
                f"should not greater than 3 but got shape {image.shape}"
            )
        self.writer.add_image(name, self.check_tensor_float(image), epoch, dataformats="CHW")

    @is_main_process
    def log_images(self, batch_imgs: Sequence, nrow: int, names: Sequence, task: str,
                   epoch: int, ds_name: str, **grid_kwargs):
        assert task in ['fusion', 'sharpening'], '@task should be fusion or sharpening'
        
        for batch_img, name in zip(batch_imgs, names):
            batch_img = get_spectral_image_ready(self.check_tensor_float(batch_img), name, task, ds_name)
            grid_img = make_grid(batch_img, nrow=nrow, **grid_kwargs)
            self.log_image(grid_img, name, epoch)

    @is_main_process
    def log_network(self, network: nn.Module, ep: int):
        if self.watch_type != "None":
            if ep % self.freq == 0:
                for (_, g), (n, p) in zip(
                    self.grad_dict.items(), network.named_parameters()
                ):
                    if self.watch_type == "all":
                        self.writer.add_histogram(n + "_data", p.flatten().float(), ep)
                    self.writer.add_histogram(n + "_grad", g.flatten().float(), ep)

    def print(self, *msg, level="INFO", dist=False, proc_id=None):
        if dist or is_main_process():
            if isinstance(level, str):
                level_int = eval(f"logging.{level}")
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.file_logger.log(level=level_int, msg=msgs)
    
    def info(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.INFO
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.file_logger.log(level=level_int, msg=msgs)
            
    def debug(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.DEBUG
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.file_logger.log(level=level_int, msg=msgs)
            
    def warning(self, *msg, dist=False, proc_id=None):
        if dist or is_main_process():
            level_int = logging.WARNING
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.file_logger.log(level=level_int, msg=msgs)
            
    def error(self, *msg, dist=False, proc_id=None, raise_error: bool=False, error_type=None):
        if dist or is_main_process():
            level_int = logging.ERROR
            msgs = f"{proc_id=} - " if proc_id is not None else ""
            for s in msg:
                msgs += s
            self.file_logger.log(level=level_int, msg=msgs)
            if raise_error:
                if error_type is not None:
                    raise error_type(msgs)
                
                raise RuntimeError(msgs)


# from aim import Distribution, Image, Repo, Run, Session, Text

# class AimLogger(object):    
#     def __init__(
#         self, run_name, resume_hash_name=None, hparams=None, cap_term_logs=True,
#         *,
#         desp=None,
#     ):
#         """Aim framewarke logger

#         Args:
#             run_name (str): experiment name
#             resume_hash_name (str, optional): set it if you want to resume training. Defaults to None.
#             hparams (dict, optional): dict of some hyperparameters of your experiments. Defaults to None.
#             cap_term_logs (bool, optional): capture terminal logs. Defaults to True.
#             desp (str, optional): short description of your experiment. Defaults to None.
#         """
#         self.run = Run(
#             resume_hash_name,
#             experiment=run_name,
#             capture_terminal_logs=cap_term_logs,
#             log_system_params=False,
#         )
#         setattr(self.run, 'description', desp)
#         self.run["hparams"] = hparams
#         self.log_info("log params:", hparams)
#         self.run_name = run_name
#         self.repo = Repo("./")

#     def _convert_uint8_img(self, img):
#         # TODO: to suit PIL package, convert the image into unit8 type
#         # ref to https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        
#         # Aim package only support uint8 type image :>
#         if isinstance(img, torch.Tensor):
#             img = img.detach().cpu().numpy()
#         img = img - img.min()
#         img = img / img.max()
#         img = (img * 255).astype(np.uint8)

#         return img

#     def log_image(self, image, name=None, epoch=None, context=None):
#         # check image type
#         if isinstance(image, np.ndarray):
#             assert image.ndim in [3, 2]
#             img = Image(self._convert_uint8_img(image))
            
#         # TODO: convert the Tensor into ndarray is really slow
#         elif isinstance(image, torch.Tensor):
#             assert image.ndim in [4, 3, 2]
#             if image.ndim == 4:
#                 nrows = math.sqrt(image.shape[0])
#                 image = make_grid(image, nrow=nrows)
#                 img = Image(image)
#             elif image.ndim in [3, 2]:
#                 img = Image(self._convert_uint8_img(image))
#             else:
#                 self.run.log_warning(f"not support image shape {image.shape}")
#         elif isinstance(image, plt.Figure):
#             img = Image(image)
#         else:
#             img = image

#         self.run.track(img, name=name, epoch=epoch, context=context)

#     def log_text(self, text, name=None, epoch=None, context=None):
#         self.run.track(Text(text), name=name, epoch=epoch, context=context)
    
#     @beartype.beartype()
#     def log_metrics(self, metrics, name=None, epoch=None, context: dict = None):
#         """log metrics or other values

#         Args:
#             metrics (dict or values): a dict or values to log
#             epoch (_type_): _description_
#             context (_type_, optional): _description_. Defaults to None.
#         """
#         self.run.track(metrics, name=name, epoch=epoch, context=context)

#     @beartype.beartype()
#     def log_distribution(
#         self,
#         distribution,
#         name=None,
#         epoch=None,
#         context: dict = None,
#     ):
#         #######################################################################
#         # !!!!
#         # warning: the context must be a dict or it will explode your aim repo
#         # I don't know why, maybe it is a bug
#         #######################################################################
#         distribution = distribution.flatten()

#         if isinstance(distribution, torch.Tensor):
#             distribution = distribution.detach().cpu().numpy()

#         # hist, bins = np.histogram(distribution, bins=64 if 64 < len(distribution) else len(distribution))
#         # bin range is
#         # bin_range = [bins[0], bins[-1]]

#         self.run.track(
#             Distribution(distribution), name=name, epoch=epoch, context=context
#         )

#     def log_network(self, network: nn.Module, epoch: int = None):
#         # refer to the warning in @log_distribution, it's important
#         # I set the @context into a dict, do not change it
#         for n, p in network.named_parameters():
#             p = p.flatten().detach().cpu().numpy()
#             # context = {"net_params_dist": n}
#             self.log_distribution(
#                 p, name="network_params", epoch=epoch, context={"net_params_dist": n}  # do not change
#             )

#     def close(self):
#         # close the run
#         # finilize and close, may one of them will take effect
#         self.run.finalize()
#         self.run.close()
#         print("logger closed")

#     def _make_msg_one_text(self, *msg):
#         if isinstance(msg[0], str) and len(msg) == 1:
#             return msg[0]
#         fin_msg = ""
#         for m in msg:
#             fin_msg += str(m)
#         return fin_msg
    
#     ######## override those functions #########
#     def log_info(self, *msg):
#         self.run.log_info(self._make_msg_one_text(*msg))

#     def log_warning(self, *msg):
#         self.run.log_warning(self._make_msg_one_text(*msg))

#     def log_error(self, *msg):
#         self.run.log_error(self._make_msg_one_text(*msg))

#     def log_debug(self, *msg):
#         self.run.log_debug(self._make_msg_one_text(*msg))
#     ###########################################

#     # repo control
#     # may be not used
#     def delete_run(self, run_hash=None):
#         # sometimes the run is locked, so we need to release it
#         # or just delete it
#         print(
#             "warning, deleting run: {}".format(run_hash if run_hash else self.run.hash)
#         )
#         run = self.repo.get_run(run_hash if run_hash else self.run.hash)
#         # run._lock.release()  # will raise Nonetype do not have the attribute
#         run.read_only = False
#         ans = input("press [y/n] to confirm deleting")
#         if ans == "y":
#             d_m = self.repo.delete_run(run_hash)
#             print("deleted {}: {}".format(run.hash, d_m))
#         elif ans == "n":
#             print("canceled")
#         else:
#             print("invalid input, canceled")
    
#     @property
#     def hash(self):
#         return self.run.hash






if __name__ == '__main__':
    logger = easy_logger()
    
    logger.info('[dim green] 1234 [/dim green]')

    
    
    
    

























