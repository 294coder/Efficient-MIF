from typing import Callable, List, Union
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from contextlib import nullcontext


class EasyProgress:
    tbar: Progress = None
    task_desp_ids: dict[str, int] = {}
    
    @classmethod
    def console(cls):
        assert cls.tbar is not None, '`tbar` has not initialized'
        return cls.tbar.console
    
    @classmethod
    def close_all_tasks(cls):
        if cls.tbar is not None:
            for task_id in cls.tbar.task_ids:
                cls.tbar.stop_task(task_id)
                # set the task_id all unvisible
                cls.tbar.update(task_id, visible=False)
                
    
    @classmethod
    def easy_progress(cls,
                      task_desciptions: list[str], 
                      task_total: list[int],
                      tbar_kwargs: dict={},
                      task_kwargs: list[dict[str, Union[str, int]]]=None,
                      is_main_process: bool=True,
                      *,
                      start_tbar: bool=True,
                      debug: bool=False) -> tuple[Progress, Union[list[int], int]]:
        """get a rich progress bar 

        Args:
            task_desciptions (list[str]): list of task descriptions of `len(task_desciptions)` tasks
            task_total (list[int]): list of length each task
            tbar_kwargs (dict, optional): kwargs for progress bar. Defaults to {}.
            task_kwargs (list[dict[str, Union[str, int]]], optional): task kwargs for each task. Defaults to None.
            is_main_process (bool, optional): if is main process. Defaults to True.
            start_tbar (bool, optional): start running progress bar when ini. Defaults to True.
            debug (bool, optional): debug mode, set progress bar to be unvisible. Defaults to False.

        Returns:
            tuple[Progress, Union[list[int], int]]: Progress bar and task ids
        """
        
        def _add_task_ids(tbar: Progress, task_desciptions, task_total, task_kwargs):
            task_ids = []
            if task_kwargs is None:
                task_kwargs = [{'visible': False}] * len(task_desciptions)
            for task_desciption, task_total, id_task_kwargs in zip(task_desciptions, task_total, task_kwargs):
                if task_desciption in list(EasyProgress.task_desp_ids.keys()):
                    task_id = EasyProgress.task_desp_ids[task_desciption]
                    task_ids.append(task_id)
                else:
                    task_id = tbar.add_task(task_desciption, total=task_total, **id_task_kwargs)
                    task_ids.append(task_id)
                    EasyProgress.task_desp_ids[task_desciption] = task_id
                
            return task_ids if len(task_ids) > 1 else task_ids[0]
        
        def _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs):
            if is_main_process:
                if task_kwargs is not None:
                    assert len(task_desciptions) == len(task_total) == len(task_kwargs)
                else:
                    assert len(task_desciptions) == len(task_total)
                
                # if (console := tbar_kwargs.pop('console', None)) is not None:
                #     console._color_system = console._detect_color_system()
                
                # if 'console' in tbar_kwargs:
                #     tbar_kwargs['console']._color_system = tbar_kwargs['console']._detect_color_system()
                    
                tbar = Progress(TextColumn("[progress.description]{task.description}"),
                                BarColumn(),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                                SpinnerColumn(),
                                TimeRemainingColumn(),
                                TimeElapsedColumn(),
                                **tbar_kwargs)
                EasyProgress.tbar = tbar
                
                task_ids = _add_task_ids(tbar, task_desciptions, task_total, task_kwargs)
                
                return tbar, task_ids
            else:
                return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        
        def _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs):
            if is_main_process:
                tbar = EasyProgress.tbar
                
                task_ids = []
                if task_kwargs is None:
                    task_kwargs = [{'visible': False}] * len(task_desciptions)
                
                task_ids = _add_task_ids(tbar, task_desciptions, task_total, task_kwargs)
                
                return tbar, task_ids
            else:
                return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        
        if not debug:
            if EasyProgress.tbar is not None:
                rets = _cached_tbar_with_new_task_ids(task_desciptions, task_total, task_kwargs)
            else:
                rets = _new_tbar_with_task_ids(task_desciptions, task_total, task_kwargs)
            if start_tbar and is_main_process and not EasyProgress.tbar.live._started:
                EasyProgress.tbar.start()
            return rets
        else:
            return nullcontext(), [None] * len(task_desciptions) if len(task_desciptions) > 1 else None
        