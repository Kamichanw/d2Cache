# This file is modified from the original lm_eval/tasks/__init__.py
import os

from typing import List, Optional, Union
from lm_eval.tasks import TaskManager as StandardTaskManager


class TaskManager(StandardTaskManager):
    def initialize_tasks(
        self,
        include_path: Optional[Union[str, List]] = None,
        include_defaults: bool = True,
    ) -> dict[str, dict]:
        """Creates a dictionary of tasks indexes.

        :param include_path: Union[str, List] = None
            An additional path to be searched for tasks recursively.
            Can provide more than one such path as a list.
        :param include_defaults: bool = True
            If set to false, default tasks (those in lm_eval/tasks/) are not indexed.
        return
            Dictionary of task names as key and task metadata
        """
        if include_defaults:
            all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"]
        else:
            all_paths = []
        if include_path is not None:
            if isinstance(include_path, str):
                include_path = [include_path]
            all_paths.extend(include_path)

        task_index = {}
        for task_dir in all_paths:
            tasks = self._get_task_and_group(task_dir)
            task_index = {
                **task_index,
                **tasks,
            }  # MODIFIED: should be later tasks override earlier ones

        return task_index
