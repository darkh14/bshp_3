import json
import logging
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import time
import os

from settings import SOURCE_FOLDER, TEMP_FOLDER
from api_types import TaskData, StatusResponse, ProcessingTaskResponse

logger = logging.getLogger(__name__)


class TaskManager:
    """Tasks and files management."""

    def __init__(self):
        self.tasks: Dict[str, TaskData] = {}
        self.lock = asyncio.Lock()

    async def create_task(self, task_id: str) -> TaskData:
        """Create new task."""
        async with self.lock:
            task = TaskData(task_id=task_id)
            self.tasks[task_id] = task
            logger.info(f"Created task: {task_id}")
            return task

    async def get_task(self, task_id: str) -> Optional[TaskData]:
        """Get task data."""
        return self.tasks.get(task_id)

    async def update_task(self, task_id: str, **kwargs) -> None:
        """Update task data."""
        async with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Task not found: {task_id}")
                return

            task = self.tasks[task_id]

            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

            # Special timer processing
            if kwargs.get("status") == "PROCESSING" and task.start_time is None:
                task.start_time = time.time()
            elif kwargs.get("status") == "READY" and task.start_time:
                task.end_time = time.time()

    async def get_task_status(self, task_id: str) -> Optional[StatusResponse]:
        """Get task status for API."""
        task = await self.get_task(task_id)
        if not task:
            return None

        return StatusResponse(
            status=task.status,
            end_time=task.end_time,
            error=task.error
        )

    async def save_upload_file(self, task_id: str, filename: str, content: bytes) -> Path:
        """Saves loaded file"""

        if not os.path.isdir(SOURCE_FOLDER):
            os.makedirs(SOURCE_FOLDER)
        folder = os.path.join(SOURCE_FOLDER, f"{task_id}")
        os.mkdir(folder)
        file_path = os.path.join(folder, f"{filename}")

        async with asyncio.Lock():
            with open(file_path, "wb") as f:
                f.write(content)

        logger.info(f"Saved uploaded file: {file_path}")
        return file_path # type: ignore

    async def cleanup_task_files(self, task_id: str) -> None:
        """Cleanup temp files."""
        patterns = [
            SOURCE_FOLDER / f"{task_id}",
            TEMP_FOLDER / f"{task_id}",
        ]

        for pattern in patterns:
            # For directories without wildcards
            if pattern.name and '*' not in pattern.name and pattern.exists() and pattern.is_dir():
                try:
                    shutil.rmtree(pattern)
                    logger.info(f"Removed directory: {pattern}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {pattern}: {e}")
            else:
                #  For patterns with wildcards
                for file in pattern.parent.glob(pattern.name):
                    try:
                        if file.is_dir():
                            shutil.rmtree(file)
                            logger.info(f"Removed directory: {file}")
                        else:
                            file.unlink()
                            logger.info(f"Removed file: {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file}: {e}")


# Глобальный экземпляр хранилища
task_manager = TaskManager()