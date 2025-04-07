import queue
import threading
import logging
import time
from typing import Callable, Dict, Any, List, Optional
import uuid
from datetime import datetime
from app.models import TaskStatus

logger = logging.getLogger(__name__)

class Job:
    """Represents a job in the queue.
    
    A job is a unit of work that will be processed by a worker thread.
    """
    
    def __init__(self, task_id: int, func: Callable, args: tuple = (), kwargs: Dict[str, Any] = None):
        """Initialize a new job.
        
        Args:
            task_id: ID of the associated task
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
        """
        self.id = str(uuid.uuid4())
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
    
    def __str__(self):
        return f"Job(id={self.id}, task_id={self.task_id}, status={self.status})"

class JobQueue:
    """Thread-safe job queue for background task processing.
    
    This class implements a singleton pattern to ensure only one queue
    exists in the application.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Create a new instance if one doesn't exist."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(JobQueue, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the job queue."""
        if self._initialized:
            return
        
        self._queue = queue.Queue()
        self._jobs = {}  # job_id -> Job
        self._task_jobs = {}  # task_id -> job_id
        self._workers = []
        self._running = False
        self._initialized = True
        logger.info("Job queue initialized")
    
    def start(self, num_workers: int = 3):
        """Start the worker threads.
        
        Args:
            num_workers: Number of worker threads to create
        """
        if self._running:
            logger.warning("Job queue already running")
            return
        
        self._running = True
        
        # Create and start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"JobQueueWorker-{i}",
                daemon=True
            )
            self._workers.append(worker)
            worker.start()
            logger.info(f"Started worker thread {worker.name}")
        
        logger.info(f"Job queue started with {num_workers} workers")
    
    def stop(self):
        """Stop the worker threads."""
        if not self._running:
            logger.warning("Job queue not running")
            return
        
        self._running = False
        
        # Add None jobs to the queue to signal workers to exit
        for _ in self._workers:
            self._queue.put(None)
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=1.0)
            logger.info(f"Worker thread {worker.name} stopped")
        
        self._workers = []
        logger.info("Job queue stopped")
    
    def _worker_loop(self):
        """Worker thread loop to process jobs from the queue."""
        while self._running:
            try:
                # Get a job from the queue
                job = self._queue.get(block=True, timeout=1.0)
                
                # None is a signal to exit
                if job is None:
                    self._queue.task_done()
                    break
                
                # Process the job
                self._process_job(job)
                
                # Mark the job as done in the queue
                self._queue.task_done()
            
            except queue.Empty:
                # No jobs in the queue, continue waiting
                continue
            
            except Exception as e:
                logger.exception(f"Error in worker thread: {e}")
    
    def _process_job(self, job: Job):
        """Process a job.
        
        Args:
            job: Job to process
        """
        logger.info(f"Processing job {job.id} for task {job.task_id}")
        
        try:
            # Update job status to in progress
            job.status = TaskStatus.IN_PROGRESS
            job.started_at = datetime.now()
            self._update_task_status(job.task_id, TaskStatus.IN_PROGRESS)
            
            # Execute the job function
            job.result = job.func(*job.args, **job.kwargs)
            
            # Update job status to completed
            job.status = TaskStatus.COMPLETED
            job.completed_at = datetime.now()
            self._update_task_status(job.task_id, TaskStatus.COMPLETED)
            
            logger.info(f"Job {job.id} for task {job.task_id} completed successfully")
        
        except Exception as e:
            # Update job status to failed
            job.status = TaskStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            self._update_task_status(job.task_id, TaskStatus.FAILED)
            
            logger.exception(f"Job {job.id} for task {job.task_id} failed: {e}")
    
    def _update_task_status(self, task_id: int, status: TaskStatus):
        """Update the status of a task in the database.
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
        """
        from app.utils.db_client import get_db
        from app.models import Task
        
        try:
            with get_db() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task:
                    task.status = status
                    db.commit()
                    logger.info(f"Updated task {task_id} status to {status}")
                else:
                    logger.warning(f"Task {task_id} not found")
        except Exception as e:
            logger.exception(f"Error updating task {task_id} status: {e}")
    
    def enqueue(self, task_id: int, func: Callable, *args, **kwargs) -> str:
        """Add a job to the queue.
        
        Args:
            task_id: ID of the associated task
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            
        Returns:
            ID of the created job
        """
        # Create a new job
        job = Job(task_id, func, args, kwargs)
        
        # Store the job
        self._jobs[job.id] = job
        self._task_jobs[task_id] = job.id
        
        # Add the job to the queue
        self._queue.put(job)
        
        logger.info(f"Enqueued job {job.id} for task {task_id}")
        return job.id
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID.
        
        Args:
            job_id: ID of the job to get
            
        Returns:
            Job if found, None otherwise
        """
        return self._jobs.get(job_id)
    
    def get_job_by_task(self, task_id: int) -> Optional[Job]:
        """Get a job by task ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Job if found, None otherwise
        """
        job_id = self._task_jobs.get(task_id)
        if job_id:
            return self._jobs.get(job_id)
        return None
    
    def get_all_jobs(self) -> List[Job]:
        """Get all jobs.
        
        Returns:
            List of all jobs
        """
        return list(self._jobs.values())
    
    def get_queue_size(self) -> int:
        """Get the current size of the queue.
        
        Returns:
            Number of jobs in the queue
        """
        return self._queue.qsize()
    
    def get_active_workers(self) -> int:
        """Get the number of active worker threads.
        
        Returns:
            Number of active worker threads
        """
        return len([w for w in self._workers if w.is_alive()])

# Create a global job queue instance
job_queue = JobQueue()