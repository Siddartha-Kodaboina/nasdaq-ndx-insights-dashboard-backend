import time
import threading
from app.queue import job_queue

def test_function(task_id, name, sleep_time=1):
    """Test function for the job queue."""
    print(f"Task {task_id}: {name} started")
    time.sleep(sleep_time)
    print(f"Task {task_id}: {name} completed")
    return f"Result from {name}"

def test_job_queue():
    """Test the job queue system."""
    print("Testing job queue...")
    
    # Start the job queue with 2 workers
    job_queue.start(num_workers=2)
    print(f"Job queue started with 2 workers")
    
    # Enqueue some jobs
    job_ids = []
    for i in range(5):
        job_id = job_queue.enqueue(i + 1, test_function, i + 1, f"Job {i + 1}", i + 1)
        job_ids.append(job_id)
        print(f"Enqueued job {job_id} for task {i + 1}")
    
    # Wait for jobs to complete
    time.sleep(1)
    print(f"Queue size: {job_queue.get_queue_size()}")
    print(f"Active workers: {job_queue.get_active_workers()}")
    
    # Wait for all jobs to complete
    time.sleep(10)
    
    # Check job statuses
    for job_id in job_ids:
        job = job_queue.get_job(job_id)
        print(f"Job {job_id} for task {job.task_id}: {job.status}, Result: {job.result}")
    
    # Stop the job queue
    job_queue.stop()
    print("Job queue stopped")
    
    print("Job queue test completed!")

if __name__ == "__main__":
    test_job_queue()