#!/usr/bin/env python
"""
Test script for task management endpoints.

This script tests the following endpoints:
1. Create Task (POST /tasks/)
2. Get Task Status (GET /tasks/{task_id}/status)
3. Get Task (GET /tasks/{task_id})
4. List Tasks (GET /tasks/)
5. Get Task Data (GET /tasks/{task_id}/data)
"""

import requests
import json
import time
from datetime import datetime, timedelta
from pprint import pprint

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000"

def test_create_task():
    """Test the create task endpoint."""
    print("\n=== Testing Create Task Endpoint ===")
    
    # Request body for creating an explore_stock task
    request_body = {
        "task_type": "explore_stock",
        "parameters": {
            "ticker": "AAPL",
            "source": "source_a",
            "from_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "to_date": datetime.now().isoformat()
        }
    }
    
    print("Request:")
    pprint(request_body)
    
    # Make the request
    response = requests.post(f"{BASE_URL}/tasks/", json=request_body)
    
    print(f"\nResponse Status: {response.status_code}")
    print("Response Body:")
    pprint(response.json())
    
    # Return the task ID for further testing
    return response.json()["id"]

def test_get_task_status(task_id):
    """Test the get task status endpoint."""
    print("\n=== Testing Get Task Status Endpoint ===")
    
    # Make the request
    response = requests.get(f"{BASE_URL}/tasks/{task_id}/status")
    
    print(f"Response Status: {response.status_code}")
    print("Response Body:")
    pprint(response.json())

def test_get_task(task_id):
    """Test the get task endpoint."""
    print("\n=== Testing Get Task Endpoint ===")
    
    # Make the request
    response = requests.get(f"{BASE_URL}/tasks/{task_id}")
    
    print(f"Response Status: {response.status_code}")
    print("Response Body:")
    pprint(response.json())

def test_list_tasks():
    """Test the list tasks endpoint."""
    print("\n=== Testing List Tasks Endpoint ===")
    
    # Make the request with various filters
    response = requests.get(
        f"{BASE_URL}/tasks/",
        params={
            "skip": 0,
            "limit": 10,
            "sort_by": "created_at",
            "sort_desc": True
        }
    )
    
    print(f"Response Status: {response.status_code}")
    print("Response Body:")
    pprint(response.json())
    
    # Test with status filter
    print("\n--- Testing with status filter ---")
    response = requests.get(
        f"{BASE_URL}/tasks/",
        params={
            "status": "pending",
            "skip": 0,
            "limit": 10
        }
    )
    
    print(f"Response Status: {response.status_code}")
    print("Response Body:")
    pprint(response.json())

def test_get_task_data(task_id):
    """Test the get task data endpoint."""
    print("\n=== Testing Get Task Data Endpoint ===")
    
    # Make the request
    response = requests.get(f"{BASE_URL}/tasks/{task_id}/data")
    
    print(f"Response Status: {response.status_code}")
    if response.status_code == 200:
        print("Response Body (first 2 items):")
        data = response.json()
        pprint(data[:2] if len(data) > 2 else data)
    else:
        print("Response Body:")
        pprint(response.json())

def main():
    """Main function to run all tests."""
    # Test create task
    task_id = test_create_task()
    
    # Test get task status
    test_get_task_status(task_id)
    
    # Test get task
    test_get_task(task_id)
    
    # Test list tasks
    test_list_tasks()
    
    # Wait for task to complete (in a real scenario)
    print("\nWaiting for task to process...")
    time.sleep(2)  # In a real scenario, you'd wait longer or poll until completion
    
    # Test get task status again
    test_get_task_status(task_id)
    
    # Test get task data (might fail if task is not completed)
    test_get_task_data(task_id)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
