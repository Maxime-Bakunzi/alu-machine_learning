#!/usr/bin/env python3
"""
Python script that provides stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


def nginx_stats():
    """
    Provides statistics about Nginx logs stored in MongoDB
    - Database: logs
    - Collection: nginx
    """
    # Connect to MongoDB
    client = MongoClient('mongodb://127.0.0.1:27017')

    # Access the logs database and nginx collection
    nginx_collection = client.logs.nginx

    # Get total number of logs
    n_logs = nginx_collection.count_documents({})
    print("{} logs".format(n_logs))

    # Print methods statistics
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = nginx_collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    # Get number of status check
    status_check = nginx_collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print("{} status check".format(status_check))


if __name__ == "__main__":
    nginx_stats()
