#!/usr/bin/env python3
"""Module for listing all documents in a MongoDB collection"""


def list_all(mongo_collection):
    """Lists all documents in a MongoDB collection

    Args:
        mongo_collection: pymongo collection object

    Returns:
        list of all documents in collection, or empty list if none
    """
    if mongo_collection is None:
        return []
    
    return list(mongo_collection.find())