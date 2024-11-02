#!/usr/bin/env python3
"""Module for inserting a new document in MongoDB collection"""


def insert_school(mongo_collection, **kwargs):
    """Inserts a new document in a collection based on kwargs

    Args:
        mongo_collection: pymongo collection object
        **kwargs: keyword arguments to create the document

    Returns:
        _id of the newly inserted document
    """
    if mongo_collection is None:
        return None
    
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id