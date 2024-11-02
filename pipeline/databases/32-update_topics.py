#!/usr/bin/env python3
"""Module for updating topics of a school documentin MongoDB"""


def update_topics(mongo_collection, name, topics):
    """Changes all topics of a school document based on the name

    Args:
        mongo_collection: pymongo collection object
        name (str): school name to update 
        topics (list): list of strings of topics approached in the school

    Returns:
        None
    """
    if mongo_collection is None:
        return
    
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )