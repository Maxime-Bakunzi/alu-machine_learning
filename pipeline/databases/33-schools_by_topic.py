#!/usr/bin/env python3
"""Module for finding schools by topics in MongoDB"""


def schools_by_topic(mongo_collection, topic):
    """Returns a list of schools having a specific topic

    Args:
        mongo_collection: pymongo collection object
        topic (str): topic searched

    Returns:
        list: list of schools that have a specified topic
    """
    if mongo_collection is None:
        return []
    
    return list(mongo_collection.find({"topics": topic}))