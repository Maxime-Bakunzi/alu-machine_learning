#!/usr/bin/env python3
"""
Script that displays information about the next upcoming SpaceX launch
using the SpaceX API.
"""
import requests
from datetime import datetime
import time


def get_upcoming_launch():
    """
    Fetches and returns formatted information about the next upcoming SpaceX  launch.

    Returns:
        str: Formatted string containing launch information
    """
    api_url = "https://api.spacexdata.com/v4/launches/upcoming"

    try:
        # Get upcoming launches
        response = requests.get(api_url)
        response.raise_for_status()
        launches = response.json()

        if not launches:
            return "No upcoming launches found"

        # Sort launches by date_unix and get the soonest one
        launches.sort(key=lambda x: x.get('date_unix', float('inf')))
        next_launch = launches[0]

        # Get rocket information
        rocket_response = requests.get(
            "https://api.spacexdata.com/v4/rockets/{}".format(
                next_launch['rocket']))
        rocket_response.raise_for_status()
        rocket_data = rocket_response.json()

        # Get launchpad information
        launchpad_response = requests.get(
            "https://api.spacexdata.com/v4/launchpads/{}".format(
                next_launch['launchpad']))
        launchpad_response.raise_for_status()
        launchpad_data = launchpad_response.json()

        # Convert UTC timestamp to local time
        launch_date = datetime.fromtimestamp(
            next_launch['date_unix']).strftime('%Y-%m-%dT%H:%M:%S%z')

        # Format the output string
        return "{} ({}) {} - {} ({})".format(
            next_launch['name'],
            launch_date,
            rocket_data['name'],
            launchpad_data['name'],
            launchpad_data['locality']
        )

    except requests.exceptions.RequestException as e:
        return "Error fetching launch data: {}".format(str(e))
    except (KeyError, ValueError) as e:
        return "Error processing launch data: {}".format(str(e))


if __name__ == '__main__':
    result = get_upcoming_launch()
    print(result)
