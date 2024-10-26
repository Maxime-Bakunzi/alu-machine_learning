# Data Collection - APIs Project

## Description
This project focuses on working with various APIs (SWAPI, GitHub, and SpaceX) to collect and process data. It demonstrates skills in API interaction, data processing, error handling, and working with rate limits. The project consists of several Python scripts, each interacting with different APIs to accomplish specific tasks.

## Project Structure
```
pipeline/
└── apis/
    ├── README.md
    ├── 0-passengers.py
    ├── 1-sentience.py
    ├── 2-user_location.py
    ├── 3-upcoming.py
    └── 4-rocket_frequency.py
```

## Requirements
### General
* All files interpreted/compiled on Ubuntu 16.04 LTS using Python 3.5
* Files must be executable
* Files should end with a new line
* First line of all files must be `#!/usr/bin/env python3`
* Code should follow pycodestyle style (version 2.4)
* All modules, classes, and functions must be documented
* String formatting must use `.format()` method

### Python Requirements
* Python 3.5+
* `requests` library
* Standard Python libraries (sys, time, collections, datetime)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Maxime-Bakunzi/alu-machine_learning.git
```

2. Navigate to the project directory:
```bash
cd alu-machine_learning/pipeline/apis
```

3. Install required packages:
```bash
pip3 install requests
```

## Scripts Description

### 0. Can I Join? (`0-passengers.py`)
* Script that uses the SWAPI (Star Wars API) to find ships that can hold a specified number of passengers
* Function: `availableShips(passengerCount)`
* Returns a list of ship names that meet the passenger capacity requirement
* Handles API pagination

### 1. Where I am? (`1-sentience.py`)
* Script that uses SWAPI to find home planets of all sentient species
* Function: `sentientPlanets()`
* Returns a list of planet names
* Handles API pagination and potential missing data

### 2. Rate me if you can! (`2-user_location.py`)
* Script that uses GitHub API to find a user's location
* Takes full GitHub API URL as command line argument
* Handles various API responses:
  * User not found: Prints "Not found"
  * Rate limit exceeded: Prints "Reset in X min"
* Includes proper rate limit handling

### 3. What will be next? (`3-upcoming.py`)
* Script that uses SpaceX API to display information about the next upcoming launch
* Shows launch name, date, rocket name, and launchpad information
* Formats output as: `<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)`
* Sorts launches by date using Unix timestamps

### 4. How many by rocket? (`4-rocket_frequency.py`)
* Script that uses SpaceX API to display launch frequency per rocket
* Counts all launches (past and upcoming)
* Sorts results by:
  1. Number of launches (descending)
  2. Rocket name (alphabetically)
* Formats output as: `<rocket name>: <launch count>`

## Usage Examples

### 0-passengers.py
```bash
./0-passengers.py
CR90 corvette
Sentinel-class landing craft
# ... other ships that can hold 4 or more passengers
```

### 1-sentience.py
```bash
./1-sentience.py
Endor
Naboo
# ... other planets with sentient species
```

### 2-user_location.py
```bash
./2-user_location.py https://api.github.com/users/holbertonschool
San Francisco, CA
```

### 3-upcoming.py
```bash
./3-upcoming.py
O3b mPower 1,2 (2022-06-30T20:00:00-04:00) Falcon 9 - CCSFS SLC 40 (Cape Canaveral)
```

### 4-rocket_frequency.py
```bash
./4-rocket_frequency.py
Falcon 9: 104
Falcon 1: 5
Falcon Heavy: 3
```

## Error Handling
* All scripts include comprehensive error handling for:
  * API connection issues
  * Rate limiting
  * Missing or malformed data
  * Invalid input parameters

## API Rate Limits
* GitHub API: 60 requests per hour for unauthenticated requests
* SWAPI: No rate limit
* SpaceX API: No rate limit

## Testing
* Each script can be tested using the provided main files
* Make sure to handle rate limits when testing GitHub API
* Consider using mock data for testing rate limit scenarios

## Authors
[Maxime Guy Bakunzi](https://github.com/Maxime-Bakunzi)

