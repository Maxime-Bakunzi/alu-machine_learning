# Databases

This project is part of the Machine Learning Specialization at African Leadership University. It covers SQL (MySQL), NoSQL (MongoDB), and their integration with Python for data handling and analysis.

## Project Overview

The project consists of 35 tasks (0-34) that explore various aspects of database management, querying, and data manipulation using both SQL and MongoDB. It provides hands-on experience with:

- MySQL database management and querying
- MongoDB document database operations
- Python scripts for database interaction
- Data analysis and statistics generation
- Integration of different database systems

## Environment & Requirements

### General Requirements
- All files interpreted/compiled on Ubuntu 16.04/18.04 LTS
- Files end with a new line
- README.md file at the root of the project folder is mandatory

### MySQL Requirements
- MySQL 5.7 (version 5.7.30)
- All SQL keywords should be in uppercase (SELECT, WHERE...)
- SQL files should include comments before queries

### MongoDB Requirements
- MongoDB (version 4.2)
- First line of Mongo files should be a comment
- PyMongo (version 3.10)

### Python Requirements
- Python 3.5 or 3.7
- Libraries: PyMongo, MySQLdb
- Files must start with `#!/usr/bin/env python3`
- Code should follow pycodestyle (version 2.5)
- All modules and functions must have documentation
- Code should not execute when imported

## Tasks

### MySQL Basic Operations (0-4)

#### 0. Create a Database
- File: `0-create_database_if_missing.sql`
- Creates the database `metal_bands` if it doesn't exist
- Does not fail if database exists

#### 1. First Table
- File: `1-first_table.sql`
- Creates table `first_table` with columns:
  - id INT
  - name VARCHAR(256)

#### 2. List all in table
- File: `2-list_values.sql`
- Lists all rows in `first_table`
- Uses SELECT statement

#### 3. Insert Value
- File: `3-insert_value.sql`
- Inserts new row in `first_table`
- Values: id = 89, name = "Best School"

#### 4. Best Score
- File: `4-best_score.sql`
- Lists records with score >= 10 in second_table
- Orders results by score (top first)

### Data Analysis and Aggregation (5-7)

#### 5. Average
- File: `5-average.sql`
- Computes score average of all records in second_table
- Result column name: average

#### 6. Temperatures #0
- File: `6-avg_temperatures.sql`
- Displays average temperature by city
- Ordered by temperature (descending)

#### 7. Temperatures #2
- File: `7-max_state.sql`
- Displays max temperature of each state
- Ordered by state name

### Advanced Queries and Joins (8-12)

#### 8. Genre ID by Show
- File: `8-genre_id_by_show.sql`
- Lists shows with at least one genre linked
- Sorted by title and genre_id

#### 9. No Genre
- File: `9-no_genre.sql`
- Lists shows without a genre linked
- Sorted by title

#### 10. Number of Shows by Genre
- File: `10-count_shows_by_genre.sql`
- Lists genres and number of shows linked
- Sorted by number of shows linked (descending)

#### 11. Rating Shows
- File: `11-rating_shows.sql`
- Lists shows by their rating sum
- Sorted by rating (descending)

#### 12. Rating Genres
- File: `12-rating_genres.sql`
- Lists genres by their rating sum
- Sorted by rating (descending)

### User Management and Data Integrity (13-16)

#### 13. Unique Users
- File: `13-uniq_users.sql`
- Creates table users with unique email constraint

#### 14. Country Users
- File: `14-country_users.sql`
- Creates table users with country enumeration

#### 15. Band Fans
- File: `15-fans.sql`
- Lists bands with their fan count
- Ordered by number of fans (descending)

#### 16. Glam Rock
- File: `16-glam_rock.sql`
- Lists Glam rock bands ranked by longevity

### Advanced Database Operations (17-21)

#### 17. Store Engine
- File: `17-store.sql`
- Creates table with specific storage engine

#### 18. Valid Email
- File: `18-valid_email.sql`
- Implements email validation trigger

#### 19. Bonus
- File: `19-bonus.sql`
- Creates stored procedure AddBonus

#### 20. Average Score
- File: `20-average_score.sql`
- Computes average score for a student

#### 21. Safe Divide
- File: `21-div.sql`
- Creates function SafeDiv for safe division

### MongoDB Basic Operations (22-29)

#### 22. List Databases
- File: `22-list_databases`
- Lists all databases in MongoDB

#### 23. Create Database
- File: `23-use_or_create_database`
- Creates/uses a database

#### 24. Insert Document
- File: `24-insert`
- Inserts document in collection

#### 25. All Documents
- File: `25-all`
- Lists all documents in collection

#### 26. Match Documents
- File: `26-match`
- Lists documents matching criteria

#### 27. Count Documents
- File: `27-count`
- Counts number of documents in collection

#### 28. Update Document
- File: `28-update`
- Updates document based on criteria

#### 29. Delete Documents
- File: `29-delete`
- Deletes documents matching criteria

### Python and MongoDB Integration (30-34)

#### 30. List all documents in Python
- File: `30-all.py`
- Python function to list all documents
- Returns empty list if no documents

#### 31. Insert School
- File: `31-insert_school.py`
- Inserts new document in collection
- Returns new _id

#### 32. Update Topics
- File: `32-update_topics.py`
- Updates document topics based on name
- Changes all topics of school document

#### 33. Schools by Topic
- File: `33-schools_by_topic.py`
- Returns list of schools having specific topic
- Uses find() method

#### 34. Log Stats
- File: `34-log_stats.py`
- Provides Nginx logs statistics from MongoDB
- Shows total logs and method counts
- Displays status check count

## Project Structure
```
pipeline/
└── databases/
    ├── 0-create_database_if_missing.sql
    ├── 1-first_table.sql
    ├── 2-list_values.sql
    ├── 3-insert_value.sql
    ├── 4-best_score.sql
    ├── 5-average.sql
    ├── 6-avg_temperatures.sql
    ├── 7-max_state.sql
    ├── 8-genre_id_by_show.sql
    ├── 9-no_genre.sql
    ├── 10-count_shows_by_genre.sql
    ├── 11-rating_shows.sql
    ├── 12-rating_genres.sql
    ├── 13-uniq_users.sql
    ├── 14-country_users.sql
    ├── 15-fans.sql
    ├── 16-glam_rock.sql
    ├── 17-store.sql
    ├── 18-valid_email.sql
    ├── 19-bonus.sql
    ├── 20-average_score.sql
    ├── 21-div.sql
    ├── 22-list_databases
    ├── 23-use_or_create_database
    ├── 24-insert
    ├── 25-all
    ├── 26-match
    ├── 27-count
    ├── 28-update
    ├── 29-delete
    ├── 30-all.py
    ├── 31-insert_school.py
    ├── 32-update_topics.py
    ├── 33-schools_by_topic.py
    └── 34-log_stats.py
```

## Setup and Usage

### MySQL Setup
```bash
# Install MySQL 5.7
sudo apt-get install mysql-server
mysql -uroot -p
```

### MongoDB Setup
```bash
# Install MongoDB 4.2
wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" > /etc/apt/sources.list.d/mongodb-org-4.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
```

### Python Dependencies
```bash
pip3 install pymongo
pip3 install mysqlclient
```

## Author
[Maxime Guy Bakunzi](https://github.com/Maxime-Bakunzi)

## Repository Information
- GitHub repository: `alu-machine_learning`
- Directory: `pipeline/databases`