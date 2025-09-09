import json
import re
import html
from datetime import datetime
from collections import defaultdict

# Convert UNIX timestamp to datetime
def unix_to_datetime(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp)

# Preprocess the body text for neural network training
def preprocess_text(text):
    # Decode HTML entities
    text = html.unescape(text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Location keywords list
location_keywords = [
    ' america ', 'united states', ' usa ', 'the us', 'u.s.', 'stateside', 'across the states',
    # US States
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 
    'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 
    'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 
    'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 
    'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 
    'utah', 'vermont', 'virginia', 'washington', ' DC ', 'west virginia', 'wisconsin', 'wyoming',
    # Major US Cities by population (Top 50)
    'new york', ' ny ', ' nyc ', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia', 'san antonio', 'san diego',
    'dallas', 'san jose', 'austin', 'jacksonville', 'fort worth', 'columbus', 'indianapolis', 'charlotte', 'san francisco', 'seattle',
    'nashville', 'denver', 'oklahoma city', 'el paso', 'boston', 'portland', 'las vegas', 'vegas', 'detroit', 'memphis', 'louisville',
    'baltimore', 'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento', 'kansas city', ' mesa ', 'atlanta', 'omaha',
    'colorado springs', 'raleigh', 'long beach', 'virginia beach', 'miami', 'oakland', 'minneapolis', 'tulsa', 'bakersfield',
    'wichita', 'arlington', 'aurora', 'tampa', 'new orleans', 'cleveland', 'honolulu', 'anaheim', 'lexington', 'stockton',
    'corpus christi', 'henderson', 'riverside', 'newark', 'st. paul', 'santa ana', 'cincinnati', 'irvine', 'orlando', 'pittsburgh',
    'st. louis', 'greensboro', 'jersey city', 'anchorage', 'lincoln', 'plano', 'durham', 'buffalo', 'chandler', 'chula vista',
    'toledo', 'madison', 'gilbert', ' reno ', 'fort wayne', 'north las vegas', 'st. petersburg', 'lubbock', 'irving', 'laredo',
    'winston-salem', 'chesapeake', 'glendale', 'garland', 'scottsdale', 'norfolk', 'boise', 'fremont', 'spokane', 'santa clarita',
    'baton rouge', 'richmond', 'hialeah',
    # Tourist spots
    'grand canyon', 'yellowstone', 'hollywood', 'niagara', 'disney world', 'yosemite', 'central park',
    # Transportation services
    'amtrak', 'greyhound', 'interstate'
]

# Filter records based on conditions and convert JSON to TSV
def filter_records(file_path, search_keywords, location_keywords, tsv_file_path, monthly_counts_file_path):
    
    # ☆ Mod
    # Updated date range to include end of day on December 31, 2022
    start_date = datetime(2012, 1, 1, 0, 0, 0)
    end_date = datetime(2022, 12, 31, 23, 59, 59)
    
    tsv_lines = []
    headers = ["created_date", "subreddit_id", "id", "author", "parent_id", "body", "score"]  # Added "score" to the headers
    processed_bodies = set()  # To track processed body text
    
    # Dictionary to count records by month (key: 'YYYY-MM', value: count)
    month_counts = defaultdict(int)
    
    # Open the file and process each line as a separate JSON object
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                record = json.loads(line.strip())
                
                # Ensure 'created_utc' is treated as an integer
                created_utc = int(record.get("created_utc", 0))
                created_date = unix_to_datetime(created_utc)
                
                # Extract additional fields
                subreddit_id = record.get('subreddit_id', '')
                comment_id = record.get('id', '')
                author = record.get('author', '')
                parent_id = record.get('parent_id', '')
                score = record.get('score', 0)  # Extracting the score field
                is_self = record.get('is_self', False)  # Check if the post is a text-only post
                
                # Only process text-only posts (is_self is True)
                if not is_self:
                    continue
                
                # Preprocess the body text
                body_text = preprocess_text(record.get('selftext', ''))
                
                # Skip if the body text has already been processed (i.e., is a duplicate)
                if body_text in processed_bodies:
                    continue
                # Add the body text to the set to track it
                processed_bodies.add(body_text)
                
                # Check if any of the search_keywords are in the processed 'body' field
                # AND any of the location_keywords are in the body field
                # Also exclude comments mentioning 'south america' or 'latin america'
                if (any(keyword in body_text.lower() for keyword in search_keywords) and 
                    any(location in body_text.lower() for location in location_keywords) and
                    'south america' not in body_text.lower() and
                    'latin america' not in body_text.lower()):
                    if start_date <= created_date <= end_date:
                        # Prepare TSV line with all required fields, including 'score'
                        tsv_line = "\t".join([
                            created_date.isoformat(),
                            subreddit_id,
                            comment_id,
                            author,
                            parent_id,
                            body_text,
                            str(score)  # Add score as string for TSV
                        ])
                        tsv_lines.append(tsv_line)
                        
                        # Extract the year and month to track the count
                        year_month = created_date.strftime('%Y-%m')  # 'YYYY-MM' format
                        month_counts[year_month] += 1
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except ValueError as e:
                print(f"Value error: {e}")

    # Write the TSV lines to the output file
    with open(tsv_file_path, 'w', encoding='utf-8') as tsv_file:
        # Write header line
        tsv_file.write("\t".join(headers) + "\n")
        # Write data lines
        for line in tsv_lines:
            tsv_file.write(line + "\n")
    
    # Write monthly counts to a separate tab-delimited file for Excel
    with open(monthly_counts_file_path, 'w', encoding='utf-8') as count_file:
        # Write header
        count_file.write("Month\tCount\n")
        # Write data lines
        for month, count in sorted(month_counts.items()):
            count_file.write(f"{month}\t{count}\n")
    
    # Output the monthly counts to the standard output as well
    print("\nMonthly record counts:")
    for month, count in sorted(month_counts.items()):
        print(f"{month}  {count}")

    return len(tsv_lines)  # Return the number of filtered records

# ☆ Mod
file_path = '/Users/ryuichi/Documents/research/extraction/data/travel_submissions.json'  # Replace with the path to your JSON file
search_keywords = ['price', 'cost', 'inflation', 'deflation', 'expensive', 'cheap', 'purchase', 'sale', 'increasing', 'decreasing', 'rising', 'falling', 'affordable', 'unaffordable']  # List of keywords
# ☆ Mod
tsv_file_path = '/Users/ryuichi/Documents/research/extraction/data/travel_submissions_2012_2022.tsv'  # Output TSV file path
# ☆ Mod
monthly_counts_file_path = '/Users/ryuichi/Documents/research/extraction/data/travel_submissions_2012_2022_counts.txt'  # Monthly counts file path

filtered_count = filter_records(file_path, search_keywords, location_keywords, tsv_file_path, monthly_counts_file_path)

# Print the number of filtered records
print(f"Number of filtered records: {filtered_count}")