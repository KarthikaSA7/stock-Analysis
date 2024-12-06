import requests
import shutil
import ast
import glob
import re
import asyncio
import http.client
import aiohttp
import csv
import pandas as pd
import json
import os
import time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Access the environment variables
chatgpt_password = os.getenv('CHATGPT_SECRET_KEY')
adword_password = os.getenv('ADWORD_KEY')
llama_password = os.getenv('LLAMA_KEY')
gtrend_password = os.getenv('GTREND_KEY')
twitter_password = os.getenv('TWITTER_KEY')
seeking_alpha_password = os.getenv('SEEKING_ALPHA_KEY')

app = Flask(__name__)

application = app

CHATGPT_URL = 'https://open-ai21.p.rapidapi.com/chatgpt'
CHATGPT_SECRET_KEY = chatgpt_password
CHATGPT_HOST = 'open-ai21.p.rapidapi.com'

ADWORD_URL = 'https://google-keyword-insight1.p.rapidapi.com/keysuggest/'
ADWORD_KEY = adword_password
ADWORD_HOST = 'google-keyword-insight1.p.rapidapi.com'

LLAMA_URL = "https://meta-llama-3-8b.p.rapidapi.com/"
LLAMA_KEY = llama_password
LLAMA_HOST = "meta-llama-3-8b.p.rapidapi.com"

GTREND_URL = 'https://serpapi.com/search.json'
GTREND_KEY = gtrend_password

TWITTER_URL = 'https://twitter154.p.rapidapi.com/search/search'
TWITTER_KEY = twitter_password
TWITTER_HOST = 'twitter154.p.rapidapi.com'

SEEKING_ALPHA_KEY = seeking_alpha_password
SEEKING_ALPHA_HOST = 'seeking-alpha.p.rapidapi.com'

@app.route('/')
def index():
    return render_template('index-main.html')

@app.route('/trends')
def trends():
    return render_template('index.html')
@app.route('/twitter')
def about():
    return render_template('index-twitter.html')
    
@app.route('/news')
def news():
    return render_template('index-news.html')

@app.route('/submit', methods=['POST'])
def analyze():
    def read_keyword_from_csv(filepath, column_name, number_of_keyword):
        column_data = []
        with open(filepath, mode='r',encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for i, row in enumerate(csv_reader):
                if column_name in row:
                    if i >= number_of_keyword:
                        break
                    column_data.append(row[column_name])
                else:
                    return None
        return column_data

    col_1 = []
    col_2 = []
    col_3 = []
    col_4 = []

    response_data = {}

    # Line Chart
    dataline = {}
    line_dataset = []
    line_colors = ['#FF0000', '#0000FF', '#008000', '#FFFF00', '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#000000', '#A02334']
    line_label = []

    label = ['Positive', 'Negative', 'Neutral']
    pie_bar_data = []
    color = ['#FFFF00', '#FFA500', '#800080']
    border_color = '#fff'
    border_width = 1

    # Pie Chart
    datapie = {}
    pie_dataset = []
    pie_response_data = {}

    # Bar Chart
    databar = {}
    bar_dataset = []
    bar_response_data = {}
    bar_data_label = 'Sentiment Percentage'

    # Initialize lists for each category
    positive_names = []
    negative_names = []
    neutral_names = []  # Assuming no neutral entries for now

    all_data = []

    data = request.get_json()
    user_input = data['userInput']
    models_choose = data['model']

    for i in models_choose:

        # Chatgpt
        if i == 'chatgpt':
            # API request setup
            url = CHATGPT_URL

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Identify keywords related to the company \"{user_input}\" based on its products and services. These keywords will be utilized to analyze public demand trends for the company's offerings, which will in turn be used for forecasting the stock price. Give me a consolidated list of top ten keywords."
                    }
                ],
                "web_access": False
            }
            headers = {
                "x-rapidapi-key": CHATGPT_SECRET_KEY,
                "x-rapidapi-host": CHATGPT_HOST,
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                response_data = response.json()

                # Parse the result to get the list of keywords
                keywords_str = response_data['result']
                keywords_list = keywords_str.split("\n")[3:13]  # Extracting only the keywords

                # Clean up the list to remove numbering and whitespace
                keywords_list = [keyword.strip().split('. ')[-1] for keyword in keywords_list]

                # Create a directory for the company if it does not exist
                folder_name = user_input
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Write the list to a CSV file in the company's folder
                csv_filename = os.path.join(folder_name, f'{user_input.lower().replace(" ", "_")}_chatgpt_keywords.csv')

                with open(csv_filename, mode='w', newline='',encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['chatgpt'])  # Write the column name
                    for keyword in keywords_list:
                        writer.writerow([keyword])

                response_data['chatgpt'] = keywords_list
                file1 = pd.read_csv(csv_filename)
                col_1 = file1.iloc[:, 0].tolist()
                print(f"ChatGpt Keywords saved to {csv_filename}")

            else:
                print("You have exceeded the MONTHLY quota for Requests on your ChatGpt plan")

        # Adword
        if i == 'adword':
            filename = f"{user_input.lower().replace(' ', '_')}_googleadwords_keywords.csv"
            folder_name = user_input

            # Define the URL and query parameters
            url = ADWORD_URL
            querystring = {
                "keyword": user_input,
                "location": "US",
                "lang": "en"
            }
            headers = {
                "x-rapidapi-key": ADWORD_KEY,
                "x-rapidapi-host": ADWORD_HOST
            }

            # Make the API request
            response = requests.get(url, headers=headers, params=querystring)

            if response.status_code == 200:
                data = response.json()

                # Convert the JSON response to a DataFrame
                df = pd.DataFrame(data)

                # Create folder if it does not exist
                os.makedirs(folder_name, exist_ok=True)

                # Path to save the CSV file
                file_path = os.path.join(folder_name, filename)

                # Save DataFrame to CSV
                df.to_csv(file_path, index=False)

                print(f"adword Data saved to {file_path}")

                adwords_keywords = read_keyword_from_csv(file_path, column_name= 'text', number_of_keyword= 11)

                # data added to response

                response_data['adword'] = adwords_keywords
                file2 = pd.read_csv(file_path)
                col_2 = file2.iloc[:, 0].tolist()
            else:
                print("You have exceeded the MONTHLY quota for Requests on your adword  plan")

        # Llama
        if i == 'llama':

            # Generate folder name and file name based on company name
            folder_name = user_input
            file_name = f"{user_input.lower().replace(' ', '_')}_metaai_keywords.csv"

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Construct the full file path
            file_path = os.path.join(folder_name, file_name)

            url = LLAMA_URL

            payload = {
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Identify keywords related to the company '{user_input}' based on its products and services. These keywords will be utilized to analyze the public demand trends for the company's offerings, which will in turn be used for forecasting the stock price. Give me a consolidated list of top ten keywords. Keywords need not be explained. I just need the list of keywords."
                    }
                ]
            }
            headers = {
                "x-rapidapi-key": LLAMA_KEY,
                "x-rapidapi-host": LLAMA_HOST,
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                # Parse the JSON response to extract keywords
                response_json = response.json()

                # Extract keywords
                try:
                    keywords_text = response_json['choices'][0]['message']['content']
                    # Extract keywords from the response text, skipping the first line
                    keywords = [line.strip() for line in keywords_text.split('\n')[1:] if line.strip() and not line.strip().isdigit()]

                    # Remove numbering from keywords
                    keywords = [keyword.split('. ')[-1] for keyword in keywords]

                    # Write keywords to CSV file
                    with open(file_path, mode='w', newline='',encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(['meta_ai'])
                        for keyword in keywords:
                            writer.writerow([keyword])

                    response_data['llama'] = keywords
                    file3 = pd.read_csv(file_path)
                    col_3 = file3.iloc[:, 0].tolist()

                    print(f"llama Keywords have been saved to {file_path}")
                except KeyError as e:
                    print(f"KeyError: {e}. Please check the structure of the response JSON.")

            else:
                print("You have exceeded the MONTHLY quota for Requests on your llama plan")

        # Gtrent
        if i == 'gtrend':
            url = GTREND_URL
            params = {
                'engine': 'google_trends',
                'q': user_input,  # Replace with your search query
                'data_type': 'RELATED_QUERIES',
                'api_key': GTREND_KEY
                # Replace with your actual API key
            }

            # Make the HTTP GET request
            response = requests.get(url, params=params)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                results = response.json()
                related_queries = results.get('related_queries', [])

                # Convert related queries to a DataFrame
                df = pd.DataFrame(related_queries)

                # Save the DataFrame to a CSV file
                df.to_csv("related_queries.csv", index=False)

                # Define the input and output file paths
                input_file = 'related_queries.csv'
                output_folder = user_input
                filename = f"{user_input.lower().replace(' ', '_')}_gtrend_queries.csv"
                output_file = os.path.join(output_folder, filename)

                # Ensure the output folder exists
                os.makedirs(output_folder, exist_ok=True)

                # Read the CSV file
                df = pd.read_csv(input_file, header=None, names=['data'])

                # Extract 'query' text from each row
                def extract_query(text):
                    try:
                        data_dict = ast.literal_eval(text)
                        return data_dict.get('query', '')
                    except (ValueError, SyntaxError):
                        return ''

                # Apply the extraction function to the DataFrame
                df['query'] = df['data'].apply(extract_query)

                # Remove empty rows
                df = df[df['query'].str.strip() != '']

                # Rename the column
                df = df[['query']].rename(columns={'query': 'GTrend Related Queries'})

                # Save the extracted queries to a new CSV file
                df.to_csv(output_file, index=False)

                print(f"gtrend Extracted queries have been saved to {output_file}.")

                gtrends_keywords = read_keyword_from_csv(output_file, column_name='GTrend Related Queries',
                                                         number_of_keyword=11)
                response_data['gtrends'] = gtrends_keywords
                file4 = pd.read_csv(output_file)
                col_4 = file4.iloc[:, 0].tolist()
                
            else:
                print(f"Error: {response.status_code}. gtrent reach monthly hitted")


    # Combine all columns into one list
    all_texts = col_1 + col_2 + col_3 + col_4

    # Count occurrences of each text
    text_counts = pd.Series(all_texts).value_counts()

    # Create a DataFrame to hold the results
    results = pd.DataFrame({
        'Text': text_counts.index,
        'Frequency': text_counts.values
    })

    # Sort by frequency (higher first)
    results_sorted = results.sort_values(by='Frequency', ascending=False)

    # Save the sorted results to a new CSV file
    results_sorted.to_csv('sorted_texts.csv', index=False)

    # Data for Sentiment Analysis
    sorted_keyword = read_keyword_from_csv('sorted_texts.csv', 'Text', 19)

    # Read keywords from CSV file
    input_file = 'sorted_texts.csv'  # Replace with your actual CSV file name

    # Read keywords from column A
    with open(input_file, mode='r',encoding='utf-8') as file:
        reader = csv.reader(file)
        keywords_list = [row[0] for row in reader if row]  # Reading column A

    # Convert the list of keywords to a single string for the API request
    keywords_string = ', '.join(keywords_list)

    # Prepare API request
    url = "https://meta-llama-3-8b.p.rapidapi.com/"

    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {
                "role": "user",
                "content": f"From the different keywords present in the file, pick exactly 10 keywords that look the most important. The keywords should reflect the company's growth prospects. In the output if you print only the keywords that sufficient, description or explanation is not required. Keywords: {keywords_string}"
            }
        ]
    }
    headers = {
        "x-rapidapi-key": LLAMA_KEY,  # Replace with your actual API key
        "x-rapidapi-host": LLAMA_HOST,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # Parse the JSON response to extract keywords
    response_json = response.json()

    # Print the entire response for debugging
    print("llama Response JSON:")

    # Extract keywords
    try:
        keywords_text = response_json['choices'][0]['message']['content']
        # Extract keywords from the response text, skipping the first line
        keywords = [line.strip() for line in keywords_text.split('\n')[1:] if
                    line.strip() and not line.strip().isdigit()]

        # Remove numbering from keywords
        keywords = [keyword.split('. ')[-1] for keyword in keywords]

        # Save keywords to a new CSV file
        output_file = 'important_keywords.csv'  # Replace with desired output file name
        with open(output_file, mode='w', newline='',encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['important_keywords'])
            for keyword in keywords:
                writer.writerow([keyword])

        print(f"Important keywords have been saved to {output_file}")
    except KeyError as e:
        print(f"Reach monthly limit.KeyError: {e}. Please check the structure of the response JSON.")

    # Handle response error from llama
    if response.status_code == 200:
        important_keywords = read_keyword_from_csv('important_keywords.csv', column_name='important_keywords', number_of_keyword=10)
        response_data['consolidated'] = important_keywords


        async def get_google_trends_data(session, keyword):
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_trends",
                "q": keyword,
                "hl": "en",
                "api_key": GTREND_KEY
            }

            async with session.get(url, params=params) as response:
                data = await response.json()
                return data, response

        def process_trends_data(data, keyword):
            # Extract timeline data
            timeline_data = data.get("interest_over_time", {}).get("timeline_data", [])

            # Create a list to store structured data
            structured_data = []

            for entry in timeline_data:
                date = entry.get('date', '')
                timestamp = entry.get('timestamp', '')
                for value in entry.get('values', []):
                    structured_data.append({
                        'Keyword': keyword,
                        'Date': date,
                        'Timestamp': timestamp,
                        'Query': value.get('query', ''),
                        'Value': value.get('value', ''),
                        'Extracted Value': value.get('extracted_value', '')
                    })

            # Convert the structured data into a DataFrame
            df = pd.DataFrame(structured_data)
            return df, structured_data

        async def process_keyword(session, keyword, index):
            keyword = keyword.lower().replace('*', '')
            chart_data = {}
            google_trends_data, response = await get_google_trends_data(session, keyword)

            if response.status == 200:
                trends_df, structured_data = process_trends_data(google_trends_data, keyword)
                filename = f"{keyword.lower().replace(' ', '_').replace('*', '')}_google_trends.csv"
                trends_df.to_csv(filename, index=False)

                data = read_keyword_from_csv(filename, column_name='Extracted Value', number_of_keyword=53)
                data = [int(i) for i in data]

                print("Final Gtrend result:", keyword)
                chart_data['label'] = keyword
                chart_data['data'] = data
                chart_data['borderColor'] = line_colors[index % len(line_colors)]  # Handle color cycling
                chart_data['borderWidth'] = 2
                chart_data['fill'] = False
                line_dataset.append(chart_data)

                labels_line = read_keyword_from_csv(filename, column_name='Date', number_of_keyword=53)
                for i in labels_line:
                    line_label.append(i)

            else:
                print(f"Failed to fetch data for {keyword}")

        async def process_sorted_keyword(session, keyword):
            keyword = keyword.lower().replace('*', '')
            google_trends_data, respose = await get_google_trends_data(session, keyword)

            if respose.status == 200:
                trends_df, structured_data = process_trends_data(google_trends_data, keyword)
                print(f"sorted keyword {keyword}")

                all_data.extend(structured_data)
                
        async def main():
            important_keywords = read_keyword_from_csv('important_keywords.csv', column_name='important_keywords', number_of_keyword=10)

            async with aiohttp.ClientSession() as session:
                tasks1 = [process_keyword(session, keyword, index) for index, keyword in enumerate(important_keywords)]
                await asyncio.gather(*tasks1)

                tasks2 = [process_sorted_keyword(session, keyword) for keyword in sorted_keyword]
                await asyncio.gather(*tasks2)
            
        asyncio.run(main())
            # Compined data
        if len(all_data) != 0:
            combined_df = pd.DataFrame(all_data)

            # Optionally, save the DataFrame to a CSV file
            combined_df.to_csv("combined_google_trends.csv", index=False)

            # Chart
            file_path = 'combined_google_trends.csv'
            df = pd.read_csv(file_path)

            # Function to determine the trend of values for each keyword
            def categorize_trend(start, end):
                if end > start:
                    return 'Positive'
                elif end < start:
                    return 'Negative'
                else:
                    return 'Neutral'

            # Group by keyword and aggregate the necessary data
            aggregated_data = df.groupby('Keyword').agg(
                Start_Date=('Date', 'first'),
                End_Date=('Date', 'last'),
                Min_Value=('Extracted Value', 'min'),
                Max_Value=('Extracted Value', 'max'),
                Start_Value=('Extracted Value', 'first'),
                End_Value=('Extracted Value', 'last')
            ).reset_index()

            # Apply the trend categorization based on start and end values
            aggregated_data['Trend'] = aggregated_data.apply(
                lambda row: categorize_trend(row['Start_Value'], row['End_Value']), axis=1
            )

            # Drop the temporary columns for start and end values
            aggregated_data = aggregated_data.drop(columns=['Start_Value', 'End_Value'])

            # Save the categorized trends with additional data to a new CSV file
            output_path = 'trends_with_dates.csv'
            aggregated_data.to_csv(output_path, index=False)

            # Define the input and output CSV file names
            sentiment_csv_file = 'trends_with_dates.csv'

            # Read the data from the input CSV file
            with open(sentiment_csv_file, mode='r',encoding='utf-8') as infile:
                reader = csv.reader(infile)

                # Skip the header row
                header = next(reader)

                for row in reader:
                    name = row[0]
                    sentiment = row[5]

                    if sentiment == 'Positive':
                        positive_names.append(name)
                    elif sentiment == 'Negative':
                        negative_names.append(name)
                    else:
                        neutral_names.append(name)  # This line is not necessary if there are no neutral entries

            # Calculate the summary statistics
            total_keywords = aggregated_data['Keyword'].nunique()
            positive_count = aggregated_data[aggregated_data['Trend'] == 'Positive'].shape[0]
            negative_count = aggregated_data[aggregated_data['Trend'] == 'Negative'].shape[0]
            neutral_count = aggregated_data[aggregated_data['Trend'] == 'Neutral'].shape[0]

            # Prepare the summary data
            summary_data = {
                'Total Keywords': [total_keywords],
                'Positive Keywords': [positive_count],
                'Negative Keywords': [negative_count],
                'Neutral Keywords': [neutral_count]
            }

            summary_df = pd.DataFrame(summary_data)

            # Save the summary to a separate CSV file
            summary_output_path = f'{user_input.lower().replace(" ", "_")}_trends_summary.csv'
            summary_df.to_csv(summary_output_path, index=False)

            Total_keywords = int(read_keyword_from_csv(summary_output_path, column_name='Total Keywords', number_of_keyword=1)[0])
            Positive_keywords = int(read_keyword_from_csv(summary_output_path, column_name='Positive Keywords', number_of_keyword=1)[0])
            Negative_keywords = int(read_keyword_from_csv(summary_output_path, column_name='Negative Keywords', number_of_keyword=1)[0])
            Neutral_keywords = int(read_keyword_from_csv(summary_output_path, column_name='Neutral Keywords', number_of_keyword=1)[0])

            # Percentage calculate
            Positive_keywords = (Positive_keywords / Total_keywords) * 100
            Negative_keywords = (Negative_keywords / Total_keywords) * 100
            Neutral_keywords = (Neutral_keywords / Total_keywords) * 100

            pie_bar_data = [Positive_keywords, Negative_keywords, Neutral_keywords]
    
            # Pie
            pie_response_data['data'] = pie_bar_data
            pie_response_data['backgroundColor'] = color
            pie_response_data['borderColor'] = border_color
            pie_response_data['borderWidth'] = border_width

            # Bar
            bar_response_data['label'] = bar_data_label
            bar_response_data['data'] = pie_bar_data
            bar_response_data['backgroundColor'] = color
            bar_response_data['borderColor'] = border_color
            bar_response_data['borderWidth'] = border_width

            pie_dataset.append(pie_response_data)
            bar_dataset.append(bar_response_data)

            datapie['labels'] = label
            datapie['datasets'] = pie_dataset

            databar['labels'] = label
            databar['datasets'] = bar_dataset

            dataline['datasets'] = line_dataset

            # Clean data
            line_label = [date.replace('â€‰â€“â€‰', '–') for date in line_label]
            line_label = line_label[:53]
            dataline['labels'] = line_label
        else:
            print("combined data is not created")

    else:
        print(f"response is not get from llama.")

    # Response
    response_data['dataLine'] = dataline
    response_data['dataPie'] = datapie
    response_data['dataBar'] = databar
    response_data['positiveNames'] = positive_names
    response_data['negativeNames'] = negative_names
    response_data['neutralNames'] = neutral_names
    response_data['type'] = "gtrend"
    json_data = jsonify(response_data)
    # Define the path to the JSON file
    file_path = 'stock_response_data.json'

    # Save the JSON data to the file
    with open(file_path, 'w',encoding='utf-8') as file:
        json.dump(response_data, file, indent=4)

    # json_file_path = 'stock_response_data.json'
    # Open the JSON file and read its contents
    # with open(json_file_path, 'r') as file:
    # Load the JSON data into a dictionary
    # response_data = json.load(file)

    # json_data = jsonify(response_data)
    response = make_response(json_data)
    response.status_code = 200
    return response

@app.route('/twitter-submit', methods=['POST'])
def twitter():
    response_data = {}
    keywords_data = []
    tweets = []
    sentiments = []
    pie_data = {}
    keyword_generated = []
    labels = ['Positive', 'Negative', 'Neutral']
    datasets = []
    chart_label = 'Tweet Sentiment'
    backgroundColor = ['#4CAF50', '#F44336', '#FFC107']

    data = request.get_json()
    print(data)
    user_input = data['userInput']

    # Construct the full file path
    file_path = 'keywords.csv'

    url = LLAMA_URL

    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {
                "role": "user",
                "content": f"Get two keywords about the company '{user_input}' that is used by financial market investors. Give me the output as a plain text list of keywords without any additional content."
            }
        ]
    }
    headers = {
        "x-rapidapi-key": LLAMA_KEY,
        "x-rapidapi-host": LLAMA_HOST,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # Parse the JSON response to extract keywords
    response_json = response.json()

    # Print the entire response for debugging
    print("Response JSON:")

    # Extract keywords
    try:
        keywords_text = response_json['choices'][0]['message']['content']
        # Extract keywords from the response text, skipping the first line
        keywords = [line.strip() for line in keywords_text.split('\n')[1:] if
                    line.strip() and not line.strip().isdigit()]

        # Remove numbering from keywords
        keywords = [keyword.split('. ')[-1] for keyword in keywords]

        # Add the company name as the first keyword
        keywords.insert(0, user_input)

        # Remove special characters and punctuation marks (except space and hyphen)
        keywords = [re.sub(r'[^\w\s-]', '', keyword) for keyword in keywords]

        # Write keywords to CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['meta_ai'])
            for keyword in keywords:
                writer.writerow([keyword])

        print(f"Keywords have been saved to {file_path}")
    except KeyError as e:
        print(f"KeyError: {e}. Please check the structure of the response JSON.")

    # Directory where the CSV files will be saved
    directory = "extracted_tweets"

    # Check if the directory exists
    if os.path.exists(directory):
        # If the directory exists, delete its contents
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # If the directory does not exist, create it
        os.makedirs(directory)

    # Read keywords from the CSV file, skipping the first row (header)
    keywords = []
    with open('keywords.csv', mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        keywords = [row[0] for row in reader]  # Assuming each row has one keyword

    # Ensure we have exactly three keywords
    if len(keywords) < 3:
        raise ValueError("The keywords.csv file must contain at least 3 keywords after the header.")
    elif len(keywords) > 3:
        keywords = keywords[:3]  # Only take the next three keywords

    # Define the list of query strings based on keywords
    query_strings = [
        {"query": keywords[0], "min_retweets": "1", "min_likes": "1", "limit": "20",
         "start_date": "2024-07-01", "language": "en"},
        {"query": keywords[1], "min_retweets": "1", "min_likes": "1", "limit": "20",
         "start_date": "2024-07-01", "language": "en"},
        {"query": keywords[2], "min_retweets": "1", "min_likes": "1", "limit": "20",
         "start_date": "2024-07-01", "language": "en"},
    ]

    # API endpoint and headers
    url = TWITTER_URL
    headers = {
        'X-RapidAPI-Key': TWITTER_KEY,
        'X-RapidAPI-Host': TWITTER_HOST
    }

    # Loop through each query string
    for query_string in query_strings:
        keyword_data = {}
        # Send GET request to the Twitter API
        response = requests.request("GET", url, headers=headers, params=query_string)
        data = response.json()

        # Specify the CSV file name based on the query and start date
        query_name = query_string["query"].replace(" ", "_").lower()
        start_date = query_string["start_date"]
        csv_file = os.path.join(directory, f"{query_name}_{start_date}.csv")

        # Overwrite the file if it exists, else create a new file
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Write header row
            writer.writerow(["target", "id", "date", "flag", "user", "text"])

            # Write data rows for English tweets only
            for tweet in data["results"]:
                if tweet.get("language") == "en":
                    writer.writerow([" ", tweet["tweet_id"], tweet["creation_date"], " ", " ", tweet["text"]])

        # Print completion message for each query
        print(f"Data for {query_string['query']} extracted to:", csv_file)

        df = pd.read_csv(csv_file)
        # Get the number of rows
        row_count = len(df)
        keyword_data['keyword'] = query_string['query']
        keyword_data['tweet-count'] = row_count
        keywords_data.append(keyword_data)

    # Directory containing the input CSV files
    input_directory = 'extracted_tweets'

    # API endpoint and headers
    url = LLAMA_URL
    headers = {
        "x-rapidapi-key": LLAMA_KEY,
        "x-rapidapi-host": LLAMA_HOST,
        "Content-Type": "application/json"
    }

    async def get_sentiment_response(session, tweet_text, requests_made):
        payload = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {
                    "role": "user",
                    "content": f"Can you read this tweet and tell me if the sentiment reflecting in this tweet about the company is positive, negative, or neutral? Here is the tweet: {tweet_text}"
                }
            ]
        }

        # Check if the request limit is about to be exceeded
        if requests_made >= 28:  # A safe threshold just before hitting the limit
            print("Rate limit approaching, sleeping for 60 seconds...")
            await asyncio.sleep(60)  # Wait for the rate limit window to reset
            requests_made = 0  # Reset request counter after waiting

        async with session.post(url, json=payload, headers=headers) as response:
            response_json = await response.json()

            if 'error' in response_json and response_json['error']['code'] == 'rate_limit_exceeded':
                print("Rate limit exceeded, retrying after 60 seconds...")
                await asyncio.sleep(60)
                return await get_sentiment_response(session, tweet_text, requests_made)

            try:
                sentiment_text = response_json['choices'][0]['message']['content']
                return sentiment_text.strip(), requests_made + 1
            except KeyError as e:
                print(f"KeyError: {e}. Please check the structure of the response JSON.")
                return "error", requests_made + 1

    def determine_sentiment(sentiment_response):
        sentiment_response = sentiment_response.lower()

        pos_index = sentiment_response.find("positive")
        neg_index = sentiment_response.find("negative")
        neu_index = sentiment_response.find("neutral")

        indices = [(pos_index, "positive"), (neg_index, "negative"), (neu_index, "neutral")]

        indices = [(idx, sentiment) if idx != -1 else (float('inf'), sentiment) for idx, sentiment in indices]

        first_sentiment = min(indices)[1]

        return first_sentiment

    async def process_file(filename):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(input_directory, f"{os.path.splitext(filename)[0]}_updated.csv")

        async with aiohttp.ClientSession() as session:
            with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames

                with open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()

                    requests_made = 0  # Track the number of requests made
                    for row in reader:
                        tweet_text = row['text']
                        sentiment_response, requests_made = await get_sentiment_response(session, tweet_text,
                                                                                         requests_made)
                        row['target'] = sentiment_response
                        row['flag'] = determine_sentiment(sentiment_response)
                        writer.writerow(row)

            print(f"Sentiment analysis completed for {filename}. Updated file saved as {output_file_path}.")

    async def main():
        tasks = []
        for filename in os.listdir(input_directory):
            if filename.endswith('.csv'):
                tasks.append(process_file(filename))

        await asyncio.gather(*tasks)

    # Run the main function
    asyncio.run(main())


    # Path to the directory containing the CSV files
    csv_directory = "extracted_tweets/"

    # Pattern to match the CSV files ending with "updated.csv"
    csv_pattern = csv_directory + "*updated.csv"

    # Get a list of all matching CSV files
    csv_files = glob.glob(csv_pattern)

    # Read and concatenate all matching CSV files
    merged_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Remove duplicates based on column F (assuming 0-indexed columns, column F is the 6th column)
    merged_df.drop_duplicates(subset=[merged_df.columns[5]], inplace=True)

    # Drop columns B and E (assuming 0-indexed columns, B is the 2nd column and E is the 5th column)
    columns_to_drop = [merged_df.columns[i] for i in [1, 4]]
    merged_df.drop(columns=columns_to_drop, inplace=True)

    # Specify the merged CSV file name
    merged_csv_file = f"{user_input}.csv"

    # Save the merged dataframe to a CSV file
    merged_df.to_csv(merged_csv_file, index=False)

    print(
        f"All data merged, duplicates removed, and specified columns dropped. Merged data saved to: {merged_csv_file}")

    tweet_df = pd.read_csv(merged_csv_file)

    # Extract the specific column
    tweets_list = tweet_df['text'].tolist()
    sentiments = tweet_df['flag'].tolist()

    for index, tweet in enumerate(tweets_list):
        text = f'Tweet {index + 1}: {tweet}'
        tweets.append(text)

    # File paths
    input_file_path = f'{user_input}.csv'
    count_file_path = f'{user_input}_tweetcounts.csv'

    # Initialize sentiment counts
    sentiment_counts = {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}

    # Read the updated CSV file and count sentiments
    with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            sentiment = row['flag']
            sentiment_counts['total'] += 1

            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1

    # Write sentiment counts to a new CSV file
    with open(count_file_path, mode='w', newline='', encoding='utf-8') as countfile:
        count_writer = csv.writer(countfile)
        count_writer.writerow(['Sentiment', 'Count'])
        for sentiment, count in sentiment_counts.items():
            count_writer.writerow([sentiment, count])

    print(f"Sentiment counts saved as {count_file_path}.")

    sentiment_count_df = pd.read_csv(count_file_path)

    # Extract the specific column
    sentiment_counts = sentiment_count_df['Count'].tolist()
    sentiment_total = sentiment_counts[0]
    sentiment_positive = sentiment_counts[1]
    sentiment_negative = sentiment_counts[2]
    sentiment_neutral = sentiment_counts[3]

    Positive = (sentiment_positive / sentiment_total) * 100
    Negative = (sentiment_negative / sentiment_total) * 100
    Neutral = (sentiment_neutral / sentiment_total) * 100

    chart_data = [Positive, Negative, Neutral]
    chart_datasets = {}
    keyword_generated = [sentiment_total, sentiment_positive, sentiment_negative, sentiment_neutral]

    chart_datasets['label'] = chart_label
    chart_datasets['data'] = chart_data
    chart_datasets['backgroundColor'] = backgroundColor
    datasets.append(chart_datasets)

    pie_data['labels'] = labels
    pie_data['datasets'] = datasets

    response_data['keywordCount'] = keywords_data
    response_data['tweets'] = tweets
    response_data['sentiments'] = sentiments
    response_data['data'] = pie_data
    response_data['keywordGenerated'] = keyword_generated
    response_data['type'] = "twitter"

    json_data = jsonify(response_data)

    # Define the path to the JSON file
    file_path = 'twitter_response_data.json'

    # Save the JSON data to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(response_data, file, indent=4)

    # json_file_path = 'twitter_response_data.json'

    # Open the JSON file and read its contents
    # with open(json_file_path, 'r') as file:
    # Load the JSON data into a dictionary
    # response_data = json.load(file)

    json_data = jsonify(response_data)
    response = make_response(json_data)
    response.status_code = 200
    return response

@app.route('/news-submit', methods=['POST'])
def news_submit():
    response_data = {}

    data = request.get_json()
    print(data)
    user_input = data['userInput']

    def get_company_news_and_analysis(company_symbol):
        # Set up connection to Seeking Alpha API
        conn = http.client.HTTPSConnection("seeking-alpha.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': SEEKING_ALPHA_KEY,
            'x-rapidapi-host': SEEKING_ALPHA_HOST
        }

        # Function to send request and extract data
        def fetch_data(endpoint):
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            return json.loads(data)

        # Fetching news articles
        news_endpoint = f"/news/v2/list-by-symbol?size=50&number=1&id={company_symbol.lower()}"
        news_data = fetch_data(news_endpoint)

        # Fetching analysis articles
        analysis_endpoint = f"/analysis/v2/list?id={company_symbol.lower()}&size=20&number=1"
        analysis_data = fetch_data(analysis_endpoint)

        # Extract relevant news data
        news_articles = news_data.get("data", [])
        extracted_news_data = []
        base_url = "https://seekingalpha.com"
        for article in news_articles:
            title = article.get("attributes", {}).get("title", "N/A")
            relative_url = article.get("links", {}).get("self", "N/A")
            full_url = f"{base_url}{relative_url}"
            date_published = article.get("attributes", {}).get("publishedAt", "N/A")
            extracted_news_data.append([title, full_url, date_published])

        # Extract relevant analysis data
        analysis_articles = analysis_data.get("data", [])
        extracted_analysis_data = []
        for article in analysis_articles:
            title = article.get("attributes", {}).get("title", "N/A")
            relative_url = article.get("links", {}).get("self", "N/A")
            full_url = f"{base_url}{relative_url}"
            date_published = article.get("attributes", {}).get("publishedAt", "N/A")
            extracted_analysis_data.append([title, full_url, date_published])

        # Prepare directory and file names
        directory = company_symbol.lower().replace(" ", "_")
        if not os.path.exists(directory):
            os.makedirs(directory)

        news_file_name = f"{directory}/{company_symbol.lower().replace(' ', '_')}_seekingalpha_news.csv"
        analysis_file_name = f"{directory}/{company_symbol.lower().replace(' ', '_')}_seekingalpha_analysis.csv"

        # Write news data to CSV
        with open(news_file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Title", "URL", "Published Date"])
            writer.writerows(extracted_news_data)

        # Write analysis data to CSV
        with open(analysis_file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Title", "URL", "Published Date"])
            writer.writerows(extracted_analysis_data)

        print(f"News data saved to {news_file_name}")
        print(f"Analysis data saved to {analysis_file_name}")

    # Function to asynchronously get the full response from the Llama model
    async def get_full_model_response(session, article_url, company_name):
        url = LLAMA_URL
        payload = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the article at {article_url} and determine the sentiment toward {company_name}. Classify it as positive only if the article contains overwhelmingly positive phrases with no trace of negativity. If there is even slight negativity or the tone is mixed, categorize it as negative. Summarize it in less than 50 words."
                }
            ]
        }
        headers = {
            "x-rapidapi-key": LLAMA_KEY,
            "x-rapidapi-host": LLAMA_HOST,
            "Content-Type": "application/json"
        }
        for attempt in range(MAX_RETRIES):
            try:
            # Make the API request with a timeout
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        # Extract the full content from the Llama model response
                        full_response = response_json['choices'][0]['message']['content'].strip()
                        return full_response

                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", RETRY_DELAY))
                        await asyncio.sleep(retry_after)
                        return await get_full_model_response(session, article_url, company_name)

                    else:
                        await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)  # Wait before retrying
                else:
                    return f"Request failed after {MAX_RETRIES} attempts: {e}"
            except (KeyError, IndexError) as e:
                return "Error in processing the request"
            except Exception as e:
                return f"Unexpected error occurred: {e}"

                # Fallback in case all retries fail
            return "Request failed, unable to retrieve model response."

    # Function to determine sentiment from the model response
    def determine_sentiment(response_text):
        keywords = ["positive", "negative", "neutral"]
        response_text_lower = response_text.lower()

        # Find the first occurrence of any keyword
        for keyword in keywords:
            if keyword in response_text_lower:
                return keyword
        return "unknown"  # Default if none of the keywords are found

    # Function to process the CSV file asynchronously
    async def process_csv_file(session, file_path, company_symbol):
        output_rows = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)
            header.extend(['model_response', 'sentiment'])  # Add new column headers
            output_rows.append(header)

            # Process each row asynchronously
            tasks = []
            for row in reader:
                article_url = row[1]  # Assuming the URL is in the second column
                tasks.append(process_row(session, article_url, company_symbol, row))

            # Wait for all tasks to complete
            processed_rows = await asyncio.gather(*tasks)
            output_rows.extend(processed_rows)

        # Write the updated data back to the CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(output_rows)

        print(f"Model responses and sentiment have been saved to {file_path}")

    # Function to process each row asynchronously
    async def process_row(session, article_url, company_symbol, row):
        full_response = await get_full_model_response(session, article_url, company_symbol)
        sentiment = determine_sentiment(full_response)  # Determine sentiment from the model response
        row.extend([full_response, sentiment])  # Add model response and sentiment to the row
        return row

    # Function to start processing files
    async def process_files(company_symbol):
        directory = company_symbol.lower().replace(" ", "_")
        news_file_name = f"{directory}/{company_symbol.lower().replace(' ', '_')}_seekingalpha_news.csv"
        analysis_file_name = f"{directory}/{company_symbol.lower().replace(' ', '_')}_seekingalpha_analysis.csv"

        async with aiohttp.ClientSession() as session:
            if os.path.exists(news_file_name):
                await process_csv_file(session, news_file_name, company_symbol)
            else:
                print(f"File '{news_file_name}' not found.")

            if os.path.exists(analysis_file_name):
                await process_csv_file(session, analysis_file_name, company_symbol)
            else:
                print(f"File '{analysis_file_name}' not found.")

    # Main entry point to start asynchronous processing
    async def main(company_symbol):
        await process_files(company_symbol)

    company_symbol = user_input.strip()
    # Prepare directory and file names
    directory = company_symbol.lower().replace(" ", "_")
    get_company_news_and_analysis(company_symbol)
    # Max retries for the request
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # Seconds to wait before retrying
    asyncio.run(main(company_symbol))

    url = LLAMA_URL
    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {
                "role": "user",
                "content": f"Get me two keywords which can be raw materials or components that is vital for the company {company_symbol}. Output must have only the two keywords. No other text or description is needed. Make sure there are two keywords alone in the output."
            }
        ]
    }
    headers = {
        "x-rapidapi-key": LLAMA_KEY,
        "x-rapidapi-host": LLAMA_HOST,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # Parse the JSON response
    response_json = response.json()
    # Extract the content message
    content = response_json['choices'][0]['message']['content']

    # Use regex to split the content based on various delimiters
    keywords = re.split(r'[,\n&*0-9]+', content.strip())

    # Filter out any empty strings and print the keywords
    filtered_keywords = [keyword.strip() for keyword in keywords if keyword.strip()]

    # Save the keywords to a CSV file
    csv_file = f"{directory}/{company_symbol.lower().replace(' ', '_')}_keycomponents.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Keyword'])  # Write the header
        for keyword in filtered_keywords:
            writer.writerow([keyword])  # Write each keyword in a new row

    print(f"Keywords have been saved to {csv_file}.")

    # Define the path to your CSV file
    key_components_file_name = f"{directory}/{company_symbol.lower().replace(' ', '_')}_keycomponents.csv"  # Update this with the name of your CSV file
    file_path = os.path.join(os.getcwd(), key_components_file_name)

    # Function to get the full response from the Llama model
    def get_full_model_res_key(keyword, company_symbol):
        url = LLAMA_URL

        payload = {
            "model": "meta-llama/Llama-3-8b-chat-hf",
            "messages": [
                {
                    "role": "user",
                    "content": f"How is the recent price trend of {keyword} and how is it impacting {company_symbol}?. Think like a financial analyst and analyze whether the price movement is going to positively or negatively impact. Clearly state whether it positive, negative or neutral for the company's growth."
                }
            ]
        }
        headers = {
            "x-rapidapi-key": LLAMA_KEY,
            "x-rapidapi-host": LLAMA_HOST,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()

        # Extract the full content from the Llama model response
        try:
            full_response = response_json['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError):
            full_response = "Error in processing the request"

        return full_response

        # Function to determine sentiment from the response text
    def extract_sentiment(response):
        sentiments = ["positive", "negative", "neutral", "POSITIVE", "NEGATIVE", "NEUTRAL"]
        # Find the first occurrence of any sentiment word in the response
        first_occurrence = min(
            [(response.find(word), word) for word in sentiments if word in response],
            default=(float('inf'), None)
        )
        return first_occurrence[1] if first_occurrence[1] else "unknown"

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{key_components_file_name}' not found.")
    else:
        # Read the CSV file, get model responses, and store the results
        output_rows = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames + ['Model_Response', 'Sentiment']
            output_rows.append(fieldnames)

            for row in reader:
                keyword = row['Keyword']  # Assuming the keyword is under the 'Keyword' column
                model_response = get_full_model_res_key(keyword, company_symbol)
                sentiment = extract_sentiment(model_response)
                row['Model_Response'] = model_response
                row['Sentiment'] = sentiment
                output_rows.append(row.values())

        # Write the updated data back to the CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(output_rows)

        print(f"Model responses and sentiments have been saved to {file_path}")

    news_df = pd.read_csv(f"{directory}/{company_symbol.lower().replace(' ', '_')}_seekingalpha_news.csv")

    analysis_df = pd.read_csv(f"{directory}/{company_symbol.lower().replace(' ', '_')}_seekingalpha_analysis.csv")

    keycomponents_df = pd.read_csv(f"{directory}/{company_symbol.lower().replace(' ', '_')}_keycomponents.csv")

    response_data = {
        'news': news_df['Title'].tolist(),
        'analyst': analysis_df['Title'].tolist(),
        'keyComponents': keycomponents_df["Keyword"].tolist(),
        'type' : "news"
    }

    # Define the path to the JSON file
    file_path = 'news_analysis_data.json'

    # Save the JSON data to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(response_data, file, indent=4)

    # json_file_path = 'news_analysis_data.json'

    # Open the JSON file and read its contents
    # with open(json_file_path, 'r') as file:
    # Load the JSON data into a dictionary
    # response_data = json.load(file)

    json_data = jsonify(response_data)
    response = make_response(json_data)
    response.status_code = 200
    return response

@app.route('/submit-analysis', methods=['POST'])
def submit_analysis():
    response_data = {}
    labels = ['Positive', 'Negative', 'Neutral']
    backgroundColor = ['#4CAF50', '#F44336', '#FFC107']
    overall_backgroundColor = ['#4CAF50', '#F44336', '#FFC107', '#F44336']
    label = 'Sentiment'
    news_ds = {}
    news_datasets = []
    news_info = {}
    analyst_ds = {}
    analyst_datasets = []
    analyst_info = {}
    key_components_ds = {}
    key_components_info = []
    overall_ds = {}
    overall_info = []

    data = request.get_json()
    print(data)
    user_input = data['userInput']

    company_symbol = user_input.strip()
    # Prepare directory and file names
    directory = company_symbol.lower().replace(" ", "_")
    # Define the directory and company symbol
    overall_directory = f""  # replace with your directory path

    # Define the path to the folder containing the CSV files
    folder_path = os.path.join(overall_directory, directory)

    # Initialize a list to store the results
    results = []

    # Iterate over each CSV file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the 'sentiment' column exists, case-insensitive
                sentiment_col = [col for col in df.columns if col.lower() == 'sentiment']
                if not sentiment_col:
                    print(f"'Sentiment' column not found in {file_name}")
                    continue

                # Count positive, negative, and neutral sentiments
                counts = df[sentiment_col[0]].str.lower().value_counts()
                positive = counts.get('positive', 0)
                negative = counts.get('negative', 0)
                neutral = counts.get('neutral', 0)

                # Append the results for this file
                results.append([file_name, positive, negative, neutral])

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results, columns=['File Name', 'Positive', 'Negative', 'Neutral'])

    overall_file_name = f"{directory}/overall_sentiment.csv"  # Update this with the name of your CSV file
    output_path = os.path.join(os.getcwd(), overall_file_name)

    # Save the results to the CSV file
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

    overall_df = pd.read_csv(overall_file_name)

    news_list = overall_df[overall_df['File Name'] == f"{company_symbol.lower().replace(' ', '_')}_seekingalpha_news.csv"][['Positive', 'Negative', 'Neutral']].values.tolist()
    news_data = news_list[0]

    # Convert the list of strings to integers
    news_data = [int(value) for value in news_data]

    # Calculate the total sum of the list
    news_total = sum(news_data)

    # Calculate the percentage of each value
    if news_total != 0:
        news_chart_data = [(value / news_total) * 100 for value in news_data]
    else:
        news_chart_data = [0, 0, 0]

    news_ds['data'] = news_chart_data
    news_ds['backgroundColor'] = backgroundColor

    news_datasets.append(news_ds)

    news_info['labels'] = labels
    news_info['datasets'] = news_datasets

    analyst_list = overall_df[overall_df['File Name'] == f"{company_symbol.lower().replace(' ', '_')}_seekingalpha_analysis.csv"][['Positive', 'Negative', 'Neutral']].values.tolist()
    analyst_data = analyst_list[0]
    analyst_data = [int(value) for value in analyst_data]

    # Calculate the total sum of the list
    analyst_total = sum(analyst_data)

    # Calculate the percentage of each value
    if analyst_total != 0:
        analyst_chart_data = [(value / analyst_total) * 100 for value in analyst_data]

    else:
        analyst_chart_data = [0, 0, 0]

    analyst_ds['data'] = analyst_chart_data
    analyst_ds['backgroundColor'] = backgroundColor

    analyst_datasets.append(analyst_ds)

    analyst_info['labels'] = labels
    analyst_info['datasets'] = analyst_datasets

    key_component_list = overall_df[overall_df['File Name'] == f"{company_symbol.lower().replace(' ', '_')}_keycomponents.csv"][['Positive', 'Negative', 'Neutral']].values.tolist()
    key_component_data = key_component_list[0]

    # Convert the list of strings to integers
    key_component_data = [int(value) for value in key_component_data]

    # Calculate the total sum of the list
    key_components_total = sum(key_component_data)

    # Calculate the percentage of each value
    if key_components_total != 0:
        key_components_chart_data = [(value / key_components_total) * 100 for value in key_component_data]
    else:
        key_components_chart_data = [0, 0, 0]
    overall = []
    for i in range(3):
        val = news_chart_data[i] + analyst_chart_data[i] + key_components_chart_data[i]
        overall.append(val)
    net_diff = abs(overall[0] - overall[1])
    overall.append(net_diff)
    overall_total = sum(overall)

    # Calculate the percentage of each value
    if overall_total != 0:
        overall_chart_data = [(value / overall_total) * 100 for value in overall]
    else:
        overall_chart_data = [0, 0, 0]
    overall_ds['label'] = label
    overall_ds['data'] = overall_chart_data
    overall_ds['backgroundColor'] = overall_backgroundColor

    overall_info.append(overall_ds)

    key_components_ds['label'] = label
    key_components_ds['data'] = key_components_chart_data
    key_components_ds['backgroundColor'] = backgroundColor

    key_components_info.append(key_components_ds)

    main_positive = []
    main_negative = []
    main_neutral = []

    for i in range(3):
        if i == 0:
            main_positive.append(news_chart_data[i])
            main_positive.append(analyst_chart_data[i])
            main_positive.append(key_components_chart_data[i])
        elif i == 1:
            main_negative.append(news_chart_data[i])
            main_negative.append(analyst_chart_data[i])
            main_negative.append(key_components_chart_data[i])
        else:
            main_neutral.append(news_chart_data[i])
            main_neutral.append(analyst_chart_data[i])
            main_neutral.append(key_components_chart_data[i])

    main_dataset = {
        "labels": ['Company News Sentiment', 'Analyst View Sentiment', 'Key Component Sentiment'],
        "datasets": [
            {
                "label": 'Positive',
                "data": main_positive,
                "backgroundColor": '#4CAF50'
            },
            {
                "label": 'Negative',
                "data": main_negative,
                "backgroundColor": '#F44336'
            },
            {
                "label": 'Neutral',
                "data": main_neutral,
                "backgroundColor": '#FFC107'
            }
        ]
    }

    response_data['mainDatasets'] = main_dataset
    response_data['news_data'] = news_info
    response_data['analysis_data'] = analyst_info
    response_data['key_components'] = key_components_info
    response_data['overall'] = overall_info
    response_data['type'] = "news"

    json_data = jsonify(response_data)

    response = make_response(json_data)
    response.status_code = 200
    return response


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8080)