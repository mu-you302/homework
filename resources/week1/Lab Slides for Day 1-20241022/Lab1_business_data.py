import csv  
  
# CSV file path  
file_path = 'business_data.csv'  
  
# List to hold the second column data  
second_column_data = []  
  
# Open the CSV file for reading  
with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:  
    csv_reader = csv.reader(csv_file)  
      
    # Skip the header row if present (optional)  
    # next(csv_reader, None)  # Uncomment this line if your CSV has a header  
      
    # Iterate over each row in the CSV file  
    for row in csv_reader:  
        if len(row) > 1:  # Ensure there's at least a second column  
            second_column_data.append(row[1])  # Append the second column data  
  
# second_column_data now contains all the data from the second column  
print(second_column_data)