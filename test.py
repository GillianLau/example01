import csv
filename = 'Motor_Vehicle_Collisions_-_Crashes.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    # print(header_row)
    # for index,column_header in enumerate(header_row):
    #     print(index,column_header)

    highs = []
    for row in reader:
        high = int(row[4])
        highs.append(high)
    print(highs)