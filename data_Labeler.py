import os
import pandas as pd

csv_folder = 'ds'  

csv_files = [file for file in os.listdir(csv_folder) if file.endswith('.csv')]

label_mapping = {
    '1': 'tốt',
    '2': 'rất tốt',
    '3': 'tệ',
    '4': 'rất tệ',
    '5': 'bình thường',
    '6': 'không liên quan'
}

for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(csv_path)
    if len(df.columns) >= 6:
        column_index = 5
        non_empty_rows = df[df.iloc[:, column_index].notna()]
        for index, row in non_empty_rows.iterrows():
            content = row.iloc[column_index]
            while True:
                user_input = input(f"{content}' (1-6): ")
                if user_input in label_mapping:
                    break  
                else:
                    print("Invalid input. Please enter a number between 1 and 6.")
            df.at[index, 'Cảm Xúc'] = label_mapping[user_input]
        df.to_csv(csv_path, index=False)
        print(f"Updated '{csv_file}' with user sentiment values.")
    else:
        print(f"'{csv_file}' does not contain at least 6 columns.")

print("Sentiment input complete for all CSV files.")
