import pandas as pd
import os

def convert_csv_to_parquet(csv_path, parquet_path):
    print(f"Converting {csv_path} to {parquet_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(parquet_path, index=False)
    print("Done.")

if __name__ == "__main__":
    csv_dir = "/Users/hazanserifli/Desktop/tactıq/raw_data/csv"
    output_dir = "/Users/hazanserifli/Desktop/tactıq/raw_data"
    
    files_to_convert = ["antalya-konya.csv", "eyup-samsun.csv","fatih-eyup.csv", "fb-rize.csv","gaziantep-kayseri.csv","gençlerbirliği-gs.csv","kasımpasa-alanya.csv","kocaeli-göztepe.csv","rize-gaziantep.csv","samsun-bjk.csv","ts-başakşehir.csv"]
    
    for filename in files_to_convert:
        csv_path = os.path.join(csv_dir, filename)
        parquet_filename = filename.replace(".csv", ".parquet")
        parquet_path = os.path.join(output_dir, parquet_filename)
        
        if os.path.exists(csv_path):
            convert_csv_to_parquet(csv_path, parquet_path)
        else:
            print(f"File not found: {csv_path}")
