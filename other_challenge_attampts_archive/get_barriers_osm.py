import pandas as pd
import requests
import time
from typing import List, Dict
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('osm_barrier_processing.log'),
        logging.StreamHandler()
    ]
)

class OSMBarrierProcessor:
    def __init__(self, batch_size: int = 100, delay: float = 1.0):
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        self.batch_size = batch_size
        self.delay = delay
        
    def build_overpass_query(self, osm_ids: List[str]) -> str:
        """Build an Overpass query for multiple OSM IDs."""
        ids_str = ", ".join(osm_ids)
        return f"""
        [out:json];
        (
            way(id:{ids_str});
        );
        out body;
        (._;>;);
        out body;
        """
    
    def get_barrier_data_batch(self, osm_ids: List[str]) -> Dict[str, Dict]:
        """Fetch barrier data for a batch of OSM IDs."""
        try:
            query = self.build_overpass_query(osm_ids)
            response = requests.post(self.overpass_url, data=query)
            
            if response.status_code != 200:
                logging.error(f"Error fetching data: {response.status_code}")
                return {}
            
            data = response.json()
            results = {}
            
            for element in data.get('elements', []):
                if 'tags' in element and 'barrier' in element['tags']:
                    osm_id = str(element['id'])
                    results[osm_id] = {
                        'barrier_type': element['tags']['barrier'],
                        'barrier_name': element['tags'].get('name', ''),
                        'barrier_height': element['tags'].get('height', ''),
                        'barrier_material': element['tags'].get('material', ''),
                        'barrier_access': element['tags'].get('access', '')
                    }
            
            return results
        
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            return {}

    def process_csv(self, csv_path: str, id_column: str):
        """Process OSM IDs from CSV and append barrier data to existing columns."""
        try:
            # Read the CSV in chunks
            chunk_size = 10000
            first_chunk = True
            
            for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
                logging.info(f"Processing chunk {chunk_num + 1}")
                
                # Convert OSM IDs to strings
                chunk[id_column] = chunk[id_column].astype(str)
                osm_ids = chunk[id_column].tolist()
                
                # Process in batches and collect barrier data
                barrier_data = {}
                for i in range(0, len(osm_ids), self.batch_size):
                    batch = osm_ids[i:i + self.batch_size]
                    batch_results = self.get_barrier_data_batch(batch)
                    barrier_data.update(batch_results)
                    time.sleep(self.delay)
                
                # Add barrier data columns to the chunk
                for col in ['barrier_type', 'barrier_name', 'barrier_height', 
                          'barrier_material', 'barrier_access']:
                    chunk[col] = chunk[id_column].map(
                        lambda x: barrier_data.get(x, {}).get(col, '')
                    )
                
                # Save the updated chunk
                chunk.to_csv(
                    'temp_output.csv',
                    mode='a' if not first_chunk else 'w',
                    header=first_chunk,
                    index=False
                )
                first_chunk = False
                
                logging.info(f"Processed chunk {chunk_num + 1}")
            
            # Replace original file with updated one
            Path('temp_output.csv').replace(csv_path)
            
        except Exception as e:
            logging.error(f"Error processing CSV: {e}")
            raise

def main():
    # Configuration
    CSV_PATH = "processed_traffic_data.csv"
    ID_COLUMN = "osm_id"
    BATCH_SIZE = 100
    DELAY = 1.0  # seconds between batches
    
    processor = OSMBarrierProcessor(batch_size=BATCH_SIZE, delay=DELAY)
    
    logging.info("Starting barrier data processing")
    processor.process_csv(CSV_PATH, ID_COLUMN)
    logging.info("Processing completed")

if __name__ == "__main__":
    main()