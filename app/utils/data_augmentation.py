import pandas as pd
import numpy as np
import random
import os

def load_original_data(filepath):
    return pd.read_csv(filepath)

def add_noise(value, noise_level=0.05):
    if value == 0:
        return value
    noise = random.uniform(-noise_level, noise_level) * value
    return max(0, value + noise)

def augment_row(row, variation_type='noise'):
    new_row = row.copy()
    
    if variation_type == 'noise':
        new_row['face_x'] = add_noise(row['face_x'], 0.1)
        new_row['face_y'] = add_noise(row['face_y'], 0.1)
        new_row['face_w'] = add_noise(row['face_w'], 0.08)
        new_row['face_h'] = add_noise(row['face_h'], 0.08)
        new_row['face_con'] = min(100, max(0, add_noise(row['face_con'], 0.05)))
        new_row['pose_x'] = add_noise(row['pose_x'], 0.15)
        new_row['pose_y'] = add_noise(row['pose_y'], 0.15)
        if row['phone'] == 1:
            new_row['phone_x'] = int(add_noise(row['phone_x'], 0.1))
            new_row['phone_y'] = int(add_noise(row['phone_y'], 0.1))
            new_row['phone_w'] = int(add_noise(row['phone_w'], 0.1))
            new_row['phone_h'] = int(add_noise(row['phone_h'], 0.1))
            new_row['phone_con'] = min(1, max(0, add_noise(row['phone_con'], 0.05)))
    
    elif variation_type == 'pose_shift':
        pose_shifts = ['forward', 'left', 'right', 'down']
        if row['pose'] in pose_shifts:
            current_idx = pose_shifts.index(row['pose'])
            if random.random() < 0.3:
                new_idx = (current_idx + random.choice([-1, 1])) % len(pose_shifts)
                new_row['pose'] = pose_shifts[new_idx]
                new_row['pose_x'] = add_noise(row['pose_x'], 0.2)
                new_row['pose_y'] = add_noise(row['pose_y'], 0.2)
    
    elif variation_type == 'face_variation':
        if row['no_of_face'] > 0:
            new_row['face_con'] = min(100, max(50, row['face_con'] + random.uniform(-10, 10)))
            scale = random.uniform(0.9, 1.1)
            new_row['face_w'] = row['face_w'] * scale
            new_row['face_h'] = row['face_h'] * scale
    
    return new_row

def generate_synthetic_attentive():
    rows = []
    for _ in range(200):
        row = {
            'no_of_face': 1,
            'face_x': random.uniform(200, 400),
            'face_y': random.uniform(150, 300),
            'face_w': random.uniform(120, 200),
            'face_h': random.uniform(120, 200),
            'face_con': random.uniform(85, 99),
            'no_of_hand': random.choice([0, 1, 2]),
            'pose': 'forward',
            'pose_x': random.uniform(-10, 10),
            'pose_y': random.uniform(-15, 5),
            'phone': 0,
            'phone_x': 0,
            'phone_y': 0,
            'phone_w': 0,
            'phone_h': 0,
            'phone_con': 0,
            'label': 1
        }
        rows.append(row)
    return rows

def generate_synthetic_inattentive():
    rows = []
    
    for _ in range(80):
        row = {
            'no_of_face': 1,
            'face_x': random.uniform(100, 450),
            'face_y': random.uniform(100, 350),
            'face_w': random.uniform(100, 250),
            'face_h': random.uniform(100, 250),
            'face_con': random.uniform(50, 80),
            'no_of_hand': random.choice([0, 1, 2]),
            'pose': random.choice(['left', 'right', 'down']),
            'pose_x': random.uniform(-20, 30),
            'pose_y': random.uniform(-30, 20),
            'phone': 0,
            'phone_x': 0,
            'phone_y': 0,
            'phone_w': 0,
            'phone_h': 0,
            'phone_con': 0,
            'label': 0
        }
        rows.append(row)
    
    for _ in range(60):
        row = {
            'no_of_face': random.choice([0, 1]),
            'face_x': random.uniform(150, 400) if random.random() > 0.3 else 0,
            'face_y': random.uniform(150, 350) if random.random() > 0.3 else 0,
            'face_w': random.uniform(100, 200) if random.random() > 0.3 else 0,
            'face_h': random.uniform(100, 200) if random.random() > 0.3 else 0,
            'face_con': random.uniform(40, 70),
            'no_of_hand': random.choice([1, 2]),
            'pose': 'down',
            'pose_x': -15,
            'pose_y': 10,
            'phone': 1,
            'phone_x': random.randint(100, 400),
            'phone_y': random.randint(200, 400),
            'phone_w': random.randint(150, 300),
            'phone_h': random.randint(200, 350),
            'phone_con': random.uniform(0.7, 0.95),
            'label': 0
        }
        rows.append(row)
    
    for _ in range(40):
        row = {
            'no_of_face': 0,
            'face_x': 0,
            'face_y': 0,
            'face_w': 0,
            'face_h': 0,
            'face_con': 0,
            'no_of_hand': random.choice([0, 1, 2]),
            'pose': random.choice(['forward', 'left', 'right']),
            'pose_x': random.uniform(-15, 25),
            'pose_y': random.uniform(-20, 15),
            'phone': random.choice([0, 1]),
            'phone_x': random.randint(200, 500) if random.random() > 0.5 else 0,
            'phone_y': random.randint(150, 400) if random.random() > 0.5 else 0,
            'phone_w': random.randint(150, 300) if random.random() > 0.5 else 0,
            'phone_h': random.randint(200, 350) if random.random() > 0.5 else 0,
            'phone_con': random.uniform(0.6, 0.9) if random.random() > 0.5 else 0,
            'label': 0
        }
        rows.append(row)
    
    return rows

def generate_synthetic_moderate():
    rows = []
    for _ in range(100):
        row = {
            'no_of_face': 1,
            'face_x': random.uniform(180, 420),
            'face_y': random.uniform(130, 320),
            'face_w': random.uniform(110, 190),
            'face_h': random.uniform(110, 190),
            'face_con': random.uniform(70, 90),
            'no_of_hand': random.choice([0, 1, 2]),
            'pose': random.choice(['forward', 'left', 'right']),
            'pose_x': random.uniform(-12, 18),
            'pose_y': random.uniform(-18, 12),
            'phone': 0,
            'phone_x': 0,
            'phone_y': 0,
            'phone_w': 0,
            'phone_h': 0,
            'phone_con': 0,
            'label': 1
        }
        rows.append(row)
    
    for _ in range(50):
        row = {
            'no_of_face': 1,
            'face_x': random.uniform(150, 450),
            'face_y': random.uniform(120, 350),
            'face_w': random.uniform(100, 220),
            'face_h': random.uniform(100, 220),
            'face_con': random.uniform(60, 85),
            'no_of_hand': random.choice([1, 2]),
            'pose': random.choice(['left', 'right']),
            'pose_x': random.uniform(-15, 25),
            'pose_y': random.uniform(-25, 15),
            'phone': 0,
            'phone_x': 0,
            'phone_y': 0,
            'phone_w': 0,
            'phone_h': 0,
            'phone_con': 0,
            'label': 0
        }
        rows.append(row)
    
    return rows

def augment_dataset(input_path, output_path, target_size=2000):
    print(f"Loading original data from {input_path}...")
    df = load_original_data(input_path)
    original_size = len(df)
    print(f"Original dataset size: {original_size}")
    
    all_rows = df.to_dict('records')
    
    print("Generating synthetic attentive samples...")
    all_rows.extend(generate_synthetic_attentive())
    
    print("Generating synthetic inattentive samples...")
    all_rows.extend(generate_synthetic_inattentive())
    
    print("Generating synthetic moderate attention samples...")
    all_rows.extend(generate_synthetic_moderate())
    
    print("Applying data augmentation...")
    while len(all_rows) < target_size:
        for row in df.to_dict('records'):
            if len(all_rows) >= target_size:
                break
            
            variation_type = random.choice(['noise', 'pose_shift', 'face_variation'])
            new_row = augment_row(row, variation_type)
            all_rows.append(new_row)
    
    augmented_df = pd.DataFrame(all_rows)
    
    label_counts = augmented_df['label'].value_counts()
    print(f"\nLabel distribution:")
    print(f"  Attentive (1): {label_counts.get(1, 0)}")
    print(f"  Inattentive (0): {label_counts.get(0, 0)}")
    
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    augmented_df.to_csv(output_path, index=False)
    print(f"\nAugmented dataset saved to {output_path}")
    print(f"Final dataset size: {len(augmented_df)}")
    
    return augmented_df

if __name__ == "__main__":
    input_path = os.path.join(os.path.dirname(__file__), 'data', 'attention_detection_dataset_v1.csv')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'attention_detection_dataset_augmented.csv')
    
    augment_dataset(input_path, output_path, target_size=2500)
