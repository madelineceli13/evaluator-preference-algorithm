import pandas as pd
import numpy as np
import re
import time
from datetime import datetime

def classify_trip_type(text):
    """
    Classify hotel review by trip type based on textual cues.
    Returns: 'family', 'business', 'couple', 'friends', 'solo', or 'unknown'
    """
    if pd.isna(text):
        return 'unknown'
    
    text_lower = text.lower()
    
    # Score each category based on keyword matches
    scores = {
        'family': 0,
        'business': 0,
        'couple': 0,
        'friends': 0,
        'solo': 0
    }
    
    # Family indicators
    # Check if family/children only in hotel name or describing the hotel's market
    hotel_family_context = (
        re.search(r'(hotel|resort|park).{0,30}(family|children|kids)', text_lower) or
        re.search(r'(family|children|kids).{0,30}(hotel|resort|park)', text_lower)
    )
    exclude_family = False
    if hotel_family_context:
        has_personal_family = re.search(r'\b(my|our|my own|bringing|brought|took|traveling with).{0,15}(child|children|kids|son|daughter|baby|toddler)', text_lower)
        if not has_personal_family:
            exclude_family = True
    
    # Check if describing other guests (not their own party)
    if re.search(r'\b(lots of|many|plenty of|full of|crowded with|other).{0,15}(children|kids|families)', text_lower):
        has_personal_family = re.search(r'\b(my|our|my own|we took|brought our).{0,15}(child|children|kids|son|daughter|baby|toddler)', text_lower)
        if not has_personal_family:
            exclude_family = True

    if exclude_family:
        family_keywords = []  # Don't count any family keywords
    else:
        family_keywords = [
            'kids', 'children', 'child', 'family', 'son', 'daughter',
            'toddler', 'baby', 'infant', 'teenagers', 'grandchildren',
            'niece', 'nephew', 'stroller', 'crib', 'playground',
            'parents', 'mom', 'dad', 'mother', 'father'
        ]

    scores['family'] = sum(1 for kw in family_keywords if kw in text_lower)
    
    # Business indicators
    # High-confidence business patterns (phrases that clearly indicate business travel)
    high_confidence_business = [
        r'\b(business trip|work trip|business travel|work travel)\b',
        r'\b(here (for|on) business|here (for|on) work)\b',
        r'\b(stayed (for|on) business|staying (for|on) business)\b',
        r'\b(in town (for|on) business|in town (for|on) work)\b',
        r'\b(my (conference|meeting|convention|seminar))\b',
        r'\b(attending (a |the )?(conference|meeting|convention|seminar))\b',
        r'\b((a|my|our|with) (work|business) (colleague|colleagues|client|clients))\b',
        r'\b(corporate (event|trip|travel|meeting))\b',
    ]
    
    for pattern in high_confidence_business:
        if re.search(pattern, text_lower):
            scores['business'] += 3  # Strong signal
    
    # Medium-confidence business keywords (only count if NOT about hotel facilities)
    medium_confidence_business = [
        'conference', 'seminar', 'training course',
        'networking event', 'corporate', 'presentation'
    ]
    
    # Only count medium-confidence keywords if NOT in hotel facility context
    for kw in medium_confidence_business:
        if kw in text_lower:
            scores['business'] += 2
    
    # Low-confidence patterns (require additional context)
    # Only count if traveling WITH colleagues, not just recommended BY them
    if re.search(r'\b(with|and|brought) (my |our )?(colleague|colleagues|client|clients|coworker|coworkers)\b', text_lower):
        # Exclude if it's just a recommendation context
        if not re.search(r'\b(recommended by|suggested by|told by).{0,15}(colleague|coworker)', text_lower):
            scores['business'] += 2
    
    couple_keywords = [
    'wife', 'husband', 'partner', 'girlfriend', 'boyfriend',
    'anniversary', 'honeymoon', 'engagement',
    'couples massage', 'date night', 'my spouse', 'fiancé', 'fiancee',
    'significant other'
    ]
    for kw in couple_keywords:
        if kw in text_lower:
            if kw in ['honeymoon', 'anniversary']:
                scores['couple'] += 2
            else:
                scores['couple'] += 1

    # Handle "romantic" separately - check for negation
    if 'romantic' in text_lower:
        # Don't count if it's negated (not romantic, wouldn't for romantic, etc.)
        if not re.search(r'\b(not|no|never|wouldn\'t|would not|isn\'t|won\'t).{0,20}(romantic|honeymoon|date)', text_lower):
            scores['couple'] += 2
    if 'romantic' in text_lower:
    # Don't count if it's negated OR in business hotel context
        is_negated = re.search(r'\b(not|no|never|wouldn\'t|would not|isn\'t|won\'t).{0,20}romantic', text_lower)
        is_business_context = re.search(r'(business (hotel|travel)|recommend.{0,20}business)', text_lower)
        if not is_negated and not is_business_context:
            scores['couple'] += 2

    # Couple pattern
    if re.search(r'my (wife|husband|partner|girlfriend|boyfriend) and i', text_lower):
        scores['couple'] += 2
    
    # Friends indicators
    friends_keywords = [
        'girls trip', 'guys trip', 'girls weekend', 'guys weekend',
        'reunion', 'buddies', 'mates'
    ]
    for kw in friends_keywords:
        if kw in text_lower:
            if kw in ['girls trip', 'guys trip', 'reunion']:
                scores['friends'] += 2
            else:
                scores['friends'] += 1

    # Only count "friends" in travel context, not recommendation context
    if re.search(r'\b(group of friends|my friends? and i|with my friends?|college friends|school friends)\b', text_lower):
        scores['friends'] += 2

    # Explicit "friends" keyword only in travel context
    if re.search(r'\b(stayed|traveling|travelled|trip|vacation).{0,30}(with )?friends\b', text_lower):
        scores['friends'] += 2
    elif re.search(r'\bfriends.{0,30}(stayed|traveling|travelled|trip|vacation)\b', text_lower):
        scores['friends'] += 2
        
        if re.search(r'(group of friends|my friend and i|with friends|college friends|school friends)', text_lower):
            scores['friends'] += 2
    # Exclude "referred by friends" or "recommended by friends"
    if re.search(r'\b(referred|recommended|suggested|sent|is good for|can visit with).{0,20}(friends|a friend)\b', text_lower):
        scores['friends'] = max(0, scores['friends'] - 2)  # Remove points if added
        
    # General "we" signal
    we_count = len(re.findall(r'\bwe\b', text_lower))
    if we_count >= 2:  # Multiple "we" mentions
        if scores['business'] == 0 and scores['family'] == 0 and scores['friends'] == 0:
            if scores['couple'] > 0:  # Only reinforce existing couple signals
                scores['couple'] += 1
    
    # Solo indicators
    solo_keywords = [
        'solo','alone', 'traveling alone', 'by myself',
        'on my own', 'single traveler', 'solo trip', 'travelling alone', 'traveling alone'
    ]
    scores['solo'] = sum(2 if kw in text_lower else 0 for kw in solo_keywords)
    
    if re.search(r'(room|stay|travel|stayed|went).{0,15}by myself', text_lower):
        if not re.search(r'(my (wife|husband|boyfriend|girlfriend|parents|family|friend)|with (my|our))', text_lower):
            scores['solo'] += 2

    # Additional solo heuristic
    i_count = len(re.findall(r'\bi\b', text_lower))
    we_count = len(re.findall(r'\bwe\b', text_lower))
    
    if i_count >= 3 and we_count == 0:
        if all(scores[cat] == 0 for cat in ['business', 'couple', 'family', 'friends']):
            scores['solo'] += 1
    
    if scores['family'] > 0 and scores['couple'] > 0:
    # Check if they mention traveling with children AND spouse together
        if re.search(r'\b(my|our) (baby|infant|toddler|child|children|kids|son|daughter)\b', text_lower):
            if re.search(r'\b(my (husband|wife)|husband and i|wife and i)\b', text_lower):
                # This is family travel, not couple travel
                scores['couple'] = 0

    # Determine winner
    max_score = max(scores.values())
    
    if max_score <= 1:
        return 'unknown'

    # Priority if tie: family > solo > business > couple > friends
    priority = ['family', 'solo', 'business', 'couple', 'friends']
    for category in priority:
        if scores[category] == max_score:
            return category
    
    return 'unknown'


def classify_with_checkpoints(df, text_column='text', batch_size=100000, 
                               checkpoint_dir='checkpoints', output_file='reviews_labeled.csv'):
    """
    Classify reviews in batches with periodic checkpointing and progress tracking.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to classify
    text_column : str
        Name of the column containing review text
    batch_size : int
        Number of rows to process per batch
    checkpoint_dir : str
        Directory to save checkpoint files
    output_file : str
        Final output filename
    """
    import os
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    total_rows = len(df)
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
    
    print(f"{'='*70}")
    print(f"TRIP TYPE CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Total rows: {total_rows:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Number of batches: {num_batches}")
    print(f"Checkpoint directory: {checkpoint_dir}/")
    print(f"{'='*70}\n")
    
    trip_types = []
    overall_start = time.time()
    
    for i in range(num_batches):
        batch_start = time.time()
        
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        
        print(f"[Batch {i+1}/{num_batches}] Processing rows {start_idx:,} to {end_idx:,}...")
        
        # Process batch
        batch = df.iloc[start_idx:end_idx]
        batch_labels = batch.apply(lambda row: classify_trip_type(str(row.get('title', '')) + ' ' + str(row.get(text_column, ''))), axis=1)
        trip_types.extend(batch_labels.tolist())
        
        batch_time = time.time() - batch_start
        rows_per_sec = len(batch_labels) / batch_time
        
        print(f"  ✓ Completed in {batch_time:.1f}s ({rows_per_sec:.0f} rows/sec)")
        
        # Show current distribution every 10 batches
        if (i + 1) % 10 == 0 or i == num_batches - 1:
            temp_counts = pd.Series(trip_types).value_counts()
            print(f"\n  Current distribution ({len(trip_types):,} rows classified):")
            for trip_type, count in temp_counts.items():
                pct = 100 * count / len(trip_types)
                print(f"    {trip_type:12s}: {count:8,} ({pct:5.1f}%)")
            print()
        
        # Save checkpoint every 10 batches (1M rows if batch_size=100K)
        if (i + 1) % 10 == 0 or i == num_batches - 1:
            checkpoint_file = f"{checkpoint_dir}/checkpoint_{end_idx:08d}.csv"
            temp_df = df.iloc[:end_idx].copy()
            temp_df['trip_type'] = trip_types
            temp_df.to_csv(checkpoint_file, index=False)
            print(f"  💾 Checkpoint saved: {checkpoint_file}")
            
            # Estimate time remaining
            elapsed = time.time() - overall_start
            rows_done = end_idx
            rows_remaining = total_rows - rows_done
            time_per_row = elapsed / rows_done
            est_remaining = time_per_row * rows_remaining
            
            print(f"  ⏱️  Elapsed: {elapsed/60:.1f} min | Est. remaining: {est_remaining/60:.1f} min\n")
    
    # Add final labels to dataframe
    df['trip_type'] = trip_types
    
    # Save final output
    print(f"\n{'='*70}")
    print("SAVING FINAL OUTPUT")
    print(f"{'='*70}")
    df.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    
    # Final statistics
    total_time = time.time() - overall_start
    avg_speed = total_rows / total_time
    
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average speed: {avg_speed:.0f} rows/sec")
    print(f"\nFinal distribution:")
    
    final_counts = df['trip_type'].value_counts()
    for trip_type, count in final_counts.items():
        pct = 100 * count / total_rows
        print(f"  {trip_type:12s}: {count:8,} ({pct:5.1f}%)")
    
    print(f"{'='*70}\n")
    
    return df


def validate_sample(df, text_column='text', sample_size=200):
    """
    Validate classification on a random sample for spot-checking.
    """
    print(f"\n{'='*70}")
    print("VALIDATION SAMPLE")
    print(f"{'='*70}\n")
    
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    
    for trip_type in ['family', 'business', 'couple', 'friends', 'solo', 'unknown']:
        type_sample = sample[sample['trip_type'] == trip_type]
        if len(type_sample) > 0:
            print(f"\n{trip_type.upper()} ({len(type_sample)} in sample):")
            print("-" * 70)
            for idx, row in type_sample.head(3).iterrows():
                title = row.get('title', 'N/A') if not pd.isna(row.get('title')) else "N/A"
                text = row[text_column] if not pd.isna(row[text_column]) else "N/A"
                print(f"  TITLE: {title}")
                print(f"  TEXT: {text}")
                print()
            # for idx, row in type_sample.head(3).iterrows():
            #     text_preview = row[text_column] if not pd.isna(row[text_column]) else "N/A"
            #     print(f"  • {text_preview}...")
            #     print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    df = pd.read_csv('data/hotelrec/subset/final_reviews_full.csv')  # Replace with your filename
    
    # Optional: Run on a small sample first to validate
    RUN_SAMPLE_FIRST = True
    
    if RUN_SAMPLE_FIRST:
        print("\n" + "="*70)
        print("RUNNING VALIDATION ON 10,000 ROW SAMPLE FIRST")
        print("="*70 + "\n")
        
        sample_df = df.sample(10000).copy()
        sample_df['trip_type'] = sample_df['text'].apply(classify_trip_type)
        
        print("\nSample distribution:")
        print(sample_df['trip_type'].value_counts())
        
        validate_sample(sample_df, text_column='text', sample_size=50)
        
        response = input("\nDoes this look good? Proceed with full dataset? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting. Adjust classify_trip_type() function as needed.")
            exit()
    
    # Run full classification with checkpoints
    df = classify_with_checkpoints(
        df, 
        text_column='text',
        batch_size=100000,  # Process 100K rows at a time
        checkpoint_dir='checkpoints',
        output_file='reviews_labeled_travel_type.csv'
    )
    
    # Show validation sample from full dataset
    validate_sample(df, text_column='text', sample_size=100)
    
    print("\n✅ CLASSIFICATION COMPLETE!")



print("="*70)
print("LOADING CLASSIFIED REVIEWS BY TRIP TYPE")
print("="*70)

# Load each CSV file
df_family = pd.read_csv('data/hotelrec/reviews_family.csv')
df_business = pd.read_csv('data/hotelrec/reviews_business.csv')
df_couple = pd.read_csv('data/hotelrec/reviews_couple.csv')
df_friends = pd.read_csv('data/hotelrec/reviews_friends.csv')
df_solo = pd.read_csv('data/hotelrec/reviews_solo.csv')

print("\nFiles loaded successfully!")
print(f"  Family: {len(df_family):,} reviews")
print(f"  Business: {len(df_business):,} reviews")
print(f"  Couple: {len(df_couple):,} reviews")
print(f"  Friends: {len(df_friends):,} reviews")
print(f"  Solo: {len(df_solo):,} reviews")

# Display 10 random samples from each category with FULL text
print("\n" + "="*70)
print("RANDOM SAMPLES - FULL REVIEWS (10 PER CATEGORY)")
print("="*70)

categories = {
    'FAMILY': df_family,
    'BUSINESS': df_business,
    'COUPLE': df_couple,
    'FRIENDS': df_friends,
    'SOLO': df_solo,
}

# Use timestamp for different samples each run
random_seed = int(time.time())
print(f"\nRandom seed: {random_seed}")

for category_name, category_df in categories.items():
    print(f"\n{'='*70}")
    print(f"{category_name} ({len(category_df):,} total reviews)")
    print("="*70)
    
    if len(category_df) == 0:
        print("  No reviews in this category")
        continue
    
    # Sample up to 10 reviews (or all if less than 10)
    sample_size = min(10, len(category_df))
    samples = category_df.sample(n=sample_size, random_state=random_seed)
    
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        title = row.get('title', 'N/A') if not pd.isna(row.get('title')) else "N/A"
        text = row.get('text', 'N/A') if not pd.isna(row.get('text')) else "N/A"
        
        print(f"\n[Sample {idx}]")
        print(f"TITLE: {title}")
        print(f"TEXT: {text}")  # FULL TEXT - no truncation
        print("-"*70)

print("\n✅ COMPLETE!")