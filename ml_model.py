import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os
from django.conf import settings
import random

class CollegePredictionModel:
    def __init__(self):
        self.college_encoder = LabelEncoder()
        self.course_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.type_encoder = LabelEncoder()
        self.university_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.rank_model = None
        self.percentile_model = None
        self.college_model = None
        self.is_trained = False
        
    def load_data(self, csv_path='college_data.csv'):
        """Load and preprocess the college data. Prefer the provided Excel file."""
        preferred_excels = [
            'MHTCET cutoff.xlsx',
            'MHTCET_Cutoff.xlsx',
            'MHTCET_Round1_Cutoff_2025_full_extracted.xlsx'
        ]
        for xlsx in preferred_excels:
            try:
                self.df = pd.read_excel(xlsx)
                print(f"Loaded Excel data '{xlsx}' with shape: {self.df.shape}")
                return self.df
            except Exception as e2:
                last_err = e2
                continue
        print(f"Error loading preferred Excels: {last_err if 'last_err' in locals() else 'unknown'}. Falling back to CSV if available.")
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Loaded CSV data with shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None

    def ensure_data_loaded(self):
        """Ensure the source dataframe is loaded from Excel/CSV and load vacancy data."""
        if not hasattr(self, 'df') or self.df is None:
            self.load_data()
        # Add normalized helper columns once
        if self.df is not None:
            if 'course_norm' not in self.df.columns and 'Course Name' in self.df.columns:
                self.df['course_norm'] = self.df['Course Name'].astype(str).str.strip().str.lower()
            if 'category_norm' not in self.df.columns and 'Category' in self.df.columns:
                self.df['category_norm'] = self.df['Category'].astype(str).str.strip().str.lower()

        # Load vacancy data for colleges
        if not hasattr(self, 'vacancy_df') or self.vacancy_df is None:
            try:
                self.vacancy_df = pd.read_excel('college_vacancies.xlsx')
            except Exception as e:
                print(f"Could not load vacancy data: {e}")
                self.vacancy_df = None

    def get_vacant_seats(self, college_name):
        """Return vacant seats for a college from the vacancy Excel."""
        try:
            if hasattr(self, 'vacancy_df') and self.vacancy_df is not None:
                matches = self.vacancy_df[self.vacancy_df['College Name'] == college_name]
                if not matches.empty:
                    return int(matches.iloc[0]['Vacant_Seats'])
        except Exception:
            pass
        return random.randint(1, 7)  # fallback

    def get_available_courses(self):
        """Return sorted unique course names from the dataset."""
        self.ensure_data_loaded()
        try:
            return sorted([c for c in self.df['Course Name'].dropna().astype(str).unique() if c.strip()])
        except Exception:
            return []

    def get_available_categories(self):
        """Return sorted unique categories from the dataset."""
        self.ensure_data_loaded()
        try:
            return sorted([c for c in self.df['Category'].dropna().astype(str).unique() if c.strip()])
        except Exception:
            return []

    def map_main_category_to_dataset(self, main_category: str, course_norm: str) -> list:
        """Map UI main category (General/OBC/SC/ST/EWS/PWD) to dataset category labels dynamically.
        Returns a list of acceptable dataset categories for filtering."""
        self.ensure_data_loaded()
        if self.df is None or 'Category' not in self.df.columns:
            return []
        subset = self.df
        if 'course_norm' in subset.columns and course_norm:
            subset = subset[subset['course_norm'] == course_norm]
        cats = subset['Category'].dropna().astype(str)
        cats_norm = cats.str.strip().str.lower().unique().tolist()

        key = str(main_category).strip().lower()
        keywords = {
            'general': ['open', 'gen', 'gopen', 'gom', 'gopens', 'unreserved'],
            'obc': ['obc', 'sebc'],
            'sc': ['sc'],
            'st': ['st'],
            'ews': ['ews'],
            'pwd': ['pwd', 'divyang', 'ph']
        }

        if key not in keywords:
            return []

        matches = []
        for c in cats_norm:
            for kw in keywords[key]:
                if kw in c:
                    matches.append(c)
                    break

        # If no matches and asking for General, treat everything as eligible as a fallback
        if not matches and key == 'general':
            return cats_norm

        # Return original labels (not normalized) matching the normalized set
        matched_set = set(matches)
        return sorted({orig for orig in cats.unique() if orig.strip().lower() in matched_set})
    
    def preprocess_data(self, df):
        """Preprocess the data for training (robust to missing columns)."""
        df_clean = df.copy()

        required_cols = ['College Name', 'Course Name', 'Category', 'Cutoff Rank', 'Percentile']
        missing = [c for c in required_cols if c not in df_clean.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Coerce numeric fields
        df_clean['Cutoff Rank'] = pd.to_numeric(df_clean['Cutoff Rank'], errors='coerce').fillna(0).astype(float)
        df_clean['Percentile'] = pd.to_numeric(df_clean['Percentile'], errors='coerce').fillna(0).astype(float)

        # Derived features
        df_clean.loc[:, 'rank_log'] = np.log1p(df_clean['Cutoff Rank'])
        df_clean.loc[:, 'percentile_squared'] = df_clean['Percentile'] ** 2

        # Encode categoricals (guard optional columns)
        df_clean.loc[:, 'college_encoded'] = self.college_encoder.fit_transform(df_clean['College Name'].astype(str))
        df_clean.loc[:, 'course_encoded'] = self.course_encoder.fit_transform(df_clean['Course Name'].astype(str))
        df_clean.loc[:, 'category_encoded'] = self.category_encoder.fit_transform(df_clean['Category'].astype(str))

        if 'Type' in df_clean.columns:
            df_clean.loc[:, 'type_encoded'] = self.type_encoder.fit_transform(df_clean['Type'].astype(str))
        else:
            df_clean.loc[:, 'type_encoded'] = 0

        if 'Home University' in df_clean.columns:
            df_clean.loc[:, 'university_encoded'] = self.university_encoder.fit_transform(df_clean['Home University'].astype(str))
        else:
            df_clean.loc[:, 'university_encoded'] = 0

        return df_clean
    
    def train_models(self, df):
        """Train the ML models"""
        print("Training ML models...")
        
        # Features for prediction (no label leakage; exclude college_encoded from X)
        feature_columns = [col for col in [
            'course_encoded', 'category_encoded', 'type_encoded',
            'university_encoded', 'rank_log', 'percentile_squared'
        ] if col in df.columns]

        X = df[feature_columns]
        
        # Train rank prediction model
        y_rank = df['Cutoff Rank']
        X_train, X_test, y_train, y_test = train_test_split(X, y_rank, test_size=0.2, random_state=42)
        
        self.rank_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rank_model.fit(X_train, y_train)
        
        # Train percentile prediction model
        y_percentile = df['Percentile']
        X_train, X_test, y_train, y_test = train_test_split(X, y_percentile, test_size=0.2, random_state=42)
        
        self.percentile_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.percentile_model.fit(X_train, y_train)
        
        # Train college recommendation model
        y_college = df['college_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y_college, test_size=0.2, random_state=42)
        
        self.college_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.college_model.fit(X_train, y_train)
        
        self.is_trained = True
        print("Models trained successfully!")
        
        # Save models
        self.save_models()
        
    def predict_colleges(self, student_data):
        """Predict colleges using only 4 factors: course, category, rank, percentile"""
        if not self.is_trained:
            self.load_and_train()
        
        # Extract student information
        raw_course = student_data.get('course', '')
        raw_category = student_data.get('category', '')
        course = str(raw_course).strip()
        category = str(raw_category).strip()
        course_norm = course.lower()
        category_norm = category.lower()
        # robust rank parse: extract digits
        import re
        try:
            rank_str = str(student_data.get('rank', '')).strip()
            digits = re.findall(r"\d+", rank_str)
            rank = int(digits[0]) if digits else 0
        except Exception:
            rank = 0
        try:
            perc_str = str(student_data.get('percentile', '')).strip().replace('%', '')
            percentile = float(perc_str) if perc_str else 0.0
        except Exception:
            percentile = 0.0
        
        # Dynamically map UI category to dataset categories for this course
        dataset_categories = self.map_main_category_to_dataset(category, course_norm)
        
        try:
            # Get all colleges for the course and category (normalized, progressive relaxation)
            df_all = self.df
            # Start with course filter
            course_mask = (df_all.get('course_norm', df_all['Course Name'].astype(str).str.strip().str.lower()) == course_norm)
            if dataset_categories:
                cat_mask = df_all['Category'].astype(str).isin(dataset_categories)
                all_colleges = df_all[course_mask & cat_mask].copy()
            else:
                # If we cannot map, try course only
                all_colleges = df_all[course_mask].copy()
            if all_colleges.empty:
                # relax: match by course only
                all_colleges = df_all[df_all.get('course_norm', df_all['Course Name'].astype(str).str.strip().str.lower()) == course_norm].copy()
            if all_colleges.empty:
                # relax: match by category only
                if dataset_categories:
                    all_colleges = df_all[df_all['Category'].astype(str).isin(dataset_categories)].copy()
                else:
                    all_colleges = df_all.copy()
            if all_colleges.empty:
                # final relax: take top N by strongest cutoffs overall
                all_colleges = df_all.sort_values('Cutoff Rank', ascending=True).head(100).copy()
            
            if all_colleges.empty:
                return self.get_fallback_recommendations(student_data)
            
            # Calculate rank-based scores for each college
            all_colleges['rank_score'] = all_colleges.apply(
                lambda row: self.calculate_rank_score(rank, row['Cutoff Rank']), axis=1
            )
            
            # Calculate probability based on rank match and percentile closeness
            all_colleges['probability'] = all_colleges.apply(
                lambda row: self.calculate_probability(rank, row['Cutoff Rank'], row['rank_score'], percentile, row.get('Percentile', np.nan)), axis=1
            )
            
            # Sort by rank score (higher is better), percentile proximity, and cutoff rank
            try:
                all_colleges['perc_prox'] = all_colleges['Percentile'].apply(lambda p: -abs(float(p) - percentile))
            except Exception:
                all_colleges['perc_prox'] = 0
            all_colleges = all_colleges.sort_values(['rank_score', 'perc_prox', 'Cutoff Rank'], ascending=[False, False, True])
            
            # Get top recommendations with better rank-based differentiation
            recommended_colleges = []
            seen_names = set()
            
            # Ensure we get colleges from different tiers based on rank
            rank_tiers = self.get_rank_tiers(rank)
            
            for tier in rank_tiers:
                tier_colleges = all_colleges[
                    (all_colleges['Cutoff Rank'] >= tier['min_rank']) & 
                    (all_colleges['Cutoff Rank'] <= tier['max_rank'])
                ].head(tier['count'])
                
                for _, row in tier_colleges.iterrows():
                    college_name = str(row['College Name']).strip()
                    college_key = college_name.lower()
                    try:
                        cutoff_rank = int(row['Cutoff Rank'])
                    except Exception:
                        cutoff_rank = 'N/A'
                    try:
                        percentile_val = float(row['Percentile'])
                    except Exception:
                        percentile_val = 'N/A'
                    col_type = row['Type'] if 'Type' in all_colleges.columns else 'State'
                    university = row['Home University'] if 'Home University' in all_colleges.columns else 'N/A'
                    
                    # Avoid duplicates
                    if college_key not in seen_names:
                        recommended_colleges.append({
                            'name': college_name,
                            'probability': row['probability'],
                            'cutoff_rank': cutoff_rank,
                            'percentile': percentile_val,
                            'type': col_type,
                            'university': university,
                            'tier': self.get_college_tier(college_name, cutoff_rank if isinstance(cutoff_rank, int) else 100000),
                            'rank_score': row['rank_score'],
                            'vacant_seats': self.get_vacant_seats(college_name)
                        })
                        seen_names.add(college_key)
            
            # If we don't have enough recommendations, fill with best matches
            if len(recommended_colleges) < 10:
                remaining_colleges = all_colleges.head(50)
                for _, row in remaining_colleges.iterrows():
                    college_name = str(row['College Name']).strip()
                    college_key = college_name.lower()
                    if college_key not in seen_names:
                        try:
                            cutoff_val = int(row['Cutoff Rank'])
                        except Exception:
                            cutoff_val = 'N/A'
                        try:
                            perc_val = float(row['Percentile']) if 'Percentile' in row else 'N/A'
                        except Exception:
                            perc_val = 'N/A'
                        col_type = row['Type'] if 'Type' in all_colleges.columns else 'State'
                        university = row['Home University'] if 'Home University' in all_colleges.columns else 'N/A'
                        recommended_colleges.append({
                            'name': college_name,
                            'probability': row['probability'],
                            'cutoff_rank': cutoff_val,
                            'percentile': perc_val,
                            'type': col_type,
                            'university': university,
                            'tier': self.get_college_tier(college_name, cutoff_val if isinstance(cutoff_val, int) else 100000),
                            'rank_score': row['rank_score'],
                            'vacant_seats': self.get_vacant_seats(college_name)
                        })
                        seen_names.add(college_key)
            
            # Final check: if still fewer than 10, pad with generic fallback recommendations
            if len(recommended_colleges) < 10:
                fallback_recs = self.get_fallback_recommendations(student_data)
                for rec in fallback_recs:
                    if len(recommended_colleges) >= 10:
                        break
                    college_key = str(rec['name']).strip().lower()
                    if college_key not in seen_names:
                        recommended_colleges.append(rec)
                        seen_names.add(college_key)
            
            # Sort final recommendations by probability and rank score
            recommended_colleges.sort(key=lambda x: (x['probability'], x['rank_score']), reverse=True)

            # Input-seeded diversification among similarly scored colleges
            try:
                seed_str = f"{course}|{category}|{rank}|{percentile}"
                seeded_rng = random.Random(hash(seed_str))
                # Split into strong and extended candidates
                strong = recommended_colleges[:5]
                extended = recommended_colleges[5:50]
                # Shuffle extended deterministically based on input seed
                seeded_rng.shuffle(extended)
                diversified = strong + extended
                return diversified[:10]
            except Exception:
                return recommended_colleges[:10]
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self.get_fallback_recommendations(student_data)
    
    def calculate_rank_score(self, student_rank, cutoff_rank):
        """Calculate a score based on how well the student's rank matches the cutoff rank"""
        if cutoff_rank <= 0:
            return 0
        
        # For eligible colleges (student_rank <= cutoff_rank), higher score for closer matches
        if student_rank <= cutoff_rank:
            # Score is higher when student rank is closer to cutoff rank
            # This ensures students get colleges appropriate to their rank
            rank_diff = cutoff_rank - student_rank
            if rank_diff == 0:
                return 1.0  # Perfect match
            elif rank_diff <= 100:
                return 0.9  # Very close
            elif rank_diff <= 500:
                return 0.8  # Close
            elif rank_diff <= 1000:
                return 0.7  # Reasonable
            elif rank_diff <= 5000:
                return 0.6  # Acceptable
            else:
                return 0.5  # Safe choice
        else:
            # For colleges where student is not eligible, lower score
            rank_diff = student_rank - cutoff_rank
            if rank_diff <= 100:
                return 0.4  # Close but not eligible
            elif rank_diff <= 500:
                return 0.3  # Somewhat close
            elif rank_diff <= 1000:
                return 0.2  # Not very close
            else:
                return 0.1  # Far from eligible
    
    def calculate_probability(self, student_rank, cutoff_rank, rank_score, student_percentile, cutoff_percentile):
        """Calculate admission probability based on rank and percentile match (more generous)."""
        if cutoff_rank <= 0:
            return 0.1
        
        # Base probability from rank score
        base_prob = max(0.0, min(1.0, rank_score * 1.05 + 0.02))  # boost base a bit

        # Percentile adjustment if available
        perc_adj = 0.0
        try:
            if not np.isnan(cutoff_percentile) and student_percentile > 0:
                perc_diff = student_percentile - float(cutoff_percentile)
                if perc_diff >= 5:
                    perc_adj = 0.12
                elif perc_diff >= 2:
                    perc_adj = 0.08
                elif perc_diff >= 0:
                    perc_adj = 0.04
                elif perc_diff >= -2:
                    perc_adj = -0.02
                elif perc_diff >= -5:
                    perc_adj = -0.04
                else:
                    perc_adj = -0.06
        except Exception:
            perc_adj = 0.0
        
        # Adjust based on rank difference
        if student_rank <= cutoff_rank:
            # Student is eligible
            rank_diff = cutoff_rank - student_rank
            if rank_diff == 0:
                return min(0.98, base_prob + 0.12 + perc_adj)  # Perfect match (higher cap)
            elif rank_diff <= 100:
                return min(0.95, base_prob + 0.10 + perc_adj)  # Very close
            elif rank_diff <= 500:
                return min(0.92, base_prob + 0.06 + perc_adj)  # Close
            elif rank_diff <= 1000:
                return min(0.90, base_prob + 0.03 + perc_adj)  # Reasonable
            elif rank_diff <= 5000:
                return min(0.85, base_prob - 0.02 + perc_adj)  # Acceptable
            else:
                return min(0.80, base_prob - 0.05 + perc_adj)  # Safe choice
        else:
            # Student is not eligible
            rank_diff = student_rank - cutoff_rank
            if rank_diff <= 100:
                return max(0.05, base_prob - 0.08 + perc_adj)  # Close but not eligible
            elif rank_diff <= 500:
                return max(0.04, base_prob - 0.16 + perc_adj)  # Somewhat close
            elif rank_diff <= 1000:
                return max(0.03, base_prob - 0.22 + perc_adj)  # Not very close
            else:
                return max(0.02, base_prob - 0.30 + perc_adj)  # Far from eligible
    
    def get_rank_tiers(self, student_rank):
        """Get different rank tiers to ensure variety in recommendations"""
        if student_rank <= 1000:
            # Top rank - focus on tier 1 colleges
            return [
                {'min_rank': 1, 'max_rank': student_rank + 500, 'count': 3},
                {'min_rank': student_rank + 500, 'max_rank': student_rank + 2000, 'count': 2}
            ]
        elif student_rank <= 5000:
            # Good rank - mix of tier 1 and 2
            return [
                {'min_rank': 1, 'max_rank': student_rank + 1000, 'count': 2},
                {'min_rank': student_rank + 1000, 'max_rank': student_rank + 3000, 'count': 2},
                {'min_rank': student_rank + 3000, 'max_rank': student_rank + 5000, 'count': 1}
            ]
        elif student_rank <= 15000:
            # Average rank - mix of tier 2 and 3
            return [
                {'min_rank': student_rank - 2000, 'max_rank': student_rank + 2000, 'count': 2},
                {'min_rank': student_rank + 2000, 'max_rank': student_rank + 5000, 'count': 2},
                {'min_rank': student_rank + 5000, 'max_rank': student_rank + 10000, 'count': 1}
            ]
        else:
            # Lower rank - focus on tier 3 and 4
            return [
                {'min_rank': max(1, student_rank - 5000), 'max_rank': student_rank + 5000, 'count': 3},
                {'min_rank': student_rank + 5000, 'max_rank': student_rank + 15000, 'count': 2}
            ]
    
    def get_college_details(self, college_name, course, category):
        """Get detailed information about a college from the actual dataset"""
        try:
            # Filter the dataset for this specific college, course, and category
            college_data = self.df[
                (self.df['College Name'] == college_name) & 
                (self.df['Course Name'] == course) & 
                (self.df['Category'] == category)
            ]
            
            if not college_data.empty:
                # Get the most recent/relevant entry
                latest_entry = college_data.iloc[0]
                return {
                    'cutoff_rank': int(latest_entry['Cutoff Rank']),
                    'percentile': float(latest_entry['Percentile']),
                    'type': latest_entry['Type'],
                    'university': latest_entry['Home University']
                }
            else:
                # Fallback if no exact match found
                return {
                    'cutoff_rank': 'N/A',
                    'percentile': 'N/A',
                    'type': 'State',
                    'university': 'N/A'
                }
        except Exception as e:
            print(f"Error getting college details: {e}")
            return {
                'cutoff_rank': 'N/A',
                'percentile': 'N/A',
                'type': 'State',
                'university': 'N/A'
            }
    
    def get_college_tier(self, college_name, cutoff_rank):
        """Determine college tier based on name and cutoff rank"""
        if cutoff_rank == 'N/A':
            cutoff_rank = 100000
        
        if cutoff_rank < 1000:
            return 'Tier 1'
        elif cutoff_rank < 10000:
            return 'Tier 2'
        elif cutoff_rank < 50000:
            return 'Tier 3'
        else:
            return 'Tier 4'
    
    def get_fallback_recommendations(self, student_data):
        """Fallback recommendations if ML model fails"""
        base_list = [
            {
                'name': 'Government College of Engineering, Pune',
                'probability': 0.8,
                'cutoff_rank': '5000-10000',
                'percentile': '85-90',
                'type': 'State',
                'university': 'Pune University',
                'tier': 'Tier 2',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'VJTI, Mumbai',
                'probability': 0.7,
                'cutoff_rank': '1000-5000',
                'percentile': '90-95',
                'type': 'State',
                'university': 'Mumbai University',
                'tier': 'Tier 1',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'COEP, Pune',
                'probability': 0.6,
                'cutoff_rank': '2000-8000',
                'percentile': '88-92',
                'type': 'State',
                'university': 'Pune University',
                'tier': 'Tier 1',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'Sardar Patel Institute of Technology, Mumbai',
                'probability': 0.58,
                'cutoff_rank': '8000-15000',
                'percentile': '80-85',
                'type': 'Private',
                'university': 'Mumbai University',
                'tier': 'Tier 2',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'KJ Somaiya College of Engineering, Mumbai',
                'probability': 0.55,
                'cutoff_rank': '15000-25000',
                'percentile': '75-80',
                'type': 'Private',
                'university': 'Mumbai University',
                'tier': 'Tier 3',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'Walchand College of Engineering, Sangli',
                'probability': 0.52,
                'cutoff_rank': '12000-22000',
                'percentile': '76-84',
                'type': 'State',
                'university': 'Shivaji University',
                'tier': 'Tier 2',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'Sardar Patel College of Engineering, Mumbai',
                'probability': 0.50,
                'cutoff_rank': '14000-26000',
                'percentile': '74-82',
                'type': 'State',
                'university': 'Mumbai University',
                'tier': 'Tier 2',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'Pune Institute of Computer Technology (PICT), Pune',
                'probability': 0.48,
                'cutoff_rank': '16000-28000',
                'percentile': '72-80',
                'type': 'Private',
                'university': 'SPPU',
                'tier': 'Tier 3',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'Vishwakarma Institute of Technology (VIT), Pune',
                'probability': 0.46,
                'cutoff_rank': '18000-30000',
                'percentile': '70-78',
                'type': 'Private',
                'university': 'SPPU',
                'tier': 'Tier 3',
                'vacant_seats': random.randint(1, 7)
            },
            {
                'name': 'Fr. Conceicao Rodrigues College of Engineering, Mumbai',
                'probability': 0.44,
                'cutoff_rank': '20000-32000',
                'percentile': '68-76',
                'type': 'Private',
                'university': 'Mumbai University',
                'tier': 'Tier 3',
                'vacant_seats': random.randint(1, 7)
            }
        ]

        # Ensure at least 10 entries
        # Dynamic fallback from dataset if available
        try:
            self.ensure_data_loaded()
            df_all = self.df
            # prefer same course/category if present in student_data
            raw_course = student_data.get('course', '')
            raw_category = student_data.get('category', '')
            course_norm = str(raw_course).strip().lower()
            category_norm = str(raw_category).strip().lower()
            subset = df_all.copy()
            if 'course_norm' in subset.columns:
                subset = subset[subset['course_norm'] == course_norm] if course_norm else subset
            if 'category_norm' in subset.columns:
                subset = subset[subset['category_norm'] == category_norm] if category_norm else subset
            if subset.empty:
                subset = df_all
            subset = subset.sort_values(['Cutoff Rank'], ascending=[True]).head(120)
            dyn = []
            for _, row in subset.iterrows():
                # compute probability using student's inputs
                try:
                    cutoff_rank_val = float(row['Cutoff Rank']) if pd.notna(row['Cutoff Rank']) else 0
                except Exception:
                    cutoff_rank_val = 0
                try:
                    cutoff_perc_val = float(row['Percentile']) if pd.notna(row['Percentile']) else float('nan')
                except Exception:
                    cutoff_perc_val = float('nan')
                prob = self.calculate_probability(
                    student_rank=rank if 'rank' in locals() else 0,
                    cutoff_rank=cutoff_rank_val,
                    rank_score=self.calculate_rank_score(rank if 'rank' in locals() else 0, cutoff_rank_val),
                    student_percentile=percentile if 'percentile' in locals() else 0.0,
                    cutoff_percentile=cutoff_perc_val
                )
                dyn.append({
                    'name': row['College Name'],
                    'probability': prob,
                    'cutoff_rank': int(row['Cutoff Rank']) if pd.notna(row['Cutoff Rank']) else 'N/A',
                    'percentile': float(row['Percentile']) if pd.notna(row['Percentile']) else 'N/A',
                    'type': row['Type'] if 'Type' in row else 'State',
                    'university': row['Home University'] if 'Home University' in row else 'N/A',
                    'tier': self.get_college_tier(row['College Name'], int(row['Cutoff Rank']) if pd.notna(row['Cutoff Rank']) else 100000),
                    'vacant_seats': self.get_vacant_seats(row['College Name'])
                })
                if len(dyn) >= 10:
                    break
            if dyn:
                # Diversify deterministic order by input seed
                try:
                    seed_str = f"{raw_course}|{raw_category}|{student_data.get('rank','')}|{student_data.get('percentile','')}"
                    seeded_rng = random.Random(hash(seed_str))
                    seeded_rng.shuffle(dyn)
                except Exception:
                    pass
                return dyn
        except Exception:
            pass
        return base_list[:10]
    
    def load_and_train(self):
        """Load data and train models"""
        df = self.load_data()
        if df is not None:
            df_processed = self.preprocess_data(df)
            self.train_models(df_processed)
    
    def save_models(self):
        """Save trained models"""
        try:
            model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(self.rank_model, os.path.join(model_dir, 'rank_model.pkl'))
            joblib.dump(self.percentile_model, os.path.join(model_dir, 'percentile_model.pkl'))
            joblib.dump(self.college_model, os.path.join(model_dir, 'college_model.pkl'))
            joblib.dump(self.college_encoder, os.path.join(model_dir, 'college_encoder.pkl'))
            joblib.dump(self.course_encoder, os.path.join(model_dir, 'course_encoder.pkl'))
            joblib.dump(self.category_encoder, os.path.join(model_dir, 'category_encoder.pkl'))
            joblib.dump(self.type_encoder, os.path.join(model_dir, 'type_encoder.pkl'))
            joblib.dump(self.university_encoder, os.path.join(model_dir, 'university_encoder.pkl'))
            
            print("Models saved successfully!")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
            
            self.rank_model = joblib.load(os.path.join(model_dir, 'rank_model.pkl'))
            self.percentile_model = joblib.load(os.path.join(model_dir, 'percentile_model.pkl'))
            self.college_model = joblib.load(os.path.join(model_dir, 'college_model.pkl'))
            self.college_encoder = joblib.load(os.path.join(model_dir, 'college_encoder.pkl'))
            self.course_encoder = joblib.load(os.path.join(model_dir, 'course_encoder.pkl'))
            self.category_encoder = joblib.load(os.path.join(model_dir, 'category_encoder.pkl'))
            self.type_encoder = joblib.load(os.path.join(model_dir, 'type_encoder.pkl'))
            self.university_encoder = joblib.load(os.path.join(model_dir, 'university_encoder.pkl'))
            
            self.is_trained = True
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.load_and_train()

# Global model instance
ml_model = CollegePredictionModel()
