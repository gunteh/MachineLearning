import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def map_to_top_ids(val, top_ids, other_val=9999):
    return val if val in top_ids else other_val


def data_exploration():
    hotel_data = pd.read_csv('hotel_bookings.csv')
    print(hotel_data.head())        
    print(hotel_data.info()) 
    print("show the basic dataset dimensionality")        
    print(hotel_data.shape)

    print("show the summaries of each")
    print(hotel_data.describe())
    print("show the summaries of each, including objects")
    print(hotel_data.describe(include='object'))

    print("any missing data?")
    print(hotel_data.isnull().sum())

    ### Plotting correlation

    # set numeric columns aas the int64 and floats
    numeric_columns = hotel_data.select_dtypes(include=['int64', 'float64'])

    # Create confusion matrix
    correlation_matrix = numeric_columns.corr()

    # Make the correlations to just the is_canceled category
    target_correlation = correlation_matrix.loc[:, 'is_canceled']

    # make a df of the targets and correlation
    target_correlaation_df = pd.DataFrame(target_correlation).rename(columns={'is_canceled': 'correlation'})

    # Add an extra filter for higher correlation to avoid background noise, so here abs > 0.05
    filtered_correlation_df = target_correlaation_df[abs(target_correlaation_df['correlation']) > 0.05]

    # Now sort vaalues by correlation, descending
    filtered_correlation_df_sorted = filtered_correlation_df.sort_values(by='correlation', ascending=False)

    # plotting the correlataion map
    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(filtered_correlation_df_sorted.T, annot=True, cmap='coolwarm', cbar=False, ax=ax)
    plt.title("is_canceled feature selection")
    plt.yticks(rotation=0)
    plt.savefig("heatmap_correlated_figures_hotels.png", dpi=100, bbox_inches='tight') #visualizaation problem, solved with saving directly
    plt.show()

    # notating the feaataures most correlated with is_canceled
    highly_correlated_features_to_cancelation = ['lead_time', 'adr', 'adults', 'arrival_date_year', 
                        'previous_cancellations', 'is_repeated_guest', 'company',
                        'booking_changes', 'total_of_special_requests',
                        'required_car_parking_spaces', 'agent']

    #determining company and agent uniquee ID's anad frequency:
    #collecting the column values for company and agent
    company_counts = hotel_data['company'].value_counts(dropna=True)
    agent_counts = hotel_data['agent'].value_counts(dropna=True)

    # make a df of the counts for company
    company_df = company_counts.reset_index()
    company_df.columns = ['id', 'count']
    company_df['type'] = 'company'

    # make a df of the counts for agent
    agent_df = agent_counts.reset_index()
    agent_df.columns = ['id', 'count']
    agent_df['type'] = 'agent'

    # Make a df of the two
    combined_df = pd.concat([company_df, agent_df])

    # Sort the df by descending values
    combined_df = combined_df.sort_values(by='count', ascending=False)

    print(combined_df.to_string(index=False)) 

    # plotting the difference
    plt.figure(figsize=(12, 6))
    sns.barplot(data=combined_df, x='id', y='count', hue='type', dodge=False)
    plt.xticks([], [])  # for the hiding of axis labels if too many IDs
    plt.xlabel('Unique IDs')
    plt.ylabel('Count of IDs')
    plt.title('Count of Unique Company and Agent IDs (Descending)')
    plt.legend(title='Type')
    plt.tight_layout()
    plt.savefig("Unqiue IDs by company and agent.png", dpi=100, bbox_inches='tight') 
    plt.show()

def training_assess_top_agents_companies(hotel_data):
    #determining cutoffs for top-n with agent and company
    print(hotel_data.head())

    # Total counts of agent  aand company
    agent_counts = hotel_data['agent'].value_counts().reset_index()
    agent_counts.columns = ['id', 'count']
    agent_counts = agent_counts[agent_counts['id'] != 0]  # Remove missing agents (was 0 after fillna)

    total_agent_count = agent_counts['count'].sum()
    agent_counts = agent_counts.sort_values('count', ascending=False)
    agent_counts['cumulative'] = agent_counts['count'].cumsum()
    agent_counts['cumulative_pct'] = agent_counts['cumulative'] / total_agent_count

    company_counts = hotel_data['company'].value_counts().reset_index()
    company_counts.columns = ['id', 'count']
    company_counts = company_counts[company_counts['id'] != 0] 

    # filtering agents by cumulative % <= 80% and count >= 30 (statistical significance)
    agent_top_filtered = agent_counts[(agent_counts['cumulative_pct'] <= 0.80) & (agent_counts['count'] >= 30)]

    # filtering company by greater  thaan 30 instances for aan id (statistical significance)
    company_top_filtered = company_counts[company_counts['count'] >= 30]

    print(f"Agent IDs meeting both criteria: {agent_top_filtered.shape[0]}")
    print(f"Companay IDs meeting minimum statistical significance: {company_top_filtered.shape[0]}")

    # Total (peaking at the future, but this function probably won't produce this exact result: 
    # Agent IDs meeting both criteria: 24
    # Same look as with agaents, and for reference with reeporting but not for the code itself: 
    # Companay IDs meeting minimum statistical significance: 29

    top_agents = set(agent_top_filtered['id'])
    top_companies = set(company_top_filtered['id'])

    return top_agents, top_companies

def training_assess_top_countries(hotel_data):
    country_counts = hotel_data['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    total = country_counts['count'].sum()
    country_counts['cumulative'] = country_counts['count'].cumsum()
    country_counts['cumulative_pct'] = country_counts['cumulative'] / total

    # Country must cover top 80% of entries and have at least 30 entries
    top_countries = country_counts[
        (country_counts['cumulative_pct'] <= 0.80) & (country_counts['count'] >= 30)
    ]['country'].tolist()

    print(f"Top countries selected (statistically significant): {len(top_countries)}")

    return set(top_countries)


def data_processing_lite(hotel_data, top_agents=None, top_companies=None, top_countries=None):

    # notating the blank missing values:

    # Fill children with 0 (assumes zero, even if not explicitly mentioning zero, just as a default (bias introduction, but only for 4 cells))
    hotel_data['children'] = hotel_data['children'].fillna(0).astype(int)

    # Fill country with 'Unknown' as it's the safest bet when in doubt
    hotel_data['country'] = hotel_data['country'].fillna('Unknown')

    # Agent and company are hundreds of unique IDs, something like 300 agents and 600 companies, filling 0 for rest
    hotel_data['agent'] = hotel_data['agent'].fillna(0).astype(int)
    hotel_data['company'] = hotel_data['company'].fillna(0).astype(int)

    if top_agents is None or top_companies is None:
        top_agents, top_companies = training_assess_top_agents_companies(hotel_data)
    if top_countries is None:
        top_countries = training_assess_top_countries(hotel_data)

    # Define mapping function
    def map_to_top_ids(val, top_ids, other_val=9999):
        return val if val in top_ids else other_val

    # Apply mapping (to training or testing data)
    hotel_data['agent_mapped'] = hotel_data['agent'].apply(map_to_top_ids, args=(top_agents,))
    hotel_data['company_mapped'] = hotel_data['company'].apply(map_to_top_ids, args=(top_companies,))

    hotel_data['country_mapped'] = hotel_data['country'].mask(~hotel_data['country'].isin(top_countries), 'Other')

    hotel_data = hotel_data.drop(columns=['agent', 'company', 'country'])

    categorical_cols = hotel_data.select_dtypes(include='object').columns

    for col in categorical_cols:
        hotel_data[col] = hotel_data[col].astype('category')

    if 'reservation_status_date' in hotel_data.columns:
        hotel_data = hotel_data.drop(columns=['reservation_status_date'])
    
    if 'reservation_status' in hotel_data.columns:
        hotel_data = hotel_data.drop(columns=['reservation_status'])

    hotel_data['adr'] = pd.to_numeric(hotel_data['adr'], errors='coerce')

    return hotel_data, top_agents, top_companies, top_countries

class LiteHotelPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, top_n_agents=20, top_n_companies=50, top_n_countries=20):
        self.top_n_agents = top_n_agents
        self.top_n_companies = top_n_companies
        self.top_n_countries = top_n_countries

    def fit(self, X, y=None):
        X = X.copy()

        # Fill missing agent/company/country dataa for top selections
        X['agent'] = X['agent'].fillna(0).astype(int)
        X['company'] = X['company'].fillna(0).astype(int)
        X['country'] = X['country'].fillna('Unknown')

        # Learning top aagents, top companies, and top countries from data
        self.top_agents_ = X['agent'].value_counts().head(self.top_n_agents).index.tolist()
        self.top_companies_ = X['company'].value_counts().head(self.top_n_companies).index.tolist()
        self.top_countries_ = X['country'].value_counts().head(self.top_n_countries).index.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing values
        X['children'] = X['children'].fillna(0).astype(int)
        X['agent'] = X['agent'].fillna(0).astype(int)
        X['company'] = X['company'].fillna(0).astype(int)
        X['country'] = X['country'].fillna('Unknown')

        # Apply mapping (to training or testing data)
        X['agent_mapped'] = X['agent'].apply(map_to_top_ids, args=(self.top_agents_,))
        X['company_mapped'] = X['company'].apply(map_to_top_ids, args=(self.top_companies_,))

        X['country_mapped'] = X['country'].mask(~X['country'].isin(self.top_countries_), 'Other')

        X = X.drop(columns=['agent', 'company', 'country'])

        X = X.drop(columns=['reservation_status', 'reservation_status_date'], errors='ignore')

        # Convert categorical columns as needed for object columns
        categorical_cols = X.select_dtypes(include='object').columns
        for col in categorical_cols:
            X[col] = X[col].astype('category')

        # Ensure adr is numeric if needed
        X['adr'] = pd.to_numeric(X['adr'], errors='coerce')

        return X

"""def make_hotel_scaled_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = make_df_scaled_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ('preprocessing', LiteHotelPreprocessor()),
        ('preprocessing2', preprocessor),
        ('model', model)
    ])"""

def make_hotel_unscaled_pipeline(model, X_sample):
    # Apply the manual preprocessing first
    cleanedf = LiteHotelPreprocessor()
    X_cleaned = cleanedf.transform(X_sample)

    # gather the columns from that state
    numeric_cols = X_cleaned.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build scaled preprocessor
    preprocessor = make_df_unscaled_preprocessor(numeric_cols, categorical_cols)

    # Final pipeline (cleanedf already fit above)
    return Pipeline([
        ('preprocessing', cleanedf), 
        ('column_transform', preprocessor),
        ('model', model)
    ])

def make_hotel_scaled_pipeline(model, X_sample):
    # Apply the manual preprocessing first
    cleanedf = LiteHotelPreprocessor()
    X_cleaned = cleanedf.transform(X_sample)

    # gather the columns from that state
    numeric_cols = X_cleaned.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build scaled preprocessor
    preprocessor = make_df_scaled_preprocessor(numeric_cols, categorical_cols)

    # Final pipeline (cleanedf already fit above)
    return Pipeline([
        ('preprocessing', cleanedf), 
        ('column_transform', preprocessor),
        ('model', model)
    ])

"""def make_hotel_unscaled_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = make_df_unscaled_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ('preprocessing', LiteHotelPreprocessor()),
        ('preprocessing2', preprocessor),
        ('model', model)
    ])"""

"""def make_accidents_scaled_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = make_df_scaled_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ('preprocessing', VehicleAccidentsPreprocessor()),
        ('preprocessing2', preprocessor),
        ('model', model)
    ])
"""
"""def make_accidents_unscaled_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = make_df_unscaled_preprocessor(numeric_cols, categorical_cols)
    return Pipeline([
        ('preprocessing', VehicleAccidentsPreprocessor()),
        ('preprocessing2', preprocessor),
        ('model', model)
    ])"""

def make_accidents_unscaled_pipeline(model, X_sample):
    # Apply the manual preprocessing first
    cleanedf = VehicleAccidentsPreprocessor()
    X_cleaned = cleanedf.transform(X_sample)

    # gather the columns from that state
    numeric_cols = X_cleaned.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build scaled preprocessor
    preprocessor = make_df_unscaled_preprocessor(numeric_cols, categorical_cols)

    # Final pipeline (cleanedf already fit above)
    return Pipeline([
        ('preprocessing', cleanedf), 
        ('column_transform', preprocessor),
        ('model', model)
    ])

def make_accidents_scaled_pipeline(model, X_sample):
    # Apply the manual preprocessing first
    cleanedf = VehicleAccidentsPreprocessor()
    X_cleaned = cleanedf.transform(X_sample)

    # gather the columns from that state
    numeric_cols = X_cleaned.select_dtypes(include='number').columns.tolist()
    categorical_cols = X_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build scaled preprocessor with the new columns
    preprocessor = make_df_scaled_preprocessor(numeric_cols, categorical_cols)

    # pipeline work
    return Pipeline([
        ('preprocessing', cleanedf),
        ('column_transform', preprocessor),
        ('model', model)
    ])

def data_processing_full(hotel_data):

    # notating the blank missing values:

    # Fill children with 0 (assumes zero, even if not explicitly mentioning zero, just as a default (bias introduction, but only for 4 cells))
    hotel_data['children'] = hotel_data['children'].fillna(0).astype(int)

    # Fill country with 'Unknown' as it's the safest bet when in doubt
    hotel_data['country'] = hotel_data['country'].fillna('Unknown')

    # Agent and company are hundreds of unique IDs, something like 300 agents and 600 companies, filling 0 for rest
    hotel_data['agent'] = hotel_data['agent'].fillna(0).astype(int)
    hotel_data['company'] = hotel_data['company'].fillna(0).astype(int)

    categorical_cols = hotel_data.select_dtypes(include='object').columns

    for col in categorical_cols:
        hotel_data[col] = hotel_data[col].astype('category')

    if 'reservation_status_date' in hotel_data.columns:
        hotel_data = hotel_data.drop(columns=['reservation_status_date'])
    
    if 'reservation_status' in hotel_data.columns:
        hotel_data = hotel_data.drop(columns=['reservation_status'])

    hotel_data['adr'] = pd.to_numeric(hotel_data['adr'], errors='coerce')

    return hotel_data

def vehicle_accidents_processing(df):
    # Drop all na's
    df = df.dropna(subset=['Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng'])

    # Read and fill na's to many columns
    df['Wind_Direction'] = df['Wind_Direction'].fillna('Unknown')
    df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].fillna('Unknown')    

    # Filling Na's with median for weather columns, due to high skew
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    for col in weather_cols:
        df[col] = df[col].fillna(df[col].median())

    # Converting time and date to datetimee
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])

    # Datetime manipulations (if needed)
    df['Start_Hour'] = df['Start_Time'].dt.hour
    df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek  # 0 = Monday
    df['Start_Month'] = df['Start_Time'].dt.month
    df['Is_Weekend'] = df['Start_DayOfWeek'].isin([5, 6]).astype(int)
    df['Duration_Hours'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600.0


    df = df.drop(columns=['Start_Time', 'End_Time'])

    # Convert the state and city to top values and apply "other"
    top_states = df['State'].value_counts().nlargest(10).index
    df['State_Simplified'] = df['State'].where(df['State'].isin(top_states), 'Other')

    top_cities = df['City'].value_counts().nlargest(20).index
    df['City_Simplified'] = df['City'].where(df['City'].isin(top_cities), 'Other')

    # dropping a bunch of high cardinatliy columns
    dropping_cols = [
        'Number', 'Street', 'Zipcode', 'Country', 'Timezone',
        'Airport_Code', 'Weather_Timestamp', 'Nautical_Twilight',
        'Astronomical_Twilight', 'Civil_Twilight', 'ID'
    ]
    df = df.drop(columns=[col for col in dropping_cols if col in df.columns])

    # Just in case: reset index for easier refereencee (only if needed)
    df = df.reset_index(drop=True)

    return df

class VehicleAccidentsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # If hyperparameter changes are needed, then use it here
        pass

    def fit(self, X, y=None):
        # For if fitting values/columns are needed
        return self

    def transform(self, X):
        df = X.copy()

        print("running...")

        # Drop all na's from essential datetime and geo columns
        df = df.dropna(subset=['Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng'])

        # Read and fill na's to many columns
        df['Wind_Direction'] = df['Wind_Direction'].fillna('Unknown')
        df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
        df['Sunrise_Sunset'] = df['Sunrise_Sunset'].fillna('Unknown')    

        # Filling Na's with median for weather columns, due to high skew
        weather_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
        for col in weather_cols:
            df[col] = df[col].fillna(df[col].median())

        # Converting time and date to datetime
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
        df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')

        # Datetime manipulations (if needed)
        df['Start_Hour'] = df['Start_Time'].dt.hour
        df['Start_DayOfWeek'] = df['Start_Time'].dt.dayofweek  # 0 = Monday
        df['Start_Month'] = df['Start_Time'].dt.month
        df['Is_Weekend'] = df['Start_DayOfWeek'].isin([5, 6]).astype(int)
        df['Duration_Hours'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600.0

        # Convert the state and city to top values and apply "Other"
        top_states = df['State'].value_counts().nlargest(10).index
        df['State_Simplified'] = df['State'].where(df['State'].isin(top_states), 'Other')

        top_cities = df['City'].value_counts().nlargest(20).index
        df['City_Simplified'] = df['City'].where(df['City'].isin(top_cities), 'Other')

        # Dropping a bunch of high cardinality columns
        dropping_cols = [
            'Number', 'Street', 'Zipcode', 'Country', 'Timezone',
            'Airport_Code', 'Weather_Timestamp', 'Nautical_Twilight',
            'Astronomical_Twilight', 'Civil_Twilight', 'Start_Time', 'End_Time', 'ID'
        ]
        df = df.drop(columns=[col for col in dropping_cols if col in df.columns])

        # Just in case: reset index for easier reference (only if needed)
        df = df.reset_index(drop=True)

        print(df.select_dtypes(include='datetime').columns)

        return df

def data_exploration_accidents():
    hotel_data = pd.read_csv('US_Accidents_March23.csv')
    print(hotel_data.head())        
    print(hotel_data.info()) 
    print("show the basic dataset dimensionality")        
    print(hotel_data.shape)

    print("show the summaries of each")
    print(hotel_data.describe())
    print("show the summaries of each, including objects")
    print(hotel_data.describe(include='object'))

    print("any missing data?")
    print(hotel_data.isnull().sum())

    ### Plotting correlation

    # set numeric columns aas the int64 and floats
    numeric_columns = hotel_data.select_dtypes(include=['int64', 'float64'])

    # Create confusion matrix
    correlation_matrix = numeric_columns.corr()

    # Make the correlations to just the is_canceled category
    target_correlation = correlation_matrix.loc[:, 'Severity']

    # make a df of the targets and correlation
    target_correlaation_df = pd.DataFrame(target_correlation).rename(columns={'Severity': 'correlation'})

    # Add an extra filter for higher correlation to avoid background noise, so here abs > 0.05
    filtered_correlation_df = target_correlaation_df[abs(target_correlaation_df['correlation']) > 0.05]

    # Now sort vaalues by correlation, descending
    filtered_correlation_df_sorted = filtered_correlation_df.sort_values(by='correlation', ascending=False)

    # plotting the correlataion map
    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(filtered_correlation_df_sorted.T, annot=True, cmap='coolwarm', cbar=False, ax=ax)
    plt.title("Severity feature selection")
    plt.yticks(rotation=0)
    plt.savefig("heatmap_correlated_figures_Severity.png", dpi=100, bbox_inches='tight') #visualizaation problem, solved with saving directly
    plt.show()

    sns.countplot(x='Severity', data=us_accidents)
    plt.title("Class Distribution of Severity")

"""numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

numerical_transformer_unscaled = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(handle_unknown='value', handle_missing='value'))
])

alpha_num_preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

alpha_num_preprocessor_unscaled = ColumnTransformer([
    ('num', numerical_transformer_unscaled, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])"""

def make_df_scaled_preprocessor(model, numeric_cols, categorical_cols):
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(handle_unknown='value', handle_missing='value'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def make_df_unscaled_preprocessor(numeric_cols, categorical_cols):
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(handle_unknown='value', handle_missing='value'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor

def make_df_scaled_preprocessor(numeric_cols, categorical_cols):
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', TargetEncoder(handle_unknown='value', handle_missing='value'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor   


if __name__ == "__main__":
    pass
    #data_exploration()

    data_exploration_accidents()