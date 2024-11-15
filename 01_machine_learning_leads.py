import pandas as pd
import sqlalchemy as sql
import pycaret.classification as clf



sql_engine = sql.create_engine("sqlite:///database/leads_scored.db")

conn = sql_engine.connect()

# Table Names

metadata = sql.MetaData()

metadata.reflect(bind=sql_engine)

metadata.tables.keys()

pd.read_sql_table('leads_scored', conn)

pd.read_sql_table('products', conn)

pd.read_sql_table('transactions', conn)


df = pd.read_sql_table('leads_scored', conn) \
    .drop(columns=['predict', 'p0', 'p1', 'mailchimp_id', 'made_purchase', 'user_full_name'])

target = pd.read_sql_table('transactions', conn)['user_email'].unique()

df['purchased'] = df['user_email'].isin(target).astype(int)

clf.setup(
    data = df, 
    target = "purchased",
    session_id = 123,  
)

model_xgb = clf.create_model('xgboost')

clf.predict_model(model_xgb, data = df, raw_score=True)
pd.read_sql_table('leads_scored', conn) 

