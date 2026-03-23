import pandas as pd
run='outputs/serious_run_french_001'  # change this

contrast=pd.read_csv(f'{run}/tables/sample_neuron_contrast.csv')
dist=pd.read_csv(f'{run}/tables/sample_layer_condition_distance.csv')
tend=pd.read_csv(f'{run}/tables/neuron_tendency.csv')

# A) Most condition-dominant neurons overall
top=contrast.sort_values('dominant_margin', ascending=False).head(30)
print('\nTop dominant neurons:')
print(top[['source_id','layer','neuron','dominant_condition','dominant_margin']].to_string(index=False))

# B) Which condition pairs are most different by layer (average)
g=dist.groupby(['layer','condition_a','condition_b'])['cosine_distance'].mean().reset_index()
print('\nMost different layer-condition pairs:')
print(g.sort_values('cosine_distance', ascending=False).head(20).to_string(index=False))

# C) Strongest recurring neurons by condition
tt=tend.sort_values(['event_count','activation_mean'], ascending=[False,False])
print('\nTop recurring neurons by condition:')
print(tt.groupby('condition').head(10)[['condition','domain','layer','neuron','event_count','activation_mean']].to_string(index=False))
