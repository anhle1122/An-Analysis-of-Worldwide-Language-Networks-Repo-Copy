import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms import bipartite
import os

df = pd.read_csv("spoken_languages_by_country.csv")

# Create filtered bipartite graph
filtered_df = df[df['percent'] > 5]
B = nx.Graph()
countries = filtered_df['country'].unique()
languages = filtered_df['language'].unique()
B.add_nodes_from(countries, bipartite=0, type='country')
B.add_nodes_from(languages, bipartite=1, type='language')

for _, row in filtered_df.iterrows():
    B.add_edge(row['country'], row['language'], weight=row['percent'])

# === FIGURE 4: Bipartite Graph ===
pos = {}
y_gap = 1.5
for i, node in enumerate(sorted(countries)):
    pos[node] = (-1, i * y_gap)
for i, node in enumerate(sorted(languages)):
    pos[node] = (1, i * y_gap)

plt.figure(figsize=(30, 60))
nx.draw_networkx_nodes(B, pos, nodelist=countries, node_color='skyblue', node_size=800)
nx.draw_networkx_nodes(B, pos, nodelist=languages, node_color='orange', node_size=500)
nx.draw_networkx_edges(B, pos, edge_color='gray', alpha=0.4, width=1.2)
nx.draw_networkx_labels(B, pos, font_size=11)
plt.title("Country–Language Bipartite Graph (Filtered >5%)", fontsize=30)
plt.axis('off')
plt.savefig("bipartite_graph.png", dpi=300)
plt.close()

# === FIGURE 5: Edge Weight Histogram ===
weights = [d['weight'] for _, _, d in B.edges(data=True)]
plt.figure(figsize=(10, 6))
plt.hist(weights, bins=50, color='teal')
plt.title("Edge Weight Distribution", fontsize=16)
plt.xlabel("Percent of population", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.tight_layout()
plt.savefig("weight_distribution.png", dpi=300)
plt.close()

# === FIGURE 7: Heatmap of Top 30 Languages ===
heat_df = df.pivot_table(index='language', columns='country', values='percent', aggfunc='mean').fillna(0)
top_langs = df.groupby('language')['percent'].sum().sort_values(ascending=False).head(30).index
heat_df = heat_df.loc[heat_df.index.isin(top_langs)]

plt.figure(figsize=(16, 10))
sns.heatmap(heat_df, cmap='BuGn', linewidths=0.2, linecolor='gray', cbar_kws={'label': 'Percent (%)'})
plt.xticks(rotation=90)
plt.yticks(fontsize=8)
plt.title("Heatmap: Top 30 Spoken Languages Across Countries (WVS Q272)", fontsize=16)
plt.tight_layout()
plt.savefig("language_country_heatmap.png", dpi=300)
plt.close()
