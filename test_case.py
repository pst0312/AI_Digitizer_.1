import plotly.express as px
import pandas as pd

graph = pd.read_csv("output_pdf/Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2/page_6/graphs/page6.3_points.csv")

fig = px.scatter(graph, x="x", y="y", title="Graph of page 6.3")
fig.show()