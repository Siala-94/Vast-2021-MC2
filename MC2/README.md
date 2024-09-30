# 2021 Vast Mini-Challenge 2 - Visual Analytics for Missing Employees on Kronos Island

## Project Overview

This project is a solution for the 2021 VAST Mini-Challenge 2 -as part of the course advanced visual analytics at Linköpings Universit- aimed at investigating the movements, transactions, and relationships of GASTech employees on Kronos Island following the disappearance of several employees. The main goal is to find anomalies, relationships, and suspicious activities using visual analytics techniques to help law enforcement conduct thorough investigations.

## Project Objectives

The project focuses on answering the following questions:

What are the most popular places, and are there any anomalies?
Are there discrepancies between vehicle movements and transaction data?
Can ownership of each credit and loyalty card be inferred?
What formal or informal relationships exist between GASTech personnel?
Can any suspicious activities be identified, and where are these activities occurring?

## Data Sources

### Several datasets and mapping resources were provided:

car-assignment.csv: Employee vehicle assignments, including Car ID and employment details.
gps.csv: GPS data for vehicles, including timestamp, latitude, and longitude.
loyalty-data.csv: Loyalty card transactions with timestamps, locations, prices, and loyalty numbers.
cc-data.csv: Credit card transactions with timestamps, locations, prices, and the last 4 digits of credit card numbers.
Map of Abila (JPEG) and Shapefile of Abila and Kronos: Geospatial resources to help map locations.
Preprocessing Steps

### The following preprocessing steps were performed:

Card Data Comparison: Compared credit card and loyalty card data to infer a relationship between the two, allowing the datasets to be merged.
Location Mapping: Added missing location data (latitude and longitude) from the provided maps and shapefiles to ensure accurate geospatial visualization.
GPS and Car Assignment Merge: Merged GPS and car-assignment data to track employee movements with their vehicles.
Inferring Credit Card Owners: Used a majority vote algorithm to assign credit card ownership by identifying the employee closest to the transaction location within a set time frame.
Final Cleaning: Removed outliers and missing data for transactions without location coordinates.
Analysis & Methodology

### The project utilizes two main components:

Plotly Dashboard: An interactive dashboard built using Dash by Plotly, providing insights into card transactions and GPS data. Users can filter data by location, individual, and time for in-depth analysis.
Jupyter Notebook: Used for additional analyses and graph generation, such as heatmaps of transaction popularity.
Results

Popular Places & Anomalies: Transaction data reveals the most popular locations, with anomalies such as transaction spikes during weekends and inconsistencies in timing and location.
Vehicle vs. Transaction Discrepancies: Discrepancies were found where employees were recorded at one location via GPS but had transactions at another.
Inferring Card Ownership: Most credit card transactions could be aligned with GPS data, but some errors occurred, requiring further investigation.
Personnel Relationships: Visualizations show patterns of employee behavior, such as social gatherings outside of work, especially among engineers and executives.
Suspicious Activity: Certain employees exhibited unusual behavior, such as late-night visits to work and meetings at odd hours.
Design & Implementation

### The project was developed using Python for both preprocessing and visualization:

Dash by Plotly: Chosen for its compatibility with Python and ease of use in creating interactive visualizations.
Jupyter Notebook: Used for iterative data analysis and processing.
Limitations & Future Work

Credit Card Ownership Inference: The current method could be improved by correlating transactions more accurately with GPS data.
Visualization Improvements: Future iterations could benefit from more targeted visualizations that directly answer specific project questions.
Conclusion

The project successfully answers the key questions posed in the challenge, though with room for further refinement in credit card ownership attribution and visualization design.
