# **Drug Repurposing for Alzheimer's Disease using Machine Learning and Cheminformatics**

## **Overview**
This project aims to identify potential drug candidates for **Alzheimer’s disease** by predicting **drug-amyloid beta interactions** using **machine learning and cheminformatics techniques**. The accumulation of **amyloid beta** is a key pathological hallmark of Alzheimer's, making it a prime target for therapeutic intervention. Instead of developing new drugs from scratch, this system focuses on **drug repurposing**—the process of finding **new uses for existing drugs**—which can significantly **reduce time and costs** associated with drug discovery.

## **How It Works**
This system **predicts the interaction strength** between known drugs and amyloid beta using a combination of:
- **Molecular descriptor analysis** (RDKit)
- **Machine learning prediction** (RandomForestRegressor)
- **Drug data extraction** (Therapeutics Data Commons - PyTDC)
- **Chemical structure identification** (PubChem API)

It generates a **ranked list of drugs** based on their **predicted binding affinity to amyloid beta**, helping researchers prioritize compounds for further experimental validation.

---

## **Features**
## **1. Drug-Target Interaction Prediction**
- Extracts **molecular descriptors** (molecular weight, LogP, hydrogen donors/acceptors, TPSA) from **SMILES representations** of drugs.
- Processes drug-target interaction data from **PyTDC (DAVIS dataset)**.
- Utilizes a **Random Forest model** to predict drug binding affinity to **amyloid beta**.

## **2. Data Analysis and Visualization**
- **Statistical Summary**: Provides mean, median, min/max, and standard deviation of predicted scores.
- **Top Drug Candidates**: Identifies and ranks the top-scoring drugs for **amyloid beta binding**.
- **Visualization**:
  - **Histogram of predicted scores** to show distribution.
  - **Bar chart of top drugs** to highlight potential candidates.

## **3. Output Data for Further Analysis**
- Saves results as **drug_repurposing_results.csv**, containing:
  - **Drug ID**
  - **SMILES Representation**
  - **Predicted Binding Score**
- Enables **easy integration** with experimental validation pipelines.

## **Installation**
To run this system on **Google Colab** or locally, install the required dependencies:

```bash
pip install PyTDC rdkit pubchempy scikit-learn pandas numpy matplotlib
```
