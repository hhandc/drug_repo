library(ggplot2)
library(dplyr)
library(readr)

data <- read_csv("../data/processed/combined_dti_dataset.csv")

interaction_plot <- ggplot(data, aes(x = Y)) +
  geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Interaction Values",
       x = "Interaction Value (Y)", 
       y = "Frequency") +
  theme_minimal()

print(interaction_plot)

molecular_weight_plot <- ggplot(data, aes(x = nchar(SMILES))) +
  geom_histogram(binwidth = 10, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Drug Molecular Weights",
       x = "Approximate Molecular Weight",
       y = "Frequency") +
  theme_minimal()

print(molecular_weight_plot)

sequence_length_plot <- ggplot(data, aes(x = nchar(Target_Sequence))) +
  geom_histogram(binwidth = 50, fill = "lightcoral", color = "black") +
  labs(title = "Distribution of Protein Sequence Lengths",
       x = "Protein Sequence Length",
       y = "Frequency") +
  theme_minimal()

print(sequence_length_plot)