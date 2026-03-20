# TalentLens — Dataset Instructions

This app uses two public datasets. Download them and place the files here.

## 1. Kaggle Resume Dataset
URL: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
File needed: `Resume.csv`
Place at: `data/Resume.csv`

Columns used: `Resume_str` (raw text), `Category` (job category)
The app extracts a vocabulary of real skills from these ~2,400 resumes at startup.

## 2. LinkedIn Job Postings
URL: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
File needed: `postings.csv` (from the dataset zip)
Place at: `data/postings.csv`

Columns used: `title`, `description`, `skills_desc`
The app uses this to let users search real job postings instead of typing skills manually.

## Running without datasets
The app works without these files using a built-in skill taxonomy derived from
the same datasets (pre-extracted and bundled in data/skills_taxonomy.json).
