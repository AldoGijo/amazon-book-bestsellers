# 📚 Amazon Bestseller Book Explorer & Predictor

An interactive **Data Science & Machine Learning** portfolio project that analyzes Amazon's top-performing books. This project features a **Streamlit Dashboard** that combines rule-based user filtering with a predictive **Machine Learning backend** to estimate a book's marketplace longevity.

---

## 🚀 Live Application
🔗 **[Click here to view the live dashboard on Streamlit Cloud](https://streamlit.app)** *(Update this link after deploying!)*

---

## 🛠️ Project Lifecycle Breakdown

### 1. Data Engineering & Cleaning
- **Decimal Restoration**: Caught and corrected critical truncation bugs by maintaining exact floating-point precision for `Price (USD)` and `Rating`.
- **Temporal Parsing**: Extracted and normalized publication timelines into clean datetime structures to calculate the exact `Years_Since_Release`.
- **Dimensionality Reduction**: Removed zero-variance identifiers (`Amazon URL`, `ISBN`) to protect downstream models from data leakage and severe overfitting.

### 2. Feature Engineering
- **High-Cardinality Resolution**: Avoided categorical column explosion (168 unique authors, 56 unique publishers) by implementing target frequency-mapping algorithms (`Author_Book_Count`, `Publisher_Book_Count`).
- **Categorical Processing**: Implemented algorithmic Label Encoding for binary categorizations and One-Hot Encoding for structural book formats.

### 3. Machine Learning Backend
- **Algorithm**: `Random Forest Regressor` chosen for its exceptional capacity to process non-linear features and mixed scales without data standardisation.
- **Key Insight**: Model evaluation unlocked the core drivers of book longevity:
  - **Total Reviews**: Accounts for **60%** of predictive weight (the ultimate viral growth engine).
  - **Years Since Release**: Accounts for **32%** of predictive weight (legacy compounding advantage).
  - **Pricing/Formatting**: Holds **<1%** of predictive weight (market demand supersedes structural costs).

---

## 🖥️ Dashboard Architecture (`app.py`)

The application splits the workflow into two clean operations:
1. **🎯 Rule-Based Filter**: Queries the original string data using Pandas matrix operations to let users instantly sort, budget, and filter books by Author, Genre, and Publisher.
2. **🤖 ML Success Predictor**: Deserializes the trained Random Forest model artifact (`.pkl`) to run a real-time predictive simulation, estimating the total bestseller weeks for a theoretical new book concept.

---

## 🏃‍♂️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📦 Repository Structure
```text
├── app.py                         # Streamlit application script
├── Amazon_BestSelling_Books_500.csv # Original text-base dataset
├── book_success_model.pkl         # Trained Random Forest model binary
├── requirements.txt               # App cloud dependencies
└── README.md                      # Documentation
```
