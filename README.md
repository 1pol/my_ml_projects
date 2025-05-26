
# 🏠 Bengaluru House Price Prediction

A machine learning project that predicts house prices in Bengaluru based on features like location, square footage, number of BHKs, and bathrooms. Built using Python, Pandas, Scikit-learn, and Flask.

---

## 🚀 Features

- Linear regression model to predict prices
- Cleaned and preprocessed real-world Bengaluru housing dataset
- User-friendly Flask web interface
- Trained model serialized with `pickle` and deployed locally

---

## 🛠️ Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- Flask
- Pickle

---

## 📁 Project Structure

```
├── Bengaluru_House_Data.csv      # Dataset
├── columns.pkl                   # Column transformer info
├── model.pkl                     # Trained ML model
├── houseprice.py                 # Preprocessing and training code
├── main.py                       # Flask web app
├── README.md                     # Project overview
```

---

## ⚙️ How to Run

1. Clone the repository:
   ```
   git clone https://github.com/1pol/my_ml_projects.git
   cd my_ml_projects
   ```

2. Install dependencies (create a virtual env if desired):
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```
   python main.py
   ```

4. Open your browser at:
   ```
   http://127.0.0.1:5000/
   ```

---

## 📈 Dataset

Source: Real estate data of Bengaluru containing features like area type, location, size, total square feet, price, etc.

---

## 🙋‍♂️ Author

Harshit Polmersetty  
📧 Email: harshit.pst049@gmail.com

---

## 📌 License

This project is licensed under the MIT License.
