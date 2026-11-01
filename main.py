"""
Student Performance Analysis and Prediction System
Mini Project - Data Mining using Python

The application generates a synthetic student dataset, trains a Decision Tree
classifier, evaluates it, and provides a Tkinter interface for predictions.

Note: The dataset is artificially generated for academic demonstration.
It does not represent real students or guarantee real-world prediction accuracy.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_performance_data.csv"
MODEL_FEATURES = [
    "Attendance",
    "Study_Hours",
    "Internal_Marks",
    "Assignment_Score",
    "Previous_Marks",
]


def generate_dataset(number_of_students=500, random_state=42):
    """Create reproducible synthetic student records and save them as CSV."""
    rng = np.random.default_rng(random_state)

    attendance = rng.uniform(35, 100, number_of_students).round(1)
    study_hours = rng.uniform(0.5, 8, number_of_students).round(1)
    internal_marks = rng.uniform(5, 50, number_of_students).round(1)
    assignment_score = rng.uniform(10, 100, number_of_students).round(1)
    previous_marks = rng.uniform(10, 100, number_of_students).round(1)

    # A transparent synthetic rule creates a final performance score.
    # Random noise makes the dataset less perfectly separable.
    score = (
        0.20 * attendance
        + 4.0 * study_hours
        + 0.55 * internal_marks
        + 0.15 * assignment_score
        + 0.25 * previous_marks
        + rng.normal(0, 7, number_of_students)
    )

    # The threshold is used only to create demonstration labels.
    result = np.where(score >= 65, "Pass", "Fail")

    data = pd.DataFrame({
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Internal_Marks": internal_marks,
        "Assignment_Score": assignment_score,
        "Previous_Marks": previous_marks,
        "Result": result,
    })
    data.to_csv(DATA_FILE, index=False)
    return data


def train_model(data):
    """Split data, train a Decision Tree, and calculate evaluation metrics."""
    X = data[MODEL_FEATURES]
    y = data["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=["Fail", "Pass"])
    report = classification_report(
        y_test, predictions, labels=["Fail", "Pass"], zero_division=0
    )
    return model, accuracy, matrix, report


class StudentPerformanceApp:
    def __init__(self, root, model, accuracy, matrix, report, data):
        self.root = root
        self.model = model
        self.accuracy = accuracy
        self.matrix = matrix
        self.report = report
        self.data = data
        self.entries = {}

        self.root.title("Student Performance Analysis and Prediction")
        self.root.geometry("850x720")
        self.root.minsize(760, 650)
        self.root.configure(bg="#f3f6fa")

        self.build_interface()

    def build_interface(self):
        header = tk.Frame(self.root, bg="#17365d", padx=20, pady=18)
        header.pack(fill="x")

        tk.Label(
            header,
            text="STUDENT PERFORMANCE ANALYSIS",
            font=("Arial", 19, "bold"),
            fg="white",
            bg="#17365d",
        ).pack()
        tk.Label(
            header,
            text="Predict academic result using a Decision Tree classifier",
            font=("Arial", 10),
            fg="#e5edf7",
            bg="#17365d",
        ).pack(pady=(5, 0))

        content = tk.Frame(self.root, bg="#f3f6fa", padx=28, pady=18)
        content.pack(fill="both", expand=True)

        tk.Label(
            content,
            text="Enter Student Details",
            font=("Arial", 15, "bold"),
            bg="#f3f6fa",
            fg="#17365d",
        ).pack(anchor="w", pady=(0, 12))

        form = tk.Frame(content, bg="white", padx=20, pady=16,
                        highlightbackground="#d8e0ea", highlightthickness=1)
        form.pack(fill="x")

        fields = [
            ("Attendance (%)", "Attendance", "35 to 100"),
            ("Study Hours per Day", "Study_Hours", "0.5 to 8"),
            ("Internal Marks (out of 50)", "Internal_Marks", "5 to 50"),
            ("Assignment Score (out of 100)", "Assignment_Score", "10 to 100"),
            ("Previous Exam Marks (out of 100)", "Previous_Marks", "10 to 100"),
        ]

        for row, (label, key, hint) in enumerate(fields):
            tk.Label(
                form, text=label, font=("Arial", 10, "bold"),
                bg="white", fg="#263445"
            ).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=9)

            entry = ttk.Entry(form, width=22, font=("Arial", 10))
            entry.grid(row=row, column=1, sticky="w", pady=9)
            self.entries[key] = entry

            tk.Label(
                form, text=hint, font=("Arial", 9),
                bg="white", fg="#718096"
            ).grid(row=row, column=2, sticky="w", padx=12)

        form.columnconfigure(2, weight=1)

        buttons = tk.Frame(content, bg="#f3f6fa")
        buttons.pack(fill="x", pady=16)

        ttk.Button(buttons, text="Predict Result",
                   command=self.predict).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Clear Fields",
                   command=self.clear_fields).pack(side="left", padx=8)
        ttk.Button(buttons, text="View Model Evaluation",
                   command=self.show_evaluation).pack(side="left", padx=8)
        ttk.Button(buttons, text="Dataset Summary",
                   command=self.show_summary).pack(side="left", padx=8)

        self.result_frame = tk.Frame(
            content, bg="white", padx=18, pady=16,
            highlightbackground="#d8e0ea", highlightthickness=1
        )
        self.result_frame.pack(fill="both", expand=True)

        tk.Label(
            self.result_frame, text="Prediction Result",
            font=("Arial", 13, "bold"), bg="white", fg="#17365d"
        ).pack(anchor="w")

        self.result_label = tk.Label(
            self.result_frame,
            text="Enter the student's details and click Predict Result.",
            font=("Arial", 11), bg="white", fg="#526174",
            wraplength=700, justify="left"
        )
        self.result_label.pack(anchor="w", pady=12)

        tk.Label(
            content,
            text=f"Model test accuracy: {self.accuracy * 100:.2f}%  |  "
                 f"Training dataset: {len(self.data)} synthetic records",
            font=("Arial", 9), bg="#f3f6fa", fg="#526174"
        ).pack(anchor="w", pady=(12, 0))

        tk.Label(
            content,
            text="Academic demonstration only: predictions are based on artificial sample data.",
            font=("Arial", 9, "italic"), bg="#f3f6fa", fg="#8a5a00"
        ).pack(anchor="w", pady=(4, 0))

    def predict(self):
        values = {}
        limits = {
            "Attendance": (35, 100),
            "Study_Hours": (0.5, 8),
            "Internal_Marks": (5, 50),
            "Assignment_Score": (10, 100),
            "Previous_Marks": (10, 100),
        }

        try:
            for key, entry in self.entries.items():
                raw = entry.get().strip()
                if not raw:
                    raise ValueError(f"Please enter a value for {key.replace('_', ' ')}.")
                value = float(raw)
                low, high = limits[key]
                if not low <= value <= high:
                    raise ValueError(
                        f"{key.replace('_', ' ')} must be between {low} and {high}."
                    )
                values[key] = value
        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))
            return

        input_data = pd.DataFrame([values], columns=MODEL_FEATURES)
        prediction = self.model.predict(input_data)[0]

        if prediction == "Pass":
            self.result_label.config(
                text="PREDICTED RESULT: PASS\n\n"
                     "The model predicts that the student is likely to pass "
                     "based on the entered academic information.",
                fg="#187443",
            )
        else:
            self.result_label.config(
                text="PREDICTED RESULT: FAIL\n\n"
                     "The model predicts that the student may be at risk of "
                     "failing based on the entered academic information. "
                     "This result can be used as an indication for academic support.",
                fg="#b42318",
            )

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_label.config(
            text="Enter the student's details and click Predict Result.",
            fg="#526174",
        )

    def show_evaluation(self):
        window = tk.Toplevel(self.root)
        window.title("Model Evaluation")
        window.geometry("620x500")

        text = tk.Text(window, wrap="word", font=("Consolas", 10), padx=12, pady=12)
        text.pack(fill="both", expand=True)

        content = (
            f"DECISION TREE MODEL EVALUATION\n"
            f"{'=' * 42}\n\n"
            f"Test accuracy: {self.accuracy * 100:.2f}%\n\n"
            f"Confusion matrix (actual rows; predicted columns)\n"
            f"Class order: Fail, Pass\n"
            f"{self.matrix}\n\n"
            f"Classification report\n"
            f"{self.report}\n"
            f"\nNote: Evaluation is performed on a held-out portion of "
            f"the generated synthetic dataset. It is not evidence of "
            f"real-world predictive performance.\n"
        )
        text.insert("1.0", content)
        text.config(state="disabled")

    def show_summary(self):
        window = tk.Toplevel(self.root)
        window.title("Dataset Summary")
        window.geometry("620x480")

        summary = self.data.describe().round(2).to_string()
        counts = self.data["Result"].value_counts().to_string()

        text = tk.Text(window, wrap="none", font=("Consolas", 10), padx=12, pady=12)
        text.pack(fill="both", expand=True)
        text.insert(
            "1.0",
            "DATASET SUMMARY\n"
            "===============\n\n"
            f"Number of records: {len(self.data)}\n\n"
            "Descriptive statistics:\n"
            f"{summary}\n\n"
            "Result distribution:\n"
            f"{counts}\n"
        )
        text.config(state="disabled")


def main():
    data = generate_dataset()
    model, accuracy, matrix, report = train_model(data)

    root = tk.Tk()
    StudentPerformanceApp(root, model, accuracy, matrix, report, data)
    root.mainloop()


if __name__ == "__main__":
    main()
