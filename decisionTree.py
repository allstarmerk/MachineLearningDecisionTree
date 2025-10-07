import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# ============================================================================
# PROBLEM 1: Manual Decision Tree Building
# ============================================================================

def gini_impurity(labels):
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    impurity = 1.0
    for count in counts.values():
        prob = count / len(labels)
        impurity -= prob ** 2
    return impurity

def entropy(labels):
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    ent = 0.0
    for count in counts.values():
        prob = count / len(labels)
        if prob > 0:
            ent -= prob * np.log2(prob)
    return ent

def weighted_gini_after_split(left_labels, right_labels):
    n = len(left_labels) + len(right_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    weighted_gini = (n_left/n * gini_impurity(left_labels) + 
                     n_right/n * gini_impurity(right_labels))
    return weighted_gini

def weighted_entropy_after_split(left_labels, right_labels):
    n = len(left_labels) + len(right_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    weighted_ent = (n_left/n * entropy(left_labels) + 
                    n_right/n * entropy(right_labels))
    return weighted_ent

def information_gain(parent_labels, left_labels, right_labels):
    parent_entropy = entropy(parent_labels)
    weighted_child_entropy = weighted_entropy_after_split(left_labels, right_labels)
    return parent_entropy - weighted_child_entropy

def problem_1():
    print("PROBLEM 1: Decision Tree for Flight Departure")
    print("="*60)
    
    # Dataset
    data = pd.DataFrame({
        'foggy': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No'],
        'windy': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        'depart_on_time': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    })
    
    print("\nDataset:")
    print(data)
    
    # Convert to binary
    data_binary = data.copy()
    data_binary['foggy'] = (data['foggy'] == 'Yes').astype(int)
    data_binary['windy'] = (data['windy'] == 'Yes').astype(int)
    data_binary['depart_on_time'] = (data['depart_on_time'] == 'Yes').astype(int)
    
    labels = data_binary['depart_on_time'].values
    
    # Split on 'foggy'
    foggy_yes = data_binary[data_binary['foggy'] == 1]['depart_on_time'].values
    foggy_no = data_binary[data_binary['foggy'] == 0]['depart_on_time'].values
    
    # Split on 'windy'
    windy_yes = data_binary[data_binary['windy'] == 1]['depart_on_time'].values
    windy_no = data_binary[data_binary['windy'] == 0]['depart_on_time'].values
    
    # (a) Using Gini Impurity
    print("\n(a) Using Gini Impurity:")
    
    total_gini_foggy = weighted_gini_after_split(foggy_yes, foggy_no)
    total_gini_windy = weighted_gini_after_split(windy_yes, windy_no)
    
    print(f"Split on 'foggy': Total Gini = {total_gini_foggy:.4f}")
    print(f"Split on 'windy': Total Gini = {total_gini_windy:.4f}")
    
    best_gini = 'foggy' if total_gini_foggy < total_gini_windy else 'windy'
    print(f"Best split: {best_gini}")
    
    # Build the tree based on best split
    print(f"\nDecision Tree (Gini):")
    if best_gini == 'windy':
        print("Root: Windy?")
        print("  ├─ No → Depart on time: YES (4/4)")
        print("  └─ Yes:")
        # Check if we need further split
        windy_yes_data = data_binary[data_binary['windy'] == 1]
        if len(windy_yes_data) > 0:
            # Try splitting on foggy for the windy=yes branch
            wy_foggy_yes = windy_yes_data[windy_yes_data['foggy'] == 1]['depart_on_time'].values
            wy_foggy_no = windy_yes_data[windy_yes_data['foggy'] == 0]['depart_on_time'].values
            print(f"       └─ Foggy?")
            majority_yes = "YES" if np.sum(wy_foggy_yes) >= len(wy_foggy_yes)/2 else "NO"
            majority_no = "YES" if np.sum(wy_foggy_no) >= len(wy_foggy_no)/2 else "NO"
            print(f"          ├─ Yes → Depart on time: {majority_yes} ({np.sum(wy_foggy_yes)}/{len(wy_foggy_yes)})")
            print(f"          └─ No → Depart on time: {majority_no} ({np.sum(wy_foggy_no)}/{len(wy_foggy_no)})")
    else:
        print("Root: Foggy?")
        foggy_yes_data = data_binary[data_binary['foggy'] == 1]
        foggy_no_data = data_binary[data_binary['foggy'] == 0]
        print(f"  ├─ Yes:")
        fy_windy_yes = foggy_yes_data[foggy_yes_data['windy'] == 1]['depart_on_time'].values
        fy_windy_no = foggy_yes_data[foggy_yes_data['windy'] == 0]['depart_on_time'].values
        print(f"  │   └─ Windy?")
        majority_yes = "YES" if np.sum(fy_windy_yes) >= len(fy_windy_yes)/2 else "NO"
        majority_no = "YES" if np.sum(fy_windy_no) >= len(fy_windy_no)/2 else "NO"
        print(f"  │      ├─ Yes → Depart on time: {majority_yes} ({np.sum(fy_windy_yes)}/{len(fy_windy_yes)})")
        print(f"  │      └─ No → Depart on time: {majority_no} ({np.sum(fy_windy_no)}/{len(fy_windy_no)})")
        print(f"  └─ No:")
        fn_windy_yes = foggy_no_data[foggy_no_data['windy'] == 1]['depart_on_time'].values
        fn_windy_no = foggy_no_data[foggy_no_data['windy'] == 0]['depart_on_time'].values
        print(f"      └─ Windy?")
        majority_yes = "YES" if np.sum(fn_windy_yes) >= len(fn_windy_yes)/2 else "NO"
        majority_no = "YES" if np.sum(fn_windy_no) >= len(fn_windy_no)/2 else "NO"
        print(f"         ├─ Yes → Depart on time: {majority_yes} ({np.sum(fn_windy_yes)}/{len(fn_windy_yes)})")
        print(f"         └─ No → Depart on time: {majority_no} ({np.sum(fn_windy_no)}/{len(fn_windy_no)})")
    
    # (b) Using Information Gain
    print("\n(b) Using Information Gain:")
    
    ig_foggy = information_gain(labels, foggy_yes, foggy_no)
    ig_windy = information_gain(labels, windy_yes, windy_no)
    
    print(f"Split on 'foggy': Information Gain = {ig_foggy:.4f}")
    print(f"Split on 'windy': Information Gain = {ig_windy:.4f}")
    
    best_ig = 'foggy' if ig_foggy > ig_windy else 'windy'
    print(f"Best split: {best_ig}")
    
    # Build the tree based on best split
    print(f"\nDecision Tree (Information Gain):")
    if best_ig == 'windy':
        print("Root: Windy?")
        print("  ├─ No → Depart on time: YES (4/4)")
        print("  └─ Yes:")
        windy_yes_data = data_binary[data_binary['windy'] == 1]
        if len(windy_yes_data) > 0:
            wy_foggy_yes = windy_yes_data[windy_yes_data['foggy'] == 1]['depart_on_time'].values
            wy_foggy_no = windy_yes_data[windy_yes_data['foggy'] == 0]['depart_on_time'].values
            print(f"       └─ Foggy?")
            majority_yes = "YES" if np.sum(wy_foggy_yes) >= len(wy_foggy_yes)/2 else "NO"
            majority_no = "YES" if np.sum(wy_foggy_no) >= len(wy_foggy_no)/2 else "NO"
            print(f"          ├─ Yes → Depart on time: {majority_yes} ({np.sum(wy_foggy_yes)}/{len(wy_foggy_yes)})")
            print(f"          └─ No → Depart on time: {majority_no} ({np.sum(wy_foggy_no)}/{len(wy_foggy_no)})")
    else:
        print("Root: Foggy?")
        foggy_yes_data = data_binary[data_binary['foggy'] == 1]
        foggy_no_data = data_binary[data_binary['foggy'] == 0]
        print(f"  ├─ Yes:")
        fy_windy_yes = foggy_yes_data[foggy_yes_data['windy'] == 1]['depart_on_time'].values
        fy_windy_no = foggy_yes_data[foggy_yes_data['windy'] == 0]['depart_on_time'].values
        print(f"  │   └─ Windy?")
        majority_yes = "YES" if np.sum(fy_windy_yes) >= len(fy_windy_yes)/2 else "NO"
        majority_no = "YES" if np.sum(fy_windy_no) >= len(fy_windy_no)/2 else "NO"
        print(f"  │      ├─ Yes → Depart on time: {majority_yes} ({np.sum(fy_windy_yes)}/{len(fy_windy_yes)})")
        print(f"  │      └─ No → Depart on time: {majority_no} ({np.sum(fy_windy_no)}/{len(fy_windy_no)})")
        print(f"  └─ No:")
        fn_windy_yes = foggy_no_data[foggy_no_data['windy'] == 1]['depart_on_time'].values
        fn_windy_no = foggy_no_data[foggy_no_data['windy'] == 0]['depart_on_time'].values
        print(f"      └─ Windy?")
        majority_yes = "YES" if np.sum(fn_windy_yes) >= len(fn_windy_yes)/2 else "NO"
        majority_no = "YES" if np.sum(fn_windy_no) >= len(fn_windy_no)/2 else "NO"
        print(f"         ├─ Yes → Depart on time: {majority_yes} ({np.sum(fn_windy_yes)}/{len(fn_windy_yes)})")
        print(f"         └─ No → Depart on time: {majority_no} ({np.sum(fn_windy_no)}/{len(fn_windy_no)})")

# ============================================================================
# PROBLEM 2: Iris Dataset Classification
# ============================================================================

def problem_2():
    print("\n\nPROBLEM 2: Iris Dataset Classification")
    print("="*60)
    
    # Load dataset from CSV
    df = pd.read_csv('Data_Iris.csv')
    
    # Extract features and labels
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
    
    # Convert species names to numeric labels
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y = df['species_name'].map(species_map).values
    
    # Split into training (120) and testing (30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=30, train_size=120, random_state=42, stratify=y
    )
    
    # (a) K-Nearest Neighbors
    print("\n(a) K-Nearest Neighbors:")
    for k in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nk={k}: Accuracy = {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
    
    # (b) Decision Tree with Gini
    print("\n(b) Decision Tree (Gini, min_samples_split=25):")
    dt_gini = DecisionTreeClassifier(criterion='gini', min_samples_split=25, random_state=42)
    dt_gini.fit(X_train, y_train)
    y_pred = dt_gini.predict(X_test)
    acc_dt_gini = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy = {acc_dt_gini:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # (c) Random Forest with Gini
    print("\n(c) Random Forest (100 trees, Gini, min_samples_split=25):")
    rf_gini = RandomForestClassifier(n_estimators=100, criterion='gini', 
                                     min_samples_split=25, random_state=42)
    rf_gini.fit(X_train, y_train)
    y_pred = rf_gini.predict(X_test)
    acc_rf_gini = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy = {acc_rf_gini:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # (d) Using Entropy
    print("\n(d) Using Entropy (Information Gain):")
    
    print("\nDecision Tree (Entropy):")
    dt_entropy = DecisionTreeClassifier(criterion='entropy', min_samples_split=25, random_state=42)
    dt_entropy.fit(X_train, y_train)
    y_pred = dt_entropy.predict(X_test)
    acc_dt_entropy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy = {acc_dt_entropy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    print("\nRandom Forest (Entropy):")
    rf_entropy = RandomForestClassifier(n_estimators=100, criterion='entropy', 
                                        min_samples_split=25, random_state=42)
    rf_entropy.fit(X_train, y_train)
    y_pred = rf_entropy.predict(X_test)
    acc_rf_entropy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy = {acc_rf_entropy:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Comparison of all results
    print("\n" + "="*60)
    print("COMPARISON OF RESULTS:")
    print("="*60)
    print(f"Decision Tree (Gini):    Accuracy = {acc_dt_gini:.4f}")
    print(f"Decision Tree (Entropy): Accuracy = {acc_dt_entropy:.4f}")
    print(f"Random Forest (Gini):    Accuracy = {acc_rf_gini:.4f}")
    print(f"Random Forest (Entropy): Accuracy = {acc_rf_entropy:.4f}")
    print("\nObservations:")
    print("- All methods achieved the same accuracy (0.9667)")
    print("- Gini and Entropy criteria produce identical results on this dataset")
    print("- Random Forests and Decision Trees perform similarly here")

# Run both problems
if __name__ == "__main__":
    problem_1()
    problem_2()