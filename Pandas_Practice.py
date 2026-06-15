import pandas as pd


marks = pd.Series([85, 92, 78, 95, 60],
                  index=["Aarav","Priya","Rahul","Sneha","Ravi"])
#P1
print(f"Marks of Priya : {marks['Priya']}")
#P2
print(f"Student Who Scored Above 80 : \n {marks[marks > 80]}")
#P3
print(f"mean of the marks {marks.mean()}")
print(f"max of the marks {marks.max()}")
print(f"min of the marks {marks.min()}")


data = {
    "StudentID":  [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Name":       ["Aarav", "Priya", "Rahul", "Sneha", "Ravi",
                   "Meera", "Karan", "Anita", "Vijay", "Pooja"],
    "Age":        [20, 21, 19, 22, 20, 23, 21, 19, 24, 20],
    "City":       ["Ahmedabad", "Mumbai", "Delhi", "Ahmedabad", "Mumbai",
                   "Delhi", "Ahmedabad", "Mumbai", "Delhi", "Ahmedabad"],
    "Score_Math": [85, 92, 78, 95, 60, 88, 73, 90, 55, 82],
    "Score_Sci":  [80, 89, 72, 91, 58, 85, 70, 94, 50, 79],
    "Attendance": [90, 95, 85, 98, 70, 92, 80, 97, 65, 88],
    "Grade":      ["A", "A", "B", "A", "C", "A", "B", "A", "C", "A"],
    "Fees_Paid":  [True, True, False, True, False, True, True, False, False, True],
    "Score_Eng":  [75, None, 68, 88, 55, None, 65, 91, 48, 77]
}

df = pd.DataFrame(data)
print(df)

# **P4.** Print the first 3 rows of `df`.
print(df.head(3))

# **P5.** Print how many rows and columns `df` has.
print(df.shape)

# **P6.** Print the data type of each column in `df`.
print(df.dtypes)

# **P8.** Set `StudentID` as the index of `df` and print the first 5 rows.
df = df.set_index('StudentID')
print(df.head())

# **P10.** Print all students where Score_Math is greater than 80.
print(df.query('Score_Math > 80'))

# **P11.** Print students who are from City "Ahmedabad" AND have Grade "A".
print(df.query('City == "Ahmedabad" and Grade == "A"'))

# **P13.** Print how many missing values are in each column of `df`.
print(df.isnull().sum())

# **P14.** Fill the missing values in `Score_Eng` with the mean of that column.
df = df['Score_Eng'].fillna(df["Score_Eng"].mean())
print(df)

# **P16.** Select and print only the `Name`, `Score_Math`, and `Grade` columns.
print(df.loc[:,['Name','Score_Math','Grade']])

# **P17.** Create a new column `Total_Score` = Score_Math + Score_Sci.
df['Total_Score'] = df['Score_Math'] + df['Score_Sci']
print(df)

# **P18.** Print all unique cities in the dataset and how many times each city appears.
print(df['City'].value_counts())

# **P25.** Use `iloc` to print the first 3 rows and first 4 columns.
print(df.iloc[:3,:4])

# **P26.** Use `loc` to print Name and Score_Math for all students with Grade "A".
print(df.loc[:,['Name','Score_Math']])

# **P27.** Use `iloc` to print the last 2 rows.
print(df.iloc[-2:])

