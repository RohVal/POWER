from re import findall
"""this script is used to format the header of the csv files in the datasets folder"""


def format_header(filename: str) -> None:
    """
    This function reads the header of a csv file, formats it and writes it back to the file.

    Args:
        filename (str): the name of the csv file to format the header of

    Returns:
        None
    """

    with open(filename, mode="r") as file:
        header = file.readline()
        rows = file.readlines()

        # split the header into columns
        header = findall(r'"(.*?)"|([^,]+)', header)

        # if the first element of the tuple is not empty, use it as the column name, otherwise use
        # the second element
        header = [col[0] if col[0] else col[1] for col in header]

        # replace commas in the column names with a hyphen
        header = [col.replace(",", " -") for col in header]

        # replace ¬∞ with degrees
        header = [col.replace("¬∞", "°") for col in header]

        # add double quotes to the column names
        header = [f'"{col}"' for col in header]

        # join the columns back together, except the last one which contains a newline character
        header = ",".join(header[:-1])

    # write the formatted header back to the file along with the rows
    with open(filename, mode="w") as file:
        file.write(header)
        file.write("\n")
        file.writelines(rows)


if __name__ == "__main__":
    filenames = [
        "./dataset/Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv",
        "./dataset/Turbine_Data_Kelmarsh_2_2022-01-01_-_2023-01-01_229.csv",
        "./dataset/Turbine_Data_Kelmarsh_3_2022-01-01_-_2023-01-01_230.csv",
        "./dataset/Turbine_Data_Kelmarsh_4_2022-01-01_-_2023-01-01_231.csv",
        "./dataset/Turbine_Data_Kelmarsh_5_2022-01-01_-_2023-01-01_232.csv",
        "./dataset/Turbine_Data_Kelmarsh_6_2022-01-01_-_2023-01-01_233.csv"
    ]
    for filename in filenames:
        format_header(filename)
