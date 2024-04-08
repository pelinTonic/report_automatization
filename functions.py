import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def date_speed_plot(dataframe):
    pass

def excel_to_dateframe(path_to_excel: str, sheet: str, sort: str = None) -> pd.DataFrame: 
    """   This function reads an Excel file and converts a specified sheet into a pandas DataFrame. 
        It also sorts the DataFrame based on a specified column if provided.

    Args:
        path_to_excel (str): The path to the Excel file
        sheet (str): The name of the sheet to be converted into a DataFrame
        sort (str, optional): The column name to sort the DataFrame by. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame obtained from the specified sheet of the Excel file, sorted by the specified column if provided
    """
    dataframe = pd.read_excel(path_to_excel, sheet)
    if sort is not None:
        dataframe = dataframe.sort_values(by=sort)

    return dataframe

def unique_values(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """This function extracts the unique values from a specified column in a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame from which to extract unique values.
        column_name (str): The name of the column from which to extract unique values.

    Returns:
        pd.DataFrame: A DataFrame containing the unique values from the specified column.
    """
    unique_values = dataframe[column_name].unique()

    return unique_values

def filter_value(dataframe: pd.DataFrame, column_name: str) -> dict:
    """    This function filters a DataFrame based on unique values of a specified column. 
        It returns a dictionary where each key is a unique value from the specified column, 
        and the corresponding value is a DataFrame filtered by that unique value.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be filtered
        column_name (str): The name of the column based on whose unique values the DataFrame will be filtered.

    Returns:
        dict: A dictionary where each key is a unique value from the specified column, 
        and the corresponding value is a DataFrame filtered by that unique value.
    """

    return {group: data for group, data in dataframe.groupby(column_name)}

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:

    """
    Removes outliers from a pandas DataFrame.

    This function uses the Interquartile Range (IQR) method to identify and remove outliers from the DataFrame. 
    It first calculates the 1st quartile (25th percentile) and the 3rd quartile (75th percentile) of the data. 
    The IQR is the range between these two quartiles. Any data point that falls below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is considered an outlier.

    Args:
        df (pd.DataFrame): The input DataFrame from which outliers need to be removed.

    Returns:
        pd.DataFrame: The DataFrame after removing the outliers.
    """
    numeric_df = df.select_dtypes(include=['number'])
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = ((numeric_df >= lower_bound) & (numeric_df <= upper_bound)).all(axis=1)
    return df[mask]

def line_plots(path_to_data: str):
    #Napraviti da radi samo za jednog radnika ne za sves
    """ Generate line plots for each worker and material combination based on the provided data.

    Args:
        path_to_data (str): The path to the Excel file containing the data.

    """
    df = excel_to_dateframe(path_to_data, "Poslije stimulacija")
    workers = df["Ime"].unique()

    for worker in workers:
        worker_data = df[df["Ime"] == worker]
        materials = worker_data["Sirovina"].unique()

        for material in materials:
            material_data = worker_data[worker_data["Sirovina"] == material]
            material_data = remove_outliers(material_data)

            if len(material_data) > 1:
                fig, axs = plt.subplots(1, 1, figsize=(10, 5))
                axs.plot(material_data['Datum'], material_data["Brzina"])
                axs.set_xlabel('Datum')
                axs.set_ylabel('Brzina')
                axs.set_title(f"{worker}, {material}")
                for x, y in zip(material_data['Datum'], material_data["Brzina"]):
                    axs.text(x, y, f'{round(y,2)}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(f"{worker}_{material}_plot_without_outliers.png")
                plt.close()
            else:
                pass

def averages_per_process(df: pd.DataFrame) -> dict:
    """Calculate the average speed for each process based on the provided DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing the data.

    Returns:
        dict: A dictionary where keys are the process names ('Sirovina' values) and values are the corresponding average speeds.
        
    """
    return df.groupby("Sirovina")["Brzina"].mean().round(2).to_dict()

def averages_per_person(df):
        
    worker_names_list = unique_values(df, "Ime")
    average_dict = defaultdict(dict)

    for worker in worker_names_list:
        data = filter_value(df, "Ime")
        worker_filter = data[worker]

        grouped = worker_filter.groupby("Sirovina")
        for name, raw_material in grouped:
            average = raw_material["Brzina"].mean()
            average = round(average,2)
            average_dict[worker][name] = average
    
    return average_dict

def bar_chart(worker, avg_person):

    if worker in avg_person.keys():
        avg_person_values = avg_person[worker]

    fig, ax = plt.subplots()
    labels = [i for i in range(len(avg_person_values.keys()))]
    legend_labels = avg_person_values.keys()
    values = avg_person_values.values()
    color_map = plt.get_cmap('tab20') 
    bars = ax.bar(labels, values, color=color_map(np.arange(len(labels))))

    ax.set_ylabel("Prosječna brzina")
    ax.set_title(worker)
    ax.grid(axis="y")
    ax.bar_label(bars)
    plt.xticks(fontsize=4)
    ax.legend(bars, legend_labels, fontsize = "4")
    plt.show()

def delete_keys(avg_process_dict, avg_person_dict):

    avg_process_dict_copy = avg_process_dict.copy()
    for key in avg_process_dict_copy.keys():
        if key not in avg_person_dict:
            del avg_process_dict[key]

    return(avg_process_dict)    
   
def sort_dictionary(avg_process_dict, avg_person_dict):
    
    avg_process_dict = delete_keys(avg_process_dict, avg_person_dict)

    sorted_keys = sorted(avg_person_dict.keys())

    sort_dictionary = {key: avg_process_dict[key] for key in sorted_keys}
    sort_dictionary2 = {key: avg_person_dict[key] for key in sorted_keys}

    return sort_dictionary, sort_dictionary2

def grouped_bar_chart(worker,avg_person, avg_process):

    if worker in avg_person.keys():
        avg_person_values = avg_person[worker]
    dictionaries = sort_dictionary(avg_process, avg_person_values)
    avg_process = dictionaries[0]
    avg_person = dictionaries[1]
    plot_dictionary = {
        "Process average": tuple(avg_process.values()),
        "Worker average": tuple(avg_person.values()),
    }
    print(plot_dictionary)
    x = np.arange(len(avg_process.keys()))
    width = 0.35
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained')
    
    for key, value in plot_dictionary.items():
        offset = width*multiplier
        bars = ax.bar(x + offset, value, width, label = key)
        ax.bar_label(bars, padding=2)
        multiplier += 1

    ax.set_ylabel("Prosječna brzina / kg/h")
    ax.set_title(worker)
    ax.grid(axis="y")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(avg_process.keys(), ha ="right", rotation=45)
    plt.xticks(fontsize=8)
    ax.legend()
    plt.savefig(f"{worker}_vs_average.png")
    plt.show()

def standard_deviation(dataframe):

    raw_materials = unique_values(dataframe, "Sirovina")
    st_dev_dict = {}
    for material in raw_materials:

        data = filter_value(dataframe, "Sirovina")
        material_filter = data[material]
        st_dev = material_filter["Brzina"].std()

        st_dev_dict[material] = st_dev
        

    return st_dev_dict

def number_of_workers_by_std_dev(dataframe):

    mean_dict = defaultdict(dict)
    st_dev = standard_deviation(dataframe)
    raw_materials = unique_values(dataframe, "Sirovina")

    for material in raw_materials:

        data = filter_value(dataframe, "Sirovina")
        material_filter = data[material]
        workers = unique_values(material_filter,"Ime")
        workers_data = filter_value(material_filter, "Ime")

        for worker in workers:
            worker_filter = workers_data[worker]
            mean = worker_filter["Brzina"].mean()
            mean_dict[worker][material] = mean

    
    process_average = averages_per_process(dataframe)
    # za svakog radnika u mean_dict i za svaki proces kod svakog radnika:

    workers_in_first_st_dev = {}

    for worker, material in mean_dict.items(): 
            raw_materials = material.keys()
            for materials in raw_materials: #worker = 1; material = 1
                if process_average.get(materials):
                    worker_list = []
                    first_standard_dev_positive = process_average[materials] + st_dev[materials]
                    first_standard_dev_negative = process_average[materials] - st_dev[materials]
                    if material[materials] < first_standard_dev_positive and material[materials] >= first_standard_dev_negative:
                        worker_list.append(worker)
                        if workers_in_first_st_dev.get(materials):
                            workers_in_first_st_dev[materials] += [worker_list]
                        else:
                            workers_in_first_st_dev[materials] = [worker_list]

    return workers_in_first_st_dev
