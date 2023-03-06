# import gc
import os
from collections import defaultdict
from functools import wraps
from time import perf_counter

# import holoviews as hv
# import hvplot.pandas
import numpy as np
# import pandas as pd
# import psutil

# from bokeh.plotting import output_file, save
# from holoviews import opts
from humanize import naturalsize
# from pydictobject import DictObject
# from statsmodels.tsa.stattools import acf, pacf

# SUMMARY_KEY_MAP = DictObject(
#     {
#         "variable_names": "Variable Name",
#         "num_unique": "No of Unique",
#         "samples": "Samples",
#         "num_missing": "No of Missing",
#         "perc_missing": "Per of Missing",
#         "is_numeric": "Is Numeric",
#         "comment": "Comment",
#         "dtype": "Datatype",
#         "max_value": "Max",
#         "min_value": "Min",
#         "duplicate_col": "Duplicates",
#         "num_values": "No of Values",
#         "unique": "Unique",
#         "mode_value": "Mode",
#         "mode_freq": "Mode Freq",
#         "mean_value": "Mean",
#         "std_deviation": "Standard Deviation",
#         "percentile_25": "25th percentile",
#         "percentile_50": "Median",
#         "percentile_75": "75th percentile",
#         "variable_1": "Variable 1",
#         "variable_2": "Variable 2",
#         "corr_coef": "Corr Coef",
#         "abs_corr_coef": "Abs Corr Coef",
#     }
# )


# def split_sets(df, cols, by):
#     if isinstance(by, int):
#         row_dict = df.iloc[by][cols].to_dict()
#         for key in row_dict.keys():
#             row_dict[key] = str(row_dict[key])
#     elif by == "mean":
#         # means = []
#         row_dict = {}
#         num_cols = [col for col in cols if col in df.numeric_columns]
#         for col in num_cols:
#             row_dict[col] = df[col].mean()
#             # row_dict[col] = compute_if_dask(df[col].mean())
#         cat_cols = [col for col in cols if col not in num_cols]
#         row_dict.update(dict(zip(cat_cols, ["cat_cols"] * len(cat_cols))))
#     unique_values = set(list(row_dict.values()))
#     sets = [
#         [key for key in row_dict.keys() if row_dict[key] == val]
#         for val in unique_values
#     ]
#     return sets


# def is_missing(iterable, na_values):
#     if "Series" in str(type(iterable)) and "datetime" in str(
#         iterable.dtype
#     ):  # If datatime col, string values like 'NA' etc will raise format error.
#         na_values = [val for val in na_values if not isinstance(val, str)]
#         print(
#             f"is_missing 1: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#         )
#     return iterable.isin(na_values) | (
#         iterable.isna() if np.nan in na_values else False
#     )


# def flatten_list(args):
#     if not isinstance(args, list):
#         args = [args]
#     new_list = []
#     for x in args:
#         if isinstance(x, list):
#             new_list += flatten_list(list(x))
#         else:
#             new_list.append(x)
#     return new_list


# def duplicate_columns(df):
#     """List duplicate columns in dataframe.

#     Returns the duplicate columns that are present for each
#     variable in the dataset. If there are no such duplicate
#     columns, "No Duplicate Variables" message is displayed.

#     Parameters
#     ----------
#     dataset: pandas.DataFrame

#     Returns
#     -------
#     dups: pandas.DataFrame
#     """
#     # df = self.data
#     dups = defaultdict(list)
#     sets = split_sets(
#         df, df.columns, 0
#     )  # splitting columns based on first element matching
#     for i in range(1, 4):
#         big_sets = [set for set in sets if len(set) > 100]
#         if big_sets:
#             for set in big_sets:
#                 new_sets = split_sets(df, set, i)
#                 sets.remove(set)
#                 sets += new_sets
#         else:
#             break
#     multi_sets = [set for set in sets if len(set) > 1]
#     for set in multi_sets:
#         cols = set
#         # _LOGGER.info("processing set of - {}".format(cols))
#         while cols:
#             current_col = cols[0]
#             current_col_data = df[current_col]
#             remaining_cols = cols[1:]
#             for col2 in remaining_cols:
#                 other_col_data = df[col2]
#                 if (
#                     str(df.dtypes[current_col]) == "category"
#                     and str(df.dtypes[col2]) == "category"
#                     and (
#                         len(current_col_data.cat.categories)
#                         != len(other_col_data.cat.categories)
#                     )
#                 ):
#                     continue
#                 if (  # compute_if_dask
#                     (
#                         df[current_col].iloc[:1000].astype(str)
#                         == df[col2].iloc[:1000].astype(str)
#                     ).all()
#                 ) and (  # compute_if_dask
#                     (df[current_col].astype(str) == df[col2].astype(str)).all()
#                 ):
#                     dups[SUMMARY_KEY_MAP.variable_names].append(current_col)
#                     dups[SUMMARY_KEY_MAP.duplicate_col].append(col2)
#                     cols.remove(col2)
#             cols.remove(current_col)
#     dups = pd.DataFrame(dups)
#     if dups.empty:
#         dups = "No duplicate variables"
#     else:
#         dups = dups.loc[
#             dups[SUMMARY_KEY_MAP.variable_names].isin(
#                 dups[SUMMARY_KEY_MAP.duplicate_col]
#             )
#             == False  # noqa
#         ]
#         dups = (
#             dups.groupby(SUMMARY_KEY_MAP.variable_names)
#             .agg({SUMMARY_KEY_MAP.duplicate_col: lambda s: ", ".join(s)})
#             .reset_index()
#         )
#     # self.duplicate_columns_result = dups
#     return dups


# def compute_data_health(df):
#     """Returns data health."""
#     # df = self.data
#     print(
#         f"compute_data_health 1i: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     dtypes = pd.DataFrame(df.dtypes.rename("dtype"))
#     print(
#         f"compute_data_health 1ii: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     dtypes[SUMMARY_KEY_MAP.dtype] = "*Unknown*"
#     print(
#         f"compute_data_health 1iii: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     dtypes.loc[
#         dtypes.dtype.astype(str).str.contains("float|int"),
#         SUMMARY_KEY_MAP.dtype,
#     ] = "Numeric"
#     print(
#         f"compute_data_health 1iv: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     dtypes.loc[
#         dtypes.dtype.astype(str).str.contains("date"), SUMMARY_KEY_MAP.dtype
#     ] = "Date"
#     print(
#         f"compute_data_health 1v: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     dtypes.loc[
#         dtypes[SUMMARY_KEY_MAP.dtype].isin(["Numeric", "Date"])
#         == False,  # noqa
#         SUMMARY_KEY_MAP.dtype,
#     ] = "Others"
#     print(
#         f"compute_data_health 1vi: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )

#     no_of_columns = len(df.columns)
#     print(
#         f"compute_data_health 1vii: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     pie1 = dtypes.groupby(SUMMARY_KEY_MAP.dtype).size()
#     print(
#         f"compute_data_health 1viii: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     pie1 = (pie1 / no_of_columns).to_dict()
#     print(
#         f"compute_data_health 1ix: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     if not ("Numeric" in pie1.keys()):
#         pie1.update({"Numeric": 0})
#     print(
#         f"compute_data_health 1x: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     if not ("Others" in pie1.keys()):
#         pie1.update({"Others": 0})
#     print(
#         f"compute_data_health 1xi: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     datatype_dict = {}
#     print(
#         f"compute_data_health 1xii: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     for i in sorted(pie1):
#         datatype_dict.update({i: pie1[i]})
#     pie1 = datatype_dict
#     print(
#         f"compute_data_health 1xiii: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )

#     NA_VALUES = [np.NaN, pd.NaT, None, "NA"]
#     print(
#         f"compute_data_health 1xiv: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     missing_value = is_missing(df, NA_VALUES).sum().sum()
#     print(
#         f"compute_data_health 2: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     # pie2 = missing_value / float(compute_if_dask(df.size))
#     pie2 = missing_value / float(df.size)
#     pie2 = {"Available": (1 - pie2), "Missing": pie2}
#     # pie2 = pd.DataFrame(pie2, index=['Missing Values'])
#     # pie3
#     duplicate_value = df.duplicated().sum() / len(df)
#     print(
#         f"compute_data_health 3: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     # pie3 = (duplicate_value / float(compute_if_dask(df.shape[0])))
#     # pie3 = {"Unique": (1 - pie3), "Duplicate": pie3}

#     # pie3 should be multiplied with no_of_rows because it checks the duplicate obs. all other
#     # plots in the function work on column basis.
#     pie3 = duplicate_value
#     pie3 = {"Unique": (1 - pie3), "Duplicate": pie3}

#     pie4 = duplicate_columns(df)
#     print(
#         f"compute_data_health 4: {psutil.Process(os.getpid()).memory_info().rss / 1024**2}"
#     )
#     if type(pie4) == str:
#         pie4 = {"Unique": 1, "Duplicate": 0}
#         duplicate_var = 0
#     else:
#         duplicate_var = 1 + (", ".join(pie4.Duplicates).count(","))
#         # pie4 = duplicate_var / float(compute_if_dask(df.shape[1]))
#         pie4 = duplicate_var / float(df.shape[1])
#         pie4 = {"Unique": (1 - pie4), "Duplicate": pie4}

#     for k in pie1.keys():
#         pie1[k] = np.round(pie1[k], 1)
#     for k in pie2.keys():
#         pie2[k] = np.round(pie2[k], 1)
#     for k in pie3.keys():
#         pie3[k] = np.round(pie3[k], 1)
#     for k in pie4.keys():
#         pie4[k] = np.round(pie4[k], 1)

#     data_dict = {
#         "Datatypes": pie1,
#         "Missing Values": pie2,
#         "Duplicate Rows": pie3,
#         "Duplicate Columns": pie4,
#     }

#     df_dict = {
#         "type": flatten_list(
#             [[x] * len(data_dict[x]) for x in data_dict.keys()]
#         ),
#         "labels": list(pie1.keys())
#         + list(pie2.keys())
#         + list(pie3.keys())
#         + list(pie4.keys()),
#         "values": [i * 100 for i in list(pie1.values())]
#         + [i * 100 for i in list(pie2.values())]
#         + [i * 100 for i in list(pie3.values())]
#         + [i * 100 for i in list(pie4.values())],
#     }
#     df = pd.DataFrame(df_dict)
#     df = df.set_index(["type", "labels"])
#     return df


# def plot_data_health(df, data):
#     # this multipliers resolves the issue of duplicate columns as it's values are multiplied by 1 and others
#     # with no_of_columns. which was needed for the correct metrics.
#     final_plot = None
#     for metric in df.index.get_level_values(0).unique():
#         plot = (
#             ((df.loc[metric].T).hvplot)
#             .bar(stacked=True, title=metric, height=80, invert=True, width=400)
#             .opts(xticks=list([i for i in range(df.shape[1])]))
#         )
#         if final_plot:
#             final_plot += plot
#         else:
#             final_plot = plot
#     return final_plot.cols(1).opts(title="Data Shape:" + str(data.shape))


# def get_outlier_nums(df, lb, ub):
#     """Return a dictionary containing outlier numbers for each columns in df.

#     Parameters
#     ----------
#     df: pd.DataFrame
#         Data for which outlier number information require.

#     Returns
#     -------
#     outlier_nums: pd.DataFrame
#     """
#     outlier_nums = {}
#     for col in df.columns:
#         outlier_nums[col] = [0, 0]
#     for (col, lb) in lb.items():
#         outlier_nums[col][0] = (df[col] < lb).sum()
#     for (col, ub) in ub.items():
#         outlier_nums[col][1] = (df[col] > ub).sum()
#     del df  # to save memory
#     return outlier_nums


# def _compute_outlier_bounds(  # noqa
#     df, cols=None, method="percentile", lb=None, ub=None,
# ):
#     """Compute outlier bounds for each column.

#     Parameters
#     ----------
#     df: pd.DataFrame or np.Array
#         Dataframe/2D Array consisting of independent features.
#     cols: list, optional
#         List of column names of features, by default None
#     method: str, default is percentile
#         Accepted values are mean, median, threshold, percentile
#     lb: numeric, default is none
#         lb is the lower bound
#         * If not None, pass a dictionary of columns with lower limits for each
#     ub: numeric, default is none
#         ub is the upper bound
#         * If not None, pass a dictionary of columns with upper limits for each

#     Returns
#     -------
#     tuple
#         a tuple of dictionaries for lb and ub values of all columns.
#     """
#     if cols is None:
#         cols = df.select_dtypes("number").columns.to_list()

#     num_df = df[
#         cols
#     ]  # can anything be done to save memory; it's 0.5 times the full data???
#     if method == "mean":
#         mean = num_df.mean()
#         std = num_df.std()
#         lb = (mean - lb * std).to_dict()
#         ub = (mean + ub * std).to_dict()
#     elif method == "median":
#         fst_quant = num_df.quantile(0.25)
#         thrd_quant = num_df.quantile(0.75)
#         iqr = thrd_quant - fst_quant
#         lb = (fst_quant - lb * iqr).to_dict()
#         ub = (thrd_quant + ub * iqr).to_dict()
#     elif method == "percentile":
#         lb = num_df.quantile(lb).to_dict()
#         ub = num_df.quantile(ub).to_dict()
#     elif method == "threshold":
#         pass
#     else:
#         raise ValueError("Unsupported outlier method : " + method)
#     del num_df  # to save memory
#     return (lb, ub)


# def get_outliers_df_for_data(data, cols=None):
#     """Returns the data frame with outlier analysis table for any provided data."""
#     if pd.__version__ >= "1.0.0":
#         pd.set_option("mode.use_inf_as_na", True)
#     outlier_col_labels = [
#         "< (mean-3*std)",
#         "> (mean+3*std)",
#         "< (1stQ - 1.5 * IQR)",
#         "> (3rdQ + 1.5 * IQR)",
#     ]

#     (lb, ub) = _compute_outlier_bounds(
#         data, cols=cols, method="mean", lb=3, ub=3
#     )
#     # gc.collect()
#     mean_outliers = get_outlier_nums(data[cols], lb, ub)
#     # gc.collect()

#     (lb, ub) = _compute_outlier_bounds(
#         data, cols=cols, method="median", lb=1.5, ub=1.5
#     )
#     # gc.collect()
#     median_outliers = get_outlier_nums(data, lb, ub)
#     # gc.collect()

#     # mean_outliers = Outlier(method="mean").fit(data).get_outlier_nums(data)
#     # median_outliers = Outlier(method="median").fit(data).get_outlier_nums(data)

#     outliers_df = pd.DataFrame.from_dict(mean_outliers)
#     outliers_df = pd.concat(
#         [outliers_df, pd.DataFrame.from_dict(median_outliers)]
#     )
#     outliers_df = outliers_df.reset_index(drop=True).T
#     outliers_df.rename(
#         columns=dict(zip(list(outliers_df.columns), outlier_col_labels)),
#         inplace=True,
#     )
#     outliers_df["-inf"] = 0
#     outliers_df["+inf"] = 0
#     if pd.__version__ >= "1.0.0":
#         pd.set_option("mode.use_inf_as_na", False)
#     outliers_df["-inf"].loc[outliers_df.index] = (data == -np.inf).sum()
#     outliers_df["+inf"].loc[outliers_df.index] = (data == +np.inf).sum()
#     outliers_sum = outliers_df.sum(axis=1)
#     outliers_df = outliers_df[outliers_sum > 0]
#     if outliers_df.empty:
#         return "No Outlier Values"
#     level1_col = "Data Shape:" + str(data.shape)
#     level2_cols = outliers_df.columns.to_list()
#     col_arrays = [[level1_col] * len(level2_cols), level2_cols]
#     index = pd.MultiIndex.from_tuples(
#         list(zip(*col_arrays)), names=["", "feature"]
#     )
#     outliers_df.columns = index
#     outliers_df.columns = outliers_df.columns.droplevel()
#     return outliers_df


# def get_conf_lines(data):

#     """Marks the position along the y-axis at the upper confidence interval and lower confidence interval."""
#     conf_interval_limit = 1.96 / np.sqrt(len(data))
#     conf_interval_upper = conf_interval_limit
#     conf_interval_lower = -conf_interval_limit
#     import holoviews as hv

#     hlines = hv.HLine(conf_interval_upper).opts(
#         line_dash="dashed", color="red"
#     ) * hv.HLine(conf_interval_lower).opts(line_dash="dashed", color="red")
#     return hlines


# def get_acf_plot(data, y_var, lags=None):

#     if lags is None:
#         # nlags = 50
#         nlags = min(50, data.shape[0] - 1)
#     else:
#         nlags = max(lags)

#     def get_acf_df(df, y):
#         return pd.DataFrame.from_dict(
#             {
#                 "lags": range(0, nlags + 1),
#                 "correlation": acf(df[y], nlags=nlags),
#             }
#         )

#     if data.shape[0] > 0:
#         lag_acf = get_acf_df(data, y_var).set_index("lags")
#         plot = (lag_acf.hvplot).line(
#             xlabel="Number of lags", ylabel="Correlation"
#         )
#         hlines = get_conf_lines(data)
#         fin_plot = (plot * hlines).opts(opts.Overlay(title="ACF Plot"))
#     else:
#         temp_df = pd.DataFrame({"correlation": list(np.repeat(np.nan, 51))})
#         temp_df.index.name = "lags"
#         fin_plot = (temp_df.hvplot).line(
#             xlabel="Number of lags", ylabel="Correlation"
#         )
#     fin_plot = hv.render(fin_plot)
#     return fin_plot


# def get_pacf_plot(data, y_var, lags=None):

#     if lags is None:
#         # nlags = 50
#         nobs = data.shape[0]
#         nlags = min(50, nobs // 2 - 1)
#     else:
#         nlags = max(lags)

#     def get_pacf_df(df, y):
#         return pd.DataFrame.from_dict(
#             {
#                 "lags": range(0, nlags + 1),
#                 "correlation": pacf(df[y], nlags=nlags, method="ols"),
#             }
#         )

#     if data.shape[0] > 0:
#         lag_pacf = get_pacf_df(data, y_var).set_index("lags")
#         plot = (lag_pacf.hvplot).line(
#             xlabel="Number of lags", ylabel="Correlation"
#         )
#         hlines = get_conf_lines(data)
#         fin_plot = (plot * hlines).opts(opts.Overlay(title="PACF Plot"))
#     else:
#         temp_df = pd.DataFrame({"correlation": list(np.repeat(np.nan, 51))})
#         temp_df.index.name = "lags"
#         fin_plot = (temp_df.hvplot).line(
#             xlabel="Number of lags", ylabel="Correlation"
#         )
#     fin_plot = hv.render(fin_plot)
#     return fin_plot


# # df = pd.read_csv("data/raw/mars-oos-sample.csv")
# # df1 = get_outliers_df_for_data(df)
# # print(type(df1))
# # print(df1.columns)
# # df1.columns = df1.columns.droplevel()
# # print(df1.head())

# # output_file(filename="custom_filename.html", title="Static HTML file")
# # data_health_dict = _compute_data_health(df)
# # data_health_plot = _plot_data_health(data_health_dict, df)
# # data_health_plot = hv.render(data_health_plot)
# # save(data_health_plot)


def filter_cols(df):
    """filter the columns having 'number' dtype"""
    return df.select_dtypes(include=["number"]).columns.tolist()


def reduce_mem_usage(df, verbose=True):
    """
    An utility to reduce the memory of pandas dataframes by converting
    the columns of numeric datatypes to lower sizes without losing any
    information, based on the range of values in the column
    """
    num_cols = filter_cols(df)
    ini_mem = naturalsize(df.memory_usage(deep=True).sum(), format="%.2f")
    for col in num_cols:
        col_type = df[col].dtype
        # if col_type in numerics:
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == "int":
            if (
                c_min > np.iinfo(np.int16).min
                and c_max < np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)
            elif (
                c_min > np.iinfo(np.int32).min
                and c_max < np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype(np.int32)
            elif (
                c_min > np.iinfo(np.int64).min
                and c_max < np.iinfo(np.int64).max
            ):
                df[col] = df[col].astype(np.int64)
        else:
            if (
                c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    exit_mem = naturalsize(df.memory_usage(deep=True).sum(), format="%.2f")
    if verbose:
        print(f"Initial Memory after data read: {ini_mem}")
        print(f"Final Memory after dynamic type casting: {exit_mem}")
    return df


def reduce_mem_usage_with_summing(df, verbose=True):
    """
    An utility to reduce the memory of pandas dataframes by converting
    the columns of numeric datatypes to lower sizes without losing any
    information, based on the range of values in the column
    """
    num_cols = filter_cols(df)
    ini_mem = naturalsize(df.memory_usage(deep=True).sum(), format="%.2f")
    for col in num_cols:
        col_type = df[col].dtype
        # if col_type in numerics:
        c_min = df[col].min()
        c_max = df[col].max()
        c_sum = df[col].sum()
        if str(col_type)[:3] == "int":
            if (
                c_min > np.iinfo(np.int8).min and
                c_max < np.iinfo(np.int8).max and
                c_sum < np.iinfo(np.int8).max
            ):
                df[col] = df[col].astype(np.int8)
            elif (
                c_min > np.iinfo(np.int16).min
                and c_max < np.iinfo(np.int16).max
                and c_sum < np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype(np.int16)
            elif (
                c_min > np.iinfo(np.int32).min
                and c_max < np.iinfo(np.int32).max
                and c_sum < np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype(np.int32)
            elif (
                c_min > np.iinfo(np.int64).min
                and c_max < np.iinfo(np.int64).max
                and c_sum < np.iinfo(np.int64).max
            ):
                df[col] = df[col].astype(np.int64)
        else:
            if (
                c_min > np.finfo(np.float16).min
                and c_max < np.finfo(np.float16).max
                and c_sum < np.finfo(np.float16).max
            ):
                df[col] = df[col].astype(np.float16)
            elif (
                c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
                and c_sum < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    exit_mem = naturalsize(df.memory_usage(deep=True).sum(), format="%.2f")
    if verbose:
        print(f"Initial Memory after data read: {ini_mem}")
        print(f"Final Memory after dynamic type casting: {exit_mem}")
    return df


def timer(func):
    """Print runtime of decorated function"""

    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = perf_counter()
        val = func(*args, **kwargs)
        end = perf_counter()
        print(f"Finished {func.__name__!r} in {end - start:.3f} s")
        return val

    return wrapper_timer
