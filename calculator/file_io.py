# -*- coding: utf-8 -*-
# calculator/file_io.py


"""
Created on Thu Oct 12 13:37:33 2023

Todo:
    - might want some more file checks
    - can we rationalise the line / date formatters? Do we need the formatter
    function at all? Roll the line and date formatters together so don't need
    to call both

Contains basic file input-output functions and structural configurations for
handling both TOA5 and EddyPro files. In general, and where relevant, 
intermediary data products are pandas DataFrames. The module has functions to 
retrieve both data and headers. It is strictly tasked with parsing files 
between disk and memory, checking only structural integrity of the files. 
It does NOT evaluate data integrity!
"""

import csv
import datetime as dt
import os
from typing import Callable

import numpy as np
import pandas as pd
import pathlib

###############################################################################
### CONSTANTS ###
###############################################################################



FILE_CONFIGS = {
    'TOA5': {
        'info_line': 0,
        'header_lines': {'variable': 1, 'units': 2, 'sampling': 3},
        'separator': ',',
        'non_numeric_cols': ['TIMESTAMP'],
        'time_variables': {'TIMESTAMP': 0},
        'na_values': 'NAN',
        'unique_file_id': 'TOA5',
        'quoting': csv.QUOTE_NONNUMERIC,
        'dummy_info': [
            'TOA5', 'NoStation', 'CR1000', '9999', 'cr1000.std.99.99',
            'CPU:noprogram.cr1', '9999', 'default_table'
            ]
        },
    'EddyPro': {
        'info_line': None,
        'header_lines': {'variable': 0, 'units': 1},
        'separator': '\t',
        'non_numeric_cols': ['DATAH', 'filename', 'date', 'time'],
        'time_variables': {'date': 2, 'time': 3},
        'na_values': 'NaN',
        'unique_file_id': 'DATAH',
        'quoting': csv.QUOTE_MINIMAL,
        'dummy_info': [
            'TOA5', 'SmartFlux', 'SmartFlux', '9999', 'OS', 'CPU:noprogram',
            '9999', 'default_table'
            ]
        }
    }

INFO_FIELDS = [
    'format', 'station_name', 'logger_type', 'serial_num', 'OS_version',
    'program_name', 'program_sig', 'table_name'
    ]

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

EDDYPRO_SEARCH_STR = 'EP-Summary'



###############################################################################
### FUNCTIONS ###
###############################################################################



###############################################################################
### BEGIN FILE READ / WRITE FUNCTIONS ###
###############################################################################

#------------------------------------------------------------------------------
def get_data(
        file: str | pathlib.Path, file_type: str=None, usecols: list=None
        ) -> pd.core.frame.DataFrame:
    """Read data from file.
    
    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched.
        usecols (list, optional): The subset of columns to keep. If None, keep 
            all.
        Defaults to None.

    Returns:
        File data content.

    """

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)

    # Get dictionary containing file configurations
    MASTER_DICT = FILE_CONFIGS[file_type]

    # Set rows to skip
    REQ_TIME_VARS = list(MASTER_DICT['time_variables'].keys())
    CRITICAL_FILE_VARS = MASTER_DICT['non_numeric_cols']
    rows_to_skip = list(set([0] + list(MASTER_DICT['header_lines'].values())))
    rows_to_skip.remove(MASTER_DICT['header_lines']['variable'])

    # Usecols MUST include critical non-numeric variables (including date vars)
    # and at least ONE additional column; if this condition is not satisifed,
    # do not subset the columns on import.
    thecols = None
    if usecols and not usecols == CRITICAL_FILE_VARS:
        thecols = (
            CRITICAL_FILE_VARS +
            [col for col in usecols if not col in CRITICAL_FILE_VARS]
            )

    # Now import data
    return (
        pd.read_csv(
            file,
            skiprows=rows_to_skip,
            usecols=thecols,
            parse_dates={'DATETIME': REQ_TIME_VARS},
            keep_date_col=True,
            na_values=MASTER_DICT['na_values'],
            sep=MASTER_DICT['separator'],
            engine='c',
            on_bad_lines='warn',
            low_memory=False
            )
        .set_index(keys='DATETIME')
        .astype({x: object for x in REQ_TIME_VARS})
        .pipe(_integrity_checks, non_numeric=CRITICAL_FILE_VARS)
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _integrity_checks(df: pd.core.frame.DataFrame, non_numeric: list):
    """Check the integrity of data and indices.

    Args:
      df: dataframe containing the data.
      non_numeric: column names to ignore when coercing to numeric type.
   
    """

    # Coerce non-numeric data in numeric columns
    non_nums = df.select_dtypes(include='object')
    for col in non_nums.columns:
        if col in non_numeric: continue
        df[col] = pd.to_numeric(non_nums[col], errors='coerce')

    # Check the index type, and if bad time data exists, dump the record
    if not df.index.dtype == '<M8[ns]':
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~pd.isnull(df.index)]

    # Sort the index
    return df.sort_index()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_file_headers(
        file: str | pathlib.Path, begin: int, end: int, sep: str=','
        ) -> list:
    """Get a list of the header strings.
    
    Args:
        file: absolute path of file to parse.
        begin: line number of first header line.
        end: line number of last header line.
        sep: text separation character.

    Returns:
        List of sublists, each sublist containing the text elements of a header 
            line.
    
    """

    line_list = []
    with open(file, 'r') as f:
        for i in range(end + 1):
            line = f.readline()
            if not i < begin:
                line_list.append(line)
    return [line for line in csv.reader(line_list, delimiter=sep)]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_header_df(
        file: str | pathlib.Path, file_type: str=None
        ) -> pd.core.frame.DataFrame:
    """Get a dataframe with variables as index and units and statistical
    sampling type (if file type = TOA5) as columns.

    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched.

    Returns:
        The file header content.
    
    """

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)

    configs_dict = FILE_CONFIGS[file_type]
    return (
        pd.DataFrame(
            dict(zip(
                configs_dict['header_lines'].keys(),
                get_file_headers(
                     file=file,
                     begin=min(configs_dict['header_lines'].values()),
                     end=max(configs_dict['header_lines'].values()),
                     sep=configs_dict['separator']
                     )
                ))
            )
        .set_index(keys='variable')
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_file_info(
        file: str | pathlib.Path, file_type: str=None, 
        dummy_override: bool=False
        ) -> list(str):
    """Get the information from the first line of the TOA5 file. If EddyPro file
    OR dummy_override, just grab the defaults from the configuration dictionary.

    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched.
        dummy_override: Whether to just retrieve the default info. The default 
            is False.

    Returns:
        Dictionary of elements.
    
    """

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)

    # If dummy override (or EddyPro, which has no file info) requested,
    # return dummy info
    if dummy_override or file_type == 'EddyPro':
        return dict(zip(INFO_FIELDS, FILE_CONFIGS[file_type]['dummy_info']))

    # Otherwise get the TOA5 file info from the file
    return dict(zip(
        INFO_FIELDS,
        get_file_headers(
            file=file,
            begin=FILE_CONFIGS['TOA5']['info_line'],
            end=FILE_CONFIGS['TOA5']['info_line'],
            sep=FILE_CONFIGS['TOA5']['separator']
            )[0]
        ))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_file_type(file: str | pathlib.Path) -> str:
    """Get the file type.

    Args:
        file: absolute path of file to parse.

    Returns:
        name of file type (`TOA5` or `EddyPro`).

    Raises:
      TypeError: Raised if file type not recognised.

    """

    for file_type in FILE_CONFIGS.keys():
        id_field = (
            get_file_headers(
                file=file,
                begin=0,
                end=0,
                sep=FILE_CONFIGS[file_type]['separator']
                )
            [0][0]
            ).strip('\"')
        if id_field == FILE_CONFIGS[file_type]['unique_file_id']:
            return file_type
    raise TypeError('Unknown file type!')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_file_type_configs(
        file_type: str=None, return_field: dict | str=None) -> dict:
    """Get the configuration dictionary for the file type.

    Args:
      file_type: if specified, must be either `TOA5` or 
          `EddyPro`. If None, file_type is fetched. Default is None.
      return_field:  the specific field to return (as str). If None, return all
          (as dict) with fields as keys. Default is None.

    Raises:
        RuntimeError: raised if file_type is NOT specified and return_field IS
            specified.

    Returns:
        file type configuration field or fields.
    
    """

    if not file_type:
        return FILE_CONFIGS
        if not return_field is None:
            raise RuntimeError(
                'Cannot return individual field if file type not set!'
                )
    if return_field is None:
        return FILE_CONFIGS[file_type]
    return FILE_CONFIGS[file_type][return_field]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def write_data_to_file(
        headers: pd.DataFrame, data: pd.DataFrame, 
        abs_file_path: str | pathlib.Path, output_format:str=None, 
        info: dict=None
        ):
    """Write headers and data to file. Checks only for consistency between 
    headers and data column names (no deeper analysis of consistency with 
                                   output format).

    Args:
      headers: the dataframe containing the headers as columns.
      data: the dataframe containing the data.
      abs_file_path: absolute path (including file name) to write to.
      output_format: if specified, must be either `TOA5` or `EddyPro`. The 
          default is None.
      info: the file info to write as first header line. Only required if 
          outputting TOA5, and retrieves file type-specific dummy input if not 
          specified. The default is None.

    Returns:
        None.
    
    """

    # Cross-check header / data column consistency
    _check_data_header_consistency(
        headers=headers,
        data=data
        )

    # Add the requisite info to the output if TOA5
    row_list = []
    if not output_format:
        output_format = 'TOA5'
    if output_format == 'TOA5':
        if info is None:
            info = dict(zip(
                INFO_FIELDS,
                FILE_CONFIGS[output_format]['dummy_info']
                ))
        row_list.append(list(info.values()))

    # Construct the row list
    output_headers = headers.reset_index()
    [row_list.append(output_headers[col].tolist()) for col in output_headers]

    # Write the data to file
    file_configs = get_file_type_configs(file_type=output_format)
    with open(abs_file_path, 'w', newline='\n') as f:

        # Write the header
        writer = csv.writer(
            f,
            delimiter=file_configs['separator'],
            quoting=file_configs['quoting']
            )
        for row in row_list:
            writer.writerow(row)

        # Write the data
        data.to_csv(
            f, header=False, index=False, na_rep=file_configs['na_values'],
            sep=file_configs['separator'], quoting=file_configs['quoting']
            )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _check_data_header_consistency(headers: pd.DataFrame, data: pd.DataFrame):
    """Checks that the passed headers and data are consistent.

    Args:
        headers: the dataframe containing the headers as columns.
        data: the dataframe containing the data.

    Raises:
        RuntimeError: raised if inconsistent.

    Returns:
        None
    
    """

    headers_list = headers.index.tolist()
    data_list = data.columns.tolist()
    if not headers_list == data_list:
        headers_not_in_data = list(set(headers_list) - set(data_list))
        data_not_in_headers = list(set(data_list) - set(headers_list))
        raise RuntimeError(
            'Header and data variables dont match: '
            f'Header variables not in data: {headers_not_in_data}, '
            f'Data variables not in header: {data_not_in_headers}, '
            )
#------------------------------------------------------------------------------



###############################################################################
### END FILE READ / WRITE FUNCTIONS ###
###############################################################################



###############################################################################
### BEGIN FILE FORMATTING FUNCTIONS ###
###############################################################################



#------------------------------------------------------------------------------
def get_formatter(file_type: str, which: str) -> Callable:
    """

    Args:
        file_type: must be either `TOA5` or `EddyPro`.
        which: must be one of `read_line`, `write_line`, `read_date` and 
            `write_date`.

    Returns:
        function determined by `which` arg.
    
    """

    formatter_dict = {
        'TOA5': {
            'read_line': _TOA5_line_read_formatter,
            'write_line': _TOA5_line_write_formatter,
            'read_date': _TOA5_date_read_formatter,
            'write_date': _TOA5_date_write_formatter
            },
        'EddyPro': {
            'read_line': _EddyPro_line_read_formatter,
            'write_line': _EddyPro_line_write_formatter,
            'read_date': _EddyPro_date_read_formatter,
            'write_date': _EddyPro_date_write_formatter
            }
        }
    return formatter_dict[file_type][which]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _EddyPro_line_read_formatter(line: str) -> list(str):
    """Parses a line string of EddyPro format into a list of elements. 
    
    Args:
        line: single data line from EddyPro file.

    Returns:
        list of elements (as str).

    """

    sep = FILE_CONFIGS['EddyPro']['separator']
    return line.strip().split(sep)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _EddyPro_line_write_formatter(line_list: list(str)):
    """Parses a list of elements into a line string of EddyPro format. 
    
    Args:
        line_list: list of elements (as str).

    Returns:
        single data line for EddyPro file.

    """
    
    joiner = lambda x: FILE_CONFIGS['EddyPro']['separator'].join(x) + '\n'
    return joiner(line_list)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _EddyPro_date_read_formatter(line_list: list(str)) -> dt.datetime:
    """Parses a list of date strings of EddyPro format into a pydatetime. 
    
    Args:
        line_list: separate elements of an EddyPro data line string.

    Returns:
        the constructed python datetime.

    """

    locs = FILE_CONFIGS['EddyPro']['time_variables'].values()
    return _generic_date_constructor(
        date_elems=[line_list[loc] for loc in locs]
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _EddyPro_date_write_formatter(py_date: dt.datetime, which: str) -> str:
    """Parses a pydatetime into a string of EddyPro datetime format. 
    
    Args:
        py_date: pydatetime for conversion.
        which: must be either `date` or `time` 
            (EddyPro files list them separately).
            
    Returns:
        the constructed EddyPro datetime.

    """

    return {
        'date': py_date.strftime('%Y-%m-%d'),
        'time': py_date.strftime('%H:%M:%S')
        }[which]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _TOA5_line_read_formatter(line: str) -> dt.datetime:
    """Parses a line string of TOA5 format into a list of elements. 
    
    Args:
        line: single data line from TOA5 file.

    Returns:
        list of elements (as str).

    """

    sep = FILE_CONFIGS['TOA5']['separator']
    return [x.replace('"', '') for x in line.strip().split(sep)]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _TOA5_line_write_formatter(line_list: list(str)) -> str:
    """Parses a list of elements into a line string of TOA5 format. 
    
    Args:
        line_list: list of elements (as str).

    Returns:
        single data line for TOA5 file.

    """

    formatter = lambda x: f'"{x}"'
    joiner = lambda x: FILE_CONFIGS['TOA5']['separator'].join(x) + '\n'
    return joiner(formatter(x) for x in line_list)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _TOA5_date_read_formatter(line_list: list) -> dt.datetime:
    """Parses a list of date strings of TOA5 format into a pydatetime. 
    
    Args:
        line_list: separate elements of an TOA5 data line string.

    Returns:
        the constructed python datetime.

    """

    locs = FILE_CONFIGS['TOA5']['time_variables'].values()
    return _generic_date_constructor(
        date_elems=[line_list[loc] for loc in locs]
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _TOA5_date_write_formatter(py_date, which='TIMESTAMP'):
    """Parses a pydatetime into a string of TOA5 datetime format. 
    
    Args:
        py_date: pydatetime for conversion.
        which: must be either `date` or `time` 
            (TOA5 files list them separately).
            
    Returns:
        the constructed TOA5 datetime.

    """

    return py_date.strftime('%Y-%m-%d %H:%M:%S')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _generic_date_constructor(date_elems: list(str)) -> dt.datetime:
    """Construct a date from a list of date elements using the prescribed date 
    format.

    Args:
        date_elems: the date elements to be combined.

    Returns:
        Python datetime.

    """

    return dt.datetime.strptime(' '.join(date_elems), DATE_FORMAT)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def reformat_data(data: pd.DataFrame, output_format: str) -> pd.DataFrame:
    """Take a dataset of generic intermediary data format and turn it into 
    either TOA5 or EddyPro data.

    Args:
        data: the data to be reformatted.
        output_format: the output format (must be `TOA5` or `EddyPro`).

    Returns:
        the reformatted data.  
        
    Raises:
        TypeError: Raised if the data does not have a datetime index.

    """

    # Initialisation stuff
    if not isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise TypeError('Passed data must have a DatetimeIndex!')
    _check_format(fmt=output_format)
    funcs_dict = {'TOA5': _TOA5ify_data, 'EddyPro': _EPify_data}
    df = data.copy()

    # Remove all format-specific data columns to make data format-agnostic.
    for fmt in FILE_CONFIGS.keys():
        for var in FILE_CONFIGS[fmt]['time_variables']:
            try:
                df.drop(var, axis=1, inplace=True)
            except KeyError:
                continue
    return funcs_dict[output_format](data=df)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _TOA5ify_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert data to TOA5 format.

    Args:
        data: the data to be reformatted.

    Returns:
        the reformatted data.  
    
    """

    # Create the date outputs
    formatter = get_formatter(file_type='TOA5', which='write_date')
    date_series = pd.Series(data.index.to_pydatetime(), index=data.index)
    data.insert(0, 'TIMESTAMP', date_series.apply(formatter))
    return data
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _EPify_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert data to EddyPro format.

    Args:
        data: the data to be reformatted.

    Returns:
        the reformatted data.  
    
    """

    # Check whether DATAH and filename columns are present - if so, leave them
    # if not (e.g. if source file is TOA5), insert them!
    non_num_strings = {'DATAH': 'DATA', 'filename': 'none'}
    for i, var in enumerate(['DATAH', 'filename']):
        if not var in data.columns:
            data.insert(i, var, non_num_strings[var])

    # Create the date outputs and put them in slots 2 and 3
    formatter = get_formatter(file_type='EddyPro', which='write_date')
    date_series = pd.Series(data.index.to_pydatetime(), index=data.index)
    i = 2
    for var in FILE_CONFIGS['EddyPro']['time_variables'].keys():
        series = date_series.apply(formatter, which=var)
        data.insert(i, var, series)
        i += 1

    return data
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def reformat_headers(headers: pd.DataFrame, output_format: str) -> pd.DataFrame:
    """Create formatted header from dataframe header.

    Args:
        headers: the headers to be reformatted.
        output_format: the output format (must be `TOA5` or `EddyPro`).

    Returns:
        thereformatted headers.
    
    """

    # Initialisation stuff
    _check_format(fmt=output_format)
    funcs_dict = {'TOA5': _TOA5ify_headers, 'EddyPro': _EPify_headers}
    df = headers.copy()

    # Remove all format-specific header columns to make header format-agnostic.
    for fmt in FILE_CONFIGS.keys():
        for var in FILE_CONFIGS[fmt]['non_numeric_cols']:
            try:
                df.drop(var, inplace=True)
            except KeyError:
                continue

    # Pass to function to format header as appropriate
    return funcs_dict[output_format](headers=df)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _EPify_headers(headers: pd.DataFrame) -> pd.DataFrame:
    """Convert headers to EddyPro format.

    Args:
      headers: the headers to be reformatted.

    Returns:
        the reformatted headers.
    
    """

    # Create EddyPro-specific headers
    add_df = pd.DataFrame(
        data={'units': ['DATAU', '', '[yyyy-mm-dd]', '[HH:MM]']},
        index=pd.Index(['DATAH', 'filename', 'date', 'time'], name='variable'),
        )

    # Drop the sampling header line if it exists
    if 'sampling' in headers.columns:
        headers.drop('sampling', axis=1, inplace=True)

    # Concatenate and return
    return pd.concat([add_df, headers])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _TOA5ify_headers(headers):
    """Convert headers to TOA5 format.

    Args:
      headers: the headers to be reformatted.

    Returns:
        the reformatted headers.
    
    """

    # Create TOA5-specific headers
    add_df = pd.DataFrame(
        data={'units': 'TS', 'sampling': ''},
        index=pd.Index(['TIMESTAMP'], name='variable'),
        )

    # Add the sampling header line if it doesn't exist
    if not 'sampling' in headers.columns:
        headers = headers.assign(sampling='')
    headers.sampling.fillna('', inplace=True)

    # Concatenate and return
    return pd.concat([add_df, headers])
#------------------------------------------------------------------------------

###############################################################################
### END FILE FORMATTING FUNCTIONS ###
###############################################################################



###############################################################################
### BEGIN DATE HANDLING FUNCTIONS ###
###############################################################################



#------------------------------------------------------------------------------
def get_dates(
        file: str | pathlib.Path, file_type: str=None
        ) -> list(dt.datetime):
    """Date parser only.

    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched. Defaults to None.

    Returns:
        list of dates.
    
    """

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)

    # Get the time variables and rows to skip
    time_vars = list(FILE_CONFIGS[file_type]['time_variables'].keys())
    rows_to_skip = list(set(
        [0] +
        list(FILE_CONFIGS[file_type]['header_lines'].values())
        ))
    rows_to_skip.remove(FILE_CONFIGS[file_type]['header_lines']['variable'])
    separator = FILE_CONFIGS[file_type]['separator']

    # Get the data
    df = pd.read_csv(
        file,
        usecols=time_vars,
        skiprows=rows_to_skip,
        parse_dates={'DATETIME': time_vars},
        sep=separator,
        )

    # Check the date parser worked - if not, fix it
    if df.DATETIME.dtype == 'object':
        df.DATETIME = pd.to_datetime(df.DATETIME, errors='coerce')
        df.dropna(inplace=True)

    # Return the dates as pydatetimes
    return pd.DatetimeIndex(df.DATETIME).to_pydatetime()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_start_end_dates(file: str | pathlib.Path, file_type: str=None) -> dict:
    """Get start and end dates only.

    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched. Defaults to None.

    Returns:
        dictionary containing start and end dates.
    
    """

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)

    # Get the formatters
    line_formatter = get_formatter(file_type=file_type, which='read_line')
    date_formatter = get_formatter(file_type=file_type, which='read_date')

    # Open file in binary
    with open(file, 'rb') as f:

        # Iterate forward to find first valid start date
        start_date = None
        for line in f:
            try:
                start_date = date_formatter(line_formatter(line.decode()))
                break
            except ValueError:
                continue

        # Iterate backwards to find last valid end date
        end_date = None
        f.seek(2, os.SEEK_END)
        while True:
            try:
                if f.read(1) == b'\n':
                    pos = f.tell()
                    try:
                        end_date = (
                            date_formatter(
                                line_formatter(f.readline().decode())
                                )
                            )
                        break
                    except ValueError:
                        f.seek(pos - f.tell(), os.SEEK_CUR)
                f.seek(-2, os.SEEK_CUR)
            except OSError:
                break

    return {'start_date': start_date, 'end_date': end_date}
#------------------------------------------------------------------------------



###############################################################################
### END DATE HANDLING FUNCTIONS ###
###############################################################################



###############################################################################
### BEGIN CONCATENATION FILE RETRIEVAL FUNCTIONS ###
###############################################################################



#------------------------------------------------------------------------------
def get_eligible_concat_files(
        file: str | pathlib.Path, file_type: str=None
        ) -> list:
    """Get the list of files that can be concatenated, based on file and file type.

    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched. Defaults to None.

    Returns:
        list of available backup files.
    
    """

    funcs_dict = {
        'TOA5': get_TOA5_backups,
        'EddyPro': get_EddyPro_files
        }

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)

    # Return the files
    return funcs_dict[file_type](file=file)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_TOA5_backups(file: str | pathlib.Path) -> list:
    """Get the list of TOA5 backup files that can be concatenated.

    Args:
        file: absolute path of file to parse.

    Returns:
        list of available backup files.
    
    """

    file_to_parse = _check_file_exists(file=file)
    return list(file_to_parse.parent.glob(f'{file_to_parse.stem}*.backup'))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_EddyPro_files(file: str | pathlib.Path) -> list:
    """Get the list of EddyPro summary files that can be concatenated.

    Args:
        file: absolute path of file to parse.

    Returns:
        list of available backup files.
   
    """

    file_to_parse = _check_file_exists(file=file)
    file_list = list(file_to_parse.parent.glob(f'*{EDDYPRO_SEARCH_STR}.txt'))
    file_list.sort()
    if file in file_list:
        file_list.remove(file_to_parse)
    return file_list
#------------------------------------------------------------------------------



###############################################################################
### END CONCATENATION FILE RETRIEVAL FUNCTIONS ###
###############################################################################



###############################################################################
### BEGIN FILE INTERVAL FUNCTIONS ###
###############################################################################



#------------------------------------------------------------------------------
def get_file_interval(file: str | pathlib.Path, file_type: str=None) -> int:
    """Find the file interval (i.e. time step)

    Args:
        file: absolute path of file to parse.
        file_type: if specified, must be either `TOA5` or 
            `EddyPro`. If None, file_type is fetched. Defaults to None.

    Returns:
        the inferred file interval.
    
    """

    # If file type not supplied, detect it.
    if not file_type:
        file_type = get_file_type(file)
    return get_datearray_interval(
        datearray=np.unique(np.array(get_dates(file=file, file_type=file_type)))
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_datearray_interval(datearray: np.typing.ArrayLike) -> int:
    """Attempts to infer the likely time interval from non-monotonic time stamps.

    Args:
      datearray: the date array from which to infer the interval.

    Returns:
      the inferred interval.

    Raises:
      RuntimeError: raised if the minimum and most common values are not the
      same.

    """


    if len(datearray) == 1:
        return None
    datearray = np.unique(datearray)
    deltas, counts = np.unique(datearray[1:] - datearray[:-1], return_counts=True)
    minimum_val = deltas[0].seconds / 60
    common_val = (deltas[np.where(counts==counts.max())][0]).seconds / 60
    if minimum_val == common_val:
        return int(minimum_val)
    raise RuntimeError('Minimum and most common values do not coincide!')
#------------------------------------------------------------------------------



###############################################################################
### END FILE INTERVAL FUNCTIONS ###
###############################################################################



###############################################################################
### BEGIN FILE CHECKING FUNCTIONS ###
###############################################################################



#------------------------------------------------------------------------------
def _check_file_exists(file):
    """Check path is valid.

    Args:
      file: absolute path of file to parse.

    Returns:
        None

    """

    file_to_parse = pathlib.Path(file)
    if not file_to_parse.exists():
        raise FileNotFoundError('Passed file does not exist!')
    return file_to_parse
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _check_format(fmt):
    """Check format is valid.

    Args:
      fmt: the format string. 

    Raises:
        NotImplementedError: raised if format not recognised.
        
    Returns:
        None

    """

    if not fmt in FILE_CONFIGS.keys():
        raise NotImplementedError(f'Format {fmt} is not implemented!')
#------------------------------------------------------------------------------

###############################################################################
### END FILE CHECKING FUNCTIONS ###
###############################################################################
