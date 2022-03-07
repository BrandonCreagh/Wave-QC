"""
Implementation of relevant Long Term Tests defined in QARTOD manual
	for wave data.

More information: https://ioos.noaa.gov/project/qartod/

TODO:
- Need to add tests 17? Add parameters for test 17 to metadata file?
- Air pressure, SST tests?
"""
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'

class LongTerm():
    """Long Term QC tests Module"""

    def __init__(self, new_inds):
        """Long Term tests"""
        self.new_inds = new_inds

    @classmethod
    def test_missing(cls, data):
        """
        Test flags missing data.
        Arguments:
        - data: series from dataframe, with data which failed
            previous QC runs masked out.

        Returns:
        - result: numpy array of 0 (data present) or 8 (data missing)
            which is written to the detailed QC report "param_missing" column.
        """
        # Initialise all data as missing
        result = np.full(data.shape, 8)
        # Where data are not nan, set flag to 0
        result[~np.isnan(data)] = 0
        return result

    @classmethod
    def test_mean_stdev(cls, data, num_stdevs=4):
        """
        LT Test 15 checks that data falls within an operator defined
            range of the mean.
        Generally 24 hours of data is used to calculate means and standard
            deviations, therefore results from this test will be less stable
            for new stations or after data outages.

        Arguments:
        - data: series from dataframe, with data which failed
            previous QC runs masked out.
        - num_stdevs: integer defining the number of standard deviations
             from the mean which is deemed acceptable. Defaults to 4.

        Returns:
        - result: numpy array of 0 (pass), 3 (suspect) or 4 (fail)
            which is written to the detailed QC report "param_15" column.
        """
        # Calculate Mean and Standard deviation over data
        #  Note: data that failed QC are masked out in "data" and thus excluded
        mean = data.mean()
        stdev = data.std()

        # Create masks of data falling further than N stdevs from mean
        fail1 = data < (mean - num_stdevs * stdev)
        fail2 = data > (mean + num_stdevs * stdev)
        full_fail = fail1 | fail2

        # Initialise all data as failing
        result = np.full(data.shape, 4)
        # Where data are within acceptable range, set flag to 0
        result[~full_fail] = 0
        return result

    @classmethod
    def test_flatline(cls, data, sus_flat=3, fail_flat=5, eps=0.01):
        """
        LT Test 16 checks whether data is flatlining.

        Arguments:
        - data: series from dataframe, with data which failed
            previous QC runs masked out.
        - sus_flat: optional integer defining the number of consequetive identical
            records to define a suspected flatline
        - fail_flat: optional integer defining the number of consequetive identical
            records to justify a failure for flatlining
        - eps: optional float defining tolerance level when testing difference
            between adjacent values.

        Returns:
        - result: numpy array of 0 (no flatline), 3 (sus_flat elements in a row within
            epsilon tolerance) or 4 (fail_flat elements in a row within epsilon tolerance)
            which is written to the detailed QC report "param_16" column.
        """

        def _test_flat(data, flat_length, eps=0.01):
            """
            Private function for testing flatline over n records.

            Arguments:
            - data: array of data values to search for flatlines
            - flat_length: the number of concurrent records which would constitute a
                flatline if they are identical
            - eps: threshold value for values being equal

            Returns:
            - Boolean array containing "True" wherever a flatline is detected
            """
            # Only values of flat_length >= 2 are valid, print a warning and reset
            #    to 2 if required
            if flat_length < 2:
                print("flat_length<2 is invalid; set sus_flat, fail_flat to N>1")
                flat_length = 2

            # Find cumulative difference over flat_length iterations
            sum_flat = 0
            for diff_size in range(1, flat_length):
                sum_flat = sum_flat + abs(data.diff(diff_size))
            # Return a boolean array
            return sum_flat < eps * abs((flat_length - 1))

        # Calculate the absolute difference between each element in the array
        #   and the elements fail_flat and sus_flat slots behind it.
        fail = _test_flat(data, fail_flat, eps)
        suspect = _test_flat(data, sus_flat, eps)

        # Initialise all data as passing
        result = np.full(data.shape, 0)
        # Where data are suspect, set flag to 3
        result[suspect] = 3
        # Where data meet fail criteria, set flag to 4
        result[fail] = 4
        return result


    @classmethod
    def test_feasible_range(cls, data, min_value, max_value, critical=False):
        """
        LT Test 19 checks that data fits into a range defined by the
            user. This range is defined in the metadata file for each
            station and for each parameter.

        Arguments:
        - data: series from dataframe, with data which failed
            previous QC runs masked out.
        - min_value: numeric, feasible min value, values less than
             this fail QC
        - max_value: numeric, feasible max value, values greater than
             this fail QC
        - critical: flag which if true sets values outside the range to
             fail QC, and if false, sets values outside range to suspect.
             See QARTOD manual on in-Situ wave observations, test 19 for
             more information.

        Returns:
        - results: numpy array of 0 (values within feasible range), 3 (values
            outside feasible range but test not critical parameter), 4 (values
            outside feasible range and critical for this parameter).
        """
        # Calculate failure mask - where data value is less than minimum
        #  value or data value is greater than maximum value
        fail = (data < min_value) | (data > max_value)

        # Initialise all data as passing
        result = np.full(data.shape, 0)

        # Set values which fail check to failing (if critical) or suspect
        if critical:
            result[fail] = 4
        else:
            result[fail] = 3
        return result

    @classmethod
    def test_rate_of_change(cls, data, delta=3, deg_diff=False):
        """
        LT Test 20 checks if the 1 timestep rate of change exceeds a
            threshold value, delta.

        Arguments:
        - data: series from dataframe, with data which failed
            previous QC runs masked out.
        - delta: the rate of change acceptable within 1 timestep. This
            should be defined considering the parameter and the frequency
            of data.
        - deg_diff: if set to True, the difference will be calculated for
            angles (0-360 degrees). Default value is False.

        Returns:
        - numpy array of 0 (rate of change less than delta) or 4 (rate
            of change greater than or equal to delta).
        """
        def _degree_differencing(data):
            """
            Function to calculate the degree difference between 2 angles.

            Ex: For wave directions of 350 and 10 degrees, data.diff() would
                return a difference of 340, whilst this function will calculate
                a difference of 20 degrees.
            """
            # Copy the data series to a frame and create a column of data from the
            #   previous timestep to compare against.
            diff_data = data.copy().to_frame()
            diff_data['shift'] = diff_data.shift(1)
            # Calculate basic difference between timesteps (equivalent to data.diff())
            diff_data['diff1'] = data - diff_data['shift']
            # Next 2 calculations assume 1 angle is on either side of the 0/360 divide
            diff_data['diff2'] = (data + 360) - diff_data['shift']
            diff_data['diff3'] = data - (diff_data['shift']+ 360)
            # Find the minimum, absolute angle between the 2 values
            diff_data['diff'] = np.abs(diff_data[['diff1','diff2','diff3']]).min(axis=1)
            return diff_data['diff']

        # Calculate mask of data failing test, data.diff calculates the
        #   difference between value data[i] and data[i-1]
        if not deg_diff:
            fail = abs(data.diff()) >= delta
        # For degrees, need to account for differences in direction that
        #   span the 0/360 degree divide
        else:
            fail = _degree_differencing(data) >= delta
        # Initialise all data as passing
        result = np.full(data.shape, 0)
        # Set values which fail check to 4
        result[fail] = 4
        return result

    def run(self, df, params, station_metadata):
        """
        Control for QC on params in self.df
        returns detailed QC report for file and summary of qc'd data
        """
        clean_cols = params + [x+'_qc' for x in params]
        print("{} new records".format(len(self.new_inds)))
        # Copy data and archive flags from df
        report = df[params]

        # For each param:
        # 1. Mask data from archive which failed QC
        # 2. Run range of QC tests and construct detailed report of results
        # 3. Coalesce QC into 1 flag per value:
        #       8 = missing, 4 = fail, 3 = suspect, 0 = pass
        for param in params:
            # Read metadata for this param
            param_meta_cols = [x for x in station_metadata.index if param in x]
            if len(param_meta_cols) > 0:
                param_meta = station_metadata[param_meta_cols]
            else:
                param_meta = None

            # 1. Mask data from archive which failed QC
            if param + '_qc' in df.columns:
                failed_qc = df[param+'_qc'] > 3
            else:
                failed_qc = np.full(df[param].shape, False)
            data = df[param][~failed_qc]

            # 2. Initialise new columns to fail
            tests = ['missing', '15', '16', '19', '20']
            param_test_cols = [param + '_' + x for x in tests]
            for col in param_test_cols:
                report[col] = np.full(df.index.shape, 0)

            # 2.1 Check for nans
            report[param + '_missing'][~failed_qc] = LongTerm.test_missing(data)

            # 2.2 test 15
            report[param + '_15'][~failed_qc] = LongTerm.test_mean_stdev(data)

            # 2.3 test 16
            t16_sus = param + '_flatsuspect'
            t16_fail = param + '_flatfail'
            if t16_sus in param_meta_cols:
                sus_flat = param_meta[t16_sus]
            else:
                print("Using default value of 3 for suspect flatline definition")
                sus_flat = 3
            if t16_fail in param_meta_cols:
                fail_flat = param_meta[t16_fail]
            else:
                print("Using default value of 5 for failing flatline definition")
                fail_flat = 5

            report[param + '_16'][~failed_qc] = LongTerm.test_flatline(
                data, sus_flat=sus_flat, fail_flat=fail_flat)

            # 2.4 test 19
            t19_meta = [param + '_min', param + '_max', param + '_critical']
            if sum([1 for x in t19_meta if x in param_meta_cols]) == 3:
                min_meta = param_meta[t19_meta[0]]
                max_meta = param_meta[t19_meta[1]]
                critical_meta = param_meta[t19_meta[2]]
                report[param + '_19'][~failed_qc] = LongTerm.test_feasible_range(
                    data, min_meta, max_meta, critical_meta)
            else:
                print("Insufficient metadata to run test 19 on {}".format(param))
                report[param + '_19'][~failed_qc] = 0

            # 2.5 test 20
            t20_meta = param + '_roc'
            if t20_meta in param_meta_cols:
                delta_meta = param_meta[t20_meta]
                # Note that a different algorithm is used to calculate degree
                #   differences to find the true angle between directions
                if "dir" in param:
                    report[param + '_20'][~failed_qc] = LongTerm.test_rate_of_change(
                        data, delta_meta, deg_diff=True)
                else:
                    report[param + '_20'][~failed_qc] = LongTerm.test_rate_of_change(
                        data, delta_meta)
            else:
                print("Insufficient metadata to run test 20 on {}".format(param))
                report[param + '_20'][~failed_qc] = 0

            # Coalesce QC results for this parameter
            report[param + '_qc'] = report[param_test_cols].max(axis=1)

        # If hm0 fails test 19, both mdir and tm02 are flagged as failing QC
        report['mdir_qc'][report['hm0_19'] == 4] = 4
        report['tm02_qc'][report['hm0_19'] == 4] = 4

        # Create summary report with only raw data and overal QC flag
        report_clean = report[clean_cols]
        return report, report_clean


if __name__ == "__main__":
    if len(sys.argv) == 2:
        FNAME = sys.argv[1]
        print("Performing QC on : ", FNAME)
    else:
        print("No file found")
        sys.exit(1)

    try:
        READ_DATA = pd.read_csv(FNAME)
    except FileNotFoundError as file_not_found:
        MSG = "File {} not found?".format(FNAME)
        raise FileNotFoundError(MSG) from file_not_found

    PARAMS = ['hm0', 'mdir', 'tm02']
    DETAIL_REPORT, SUMMARY = LongTerm(new_inds=READ_DATA.index).run(READ_DATA, PARAMS)
    DETAIL_REPORT.to_csv('detailed_report.csv')
    SUMMARY.to_csv('clean_data.csv')
