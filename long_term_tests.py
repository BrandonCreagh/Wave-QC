"""
Implementation of relevant Long Term Tests defined in QARTOD manual
	for wave data.

More information: https://ioos.noaa.gov/project/qartod/

TODO:
- Tests 17, 19, 20 refactor into separate functions
- Develop metadata file containing station/variable specific parameters
    (max/min etc)
- Inter-parameter dependencies: ex if hm0 fails test 19, tm02 and mdir
    are also meant to fail
- Air pressure, SST tests?
"""
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'

class LongTerm(object):
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
    def test_flatline(cls, data, eps=0.01):
        """
        LT Test 16 checks whether data is flatlining.

        Arguments:
        - data: series from dataframe, with data which failed
            previous QC runs masked out.
        - eps: optional float defining tolerance level when testing difference
            between adjacent values.

        Returns:
        - result: numpy array of 0 (no flatline), 3 (3 elements in a row within
            epsilon tolerance) or 4 (5 elements in a row within epsilon tolerance)
            which is written to the detailed QC report "param_16" column.
        """
        # Calculate the absolute difference between each element in the array
        #   and the elements 1-4 slots behind it.
        fail = (abs(data.diff(1)) + abs(data.diff(2)) + abs(data.diff(3))
                + abs(data.diff(4))) < 4 * eps
        suspect = abs(data.diff(1)) + abs(data.diff(2)) < 2 * eps

        # Initialise all data as passing
        result = np.full(data.shape, 0)
        # Where data are suspect, set flag to 3
        result[suspect] = 3
        # Where data meet fail criteria, set flag to 4
        result[fail] = 4
        return result

    def LT_tests(df, param):
        """
        Main Long Term Tests function
        Ideally would split each test into separate function
        """
        N = 4
        results = []
        for i in range(len(df)):
            j = 0
            t19 = 0
            t20 = 0
            #test 19
            #hm0,tm02,mdir
            minwh = 0
            maxwh = 30 #look for climatological ranges - east/west??
            minwp = 0
            maxwp = 18 #This is a guess, not sure what the largest wave period should be
            minwd = 0.0
            maxwd = 360
            wvpd_max = max(df.tm02)
            wvpd_min = min(df.tm02)
            wvdir_max = max(df.mdir)
            wvdir_min = min(df.mdir)
            if df.hm0[i] < minwh or df.hm0[i] >= maxwh:
                j = 4
                t19 = 1
            elif wvpd_min < minwp or wvpd_max > maxwp or wvdir_min < minwd or maxwd > 360:
                j = 3
                t19 = 1
            #test 20
            #spike test
            #hm0
            delta = 3
            if i == len(df) - 1:
                break
            if abs(df.hm0[i] - df.hm0[i+1]) >= delta:
                j = 4
                t20 = 1

            results.append([df.index[i], df.hm0[i], df.tm02[i], df.mdir[i],
                            j, t15, t16, t19, t20])
        columns = [df.index.name, "hm0", "tm02", "mdir", "flag", "test_15",
                   "test_16", "test_19", "test_20"]
        results_df = pd.DataFrame(results, columns=columns)
            #results_df.to_csv('QC.csv', index=False)
        #print(results_df)
        results_df.set_index(df.index.name, inplace=True)
        return results_df

    def run(self, df, params):
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
            # 1. Mask data from archive which failed QC
            if param + '_qc' in df.columns:
                failed_qc = df[param+'_qc'] > 3
            else:
                failed_qc = np.full(df[param].shape, False)         
            data = df[param][~failed_qc]
            # 2. Initialise new columns to fail
            tests = ['missing', '15', '16']#, '19', '20']
            param_test_cols = [param + '_' + x for x in tests]
            for col in param_test_cols:
                report[col] = np.full(df.index.shape, 0)

            # 2.1 Check for nans
            report[param + '_missing'][~failed_qc] = LongTerm.test_missing(data)

            # 2.2 test 15
            report[param + '_15'][~failed_qc] = LongTerm.test_mean_stdev(data)

            # 2.3 test 16
            report[param + '_16'][~failed_qc] = LongTerm.test_flatline(data)

            # 2.4 test 19...

            # 2.5 test 20...

            # Coalesce QC results for this parameter
            report[param + '_qc'] = report[param_test_cols].max(axis=1)
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
