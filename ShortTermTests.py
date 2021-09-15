import sys
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)

def test_9(data, N=4):
    # Calculate the number of cumulative missing values?
    missing = data.isnull().astype(int).groupby(data.notnull().astype(int).cumsum()).cumsum()
    C2 = max(missing)
    print(C2)
    # Compare to limit N
    if C2>=N:
        flag = 4
    else:
        flag = 1
    return flag
	
def test_10(data, num_stdev=4):
    mean_value = data.mean()
    stdev_value = data.std()
    results = np.full(data.shape, 4)
    fail = abs(data - mean_value) > num_stdev * stdev_value
    results[~fail] = 1
    return results
	
def test_11(data, IMIN=-2000, IMAX=2000, LMIN=-1050, LMAX=1050):
    """
    NOTE: 
    * Default local limits are based on past 2 years of M6 data 
        from new buoy - maximum Hmax is ~21m.
    * Defauly instrument limits based on Datawell Heave range
        -20m to +20m
    """
    results = []
    for val in data:
        if (val>IMAX or val<IMIN):
            results.append(4)
        elif (val>LMAX or val<LMIN):
            results.append(3)
        else:
            results.append(1)
    return results
	

def ST_tests(df, fname):
    for param in ['Heave', 'North', 'West']:
        qc_df = df[param].to_frame()
        print("Parameter ", param)
        print("Test 9 result: ", test_9(df[param]))
        print("-"*100)
        qc_df['test10'] = test_10(df[param])
        qc_df['test11'] = test_11(df[param])
        qc_df.to_csv('{}_qc_{}.csv'.format(fname, param.lower()))
		

if __name__ == "__main__":
    if len(sys.argv) == 2:
        datafile = sys.argv[1]
        print("Performing QC on : ", datafile)
    else:
        print("No file found")
        sys.exit(1)

    try:
        df = pd.read_csv(datafile)
    except FileNotFoundError:
        msg = "File {} not found?".format(datafile)
        raise FileNotFoundError(msg)
    df.set_index('Time', inplace=True)
    ST_tests(df, datafile)

