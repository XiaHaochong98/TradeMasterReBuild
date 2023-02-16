import labeling_util as util
import argparse
import pandas as pd
import os

def main(args):
    print('labeling start')
    output_path = args.data_path
    if args.OE_BTC == True:
        raw_data = pd.read_csv(args.data_path)
        raw_data['tic'] = 'OE_BTC'
        raw_data['adjcp'] = raw_data["midpoint"]
        raw_data['date'] = raw_data["system_time"]
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        raw_data.to_csv('./temp/OE_BTC_processed.csv')
        args.data_path = './temp/OE_BTC_processed.csv'
    Labeler = util.Labeler(args.data_path, 'linear')
    Labeler.fit(args.regime_number, args.length_limit)
    Labeler.label(args.labeling_parameters)
    labeled_data = pd.concat([v for v in Labeler.data_dict.values()], axis=0)
    data = pd.read_csv(args.data_path)
    merged_data = data.merge(labeled_data, how='left', on=['date', 'tic', 'adjcp'], suffixes=('', '_DROP')).filter(
        regex='^(?!.*_DROP)')
    low, high = args.labeling_parameters
    if args.PM != 'False':
        DJI = merged_data.loc[:, ['date', 'label']]
        test = pd.read_csv(args.PM, index_col=0)
        merged = test.merge(DJI, on='date')
        process_datafile_path=output_path[:-4] + '_label_by_DJIindex_' + str(args.regime_number) + '_' + str(
            args.length_limit) + '_' + str(low) + '_' + str(high) + '.csv'
        merged.to_csv(process_datafile_path, index=False)
    else:
        process_datafile_path=output_path[:-4] + '_labeled_' + str(args.regime_number) + '_' + str(args.length_limit) + '_' + str(
                low) + '_' + str(high) + '.csv'
        merged_data.to_csv(process_datafile_path
            , index=False)
    print('labeling done')
    print('plotting start')
    # a list the path to all the modeling visulizations
    market_dynamic_labeling_visulization_paths=[]
    market_dynamic_labeling_visulization_paths.append(Labeler.plot(Labeler.tics, args.labeling_parameters, output_path))
    print('plotting done')
    if args.OE_BTC == True:
        os.remove('./temp/OE_BTC_processed.csv')
    return process_datafile_path,market_dynamic_labeling_visulization_paths

def MRL_F2B_args_converter(args):
    #TODO:
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--fitting_parameters",nargs='+', type=str)
    parser.add_argument("--labeling_parameters",  nargs="+", type=float)
    parser.add_argument('--regime_number',type=int,default=4)
    parser.add_argument('--length_limit',type=int,default=0)
    parser.add_argument('--OE_BTC',type=bool,default=False)
    parser.add_argument('--PM',type=str,default='False')
    args= parser.parse_args()
    main(args)