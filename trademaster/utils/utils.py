import inspect
import os
import re

import mmcv
from mmcv import Config
from mmcv.utils import Registry
from mmcv.utils import print_log
import numpy as np
import prettytable
import plotly.graph_objects as go
import os.path as osp
import pickle
from scipy.stats import norm
def print_metrics(stats):
    table = prettytable.PrettyTable()
    for key, value in stats.items():
        table.add_column(key, value)
    return table

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        df.info()
    return df


def get_attr(args, key=None, default_value=None):
    if isinstance(args, dict):
        return args[key] if key in args else default_value
    elif isinstance(args, object):
        return getattr(args, key, default_value) if key is not None else default_value


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')
    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')


def update_data_root(cfg, logger=None):
    """Update data root according to env FINTECH_DATASETS.

    If set env FINTECH_DATASETS, update cfg.data_root according to
    MMDET_DATASETS. Otherwise, using cfg.data_root as default.

    Args:
        cfg (mmcv.Config): The model config need to modify
        logger (logging.Logger | str | None): the way to print msg
    """
    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    if 'FINTECH_DATASETS' in os.environ:
        dst_root = os.environ['FINTECH_DATASETS']
        print_log(f'FINTECH_DATASETS has been set to be {dst_root}.'
                  f'Using {dst_root} as data root.')
    else:
        return

    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    def update(cfg, src_str, dst_str):
        for k, v in cfg.items():
            if isinstance(v, mmcv.ConfigDict):
                update(cfg[k], src_str, dst_str)
            if isinstance(v, str) and src_str in v:
                cfg[k] = v.replace(src_str, dst_str)

    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root


def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmcv.utils.config.Config):
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmcv.utils.config.Config]:
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(
        replace_value(ori_cfg._cfg_dict), filename=ori_cfg.filename)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg

def evaluate_metrics(scores_dicts):
    Excess_Profit_list = []
    daily_return_list = []
    tr_list = []
    mdd_list = []
    cr_list = []
    for scores_dict in scores_dicts:
        Excess_Profit_list.append(scores_dict['Excess Profit'])
        # print('scores_dict["total_assets"] ',scores_dict["total_assets"].shape,scores_dict["total_assets"][-1],scores_dict["total_assets"][0])
        tr_list.append(
            scores_dict["total_assets"][-1] / (scores_dict["total_assets"][0] + 1e-10) - 1)
        daily_return_list.append(scores_dict["daily_return"])
        mdd = max((max(scores_dict["total_assets"]) - scores_dict["total_assets"]) / (
            max(scores_dict["total_assets"])) + 1e-10)
        mdd_list.append(mdd)
        cr_list.append(np.sum(scores_dict["daily_return"]) / (mdd + 1e-10))
    output_dict={}
    output_dict['Excess_Profit'] = sum(Excess_Profit_list) / len(Excess_Profit_list)
    output_dict['tr'] = sum(tr_list) / len(tr_list)
    daily_return_merged = np.concatenate(daily_return_list, axis=0)
    output_dict['sharpe_ratio'] = np.mean(daily_return_merged) / (
                np.std(daily_return_merged) * (len(daily_return_merged) ** 0.5) + 1e-10)
    output_dict['vol'] = np.std(daily_return_merged)
    output_dict['mdd'] = sum(mdd_list) / len(mdd_list)
    output_dict['cr'] = sum(cr_list) / len(cr_list)
    neg_ret_lst = daily_return_merged[daily_return_merged < 0]
    output_dict['sor'] = np.sum(daily_return_merged) / (np.nan_to_num(np.std(neg_ret_lst),0) + 1e-10) / (
                np.sqrt(len(daily_return_merged)) + 1e-10)
    return output_dict

def create_radar_score_baseline(dir_name,metric_path):
    # get 0-score metrics
    # noted that for Mdd and Volatility, the lower, the better.
    # So the 0-score metric for Mdd and Volatility here is actually 100-score

    # We assume that the score of all policy range within  (-100,100)
    # Do Nonthing policy will score 0
    # the baseline policy(Blind Buy for now) should score 50(-50 if worse than Do Nothing)
    # The distribution of the score of policies is a normal distribution
    # The Do Nothing policy is 0.5 percentile and baseline policy should be the 0.75 percentile(0.675 sigma away from Do Nothing)
    # Then we can score policies based on the conversion of sigma and metric value
    metric_path_zero=metric_path + '_Do_Nothing'
    zero_scores_files = [osp.join(dir_name,filename) for filename in os.listdir(dir_name) if filename.startswith(metric_path_zero)]
    zero_scores_dicts =[]
    for file in zero_scores_files:
        with open(file, 'rb') as f:
            zero_scores_dicts.append(pickle.load(f))
    # get 50-score metrics
    metric_path_fifty=metric_path + '_Blind_Bid'
    fifty_scores_files = [osp.join(dir_name,filename) for filename in os.listdir(dir_name) if filename.startswith(metric_path_fifty)]
    fifty_scores_dicts =[]
    for file in fifty_scores_files:
        with open(file, 'rb') as f:
            fifty_scores_dicts.append(pickle.load(f))
    # We only assume the daily return follows normal distribution so to give a overall metric across multiple tests we will calculate the metrics here.
    zero_metrics=evaluate_metrics(zero_scores_dicts)
    # print('fifty_scores_dicts: ',fifty_scores_dicts)
    fifty_metrics=evaluate_metrics(fifty_scores_dicts)
    # print(zero_metrics,fifty_metrics)

    metrics_sigma_dict={}
    metrics_sigma_dict['Excess_Profit']=abs(zero_metrics['Excess_Profit']-fifty_metrics['Excess_Profit'])/0.675
    metrics_sigma_dict['tr']=abs(zero_metrics['tr']-fifty_metrics['tr'])/0.675
    metrics_sigma_dict['sharpe_ratio']=abs(zero_metrics['sharpe_ratio']-fifty_metrics['sharpe_ratio'])/0.675
    # vol and mdd for Do_Nothing is score 100(3 sigma)
    metrics_sigma_dict['vol']=abs(zero_metrics['vol']-fifty_metrics['vol'])/(3-0.675)
    metrics_sigma_dict['mdd']=abs(zero_metrics['mdd']-fifty_metrics['mdd'])/(3-0.675)
    metrics_sigma_dict['cr']=abs(zero_metrics['cr']-fifty_metrics['cr'])/0.675
    metrics_sigma_dict['sor']=abs(zero_metrics['sor']-fifty_metrics['sor'])/0.675
    return metrics_sigma_dict,zero_metrics


def calculate_radar_score(dir_name,metric_path,agent_id,metrics_sigma_dict,zero_metrics):
    metric_path = metric_path + '_'+agent_id
    # print(metric_path)
    # print(os.listdir(dir_name))
    test_scores_files = [osp.join(dir_name,filename) for filename in os.listdir(dir_name) if filename.startswith(metric_path)]
    test_scores_dicts = []
    for file in test_scores_files:
        with open(file, 'rb') as f:
            test_scores_dicts.append(pickle.load(f))
    # print('test_scores_dicts:',test_scores_dicts)
    test_metrics=evaluate_metrics(test_scores_dicts)
    #turn metrics to sigma
    profit_metric_names=['Excess_Profit','tr','sharpe_ratio','cr','sor']
    risk_metric_names = ['vol', 'mdd']
    test_metrics_scores_dict={}
    for metric_name in profit_metric_names:
        test_metrics_scores_dict[metric_name]=norm.cdf((test_metrics[metric_name]-zero_metrics[metric_name])/metrics_sigma_dict[metric_name])*200-100
    for metric_name in risk_metric_names:
        test_metrics_scores_dict[metric_name] = norm.cdf(
           3-(test_metrics[metric_name] - zero_metrics[metric_name]) / metrics_sigma_dict[metric_name]) * 200-100
    test_metrics_scores_dict["Profitability"] = (test_metrics_scores_dict["tr"] + test_metrics_scores_dict["sharpe_ratio"] + test_metrics_scores_dict["cr"] + test_metrics_scores_dict["sor"]) / 4
    test_metrics_scores_dict["Risk Control"] = (test_metrics_scores_dict["mdd"] + test_metrics_scores_dict["vol"]) / 2
    return test_metrics_scores_dict

def plot_radar_chart(data,id,radar_save_path):
    data_list_profit=[]
    data_list_risk=[]
    for metric in ['Excess_Profit','sharpe_ratio','cr','sor']:
        data_list_profit.append(data[metric])
    for metric in ['vol','mdd']:
        data_list_risk.append(data[metric])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=data_list_profit+data_list_risk,
        theta=['Excess Profit', 'Sharp Ratio',
               'Calmar Ratio','Sortino Ratio']+['Volatility', 'Max Drawdown'],
        fill=None,
        line_color='peru'
    ))
    fig.add_trace(go.Barpolar(
    r=[sum(data_list_profit)/len(data_list_profit),sum(data_list_risk)/len(data_list_risk)],
    theta=[90,270],
    width=[60,30],
    marker_color=["#E4FF87", '#709BFF'],
    marker_line_color="black",
    marker_line_width=2,
    opacity=0.8
))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[-110,110],
                visible=True
            ),
            angularaxis=dict(showticklabels=False, ticks='')
        ),
        showlegend=False
    )
    # fig.show()
    radar_save_path+='_'+id+'.png'
    print('Radar plot printed to:',radar_save_path)
    fig.write_image(radar_save_path)


