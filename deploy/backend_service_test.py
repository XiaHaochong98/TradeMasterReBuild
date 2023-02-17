import os
import sys
import uuid
from flask import Flask, request, jsonify
import time
import logging
import pytz
import json
from tools import market_dynamics_labeling
import base64
from pathlib import Path
import pickle
ROOT = str(Path(__file__).resolve().parents[1])
import os.path as osp
from mmcv import Config
from trademaster.utils import replace_cfg_vals
import subprocess
import pandas as pd
tz = pytz.timezone('Asia/Shanghai')

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
app = Flask(__name__)
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

def logger():
    logger = logging.getLogger("server")
    logger.setLevel(level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    if console not in logger.handlers:
        logger.addHandler(console)
    return logger
def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    command_output = process.stdout.read().decode('utf-8')
    return command_output

logger = logger()


class Server():
    def __init__(self, debug=True):
        if debug:
            self.debug()
        pass

    def debug(self):
        pass

    def train_scripts(self, task_name, dataset_name, optimizer_name, loss_name, agent_name):
        if task_name == "algorithmic_trading":
            return os.path.join(ROOT, "tools", "algorithmic_trading", "train.py")
        elif task_name == "order_execution":
            if agent_name == "eteo":
                return os.path.join(ROOT, "tools", "order_execution", "train_eteo.py")
            elif agent_name == "pd":
                return os.path.join(ROOT, "tools", "order_execution", "train_pd.py")
        elif task_name == "portfolio_management":
            if dataset_name == "dj30":
                if agent_name == "deeptrader":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_deeptrader.py")
                elif agent_name == "eiie":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_eiie.py")
                elif agent_name == "investor_imitator":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_investor_imitator.py")
                elif agent_name == "sarl":
                    return os.path.join(ROOT, "tools", "portfolio_management", "train_sarl.py")
            elif dataset_name == "exchange":
                return os.path.join(ROOT, "tools", "portfolio_management", "train.py")

    def get_parameters(self, request):
        logger.info("get_parameters start.")
        res = {
            "task_name": ["algorithmic_trading", "order_execution", "portfolio_management"],
            "dataset_name": ["algorithmic_trading:BTC",
                             "order_excecution:BTC",
                             "portfolio_management:dj30",
                             "portfolio_management:exchange"],
            "optimizer_name": ["adam", "adaw"],
            "loss_name": ["mae", "mse"],
            "agent_name": [
                "algorithmic_trading:dqn",
                "order_execution:eteo",
                "order_execution:pd",
                "portfolio_management:a2c",
                "portfolio_management:ddpg",
                "portfolio_management:deeptrader",
                "portfolio_management:eiie",
                "portfolio_management:investor_imitator",
                "portfolio_management:pg",
                "portfolio_management:ppo",
                "portfolio_management:sac",
                "portfolio_management:sarl",
                "portfolio_management:td3"
            ]
        }
        logger.info("get_parameters end.")
        return jsonify(res)


    def start(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name")
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name")

            session_id = str(uuid.uuid1())

            error_code = 0
            info = "request success"
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)

    def start_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")

            error_code = 0
            info = "test for start status"
            res = {
                "info": info,
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)

    def test(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")

            error_code = 0
            info = "request success"
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": session_id
            }
            logger.info(info)
            return jsonify(res)

    def test_status(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:

            session_id = request_json.get("session_id")

            error_code = 0
            info = "test for test status"
            res = {
                "info": info,
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": ""
            }
            logger.info(info)
            return jsonify(res)

    def start_market_dynamics_labeling(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            # market_dynamics_labeling parameters
            args={}
            args['dataset_name'] = request_json.get("style_test_dataset_name")
            args['number_of_market_dynamics'] = request_json.get("number_of_market_style")
            if args['number_of_market_dynamics'] not in [3,4]:
                raise Exception('only support dynamics number of 3 or 4 for now')
            args['minimun_length'] = request_json.get("minimun_length")
            args['Granularity'] = request_json.get("Granularity")
            args['bear_threshold'] = request_json.get("bear_threshold")
            args['bull_threshold'] = request_json.get("bull_threshold")
            args['task_name']=request_json.get("task_name")
            # agent training parameters
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name").split(":")[-1]
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name").split(":")[-1]
            session_id= request_json.get("session_id")
            work_dir = os.path.join(ROOT, "work_dir", session_id,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")

            cfg_path = os.path.join(ROOT, "configs", task_name,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}.py")
            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)
            cfg.work_dir = "work_dir/{}/{}".format(session_id,
                                                   f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")
            cfg.trainer.work_dir = cfg.work_dir

            #prepare data
            test_start_date = request_json.get("style_test_start_date")
            test_end_date = request_json.get("style_test_end_date")

            data = pd.read_csv(os.path.join(ROOT, cfg.data.data_path, "data.csv"), index_col=0)
            data = data[(data["date"] >= test_start_date) & (data["date"] < test_end_date)]
            data_path=os.path.join(work_dir, "style_test.csv")
            data.to_csv(data_path)
            args['dataset_path']=data_path


            #front-end args to back-end args
            args=market_dynamics_labeling.MRL_F2B_args_converter(args)

            #run market_dynamics_labeling
            process_datafile_path,market_dynamic_labeling_visualization_paths=market_dynamics_labeling.main(args)

            #TODO: Find a better way to pick a visulization for PM task

            with open(market_dynamic_labeling_visualization_paths[0], "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            # update session information:
            with open(os.path.join(work_dir,'style_test_data_path.pickle') , 'wb') as handle:
                pickle.dump(process_datafile_path, handle, protocol=pickle.HIGHEST_PROTOCOL)

            error_code = 0
            info = "request success, show market dynamics labeling visualization"

            res = {
                "error_code": error_code,
                "info": info,
                "market_dynamic_labeling_visulization": encoded_string
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "market_dynamic_labeling_visulization": ""
            }
            logger.info(info)
            return jsonify(res)

    def save_market_dynamics_labeling(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            # same as agent training
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name").split(":")[-1]
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name").split(":")[-1]
            session_id = request_json.get("session_id")
            work_dir = os.path.join(ROOT, "work_dir", session_id,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")
            with open(os.path.join(work_dir,'style_test_data_path.pickle') , 'wb') as f:
                process_datafile_path=pickle.load(f)
            cfg_path = os.path.join(ROOT, "configs", task_name,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}.py")
            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)
            # build dataset
            cfg.data.test_style_path = process_datafile_path
            cfg_path = os.path.join(work_dir, osp.basename(cfg_path))
            cfg.dump(cfg_path)
            logger.info(cfg)
            error_code = 0
            info = "request success, save market dynamics"
            res = {
                "error_code": error_code,
                "info": info
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info
            }
            logger.info(info)
            return jsonify(res)


    def run_style_test(self, request):
        request_json = json.loads(request.get_data(as_text=True))
        try:
            #
            style_test_label = request_json.get("test_dynamic_label")
            # same as agent training
            task_name = request_json.get("task_name")
            dataset_name = request_json.get("dataset_name").split(":")[-1]
            optimizer_name = request_json.get("optimizer_name")
            loss_name = request_json.get("loss_name")
            agent_name = request_json.get("agent_name").split(":")[-1]
            session_id = request_json.get("session_id")
            work_dir = os.path.join(ROOT, "work_dir", session_id,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}")

            cfg_path = os.path.join(ROOT, "configs", task_name,
                                    f"{task_name}_{dataset_name}_{agent_name}_{agent_name}_{optimizer_name}_{loss_name}.py")
            cfg = Config.fromfile(cfg_path)
            cfg = replace_cfg_vals(cfg)
            cfg_path = os.path.join(work_dir, osp.basename(cfg_path))
            log_path = os.path.join(work_dir, "style_test_"+str(style_test_label)+"_log.txt")
            train_script_path = self.train_scripts(task_name, dataset_name, optimizer_name, loss_name, agent_name)
            cmd = "conda activate python3.9 && nohup python -u {} --config {} --task_name style_test --test_style {} > {} 2>&1 &".format(
                train_script_path,
                cfg_path,
                style_test_label,
                log_path)
            executor.submit(run_cmd, cmd)
            logger.info(cmd)

            radar_plot_path=osp.join(work_dir,'radar_plot_agent_'+str(style_test_label)+'.png')
            with open(radar_plot_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            #print log output
            print_log_cmd = "tail -n 2000 {}".format(log_path)
            style_test_log_info = run_cmd(print_log_cmd)


            error_code = 0
            info = f"request success, start test market {style_test_label}\n\n"
            res = {
                "error_code": error_code,
                "info": info+style_test_log_info,
                "session_id": session_id,
                'radar_plot':encoded_string
            }
            logger.info(info)
            return jsonify(res)

        except Exception as e:
            error_code = 1
            info = "request data error, {}".format(e)
            res = {
                "error_code": error_code,
                "info": info,
                "session_id": "",
                'radar_plot':""
            }
            logger.info(info)
            return jsonify(res)


class HealthCheck():
    def __init__(self):
        super(HealthCheck, self).__init__()

    def run(self, request):
        start = time.time()
        if request.method == "GET":
            logger.info("health check start.")
            error_code = 0
            info = "health check"
            time_consuming = (time.time() - start) * 1000
            res = {
                "data": {},
                "error_code": error_code,
                "info": info,
                "time_consuming": time_consuming
            }
            logger.info("health check end.")
            return jsonify(res)

SERVER = Server()
HEALTHCHECK = HealthCheck()

@app.route("/TradeMaster/getParameters", methods=["GET"])
def getParameters():
    res = SERVER.get_parameters(request)
    return res

@app.route("/TradeMaster/start", methods=["POST"])
def start():
    res = SERVER.start(request)
    return res

@app.route("/TradeMaster/start_status", methods=["POST"])
def start_status():
    res = SERVER.start_status(request)
    return res

@app.route("/TradeMaster/test", methods=["POST"])
def test():
    res = SERVER.start(request)
    return res

@app.route("/TradeMaster/test_status", methods=["POST"])
def test_status():
    res = SERVER.start_status(request)
    return res

@app.route("/TradeMaster/healthcheck", methods=["GET"])
def health_check():
    res = HEALTHCHECK.run(request)
    return res


if __name__ == "__main__":
    # host = "0.0.0.0"
    # port = 8080
    # app.run(host, port)

    server=Server()
    Request_message_1= {
        "task_name": "algorithmic_trading",
        "dataset_name": "algorithmic_trading:BTC",
        "optimizer_name": "adam",
        "start_date": "2017-08-08",
        "end_date": "2018-08-08",
        "loss_name": "mse",
        "agent_name": "algorithmic_trading:dqn",
        "style_test_dataset_name": "algorithmic_trading:BTC",
        "number_of_market_style": "3",
        "style_test_start_date": "2018-08-09",
        "style_test_end_date": "2018-09-08",
        "minimun_length": "24",
        "Granularity": "0.5",
        "bear_threshold": "-0.25",
        "bull_threshold": "0.25",
        "session_id": "b5bcd0b6-7a10-11ea-8367-181 dea4d9837"
    }

    Request_message_1= {
        "task_name": "algorithmic_trading",
        "dataset_name": "algorithmic_trading:BTC",
        "optimizer_name": "adam",
        "start_date": "2017-08-08",
        "end_date": "2018-08-08",
        "loss_name": "mse",
        "agent_name": "algorithmic_trading:dqn",
        "style_test_dataset_name": "algorithmic_trading:BTC",
        "number_of_market_style": "3",
        "style_test_start_date": "2018-08-09",
        "style_test_end_date": "2018-09-08",
        "minimun_length": "24",
        "Granularity": "1",
        "bear_threshold": "-0.25",
        "bull_threshold": "0.25",
        "session_id": "b5bcd0b6-7a10-11ea-8367-181 dea4d9837"
    }


    server.start_market_dynamics_labeling(Request_message_1)
    server.start_market_dynamics_labeling()
    server.save_market_dynamics_labeling()
    server.run_style_test()
