import time
import logging  # 引入logging模块
import os.path
import datetime
import inspect
class Logger:
    def __init__(self,mode='w'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.getcwd() + '/Logs/'
        log_name = log_path + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

class shape:
    @staticmethod
    def log_var_shapes(**kwargs):
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"tensor_shapes_{timestamp}.txt"
        current_file = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(current_file, log_file)
        with open(log_file, "w") as file:
            file.write(f"Tensor shapes logged from: {caller_file}, line {caller_line}\n\n")
            for var_name, var_value in kwargs.items():
                if hasattr(var_value, 'shape'):
                    var_shape = var_value.shape
                    file.write(f"{var_name}: {var_shape}\n")
                elif hasattr(var_value, 'size'):
                    var_shape = var_value.size()
                    file.write(f"{var_name}: {var_shape}\n")
                else: 
                    file.write(f"{var_name}: Not a tensor, the value is{var_value}\n")
        print(f"Tensor shapes logged to: {os.path.abspath(log_file)}")