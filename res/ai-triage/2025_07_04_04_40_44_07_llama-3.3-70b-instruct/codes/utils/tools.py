import argparse, os, shutil, sys, yaml
from datetime import datetime

import smtplib
from box import Box
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def get_cur_time():
    # year_month_day_hour_minute_seconds
    cur_time = datetime.now()
    cur_time = '{:%Y_%m_%d_%H_%M_%S}_{:02.0f}'.format(cur_time, 
                                                      cur_time.microsecond / 10000.0)
    return cur_time

def ignore_pycache(dir, files):
    return [f for f in files if f == '__pycache__']

#*------------------------------------------------------------------------------

def create_folders(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
    print("----Finish creating folders: ", end='')
    print(*paths, sep=', ')

def parse_agrs(config_path="./opt.yaml"):
    # parse all args from config file (yaml) and update save_dir with current time
    with open(config_path, "r") as f:
        opt = Box(yaml.safe_load(f))
    
    opt.cur_time    = get_cur_time()
    opt.save_dir    = os.path.join(opt.save_dir, f"{opt.cur_time}_{opt.exp_name}")
    
    opt.save_dir   += f"_{opt.task}"
    opt.dataroot    = os.path.join(opt.dataroot, opt.task.upper())

    opt.weights_dir = os.path.join(opt.save_dir, "weights")
    opt.codes_dir   = os.path.join(opt.save_dir, "codes")
    opt.logs_dir    = os.path.join(opt.save_dir, "logs")
    opt.imgs_dir    = os.path.join(opt.save_dir, "imgs")
    print("----Finish parsing args.")
    return opt

def read_from_cml():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--location",
        type=str,
        choices=["server", "local"],
        required=True,
        help="Training location: 'server' or 'local'"
    )
    return parser.parse_args()

def save_codes_args(source_paths, opt, save_dir, codes_dir):
    # Save some important code files and arguments
    # 'util/' 'train.py' 'opt.yaml' etc.
    
    for path in source_paths:
        if os.path.exists(path):
            dest_path = os.path.join(codes_dir, path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            if os.path.isfile(path):
                shutil.copy2(path, dest_path)
            elif os.path.isdir(path): # 如果是folder，cp everything under the folder
                shutil.copytree(path, dest_path, dirs_exist_ok=True, ignore=ignore_pycache) # 覆盖
        else:
            raise ValueError(f"Warning: {path} does not exist and will not be copied.")

    try:
        with open(os.path.join(save_dir, 'opt.yaml'), 'w') as f:
            yaml.safe_dump(opt.to_dict(), f, sort_keys=False)
    except Exception as e:
        raise ValueError(f"[Error] Failed to save arguments: {e}")
    print("----Finish saving codes and args.")
    
def send_email(subject, body, sender_email, receiver_email, app_password):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))
    # QQ SMTP server
    server = smtplib.SMTP_SSL('smtp.qq.com', 465)
    server.login(sender_email, app_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()


