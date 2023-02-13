from utility.module import requests

def read_from():
    f = open('./utility/info/from.txt', 'r')
    data = f.read()
    f.close()
    return data

def read_token():
    f = open('./utility/info/token.txt', 'r')
    data = f.read()
    f.close()
    return data

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    from_text = read_from()
    line_notify_token = read_token()
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'{from_text}:{notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)