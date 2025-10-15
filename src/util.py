from termcolor import colored

def log_title(message: str):
    """
    打印带样式的标题
    """
    total_length = 120
    message_length = len(message)
    padding = max(0, total_length - message_length - 4)
    padded_message = f"{'=' * (padding // 2)} {message} {'=' * ((padding + 1) // 2)}"
    print(colored(padded_message, 'cyan', attrs=['bold']))