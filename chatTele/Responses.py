from datetime import datetime

def sample_responses(input_text):
    user_message = str(input_text).lower()

    if user_message in ("hello", "hi",):
        return "Hey! How's it going?"

    if user_message in ("who are you","who are you?",):
        return "No?"

    if user_message in ("hello", "time?",):
        now = datetime.now()
        date_time = now.strftime("%d/%m/%y, %H:%M:%S")

        return date_time

    return "idk you"

