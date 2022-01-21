import logging
logging.basicConfig(level=logging.disable(logging.DEBUG))

import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import datetime

os.environ["SLACK_TOKEN"] = "xoxb-2980952661189-3007645529936-QWbRBC3zCdXR30DitUZr26Xq"

client = WebClient(token=os.getenv("SLACK_TOKEN"))
logger = logging.getLogger(__name__)

def list_channel_id(channel_name):
    conversation_id = None
    try:
        for response in client.conversations_list():
            if conversation_id is not None:
                break
            for channel in response["channels"]:
                if channel["name"] == channel_name:
                    conversation_id = channel["id"]
                    #Print result
                    print(f"Found conversation ID: {conversation_id}")
                    break
    except SlackApiError as e:
        print(f"Error: {e}")

def list_channels_in_workspace():
    conversation_id = None
    channel_list = {}
    try:
        for response in client.conversations_list():
            if conversation_id is not None:
                break
            for channel in response["channels"]:
                channel_id = channel["id"]
                channel_name = channel["name"]
                #Print result
                #print(f"{channel_name}: {channel_id}")
                channel_list[channel_name] = channel_id

    except SlackApiError as e:
        print(f"Error: {e}")

    return channel_list

def list_conversation_history(channel_id):
    # Store conversation history
    conversation_history = []
    conversation_history_list = {}
    try:
        result = client.conversations_history(channel = channel_id)

        conversation_history = result["messages"]

        for ch_info in conversation_history:
            timestamp = ch_info["ts"]
            user = ch_info["user"]
            text = ch_info["text"]
            human_time = convert_to_human_datetime(timestamp)
            #print(f"{timestamp}: {user} >> {text}")
            conversation_history_list[timestamp] = [human_time, user, text]

        # Print results
        logger.info("{} messages found in {}".format(len(conversation_history), id))

    except SlackApiError as e:
        logger.error("Error creating conversation: {}".format(e))

    return conversation_history_list

def convert_to_human_datetime(timestamp):
    ts = int(timestamp.split('.')[0])
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def create_channel(channel_name):
    response = client.conversations_create(
        channel_name, is_private = False)
    return response["channel"]["id"]

#list_channel_id("bot-lab")
print(list_conversation_history("C0307J0BL5N"))
#list_channels_in_workspace()

