#%% imports
import os
from dotenv import loadenv
import json
from csv import reader
#%%

loadenv()

def parse_discord():

    # %% validate message
    
    def is_valid(msg):
        invalid = ['!', '?']
        return msg != "" and msg[0] not in invalid
    # %% paths
    file_names = ['channel1_csv',
    'channel2_csv',
    'channel3_csv']

    output_path = '../data_uncompiled/uncompiled/'
    raw_path = '../data_uncompiled/raw_data/'

    # %% filtering messages
    user = os.getenv('DISC_ID')
    for name in file_names:
        data_path = os.path.join(raw_path, name + '.csv')
        #open output file
        out = os.path.join(output_path, name + '.txt')
        out_file = open(out, 'w', encoding='Latin1')
        #get messages from csv
        
        with open(data_path, 'r', encoding='Latin1') as csv_file:
            csv_reader = reader(csv_file)
            for row in csv_reader:
                text = row[3]
                if row[1] == user and is_valid(text):
                    out_file.write(text + '\n')

#%%
def parse_telegram():
    # %% Setting up output directory
    if not(os.path.exists(os.path.join('data', 'uncompiled'))):
        os.makedirs(os.path.join('data', 'uncompiled'))
    # %% paths to telegram data
    output_path = '../data_uncompiled/uncompiled/'
    raw_path = '../data_uncompiled/raw_data/'
    data_paths = ['Telgram1.json',
    'Telgram2.json',
    'Telgram3.json',
    'Telgram4.json',
]

    # %% filtering
    names = ["gasprobs", "gas"]
    for path in data_paths:
        # %% Open output file
        out = os.path.join(output_path, path + '.txt')
        out_file = open(out, 'w', encoding='Latin1')
        # %% get messages
        data_path = os.path.join(raw_path,path)
        with open(data_path, 'r', encoding='Latin1') as data_file:
            json_dict = json.load(data_file)
        message_lst = json_dict['messages']

        # %% parse message lst
        for msg in message_lst:
            if msg['type'] == "message":
                text = msg['text']
                if msg['from'] in names and text != "" and type(text) == str:
                    out_file.write(text + '\n')
        out_file.close()

# %% Auto filter messages by length and content
def acceptable_MSG(msg):
    return (msg.count(' ') > 1 or len(msg) > 20) and (not('http' in msg) and msg[0] not in ['\\', '['])

def filter():
    out_file = open('../data_uncompiled/unsanitized.txt', 'w', encoding='Latin1')
    for filename in os.listdir('../data_uncompiled/uncompiled/'):
        with open('../data_uncompiled/uncompiled/'+filename, 'r', encoding='Latin1') as file:
            prev_msg = None
            for msg in file:
                #get rid of blocks of repeating text
                #get rid of /tts and 'tts
                if(acceptable_MSG(msg)):
                    msg = msg.replace('/tts','')
                    msg = msg.replace('\'tts','')
                    if(not(prev_msg == msg)):
                        out_file.write(msg)
                        prev_msg = msg

#%% create dataset
def create_dataset():
    with open('../data_uncompiled/sanitized.txt', 'r', encoding='Latin1') as sanitized:
        count = 1
        for msg in sanitized:
            data = data.strip()
            out_file = open('../data/msg'+str(count) + '.txt', 'w', encoding='Latin1')
            out_file.write(msg)
            count+=1

# create_dataset()

#%%
def createGPT2Data():
    raw_path = '../data_uncompiled/raw_data/'
    data_paths = ['Telgram1.json',
    'Telgram2.json',
    'Telgram3.json',
    'Telgram4.json',
]

    
    for path in data_paths:
        with open(os.path.join(raw_path,path),'r',encoding='utf-8') as f:
            messages = json.load(f)
        messages = messages['messages']

        punctuation_arr = ['.','?','!']
        prev_sender = None
        formatted_text = ""
        print(messages[0])
        for msg in messages:
            if msg['type'] != 'message':
                pass
            else:
                curr_msg = msg['text']
                curr_sender = msg['from']
                if type(curr_msg) !=  str or len(curr_msg) == 0 or(prev_sender == None and (curr_sender[0] == 'A')):
                    pass
                else:
                    if not curr_msg[-1] in punctuation_arr:
                        curr_msg += '.'
                    curr_msg += ' '
                    if curr_sender != prev_sender:
                        formatted_text+="\n"
                        if curr_sender[0] == 'A':
                            formatted_text+="A: " + curr_msg
                        else:
                            formatted_text +=  "\nUser: " + curr_msg
                        prev_sender = curr_sender
                    else:
                        formatted_text += curr_msg
        
        with open(os.path.join("../data/","gpt2_train.txt"),'a',encoding='utf-8') as out_file:
            out_file.write(formatted_text)

createGPT2Data()
# %%
